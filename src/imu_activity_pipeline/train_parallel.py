"""Parallel training launcher for multi-scale, multi-seed ensembles.

Purpose:
    Prepares shared window datasets once per scale, then launches separate worker
    processes so multiple seed-specific models can train concurrently.
Inputs:
    Reads training data, model configuration, visible GPU settings, and optional
    `--skip-existing` command-line control.
Outputs:
    Writes per-model checkpoints, logs, normalization parameters, and final
    ensemble selection metadata under `saved_models/`.
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split

from .config import *
from .sensor_data_processing import (load_sensor_data, load_gold_labels, normalize_imu,
                        create_windows, assign_window_labels)
from .train import train_single_model


def load_checkpoint(path, map_location='cpu'):
    """Load trusted local checkpoints without torch.load safety warnings."""
    try:
        from numpy.core.multiarray import scalar as np_scalar
        torch.serialization.add_safe_globals([
            np_scalar,
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.int32)),
        ])
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _streaming_cache_paths(ws_suffix):
    """Return file paths for disk-backed arrays used by large windows."""
    base_dir = os.path.join(MODEL_DIR, f'_stream_cache_{ws_suffix}')
    return {
        'dir': base_dir,
        'meta': os.path.join(base_dir, 'meta.json'),
        'x_train': os.path.join(base_dir, 'X_train.npy'),
        'y_train': os.path.join(base_dir, 'y_train.npy'),
        'x_val': os.path.join(base_dir, 'X_val.npy'),
        'y_val': os.path.join(base_dir, 'y_val.npy'),
    }


def load_prepared_data(ws_suffix):
    """Load prepared train/val arrays.

    For 8s windows, load disk-backed .npy arrays via memmap to avoid
    materializing the full normalized cache in RAM.
    """
    if ws_suffix == '8s':
        paths = _streaming_cache_paths(ws_suffix)
        if not all(os.path.exists(paths[k]) for k in ('meta', 'x_train', 'y_train', 'x_val', 'y_val')):
            raise FileNotFoundError(f"8s streaming cache not found under {paths['dir']}")
        X_train = np.load(paths['x_train'], mmap_mode='r')
        y_train = np.load(paths['y_train'], mmap_mode='r')
        X_val = np.load(paths['x_val'], mmap_mode='r')
        y_val = np.load(paths['y_val'], mmap_mode='r')
        return X_train, y_train, X_val, y_val

    cache_file = os.path.join(MODEL_DIR, f'_cache_{ws_suffix}.npz')
    d = np.load(cache_file)
    return d['X_train'], d['y_train'], d['X_val'], d['y_val']


def prepare_and_save_data_streaming(ws_suffix, ws_samples, ws_step):
    """Prepare large-window data without creating a full in-memory normalized cache."""
    paths = _streaming_cache_paths(ws_suffix)
    if all(os.path.exists(paths[k]) for k in ('meta', 'x_train', 'y_train', 'x_val', 'y_val')):
        print(f"Loading streaming cache for {ws_suffix} from {paths['dir']}...")
        return load_prepared_data(ws_suffix)

    gold_labels = load_gold_labels(TRAIN_ANNOTATIONS_FILE)
    print("=" * 60)
    print(f"STEP 1: Loading training data (window_size={ws_samples}, step={ws_step})...")
    print("=" * 60)
    print(f"Gold labels: {len(gold_labels)} segments from {gold_labels['user_id'].nunique()} users")

    train_dir = TRAIN_DATA_DIR
    users = sorted([f.replace('.txt', '') for f in os.listdir(train_dir) if f.endswith('.txt')])
    gold_users = set(gold_labels['user_id'].unique())
    users = [u for u in users if u in gold_users]
    print(f"Users with gold labels: {len(users)}")

    skipped = 0
    valid_users = []
    user_window_counts = {}
    class_counts = np.zeros(6, dtype=np.int64)
    channel_sum = np.zeros(6, dtype=np.float64)
    channel_sumsq = np.zeros(6, dtype=np.float64)
    total_points = 0

    for i, user_id in enumerate(users):
        file_path = os.path.join(train_dir, f"{user_id}.txt")

        with open(file_path, 'rb') as f:
            header_bytes = f.read(100)
        try:
            header_text = header_bytes.decode('utf-8')
            if 'ACC_TIME' not in header_text:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        data = load_sensor_data(file_path, apply_filter=True)
        if data is None or len(data) < ws_samples:
            skipped += 1
            continue

        timestamps, windows = create_windows(data, ws_samples, ws_step)
        if len(windows) == 0:
            skipped += 1
            continue

        _, class_labels = assign_window_labels(timestamps, gold_labels, user_id)
        labels_6class = np.where(class_labels >= 0, class_labels + 1, 0)

        valid_users.append(user_id)
        user_window_counts[user_id] = len(labels_6class)
        class_counts += np.bincount(labels_6class, minlength=6)
        channel_sum += windows.sum(axis=(0, 1), dtype=np.float64)
        channel_sumsq += np.square(windows, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
        total_points += windows.shape[0] * windows.shape[1]

        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(users)} users")

    print(f"Skipped {skipped} users (binary format or read error)")
    print(f"Loaded {len(valid_users)} users")
    print(f"Total windows: {int(class_counts.sum())}")
    label_names = ['Background'] + ACTIVITIES
    for i in range(6):
        count = int(class_counts[i])
        pct = 100 * count / max(int(class_counts.sum()), 1)
        print(f"  {label_names[i]}: {count} ({pct:.1f}%)")

    mean = channel_sum / total_points
    var = np.maximum(channel_sumsq / total_points - mean ** 2, 1e-8)
    std = np.sqrt(var)

    os.makedirs(paths['dir'], exist_ok=True)
    with open(os.path.join(MODEL_DIR, f'norm_params_{ws_suffix}.pkl'), 'wb') as f:
        pickle.dump({'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}, f)

    unique_users = list(set(valid_users))
    train_users, val_users = train_test_split(unique_users, test_size=VAL_SPLIT, random_state=42)
    train_user_set = set(train_users)
    val_user_set = set(val_users)

    train_count = sum(user_window_counts[u] for u in train_users)
    val_count = sum(user_window_counts[u] for u in val_users)
    print(f"Train: {train_count} windows from {len(train_users)} users")
    print(f"Val: {val_count} windows from {len(val_users)} users")

    x_train_mm = np.lib.format.open_memmap(
        paths['x_train'], mode='w+', dtype=np.float32, shape=(train_count, ws_samples, 6)
    )
    y_train_mm = np.lib.format.open_memmap(
        paths['y_train'], mode='w+', dtype=np.int64, shape=(train_count,)
    )
    x_val_mm = np.lib.format.open_memmap(
        paths['x_val'], mode='w+', dtype=np.float32, shape=(val_count, ws_samples, 6)
    )
    y_val_mm = np.lib.format.open_memmap(
        paths['y_val'], mode='w+', dtype=np.int64, shape=(val_count,)
    )

    print("\nNormalizing data in per-user chunks...")
    train_pos = 0
    val_pos = 0
    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)

    for idx, user_id in enumerate(valid_users, start=1):
        file_path = os.path.join(train_dir, f"{user_id}.txt")
        data = load_sensor_data(file_path, apply_filter=True)
        timestamps, windows = create_windows(data, ws_samples, ws_step)
        _, class_labels = assign_window_labels(timestamps, gold_labels, user_id)
        labels_6class = np.where(class_labels >= 0, class_labels + 1, 0)
        norm_windows = ((windows - mean32) / std32).astype(np.float32, copy=False)

        n = len(labels_6class)
        if user_id in train_user_set:
            x_train_mm[train_pos:train_pos + n] = norm_windows
            y_train_mm[train_pos:train_pos + n] = labels_6class
            train_pos += n
        elif user_id in val_user_set:
            x_val_mm[val_pos:val_pos + n] = norm_windows
            y_val_mm[val_pos:val_pos + n] = labels_6class
            val_pos += n

        if idx % 10 == 0 or idx == len(valid_users):
            x_train_mm.flush()
            y_train_mm.flush()
            x_val_mm.flush()
            y_val_mm.flush()
            print(f"  Normalized {idx}/{len(valid_users)} users")

    meta = {
        'ws_suffix': ws_suffix,
        'window_size': ws_samples,
        'window_step': ws_step,
        'train_shape': [train_count, ws_samples, 6],
        'val_shape': [val_count, ws_samples, 6],
        'dtype': 'float32',
        'label_dtype': 'int64',
    }
    with open(paths['meta'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    x_train_mm.flush()
    y_train_mm.flush()
    x_val_mm.flush()
    y_val_mm.flush()

    return load_prepared_data(ws_suffix)


def prepare_and_save_data(ws_suffix, ws_samples, ws_step):
    """Load, window, normalize, and split data for a given window config.
    Save to disk for parallel workers to load."""
    if ws_suffix == '8s':
        return prepare_and_save_data_streaming(ws_suffix, ws_samples, ws_step)

    from .train import prepare_training_data

    cache_file = os.path.join(MODEL_DIR, f'_cache_{ws_suffix}.npz')

    if os.path.exists(cache_file):
        print(f"Loading cached data for {ws_suffix}...")
        return load_prepared_data(ws_suffix)

    windows, labels, user_ids = prepare_training_data(
        window_size=ws_samples, window_step=ws_step
    )

    print("\nNormalizing data...")
    norm_windows, mean, std = normalize_imu(windows)
    norm_params = {'mean': mean, 'std': std}
    norm_file = os.path.join(MODEL_DIR, f'norm_params_{ws_suffix}.pkl')
    with open(norm_file, 'wb') as f:
        pickle.dump(norm_params, f)
    if ws_suffix == "3s":
        with open(os.path.join(MODEL_DIR, 'norm_params.pkl'), 'wb') as f:
            pickle.dump(norm_params, f)
    del windows

    unique_users = list(set(user_ids))
    train_users, val_users = train_test_split(unique_users, test_size=VAL_SPLIT, random_state=42)

    user_array = np.array(user_ids)
    train_mask = np.isin(user_array, train_users)
    val_mask = np.isin(user_array, val_users)

    X_train = norm_windows[train_mask]
    y_train = labels[train_mask]
    X_val = norm_windows[val_mask]
    y_val = labels[val_mask]

    print(f"Train: {len(X_train)} windows from {len(train_users)} users")
    print(f"Val: {len(X_val)} windows from {len(val_users)} users")

    # Cache to disk
    print(f"Caching data to {cache_file}...")
    np.savez(cache_file, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    return X_train, y_train, X_val, y_val


def train_worker(gpu_id, data_file, seed, model_name, ws_samples):
    """Worker function: load data from disk, train one model."""
    print(f"[Worker {model_name}] Starting on GPU {gpu_id}...")

    # Load data
    ws_suffix = os.path.basename(data_file).replace('_cache_', '').replace('.npz', '')
    X_train, y_train, X_val, y_val = load_prepared_data(ws_suffix)

    # Train
    val_f1 = train_single_model(
        X_train, y_train, X_val, y_val, seed, model_name,
        window_size=ws_samples
    )
    print(f"[Worker {model_name}] Done! val_f1={val_f1:.4f}")
    return model_name, val_f1


def main():
    """Prepare shared data and launch parallel per-seed training workers."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip models that already have .pth files')
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL TRAINING PIPELINE")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print("=" * 60)

    visible_gpus = [gpu.strip() for gpu in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    if not visible_gpus:
        visible_gpus = [str(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

    # Collect all models to train
    all_jobs = []  # (ws_suffix, ws_samples, ws_step, seed, model_name)
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        for seed in ENSEMBLE_SEEDS:
            model_name = f'combined_model_{ws_suffix}_seed{seed}'
            model_path = os.path.join(MODEL_DIR, f'{model_name}.pth')

            if args.skip_existing and os.path.exists(model_path):
                print(f"  Skipping {model_name} (already exists)")
                continue
            all_jobs.append((ws_suffix, ws_samples, ws_step, seed, model_name))

    print(f"\nModels to train: {len(all_jobs)}")
    for _, _, _, seed, name in all_jobs:
        print(f"  - {name}")

    if not all_jobs:
        print("Nothing to train!")
        return

    # Group by window config (need same data)
    from collections import defaultdict
    groups = defaultdict(list)
    for ws_suffix, ws_samples, ws_step, seed, model_name in all_jobs:
        groups[ws_suffix].append((ws_samples, ws_step, seed, model_name))

    all_results = {}

    # Also load existing results
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        for seed in ENSEMBLE_SEEDS:
            model_name = f'combined_model_{ws_suffix}_seed{seed}'
            model_path = os.path.join(MODEL_DIR, f'{model_name}.pth')
            if os.path.exists(model_path) and model_name not in [j[4] for j in all_jobs]:
                ckpt = load_checkpoint(model_path, map_location='cpu')
                all_results[model_name] = ckpt.get('val_f1', 0)

    for ws_suffix, jobs in groups.items():
        ws_samples = jobs[0][0]
        ws_step = jobs[0][1]

        print(f"\n{'#' * 60}")
        print(f"# Preparing {ws_suffix} data, then training {len(jobs)} models in parallel")
        print(f"{'#' * 60}")

        # Prepare data (shared across all seeds)
        X_train, y_train, X_val, y_val = prepare_and_save_data(ws_suffix, ws_samples, ws_step)
        cache_file = os.path.join(MODEL_DIR, f'_cache_{ws_suffix}.npz')

        # Free memory before spawning workers
        del X_train, y_train, X_val, y_val

        # Launch parallel workers via subprocess for CUDA safety
        import subprocess
        processes = []
        for job_idx, (ws_samples_j, ws_step_j, seed, model_name) in enumerate(jobs):
            assigned_gpu = visible_gpus[job_idx % len(visible_gpus)] if visible_gpus else ""
            cmd = [
                sys.executable, '-c',
                f"""
import os
import sys
sys.path.insert(0, os.path.join('{BASE_DIR}', 'src'))
from imu_activity_pipeline.train import train_single_model
from imu_activity_pipeline.train_parallel import load_prepared_data
X_train, y_train, X_val, y_val = load_prepared_data('{ws_suffix}')
val_f1 = train_single_model(X_train, y_train, X_val, y_val,
                             {seed}, '{model_name}', window_size={ws_samples_j})
print(f'RESULT:{model_name}:{{val_f1:.4f}}')
"""
            ]
            log_file = os.path.join(MODEL_DIR, f'_log_{model_name}.txt')
            print(f"  Launching {model_name} on GPU {assigned_gpu or 'cpu'} (log: {log_file})")
            f_log = open(log_file, 'w')
            env = os.environ.copy()
            if assigned_gpu:
                env["CUDA_VISIBLE_DEVICES"] = assigned_gpu
            p = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT, env=env)
            processes.append((p, f_log, model_name, log_file))

        # Wait for all to finish
        print(f"\n  Waiting for {len(processes)} parallel training jobs...")
        for p, f_log, model_name, log_file in processes:
            p.wait()
            f_log.close()
            # Parse result
            with open(log_file, 'r') as f:
                log_content = f.read()
            for line in log_content.strip().split('\n'):
                if line.startswith('RESULT:'):
                    _, name, f1_str = line.split(':')
                    all_results[name] = float(f1_str)
            if p.returncode != 0:
                print(f"  WARNING: {model_name} exited with code {p.returncode}")
                print(f"  Check log: {log_file}")
            else:
                print(f"  {model_name} completed successfully")

    # Clean up cache files
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        cache_file = os.path.join(MODEL_DIR, f'_cache_{ws_suffix}.npz')
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Save backward-compatible best model
    models_3s = {k: v for k, v in all_results.items() if '_3s_' in k}
    if models_3s:
        import shutil
        best_model_name = max(models_3s, key=models_3s.get)
        src = os.path.join(MODEL_DIR, f'{best_model_name}.pth')
        dst = os.path.join(MODEL_DIR, 'combined_model_best.pth')
        if os.path.exists(src):
            shutil.copy2(src, dst)

    selected_models = {}
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        candidates = {k: v for k, v in all_results.items() if f'_{ws_suffix}_' in k}
        if candidates:
            selected_models[ws_suffix] = max(candidates, key=candidates.get)

    # Save ensemble config
    ensemble_config = {
        'models': list(all_results.keys()),
        'selected_models': selected_models,
        'val_f1': all_results,
        'window_configs': [
            {'suffix': ws_suffix, 'window_size': ws_samples, 'window_step': ws_step, 'window_sec': ws_sec}
            for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS
        ],
    }
    with open(os.path.join(MODEL_DIR, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ALL RESULTS:")
    for name, f1 in sorted(all_results.items()):
        print(f"  {name}: val F1={f1:.4f}")
    print("=" * 60)
    print("TRAINING COMPLETE!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
