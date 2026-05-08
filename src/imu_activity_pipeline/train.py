"""Sequential training pipeline for multi-scale IMU activity models.

Purpose:
    Trains the configured window-scale models one after another using balanced
    sampling, Mixup, focal loss, triplet loss, label smoothing, validation-based
    checkpointing, and ensemble metadata generation.
Inputs:
    Reads training signals and annotations through `sensor_data_processing.py` and training
    hyperparameters from `config.py`.
Outputs:
    Writes checkpoints, normalization parameters, training curves, training
    summaries, and `ensemble_config.json` under `saved_models/`.
"""
import os
import sys
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter

from .config import *
from .sensor_data_processing import (load_sensor_data, load_gold_labels, normalize_imu,
                        create_windows, assign_window_labels, apply_augmentation)
from .neural_network_models import CombinedModel, FocalLoss, TripletLoss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class IMUWindowDataset(Dataset):
    """Window dataset with optional online augmentation."""

    def __init__(self, windows, labels, augment=False):
        self.windows = windows  # numpy array
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.augment:
            window = apply_augmentation(window, p=0.5)
        return torch.FloatTensor(window), self.labels[idx]


def mixup_data(x, y, alpha=AUG_MIXUP_ALPHA):
    """Apply Mixup augmentation at batch level.
    Returns mixed inputs, pairs of targets, and mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for Mixup: weighted combination of losses for both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def prepare_training_data(window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
    """Load all training data with Butterworth filtering.

    Args:
        window_size: number of samples per window
        window_step: step size between windows
    """
    print("=" * 60)
    print(f"STEP 1: Loading training data (window_size={window_size}, step={window_step})...")
    print("=" * 60)

    gold_labels = load_gold_labels(TRAIN_ANNOTATIONS_FILE)
    print(f"Gold labels: {len(gold_labels)} segments from {gold_labels['user_id'].nunique()} users")

    all_windows = []
    all_labels = []
    all_user_ids = []

    train_dir = TRAIN_DATA_DIR
    users = sorted([f.replace('.txt', '') for f in os.listdir(train_dir) if f.endswith('.txt')])
    gold_users = set(gold_labels['user_id'].unique())
    users = [u for u in users if u in gold_users]
    print(f"Users with gold labels: {len(users)}")

    skipped = 0
    for i, user_id in enumerate(users):
        file_path = os.path.join(train_dir, f"{user_id}.txt")

        with open(file_path, 'rb') as f:
            header_bytes = f.read(100)
        try:
            header_text = header_bytes.decode('utf-8')
            if 'ACC_TIME' not in header_text:
                skipped += 1
                continue
        except:
            skipped += 1
            continue

        data = load_sensor_data(file_path, apply_filter=True)
        if data is None or len(data) < window_size:
            skipped += 1
            continue

        timestamps, windows = create_windows(data, window_size, window_step)
        if len(windows) == 0:
            skipped += 1
            continue

        binary_labels, class_labels = assign_window_labels(timestamps, gold_labels, user_id)
        labels_6class = np.where(class_labels >= 0, class_labels + 1, 0)

        all_windows.append(windows)
        all_labels.append(labels_6class)
        all_user_ids.extend([user_id] * len(timestamps))

        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(users)} users")

    print(f"Skipped {skipped} users (binary format or read error)")
    print(f"Loaded {len(all_windows)} users")

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Total windows: {len(all_labels)}")
    label_names = ['Background'] + ACTIVITIES
    for i in range(6):
        count = np.sum(all_labels == i)
        print(f"  {label_names[i]}: {count} ({100 * count / len(all_labels):.1f}%)")

    return all_windows, all_labels, all_user_ids


def get_training_artifact_paths(model_name):
    """Return all history/curve artifact paths for a model."""
    artifact_dir = os.path.join(MODEL_DIR, "training_artifacts", model_name)
    os.makedirs(artifact_dir, exist_ok=True)
    return {
        "dir": artifact_dir,
        "history_json": os.path.join(artifact_dir, "history.json"),
        "history_csv": os.path.join(artifact_dir, "history.csv"),
        "curve_png": os.path.join(artifact_dir, "curves.png"),
        "curve_pdf": os.path.join(artifact_dir, "curves.pdf"),
        "summary_json": os.path.join(artifact_dir, "summary.json"),
    }


def persist_training_history(model_name, history, summary=None):
    """Persist per-epoch history to JSON/CSV so training progress survives interruptions."""
    paths = get_training_artifact_paths(model_name)
    with open(paths["history_json"], "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)
    if summary is not None:
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return paths


def save_training_curves(model_name, history):
    """Render training loss, evaluation metrics, and LR decay curves."""
    if not history:
        return None

    paths = get_training_artifact_paths(model_name)
    df = pd.DataFrame(history)
    epochs = df["epoch"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(epochs, df["train_loss"], marker="o", linewidth=1.8, label="Train Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} Training Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, df["val_f1"], marker="o", linewidth=1.8, label="Val Macro-F1")
    axes[1].plot(epochs, df["val_acc"], marker="s", linewidth=1.4, label="Val Acc")
    axes[1].plot(epochs, df["train_acc"], marker="^", linewidth=1.2, label="Train Acc")
    axes[1].plot(epochs, df["act_acc"], marker="d", linewidth=1.2, label="Act Acc")
    axes[1].plot(epochs, df["act_recall"], marker="x", linewidth=1.2, label="Act Recall")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f"{model_name} Evaluation Metrics")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, fontsize=9)

    axes[2].plot(epochs, df["lr"], marker="o", linewidth=1.8, color="tab:green", label="Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title(f"{model_name} Learning Rate Decay")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(paths["curve_png"], dpi=180, bbox_inches="tight")
    fig.savefig(paths["curve_pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def train_single_model(X_train, y_train, X_val, y_val, seed, model_name, window_size=WINDOW_SIZE):
    """Train a single model with Focal Loss + Triplet Loss + Mixup + Label Smoothing.

    The total objective is CE loss plus weighted focal and triplet losses.
    """
    print(f"\n{'=' * 60}")
    print(f"Training model: {model_name} (seed={seed}, window_size={window_size})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    # Apply augmentation only to the training split.
    train_dataset = IMUWindowDataset(X_train, y_train, augment=True)
    val_dataset = IMUWindowDataset(X_val, y_val, augment=False)

    # Class-balanced sampler
    class_counts = Counter(y_train.tolist())
    total = len(y_train)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = [class_weights[y] for y in y_train.tolist()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = CombinedModel(input_channels=6, num_classes=6, window_size=window_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Total loss combines CE, focal loss, and triplet loss.
    weight_tensor = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(6)]).to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=LABEL_SMOOTHING)
    focal_criterion = FocalLoss(alpha=weight_tensor, gamma=STAGE2_FOCAL_GAMMA)
    triplet_criterion = TripletLoss(margin=STAGE2_TRIPLET_MARGIN)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_STAGE2, eta_min=1e-6)

    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0
    history = []
    run_started_at = datetime.now().astimezone().isoformat()

    print(
        "Training schedule: "
        f"epochs={NUM_EPOCHS_STAGE2}, "
        f"patience={EARLY_STOPPING_PATIENCE}, "
        f"min_epochs_before_early_stop={MIN_EPOCHS_BEFORE_EARLY_STOP}"
    )

    for epoch in range(NUM_EPOCHS_STAGE2):
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Apply Mixup augmentation
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=AUG_MIXUP_ALPHA)

            optimizer.zero_grad()

            # Forward with embeddings for Triplet Loss
            logits, embeddings = model(mixed_x, return_embedding=True)

            # Compute the mixed-label objective after Mixup.
            loss_ce = mixup_criterion(ce_criterion, logits, y_a, y_b, lam)
            loss_focal = mixup_criterion(focal_criterion, logits, y_a, y_b, lam)
            loss_triplet = triplet_criterion(embeddings, batch_y)  # triplet uses original labels

            loss = loss_ce + 0.2 * loss_focal + 0.1 * loss_triplet

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(batch_y.numpy())

        val_preds_all = np.array(val_preds_all)
        val_labels_all = np.array(val_labels_all)

        val_acc = np.mean(val_preds_all == val_labels_all)
        val_f1 = f1_score(val_labels_all, val_preds_all, average='macro')

        act_mask = val_labels_all > 0
        act_acc = np.mean(val_preds_all[act_mask] == val_labels_all[act_mask]) if act_mask.sum() > 0 else 0
        bg_mask = val_labels_all == 0
        act_recall = np.mean(val_preds_all[act_mask] > 0) if act_mask.sum() > 0 else 0

        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS_STAGE2} | "
              f"Loss: {train_loss / train_total:.4f} | "
              f"Acc: {100 * train_correct / train_total:.1f}%/{100 * val_acc:.1f}% | "
              f"F1: {val_f1:.4f} | ActAcc: {100 * act_acc:.1f}% | ActRecall: {100 * act_recall:.1f}% | "
              f"LR: {current_lr:.6g}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'seed': seed,
                'window_size': window_size,
            }, os.path.join(MODEL_DIR, f'{model_name}.pth'))
            print(f"  -> Saved best model (F1={val_f1:.4f})")
        else:
            no_improve += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss / train_total),
            "train_acc": float(train_correct / train_total),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "act_acc": float(act_acc),
            "act_recall": float(act_recall),
            "lr": float(current_lr),
            "best_val_f1_so_far": float(best_val_f1),
            "best_epoch_so_far": int(best_epoch),
            "no_improve": int(no_improve),
        }
        history.append(epoch_record)
        persist_training_history(model_name, history)

        if (epoch + 1) >= MIN_EPOCHS_BEFORE_EARLY_STOP and no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    run_finished_at = datetime.now().astimezone().isoformat()
    summary = {
        "model_name": model_name,
        "seed": seed,
        "window_size": window_size,
        "epochs_configured": NUM_EPOCHS_STAGE2,
        "epochs_completed": len(history),
        "early_stopped": len(history) < NUM_EPOCHS_STAGE2,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "patience": EARLY_STOPPING_PATIENCE,
        "min_epochs_before_early_stop": MIN_EPOCHS_BEFORE_EARLY_STOP,
    }
    persist_training_history(model_name, history, summary=summary)
    curve_paths = save_training_curves(model_name, history)

    print(f"Best model: epoch {best_epoch}, val F1={best_val_f1:.4f}")
    if curve_paths is not None:
        print(f"Training artifacts saved to: {curve_paths['dir']}")
    return best_val_f1


def main():
    """Train all configured window-scale ensembles sequentially."""
    print("=" * 60)
    print("IMU ACTIVITY SEGMENTATION TRAINING PIPELINE")
    print("  Improvements: Mixup, Label Smoothing, Time Warp, Multi-Scale")
    print("=" * 60)

    all_ensemble_results = {}

    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"# Training {ws_suffix} window models (window_size={ws_samples})")
        print(f"{'#' * 60}")

        # Step 1: Load data (with Butterworth filtering)
        windows, labels, user_ids = prepare_training_data(
            window_size=ws_samples, window_step=ws_step
        )

        # Step 2: Normalize
        print("\nNormalizing data...")
        norm_windows, mean, std = normalize_imu(windows)
        norm_params = {'mean': mean, 'std': std}
        norm_file = os.path.join(MODEL_DIR, f'norm_params_{ws_suffix}.pkl')
        with open(norm_file, 'wb') as f:
            pickle.dump(norm_params, f)
        # Also save default 3s as the backward-compatible name
        if ws_suffix == "3s":
            with open(os.path.join(MODEL_DIR, 'norm_params.pkl'), 'wb') as f:
                pickle.dump(norm_params, f)
        del windows

        # Step 3: Split by user
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
        del norm_windows, labels

        # Step 4: Train ensemble models.
        print(f"\nTraining {ENSEMBLE_NUM_MODELS} models for {ws_suffix} ensemble...")
        for i, seed in enumerate(ENSEMBLE_SEEDS):
            model_name = f'combined_model_{ws_suffix}_seed{seed}'
            val_f1 = train_single_model(
                X_train, y_train, X_val, y_val, seed, model_name,
                window_size=ws_samples
            )
            all_ensemble_results[model_name] = val_f1

    # Save the best single 3s model as the backward-compatible default
    models_3s = {k: v for k, v in all_ensemble_results.items() if '_3s_' in k}
    if models_3s:
        best_model_name = max(models_3s, key=models_3s.get)
        import shutil
        src = os.path.join(MODEL_DIR, f'{best_model_name}.pth')
        dst = os.path.join(MODEL_DIR, 'combined_model_best.pth')
        shutil.copy2(src, dst)

    selected_models = {}
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        candidates = {k: v for k, v in all_ensemble_results.items() if f'_{ws_suffix}_' in k}
        if candidates:
            selected_models[ws_suffix] = max(candidates, key=candidates.get)

    # Save ensemble config
    ensemble_config = {
        'models': list(all_ensemble_results.keys()),
        'selected_models': selected_models,
        'val_f1': all_ensemble_results,
        'window_configs': [
            {'suffix': ws_suffix, 'window_size': ws_samples, 'window_step': ws_step, 'window_sec': ws_sec}
            for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS
        ],
    }
    with open(os.path.join(MODEL_DIR, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ENSEMBLE RESULTS:")
    for ws_sec, ws_samples, ws_step, ws_suffix in WINDOW_CONFIGS:
        print(f"\n  {ws_suffix} models:")
        for name, f1 in all_ensemble_results.items():
            if f'_{ws_suffix}_' in name:
                print(f"    {name}: val F1={f1:.4f}")
    print("=" * 60)
    print("TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
