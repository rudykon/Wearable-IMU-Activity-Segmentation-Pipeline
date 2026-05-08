"""Post-processing parameter sweep without model retraining.

Purpose:
    Searches temporal smoothing, Viterbi, gap repair, confidence, and Top-K style
    settings to understand how decoding choices affect segment-level F1.
Inputs:
    Reuses saved checkpoints, internal_eval annotations, and configured signal
    directories.
Outputs:
    Prints or saves parameter combinations and their evaluation metrics.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os, sys, copy, itertools, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

from imu_activity_pipeline.config import *
from imu_activity_pipeline.sensor_data_processing import load_gold_labels, load_sensor_data
from imu_activity_pipeline.inference import (
    load_ensemble_models, predict_multiscale_ensemble,
    smooth_predictions, extract_segments,
    merge_same_class_segments, filter_short_segments,
    resolve_overlaps, refine_boundaries,
)
from imu_activity_pipeline.evaluate import calculate_iou
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt


# ── Custom Viterbi with tunable params ──────────────────────────────────
def viterbi_tunable(probs, self_trans=0.90, bg_to_act=0.01,
                    act_to_bg=0.05, cross_act=0.005):
    N, n_states = probs.shape
    if N == 0:
        return np.array([])

    trans = np.full((n_states, n_states), 0.001)
    for i in range(n_states):
        trans[i, i] = self_trans
    for i in range(1, n_states):
        trans[0, i] = bg_to_act
        trans[i, 0] = act_to_bg
    for i in range(1, n_states):
        for j in range(1, n_states):
            if i != j:
                trans[i, j] = cross_act
    trans = trans / trans.sum(axis=1, keepdims=True)

    log_trans = np.log(trans + 1e-10)
    log_probs = np.log(probs + 1e-10)

    V = np.full((N, n_states), -np.inf)
    backpointer = np.zeros((N, n_states), dtype=int)
    V[0] = log_probs[0]

    for t in range(1, N):
        transitions = V[t-1][:, None] + log_trans
        V[t] = np.max(transitions, axis=0) + log_probs[t]
        backpointer[t] = np.argmax(transitions, axis=0)

    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(V[-1])
    for t in range(N-2, -1, -1):
        path[t] = backpointer[t+1, path[t+1]]
    return path


# ── Weighted ensemble ───────────────────────────────────────────────────
def predict_weighted_ensemble(data, model_groups, device, weight_by_f1=False):
    """Compatibility wrapper for an older sweep path.

    The project now centralizes window creation, normalization, per-scale
    prediction, timestamp alignment, and fusion in predict_multiscale_ensemble.
    Keeping this helper as a thin wrapper avoids the stale pre-refactor call
    signature that passed already-sliced IMU arrays into create_windows().
    """
    if weight_by_f1:
        print("  Note: weight_by_f1 is deprecated; using current configured fusion.")
    return predict_multiscale_ensemble(data, model_groups, device)


# ── Top-K segment selection ──────────────────────────────────────────────
def select_top_k_segments(segments, k=3):
    """Keep only top-k segments by confidence, one per class if possible."""
    if len(segments) <= k:
        return segments
    # First, pick best segment per class
    by_class = {}
    for s in segments:
        cls = s['class_idx']
        if cls not in by_class or s['confidence'] > by_class[cls]['confidence']:
            by_class[cls] = s
    selected = list(by_class.values())
    if len(selected) >= k:
        selected.sort(key=lambda s: s['confidence'], reverse=True)
        return selected[:k]
    # Fill remaining slots
    remaining = [s for s in segments if s not in selected]
    remaining.sort(key=lambda s: s['confidence'], reverse=True)
    for s in remaining:
        if len(selected) >= k:
            break
        selected.append(s)
    return sorted(selected, key=lambda s: s['start_ts'])


# ── Pipeline with tunable params ────────────────────────────────────────
def run_pipeline(user_data, model_groups, device, params):
    """Run inference with given params, return list of (user_id, segments)."""
    all_results = []

    for uid, data in user_data.items():
        timestamps, probs = predict_multiscale_ensemble(data, model_groups, device)
        if len(timestamps) == 0:
            continue

        # Smooth
        sw = params.get('smooth_window', 5)
        smoothed = np.copy(probs)
        for c in range(probs.shape[1]):
            smoothed[:, c] = uniform_filter1d(probs[:, c], size=sw, mode='nearest')

        # Optional: median filter
        if params.get('median_filter', 0) > 0:
            mf = params['median_filter']
            if mf % 2 == 0:
                mf += 1
            for c in range(smoothed.shape[1]):
                smoothed[:, c] = medfilt(smoothed[:, c], kernel_size=mf)

        # Viterbi
        path = viterbi_tunable(
            smoothed,
            self_trans=params.get('self_trans', 0.90),
            bg_to_act=params.get('bg_to_act', 0.01),
            act_to_bg=params.get('act_to_bg', 0.05),
            cross_act=params.get('cross_act', 0.005),
        )

        # Extract segments
        segments = extract_segments(path, timestamps, smoothed)

        # Merge
        gap = params.get('gap_threshold', 60)
        segments = merge_same_class_segments(segments, gap_threshold_sec=gap)

        # Refine boundaries
        all_ts = data[:, 0]
        segments = refine_boundaries(segments, data, all_ts)

        # Resolve overlaps
        segments = resolve_overlaps(segments)

        # Filter short
        min_dur = params.get('min_duration', 120)
        segments = filter_short_segments(segments, min_duration_sec=min_dur)

        # Optional: top-k selection
        if params.get('top_k', 0) > 0:
            segments = select_top_k_segments(segments, k=params['top_k'])

        # Optional: confidence filter
        conf_min = params.get('conf_min', 0.0)
        if conf_min > 0:
            segments = [s for s in segments if s['confidence'] >= conf_min]

        all_results.append((uid, segments))

    return all_results


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate_results(all_results, gold_df):
    """Evaluate results against gold standard, return avg F1."""
    user_f1s = []
    details = {}
    split_users = gold_df['user_id'].unique()

    for user_id in split_users:
        g_segs = gold_df[gold_df['user_id'] == user_id]
        G = [{'start': r['start'], 'end': r['end'], 'category': r['category']}
             for _, r in g_segs.iterrows()]

        P = []
        for uid, segs in all_results:
            if uid == user_id:
                for s in segs:
                    P.append({'start': s['start_ts'], 'end': s['end_ts'],
                              'category': s['class_name']})

        matches = []
        for i, p in enumerate(P):
            for j, g in enumerate(G):
                if p['category'] == g['category']:
                    iou = calculate_iou(p['start'], p['end'], g['start'], g['end'])
                    if iou > 0.5:
                        matches.append({'p': i, 'g': j, 'iou': iou})
        matches.sort(key=lambda x: x['iou'], reverse=True)

        mp, mg = set(), set()
        TP = 0
        for m in matches:
            if m['p'] not in mp and m['g'] not in mg:
                TP += 1
                mp.add(m['p'])
                mg.add(m['g'])
        FP = len(P) - len(mp)
        FN = len(G) - len(mg)

        if TP + FP + FN == 0:
            f1 = 1.0
        elif TP == 0:
            f1 = 0.0
        else:
            prec = TP / (TP + FP)
            rec = TP / (TP + FN)
            f1 = 2 * prec * rec / (prec + rec)

        user_f1s.append(f1)
        details[user_id] = {'TP': TP, 'FP': FP, 'FN': FN, 'F1': f1}

    return np.mean(user_f1s), details


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    model_groups, device = load_ensemble_models()

    print("Loading internal_eval data...")
    from imu_activity_pipeline.signal_file_reader import DataReader
    reader = DataReader(INTERNAL_EVAL_DATA_DIR)
    internal_eval_files = reader.read_data()

    user_data = {}
    for uid in sorted(internal_eval_files.keys()):
        fpath = os.path.join(INTERNAL_EVAL_DATA_DIR, f"{uid}.txt")
        data = load_sensor_data(fpath)
        if data is not None and len(data) >= WINDOW_SIZE:
            user_data[uid] = data
    print(f"Loaded {len(user_data)} users")

    gold_df = load_gold_labels(INTERNAL_EVAL_GOLD_FILE)
    gold_df['user_id'] = gold_df['user_id'].astype(str)

    # ── Baseline ──
    baseline_params = {
        'smooth_window': 5, 'self_trans': 0.90, 'bg_to_act': 0.01,
        'act_to_bg': 0.05, 'cross_act': 0.005,
        'gap_threshold': 60, 'min_duration': 120,
        'top_k': 0, 'conf_min': 0.0, 'median_filter': 0,
    }
    print("\n" + "=" * 70)
    print("BASELINE:")
    results = run_pipeline(user_data, model_groups, device, baseline_params)
    f1, det = evaluate_results(results, gold_df)
    print(f"  F1 = {f1:.4f}")
    bad_users = {k: v for k, v in det.items() if v['F1'] < 1.0}
    for u, d in sorted(bad_users.items(), key=lambda x: x[1]['F1']):
        print(f"    {u}: TP={d['TP']} FP={d['FP']} FN={d['FN']} F1={d['F1']:.3f}")

    # ── Parameter sweep ──
    sweep_configs = {
        # Viterbi self-transition
        'self_trans': [0.90, 0.93, 0.95, 0.97],
        # Merge gap threshold
        'gap_threshold': [60, 90, 120, 180],
        # Min segment duration
        'min_duration': [120, 150, 180],
        # Top-K segments per user
        'top_k': [0, 3, 4],
        # Smoothing window
        'smooth_window': [3, 5, 7, 9],
        # Cross-activity transition
        'cross_act': [0.001, 0.003, 0.005],
        # Confidence minimum
        'conf_min': [0.0, 0.3, 0.4],
        # Median filter
        'median_filter': [0, 3, 5],
    }

    # Phase 1: sweep each param independently
    print("\n" + "=" * 70)
    print("PHASE 1: Independent parameter sweep")
    print("=" * 70)

    best_per_param = {}
    for param_name, values in sweep_configs.items():
        print(f"\n--- Sweeping {param_name} ---")
        best_f1 = 0
        best_val = None
        for val in values:
            p = baseline_params.copy()
            p[param_name] = val
            res = run_pipeline(user_data, model_groups, device, p)
            f1, _ = evaluate_results(res, gold_df)
            marker = " <-- best" if f1 > best_f1 else ""
            print(f"  {param_name}={val}: F1={f1:.4f}{marker}")
            if f1 > best_f1:
                best_f1 = f1
                best_val = val
        best_per_param[param_name] = (best_val, best_f1)
        print(f"  BEST: {param_name}={best_val} -> F1={best_f1:.4f}")

    # Phase 2: combine best params
    print("\n" + "=" * 70)
    print("PHASE 2: Combined best parameters")
    print("=" * 70)

    combined = baseline_params.copy()
    for param_name, (best_val, _) in best_per_param.items():
        combined[param_name] = best_val
    print(f"Combined params: {combined}")

    results = run_pipeline(user_data, model_groups, device, combined)
    f1_combined, det_combined = evaluate_results(results, gold_df)
    print(f"\nCombined F1 = {f1_combined:.4f}")
    for u, d in sorted(det_combined.items(), key=lambda x: x[1]['F1']):
        if d['F1'] < 1.0:
            print(f"  {u}: TP={d['TP']} FP={d['FP']} FN={d['FN']} F1={d['F1']:.3f}")

    # Phase 3: fine-tune top-3 most impactful params together
    print("\n" + "=" * 70)
    print("PHASE 3: Grid search on top impactful params")
    print("=" * 70)

    # Sort by improvement over baseline
    baseline_f1_val = 0.8519
    impacts = [(k, v[1] - baseline_f1_val) for k, v in best_per_param.items()]
    impacts.sort(key=lambda x: x[1], reverse=True)
    print("Parameter impact ranking:")
    for k, imp in impacts:
        print(f"  {k}: +{imp:.4f} (best={best_per_param[k][0]})")

    # Grid search on top 3
    top3 = [k for k, _ in impacts[:3]]
    print(f"\nGrid search on: {top3}")
    best_grid_f1 = 0
    best_grid_params = None

    grid_values = {k: sweep_configs[k] for k in top3}
    keys = list(grid_values.keys())
    combos = list(itertools.product(*[grid_values[k] for k in keys]))
    print(f"Testing {len(combos)} combinations...")

    for combo in combos:
        p = combined.copy()
        for k, v in zip(keys, combo):
            p[k] = v
        res = run_pipeline(user_data, model_groups, device, p)
        f1, _ = evaluate_results(res, gold_df)
        if f1 > best_grid_f1:
            best_grid_f1 = f1
            best_grid_params = p.copy()
            desc = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
            print(f"  NEW BEST: F1={f1:.4f} ({desc})")

    print(f"\n{'=' * 70}")
    print(f"FINAL BEST F1 = {best_grid_f1:.4f}")
    print(f"BEST PARAMS: {best_grid_params}")
    print(f"{'=' * 70}")

    # Show per-user breakdown
    results = run_pipeline(user_data, model_groups, device, best_grid_params)
    f1_final, det_final = evaluate_results(results, gold_df)
    print(f"\nPer-user breakdown (F1={f1_final:.4f}):")
    for u, d in sorted(det_final.items()):
        status = "✓" if d['F1'] >= 1.0 else "✗"
        print(f"  {status} {u}: TP={d['TP']} FP={d['FP']} FN={d['FN']} F1={d['F1']:.3f}")


if __name__ == '__main__':
    main()
