"""Internal-evaluation split utility for trained checkpoints.

Purpose:
    Evaluates trained checkpoints on the canonical `internal_eval` data split,
    runs inference-style post-processing, and reports segment-level metrics for
    development diagnostics.
Inputs:
    Uses saved models, normalization parameters, `data/signals/internal_eval/`,
    and `data/annotations/internal_eval_annotations.csv`.
Outputs:
    Prints internal-evaluation metrics and writes JSON/figure artifacts for
    local analysis.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from imu_activity_pipeline.config import *
from imu_activity_pipeline.sensor_data_processing import load_sensor_data, load_gold_labels, create_windows, normalize_imu
from imu_activity_pipeline.inference import (predict_windows, smooth_predictions,
                       viterbi_decode, extract_segments, merge_same_class_segments,
                       filter_short_segments, resolve_overlaps, refine_boundaries)
from imu_activity_pipeline.evaluate import calculate_iou

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CombinedModelCompat(torch.nn.Module):
    """CombinedModel compatible with the saved checkpoint structure."""

    def __init__(self, input_channels=6, num_classes=6, window_size=300):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
        )
        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
        )

        self.conv_merge = torch.nn.Sequential(
            torch.nn.Conv1d(192, 256, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )

        self.lstm = torch.nn.LSTM(192, 128, batch_first=True, bidirectional=True,
                                   num_layers=2, dropout=0.3)

        # Saved model has combined classifier (embedding + head in one Sequential)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 + 256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x, return_embedding=False):
        xp = x.permute(0, 2, 1)
        b1 = self.branch1(xp)
        b2 = self.branch2(xp)
        b3 = self.branch3(xp)
        merged = torch.cat([b1, b2, b3], dim=1)

        cnn_out = self.conv_merge(merged)
        cnn_feat = cnn_out.squeeze(-1)

        lstm_in = merged.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        combined = torch.cat([cnn_feat, lstm_feat], dim=1)
        logits = self.classifier(combined)
        return logits


def load_model_compat():
    """Load model with compatibility for saved checkpoint."""
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(MODEL_DIR, 'norm_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)

    model = CombinedModelCompat(input_channels=6, num_classes=6, window_size=WINDOW_SIZE).to(device)
    checkpoint_path = os.path.join(MODEL_DIR, 'combined_model_best.pth')
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model (val_f1={checkpoint.get('val_f1', 'N/A')}, "
          f"val_acc={checkpoint.get('val_acc', 'N/A')}, epoch={checkpoint.get('epoch', 'N/A')})")
    return model, norm_params, device
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_internal_eval_users():
    """Return users with readable signals and labels in the internal_eval split."""
    gold_labels = load_gold_labels(INTERNAL_EVAL_GOLD_FILE)
    users = sorted([f.replace('.txt', '') for f in os.listdir(INTERNAL_EVAL_DATA_DIR) if f.endswith('.txt')])
    gold_users = set(gold_labels['user_id'].unique())

    valid_users = []
    for user_id in users:
        if user_id not in gold_users:
            continue
        file_path = os.path.join(INTERNAL_EVAL_DATA_DIR, f"{user_id}.txt")
        try:
            with open(file_path, 'rb') as f:
                header_bytes = f.read(100)
            header_text = header_bytes.decode('utf-8')
            if 'ACC_TIME' not in header_text:
                continue
        except:
            continue
        data = load_sensor_data(file_path, apply_filter=True)
        if data is None or len(data) < WINDOW_SIZE:
            continue
        timestamps, windows = create_windows(data, WINDOW_SIZE, WINDOW_STEP)
        if len(windows) == 0:
            continue
        valid_users.append(user_id)

    return valid_users, gold_labels


def run_internal_eval():
    print("=" * 60)
    print("INTERNAL_EVAL SPLIT EVALUATION")
    print("=" * 60)

    internal_eval_users, gold_labels = get_internal_eval_users()
    print(f"internal_eval users ({len(internal_eval_users)}): {internal_eval_users}")

    model, norm_params, device = load_model_compat()

    all_results = []  # (user_id, category, start, end)
    per_user_metrics = {}
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for user_id in sorted(internal_eval_users):
        file_path = os.path.join(INTERNAL_EVAL_DATA_DIR, f"{user_id}.txt")
        data = load_sensor_data(file_path, apply_filter=True)
        if data is None or len(data) < WINDOW_SIZE:
            print(f"  Skipping {user_id}")
            continue

        # Run full inference pipeline
        timestamps, windows = create_windows(data, WINDOW_SIZE, WINDOW_STEP)
        norm_windows, _, _ = normalize_imu(windows, norm_params['mean'], norm_params['std'])
        probs = predict_windows(model, norm_windows, device)
        smoothed_probs = smooth_predictions(probs, timestamps)
        path = viterbi_decode(smoothed_probs)
        segments = extract_segments(path, timestamps, smoothed_probs)
        segments = merge_same_class_segments(segments, gap_threshold_sec=SHORT_GAP_THRESHOLD)
        all_timestamps = data[:, 0]
        segments = refine_boundaries(segments, data, all_timestamps)
        segments = resolve_overlaps(segments)
        segments = filter_short_segments(segments, min_duration_sec=MIN_SEGMENT_FOR_OUTPUT)

        # Predicted segments
        P = []
        for seg in segments:
            P.append({
                'start': seg['start_ts'],
                'end': seg['end_ts'],
                'category': seg['class_name'],
            })
            all_results.append([user_id, seg['class_name'], seg['start_ts'], seg['end_ts']])

        # Ground truth segments
        user_gold = gold_labels[gold_labels['user_id'] == user_id]
        G = []
        for _, row in user_gold.iterrows():
            G.append({
                'start': row['start'],
                'end': row['end'],
                'category': row['category'],
            })

        # Compute matches (IoU >= 0.5, same category)
        matches = []
        for i, p in enumerate(P):
            for j, g in enumerate(G):
                if p['category'] == g['category']:
                    iou = calculate_iou(p['start'], p['end'], g['start'], g['end'])
                    if iou > 0.5:
                        matches.append({'p_idx': i, 'g_idx': j, 'iou': iou,
                                        'category': p['category']})

        matches.sort(key=lambda x: x['iou'], reverse=True)
        matched_p = set()
        matched_g = set()
        matched_cats = []

        for m in matches:
            if m['p_idx'] not in matched_p and m['g_idx'] not in matched_g:
                matched_p.add(m['p_idx'])
                matched_g.add(m['g_idx'])
                matched_cats.append(m['category'])
                per_class_tp[m['category']] += 1

        TP = len(matched_p)
        FP = len(P) - TP
        FN = len(G) - len(matched_g)

        # Track per-class FP/FN
        for i, p in enumerate(P):
            if i not in matched_p:
                per_class_fp[p['category']] += 1
        for j, g in enumerate(G):
            if j not in matched_g:
                per_class_fn[g['category']] += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_user_metrics[user_id] = {
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1,
            'n_pred': len(P), 'n_gold': len(G),
        }
        print(f"  {user_id}: TP={TP}, FP={FP}, FN={FN}, "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.4f} "
              f"(pred={len(P)}, gold={len(G)})")

    # Overall metrics
    user_f1s = [m['f1'] for m in per_user_metrics.values()]
    avg_f1 = np.mean(user_f1s) if user_f1s else 0

    total_tp = sum(m['TP'] for m in per_user_metrics.values())
    total_fp = sum(m['FP'] for m in per_user_metrics.values())
    total_fn = sum(m['FN'] for m in per_user_metrics.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS (Segment-level, IoU>=0.5)")
    print(f"{'='*60}")
    print(f"  Mean User F1 (macro): {avg_f1:.4f}")
    print(f"  Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"  Total TP={total_tp}, FP={total_fp}, FN={total_fn}")

    # Per-class metrics
    print(f"\n  Per-class metrics:")
    per_class_results = {}
    for cat in ACTIVITIES:
        tp = per_class_tp.get(cat, 0)
        fp = per_class_fp.get(cat, 0)
        fn = per_class_fn.get(cat, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class_results[cat] = {'TP': tp, 'FP': fp, 'FN': fn,
                                   'precision': p, 'recall': r, 'f1': f}
        print(f"    {cat}: TP={tp}, FP={fp}, FN={fn}, P={p:.3f}, R={r:.3f}, F1={f:.4f}")

    # ==================== Generate charts ====================
    fig_dir = os.path.join(BASE_DIR, 'Report', 'internal_eval_evaluation_figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Chart 1: Per-user F1 bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    users_sorted = sorted(per_user_metrics.keys())
    f1_values = [per_user_metrics[u]['f1'] for u in users_sorted]
    colors = ['#2ecc71' if f >= 0.8 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_values]
    bars = ax.bar(range(len(users_sorted)), f1_values, color=colors)
    ax.set_xticks(range(len(users_sorted)))
    ax.set_xticklabels(users_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Segment F1-Score')
    ax.set_title(f'Per-User Segment F1-Score (Mean={avg_f1:.4f})')
    ax.axhline(y=avg_f1, color='blue', linestyle='--', linewidth=1, label=f'Mean={avg_f1:.4f}')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'per_user_f1.png'), dpi=150)
    plt.close()

    # Chart 2: Per-class P/R/F1 grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ACTIVITIES))
    width = 0.25
    p_vals = [per_class_results[c]['precision'] for c in ACTIVITIES]
    r_vals = [per_class_results[c]['recall'] for c in ACTIVITIES]
    f_vals = [per_class_results[c]['f1'] for c in ACTIVITIES]
    ax.bar(x - width, p_vals, width, label='Precision', color='#3498db')
    ax.bar(x, r_vals, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f_vals, width, label='F1', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVITIES, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Segment-Level Metrics (IoU>=0.5)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'per_class_prf.png'), dpi=150)
    plt.close()

    # Chart 3: TP/FP/FN stacked per class
    fig, ax = plt.subplots(figsize=(10, 5))
    tp_vals = [per_class_results[c]['TP'] for c in ACTIVITIES]
    fp_vals = [per_class_results[c]['FP'] for c in ACTIVITIES]
    fn_vals = [per_class_results[c]['FN'] for c in ACTIVITIES]
    ax.bar(x, tp_vals, width=0.6, label='TP (correct)', color='#2ecc71')
    ax.bar(x, fp_vals, width=0.6, bottom=tp_vals, label='FP (false alarm)', color='#e74c3c')
    ax.bar(x, fn_vals, width=0.6, bottom=[t+f for t, f in zip(tp_vals, fp_vals)],
           label='FN (missed)', color='#f39c12')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVITIES, fontsize=10)
    ax.set_ylabel('Segment Count')
    ax.set_title('Detection Results by Class')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'per_class_tp_fp_fn.png'), dpi=150)
    plt.close()

    # Chart 4: Summary metrics overview
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall P/R/F1
    metrics_names = ['Micro\nPrecision', 'Micro\nRecall', 'Micro\nF1', 'Mean\nUser F1']
    metrics_vals = [micro_p, micro_r, micro_f1, avg_f1]
    colors_m = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = axes[0].bar(metrics_names, metrics_vals, color=colors_m)
    for bar, val in zip(bars, metrics_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Overall Metrics')

    # Right: segment counts
    count_names = ['Total TP', 'Total FP', 'Total FN']
    count_vals = [total_tp, total_fp, total_fn]
    colors_c = ['#2ecc71', '#e74c3c', '#f39c12']
    bars2 = axes[1].bar(count_names, count_vals, color=colors_c)
    for bar, val in zip(bars2, count_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha='center', fontsize=11, fontweight='bold')
    axes[1].set_title('Segment Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'overall_summary.png'), dpi=150)
    plt.close()

    print(f"\nCharts saved to {fig_dir}/")

    # Save results as JSON for later reporting.
    results_json = {
        'avg_user_f1': float(avg_f1),
        'micro_precision': float(micro_p),
        'micro_recall': float(micro_r),
        'micro_f1': float(micro_f1),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'split': 'internal_eval',
        'n_internal_eval_users': len(internal_eval_users),
        'internal_eval_users': sorted(internal_eval_users),
        'per_user': {u: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                         for k, v in m.items()}
                     for u, m in per_user_metrics.items()},
        'per_class': {c: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                          for k, v in m.items()}
                      for c, m in per_class_results.items()},
    }
    with open(os.path.join(fig_dir, 'internal_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"Results JSON saved to {fig_dir}/internal_eval_results.json")
    return results_json


if __name__ == '__main__':
    run_internal_eval()
