#!/usr/bin/env python3
"""Internal-evaluation figure generator for prediction and error analysis.

Purpose:
    Converts `internal_eval` prediction/evaluation tables into reader-friendly
    figures such as per-user F1, per-class precision/recall/F1, ablations,
    timelines, and parameter-sensitivity plots.
Inputs:
    Reads `predictions_internal_eval.xlsx` and
    `data/annotations/internal_eval_annotations.csv` by default.
Outputs:
    Saves static internal-evaluation figures for local review or documentation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

# ── plot style ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'legend.fontsize': 7, 'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC', 'legend.framealpha': 0.9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.2,
    'grid.linewidth': 0.4, 'grid.linestyle': '--',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 0.8,
})

COLORS_5 = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
LABELS_5 = ['Badminton', 'Rope Skipping', 'Dumbbell Fly', 'Running', 'Table Tennis']
ACTIVITIES = ['羽毛球', '跳绳', '飞鸟', '跑步', '乒乓球']
ACT_EN = dict(zip(ACTIVITIES, LABELS_5))

BASE_DIR = str(ROOT)
OUT_DIR = os.path.join(BASE_DIR, 'Report/internal_eval_evaluation_figures')
os.makedirs(OUT_DIR, exist_ok=True)

from imu_activity_pipeline.config import INTERNAL_EVAL_GOLD_FILE
from imu_activity_pipeline.sensor_data_processing import load_gold_labels
from imu_activity_pipeline.evaluate import calculate_iou, load_predictions


def load_data():
    prediction_file = os.path.join(BASE_DIR, 'predictions_internal_eval.xlsx')
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(
            f"Expected internal_eval predictions at {prediction_file}. "
            "Generate them with the internal_eval split first."
        )
    print(f"Prediction file: {prediction_file}")
    print(f"Gold file: {INTERNAL_EVAL_GOLD_FILE}")
    sub = load_predictions(prediction_file)
    gold = load_gold_labels(INTERNAL_EVAL_GOLD_FILE)
    sub['user_id'] = sub['user_id'].astype(str)
    gold['user_id'] = gold['user_id'].astype(str)
    return sub, gold


def compute_metrics(sub, gold):
    """Compute detailed metrics: per-user, per-class, overall."""
    split_users = sorted(gold['user_id'].unique())
    user_metrics = {}
    all_matches = []  # (user, pred_cat, gold_cat, iou, matched)

    for uid in split_users:
        g = gold[gold['user_id'] == uid]
        p = sub[sub['user_id'] == uid]

        G = [{'start': r['start'], 'end': r['end'], 'category': r['category']}
             for _, r in g.iterrows()]
        P = [{'start': r['start'], 'end': r['end'], 'category': r['category']}
             for _, r in p.iterrows()]

        matches = []
        for i, pp in enumerate(P):
            for j, gg in enumerate(G):
                if pp['category'] == gg['category']:
                    iou = calculate_iou(pp['start'], pp['end'], gg['start'], gg['end'])
                    if iou > 0.5:
                        matches.append({'p': i, 'g': j, 'iou': iou})
        matches.sort(key=lambda x: x['iou'], reverse=True)

        mp, mg = set(), set()
        TP = 0
        for m in matches:
            if m['p'] not in mp and m['g'] not in mg:
                TP += 1; mp.add(m['p']); mg.add(m['g'])
                all_matches.append({
                    'user': uid, 'pred_cat': P[m['p']]['category'],
                    'gold_cat': G[m['g']]['category'], 'iou': m['iou'], 'type': 'TP'
                })

        FP = len(P) - len(mp)
        FN = len(G) - len(mg)

        # Track FP details
        for i, pp in enumerate(P):
            if i not in mp:
                all_matches.append({
                    'user': uid, 'pred_cat': pp['category'],
                    'gold_cat': None, 'iou': 0, 'type': 'FP'
                })
        for j, gg in enumerate(G):
            if j not in mg:
                all_matches.append({
                    'user': uid, 'pred_cat': None,
                    'gold_cat': gg['category'], 'iou': 0, 'type': 'FN'
                })

        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        user_metrics[uid] = {'TP': TP, 'FP': FP, 'FN': FN,
                             'precision': prec, 'recall': rec, 'f1': f1}

    # Per-class metrics
    class_metrics = {}
    for cat in ACTIVITIES:
        tp = sum(1 for m in all_matches if m['type'] == 'TP' and m['gold_cat'] == cat)
        fp = sum(1 for m in all_matches if m['type'] == 'FP' and m['pred_cat'] == cat)
        fn = sum(1 for m in all_matches if m['type'] == 'FN' and m['gold_cat'] == cat)
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        class_metrics[cat] = {'TP': tp, 'FP': fp, 'FN': fn,
                              'precision': prec, 'recall': rec, 'f1': f1}

    # Overall
    total_tp = sum(v['TP'] for v in user_metrics.values())
    total_fp = sum(v['FP'] for v in user_metrics.values())
    total_fn = sum(v['FN'] for v in user_metrics.values())
    avg_f1 = np.mean([v['f1'] for v in user_metrics.values()])

    overall = {
        'avg_f1': avg_f1, 'total_tp': total_tp,
        'total_fp': total_fp, 'total_fn': total_fn,
        'micro_prec': total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0,
        'micro_rec': total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0,
    }
    overall['micro_f1'] = (2 * overall['micro_prec'] * overall['micro_rec'] /
                           (overall['micro_prec'] + overall['micro_rec'])
                           if overall['micro_prec'] + overall['micro_rec'] > 0 else 0)

    return user_metrics, class_metrics, overall, all_matches


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Per-user F1
# ═══════════════════════════════════════════════════════════════════════
def fig_per_user_f1(user_metrics):
    users = sorted(user_metrics.keys())
    f1s = [user_metrics[u]['f1'] for u in users]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ['#55A868' if f >= 1.0 else '#C44E52' if f < 0.7 else '#CCB974' for f in f1s]
    bars = ax.bar(range(len(users)), f1s, color=colors, edgecolor='white', linewidth=0.4, width=0.7)

    ax.axhline(y=np.mean(f1s), color='#4C72B0', linestyle='--', linewidth=1.0,
               label=f'Mean F1 = {np.mean(f1s):.4f}')
    ax.set_xticks(range(len(users)))
    ax.set_xticklabels(users, rotation=45, ha='right', fontsize=6.5)
    ax.set_ylabel('Segmental F1-Score')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='lower left', fontsize=7)

    for i, (bar, f1) in enumerate(zip(bars, f1s)):
        if f1 < 1.0:
            ax.text(bar.get_x() + bar.get_width()/2, f1 + 0.02,
                    f'{f1:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_per_user_f1.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_per_user_f1.pdf'))
    plt.close(fig)
    print('  -> fig_per_user_f1')


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Per-class Precision/Recall/F1
# ═══════════════════════════════════════════════════════════════════════
def fig_per_class_prf(class_metrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ACTIVITIES))
    width = 0.25

    precs = [class_metrics[c]['precision'] for c in ACTIVITIES]
    recs = [class_metrics[c]['recall'] for c in ACTIVITIES]
    f1s = [class_metrics[c]['f1'] for c in ACTIVITIES]

    b1 = ax.bar(x - width, precs, width, label='Precision', color='#4C72B0', edgecolor='white', linewidth=0.3)
    b2 = ax.bar(x, recs, width, label='Recall', color='#55A868', edgecolor='white', linewidth=0.3)
    b3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#C44E52', edgecolor='white', linewidth=0.3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_5, fontsize=8)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_per_class_prf.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_per_class_prf.pdf'))
    plt.close(fig)
    print('  -> fig_per_class_prf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Error analysis (FP/FN by class)
# ═══════════════════════════════════════════════════════════════════════
def fig_error_analysis(class_metrics, all_matches):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) TP/FP/FN stacked bar
    ax = axes[0]
    x = np.arange(len(ACTIVITIES))
    tps = [class_metrics[c]['TP'] for c in ACTIVITIES]
    fps = [class_metrics[c]['FP'] for c in ACTIVITIES]
    fns = [class_metrics[c]['FN'] for c in ACTIVITIES]

    ax.bar(x, tps, 0.5, label='TP', color='#55A868', edgecolor='white', linewidth=0.3)
    ax.bar(x, fps, 0.5, bottom=tps, label='FP', color='#C44E52', edgecolor='white', linewidth=0.3)
    ax.bar(x, fns, 0.5, bottom=[t+f for t, f in zip(tps, fps)],
           label='FN', color='#8172B2', edgecolor='white', linewidth=0.3)

    for i in range(len(ACTIVITIES)):
        total = tps[i] + fps[i] + fns[i]
        if fps[i] > 0:
            ax.text(i, tps[i] + fps[i]/2, f'{fps[i]}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')
        if fns[i] > 0:
            ax.text(i, tps[i] + fps[i] + fns[i]/2, f'{fns[i]}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_5, fontsize=7, rotation=15, ha='right')
    ax.set_ylabel('Number of Segments')
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)

    # (b) IoU distribution of matched segments
    ax = axes[1]
    tp_matches = [m for m in all_matches if m['type'] == 'TP']
    ious = [m['iou'] for m in tp_matches]
    if ious:
        ax.hist(ious, bins=20, range=(0.5, 1.0), color='#4C72B0',
                edgecolor='white', linewidth=0.4, alpha=0.85)
        ax.axvline(x=np.mean(ious), color='#C44E52', linestyle='--', linewidth=0.8,
                   label=f'Mean IoU = {np.mean(ious):.3f}')
        ax.axvline(x=np.median(ious), color='#55A868', linestyle=':', linewidth=0.8,
                   label=f'Median IoU = {np.median(ious):.3f}')
    ax.set_xlabel('IoU')
    ax.set_ylabel('Count')
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, 'fig_error_analysis.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_error_analysis.pdf'))
    plt.close(fig)
    print('  -> fig_error_analysis')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Ablation - before/after post-processing tuning
# ═══════════════════════════════════════════════════════════════════════
def fig_ablation(user_metrics):
    """Compare baseline vs tuned post-processing."""
    # Baseline results (from earlier run with default params)
    baseline = {
        'HDU21012': 0.667, 'HNU21004': 1.0, 'HNU21011': 0.857,
        'HNU21014': 1.0, 'HNU21020': 0.667, 'HNU21024': 1.0,
        'HNU21033': 1.0, 'HNU21034': 1.0, 'HNU21036': 1.0,
        'HNU21044': 1.0, 'HNU21050': 0.667, 'HNU21051': 0.571,
        'HNU21060': 0.182, 'HNU21078': 1.0, 'HNU21081': 0.571,
        'HNU21086': 1.0, 'HNU21087': 1.0, 'HNU21090': 1.0,
        'HNU21093': 1.0, 'HNU21101': 0.857,
    }

    users = sorted(user_metrics.keys())
    bl_f1 = [baseline.get(u, 0) for u in users]
    tuned_f1 = [user_metrics[u]['f1'] for u in users]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Per-user comparison
    ax = axes[0]
    x = np.arange(len(users))
    width = 0.35
    ax.bar(x - width/2, bl_f1, width, label=f'Baseline (Mean={np.mean(bl_f1):.3f})',
           color='#BDBDBD', edgecolor='white', linewidth=0.3)
    ax.bar(x + width/2, tuned_f1, width, label=f'Tuned (Mean={np.mean(tuned_f1):.3f})',
           color='#4C72B0', edgecolor='white', linewidth=0.3)

    # Highlight improved users
    for i in range(len(users)):
        if tuned_f1[i] > bl_f1[i]:
            ax.annotate('', xy=(i + width/2, tuned_f1[i]),
                        xytext=(i - width/2, bl_f1[i]),
                        arrowprops=dict(arrowstyle='->', color='#55A868',
                                        lw=1.0, connectionstyle='arc3,rad=0.2'))

    ax.set_xticks(x)
    ax.set_xticklabels(users, rotation=45, ha='right', fontsize=5.5)
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.1)
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, loc='lower left')

    # (b) Improvement summary
    ax = axes[1]
    params_names = ['Smooth\nWindow', 'Cross-Act\nTransition', 'Median\nFilter',
                    'Top-K\nSelection', 'Self\nTransition', 'Min\nDuration',
                    'Confidence\nThreshold']
    # Individual parameter improvements from sweep
    improvements = [0.0564, 0.0395, 0.0203, 0.0181, 0.0110, 0.0035, 0.0035]
    colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#BDBDBD', '#E8A0BF']

    bars = ax.barh(range(len(params_names)), improvements, color=colors_bar,
                   edgecolor='white', linewidth=0.3, height=0.6)
    ax.set_yticks(range(len(params_names)))
    ax.set_yticklabels(params_names, fontsize=7)
    ax.set_xlabel('F1 Improvement over Baseline')
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'+{val:.4f}', ha='left', va='center', fontsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation.pdf'))
    plt.close(fig)
    print('  -> fig_ablation')


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Timeline visualization for selected users
# ═══════════════════════════════════════════════════════════════════════
def fig_timeline(sub, gold):
    """Show predicted vs ground truth timelines for interesting users."""
    # Select users: 2 perfect + 2 imperfect
    interesting = ['HNU21060', 'HNU21033', 'HDU21012', 'HNU21050']

    fig, axes = plt.subplots(len(interesting), 1, figsize=(12, 1.8 * len(interesting)))
    panel_labels = list('abcd')

    for idx, uid in enumerate(interesting):
        ax = axes[idx]
        g = gold[gold['user_id'] == uid].sort_values('start')
        p = sub[sub['user_id'] == uid].sort_values('start')

        # Time range
        all_starts = list(g['start']) + list(p['start'])
        all_ends = list(g['end']) + list(p['end'])
        t_min = min(all_starts) - 60000
        t_max = max(all_ends) + 60000

        # Plot ground truth (top row)
        for _, row in g.iterrows():
            cat_idx = ACTIVITIES.index(row['category']) if row['category'] in ACTIVITIES else 0
            s = (row['start'] - t_min) / 60000  # convert to minutes
            e = (row['end'] - t_min) / 60000
            ax.barh(1, e - s, left=s, height=0.35, color=COLORS_5[cat_idx],
                    edgecolor='black', linewidth=0.4, alpha=0.85)

        # Plot predictions (bottom row)
        for _, row in p.iterrows():
            cat_idx = ACTIVITIES.index(row['category']) if row['category'] in ACTIVITIES else 0
            s = (row['start'] - t_min) / 60000
            e = (row['end'] - t_min) / 60000
            ax.barh(0, e - s, left=s, height=0.35, color=COLORS_5[cat_idx],
                    edgecolor='black', linewidth=0.4, alpha=0.85)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Predicted', 'Ground Truth'], fontsize=7)
        ax.set_ylim(-0.5, 1.7)
        ax.set_title(panel_labels[idx], loc='left', fontsize=11, fontweight='bold')
        ax.text(0.99, 0.95, uid, transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold', style='italic')
        ax.set_xlabel('Time (min)' if idx == len(interesting) - 1 else '')
        ax.grid(axis='y', alpha=0)

    # Legend at bottom
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', linewidth=0.4, label=l)
                       for c, l in zip(COLORS_5, LABELS_5)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(h_pad=0.8)
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(os.path.join(OUT_DIR, 'fig_timeline.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_timeline.pdf'))
    plt.close(fig)
    print('  -> fig_timeline')


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Post-processing parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════
def fig_param_sensitivity():
    """Show how F1 changes with key parameters."""
    # Data from our sweep results
    data = {
        'Smoothing Window': {
            'x': [3, 5, 7, 9, 11, 13],
            'y': [0.8171, 0.8519, 0.8722, 0.9083, 0.9333, 0.9333],
            'best': 7,
        },
        'Cross-Activity Transition': {
            'x': [0.001, 0.003, 0.005],
            'y': [0.8914, 0.8629, 0.8519],
            'best': 0.001,
        },
        'Self Transition': {
            'x': [0.90, 0.93, 0.95, 0.97],
            'y': [0.8519, 0.8519, 0.8519, 0.8629],
            'best': 0.97,
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    panels = ['a', 'b', 'c']
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for ax, (name, d), panel, color in zip(axes, data.items(), panels, colors):
        ax.plot(d['x'], d['y'], 'o-', color=color, markersize=5, linewidth=1.2)
        # Mark best
        best_idx = d['x'].index(d['best'])
        ax.plot(d['best'], d['y'][best_idx], 's', color='#C44E52',
                markersize=8, zorder=5, markeredgecolor='black', markeredgewidth=0.5)
        ax.axhline(y=0.8519, color='#BDBDBD', linestyle=':', linewidth=0.8,
                   label='Baseline (0.8519)')
        ax.set_xlabel(name)
        ax.set_ylabel('F1-Score')
        ax.set_title(panel, loc='left', fontsize=11, fontweight='bold')
        ax.legend(fontsize=6.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT_DIR, 'fig_param_sensitivity.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_param_sensitivity.pdf'))
    plt.close(fig)
    print('  -> fig_param_sensitivity')


def main():
    print('=' * 60)
    print('  internal_eval Evaluation Figures - Ensemble Results')
    print('=' * 60)

    sub, gold = load_data()
    user_metrics, class_metrics, overall, all_matches = compute_metrics(sub, gold)

    print(f'\nOverall: F1={overall["avg_f1"]:.4f}, '
          f'TP={overall["total_tp"]}, FP={overall["total_fp"]}, FN={overall["total_fn"]}')
    print(f'Micro: P={overall["micro_prec"]:.4f}, R={overall["micro_rec"]:.4f}, '
          f'F1={overall["micro_f1"]:.4f}')

    print('\nPer-class:')
    for cat in ACTIVITIES:
        m = class_metrics[cat]
        print(f'  {ACT_EN[cat]:15s}: P={m["precision"]:.3f} R={m["recall"]:.3f} '
              f'F1={m["f1"]:.3f} (TP={m["TP"]}, FP={m["FP"]}, FN={m["FN"]})')

    print('\nGenerating figures...')
    fig_per_user_f1(user_metrics)
    fig_per_class_prf(class_metrics)
    fig_error_analysis(class_metrics, all_matches)
    fig_ablation(user_metrics)
    fig_timeline(sub, gold)
    fig_param_sensitivity()

    # Save metrics as JSON
    results = {
        'split': 'internal_eval',
        'overall': overall,
        'per_user': {k: v for k, v in user_metrics.items()},
        'per_class': {ACT_EN.get(k, k): v for k, v in class_metrics.items()},
    }
    with open(os.path.join(OUT_DIR, 'internal_eval_ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nAll figures saved to {OUT_DIR}')
    print('Done!')


if __name__ == '__main__':
    main()
