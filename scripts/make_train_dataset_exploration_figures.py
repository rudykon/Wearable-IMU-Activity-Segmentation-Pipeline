#!/usr/bin/env python3
"""Exploratory data-analysis figure generator for the IMU dataset.

Purpose:
    Summarizes data completeness, activity distributions, signal statistics,
    representative traces, and augmentation examples for repository analysis and
    documentation.
Inputs:
    Reads signal files and annotations from the configured `data/` layout.
Outputs:
    Saves EDA figures and supporting summaries to the configured figure/output
    directories.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# ── plot style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.edgecolor': '#CCCCCC',
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.4,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 0.8,
})

# ── Color palette (color-accessible, muted) ─────────────────
COLORS_6 = ['#BDBDBD', '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
COLORS_5 = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
LABELS_5 = ['Badminton', 'Rope Skipping', 'Dumbbell Fly', 'Running', 'Table Tennis']
LABELS_6 = ['Background'] + LABELS_5

# Chinese activity names (for matching data)
ACTIVITIES = ['羽毛球', '跳绳', '飞鸟', '跑步', '乒乓球']
ACT_EN = dict(zip(ACTIVITIES, LABELS_5))

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = str(ROOT)
OUT_DIR = os.path.join(BASE_DIR, 'Report/eda_figures')
os.makedirs(OUT_DIR, exist_ok=True)

from imu_activity_pipeline.config import TRAIN_ANNOTATIONS_FILE, TRAIN_DATA_DIR
from imu_activity_pipeline.sensor_data_processing import load_gold_labels, load_sensor_data


# ═════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═════════════════════════════════════════════════════════════════════════

def load_all_data():
    print('[1/5] Loading data...')
    gold = load_gold_labels(TRAIN_ANNOTATIONS_FILE)
    gold['user_id'] = gold['user_id'].astype(str)
    gold['start'] = gold['start'].astype(np.int64)
    gold['end'] = gold['end'].astype(np.int64)

    txt_files = sorted([f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.txt')])
    all_files = [f.replace('.txt', '') for f in txt_files]

    user_data = {}
    readable_users, binary_users = [], []

    for uid in all_files:
        fpath = os.path.join(TRAIN_DATA_DIR, f'{uid}.txt')
        with open(fpath, 'rb') as fp:
            header = fp.read(100)
        try:
            if 'ACC_TIME' in header.decode('utf-8'):
                readable_users.append(uid)
            else:
                binary_users.append(uid)
        except Exception:
            binary_users.append(uid)

    print(f'  Total: {len(all_files)} | Readable: {len(readable_users)} | Binary: {len(binary_users)}')

    for uid in readable_users:
        fpath = os.path.join(TRAIN_DATA_DIR, f'{uid}.txt')
        data = load_sensor_data(fpath, apply_filter=False)
        if data is not None and len(data) > 0:
            user_data[uid] = data

    print(f'  Loaded {len(user_data)} users')
    return user_data, gold, readable_users, binary_users


# ═════════════════════════════════════════════════════════════════════════
# 2. Data Completeness
# ═════════════════════════════════════════════════════════════════════════

def analyze_completeness(user_data, gold, readable_users, binary_users):
    print('\n[2/5] Data completeness analysis...')

    records = []
    for uid, data in user_data.items():
        n = len(data)
        ts = data[:, 0]
        dur_s = (ts[-1] - ts[0]) / 1000.0
        dur_min = dur_s / 60.0
        rate = n / dur_s if dur_s > 0 else 0
        gyro_zero = np.mean(np.all(data[:, 4:7] == 0, axis=1)) * 100
        records.append({
            'user_id': uid, 'duration_min': dur_min,
            'effective_rate_hz': rate, 'gyro_zero_pct': gyro_zero,
        })

    df = pd.DataFrame(records).sort_values('user_id').reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Duration distribution
    ax = axes[0]
    sorted_df = df.sort_values('duration_min', ascending=True)
    colors_bar = ['#C44E52' if d < 30 else '#4C72B0' for d in sorted_df['duration_min']]
    ax.barh(range(len(sorted_df)), sorted_df['duration_min'].values,
            color=colors_bar, edgecolor='none', height=0.8)
    ax.set_yticks(range(0, len(sorted_df), 10))
    ax.set_yticklabels(sorted_df['user_id'].values[::10], fontsize=5)
    ax.set_xlabel('Duration (min)')
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold')
    ax.axvline(x=df['duration_min'].mean(), color='#666666', linestyle='--',
               linewidth=0.7, label=f'Mean={df["duration_min"].mean():.0f} min')
    ax.legend(fontsize=6.5)
    ax.invert_yaxis()

    # (b) Sampling rate
    ax = axes[1]
    ax.hist(df['effective_rate_hz'], bins=20, color='#4C72B0',
            edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(x=100, color='#C44E52', linestyle='--', linewidth=0.8, label='Nominal 100 Hz')
    ax.axvline(x=df['effective_rate_hz'].mean(), color='#8172B2',
               linestyle=':', linewidth=0.8, label=f'Mean={df["effective_rate_hz"].mean():.1f} Hz')
    ax.set_xlabel('Effective Sampling Rate (Hz)')
    ax.set_ylabel('Number of Users')
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5)

    # (c) Gyroscope zero ratio
    ax = axes[2]
    ax.hist(df['gyro_zero_pct'], bins=20, color='#8172B2',
            edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.set_xlabel('Gyroscope All-Zero Row Ratio (%)')
    ax.set_ylabel('Number of Users')
    ax.set_title('c', loc='left', fontsize=11, fontweight='bold')

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, 'fig1_data_completeness.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_data_completeness.pdf'))
    plt.close(fig)
    print('  -> Saved fig1_data_completeness')
    return df


# ═════════════════════════════════════════════════════════════════════════
# 3. Activity Distribution
# ═════════════════════════════════════════════════════════════════════════

def analyze_activity_distribution(gold):
    print('\n[3/5] Activity distribution analysis...')

    gold = gold.copy()
    gold['duration_min'] = (gold['end'] - gold['start']) / 1000.0 / 60.0

    seg_counts = gold['category'].value_counts()
    dur_by_cat = gold.groupby('category')['duration_min'].sum()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Segment counts
    ax = axes[0]
    counts = [seg_counts.get(c, 0) for c in ACTIVITIES]
    bars = ax.bar(range(len(LABELS_5)), counts, color=COLORS_5,
                  edgecolor='white', linewidth=0.4, width=0.6)
    ax.set_xticks(range(len(LABELS_5)))
    ax.set_xticklabels(LABELS_5, fontsize=7.5, rotation=15, ha='right')
    ax.set_ylabel('Number of Segments')
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')
    mean_cnt = np.mean(counts)
    ax.axhline(y=mean_cnt, color='#666666', linestyle='--', linewidth=0.6,
               label=f'Mean={mean_cnt:.0f}')
    ax.legend(fontsize=6.5)

    # (b) Duration pie
    ax = axes[1]
    durs = [dur_by_cat.get(c, 0) for c in ACTIVITIES]
    wedges, texts, autotexts = ax.pie(
        durs, labels=LABELS_5, colors=COLORS_5, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75, textprops={'fontsize': 7.5})
    for at in autotexts:
        at.set_fontsize(7)
        at.set_fontweight('bold')
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold')

    # (c) User-activity heatmap
    ax = axes[2]
    pivot = gold.pivot_table(index='user_id', columns='category',
                             values='duration_min', aggfunc='count', fill_value=0)
    pivot = pivot.reindex(columns=ACTIVITIES)
    user_totals = pivot.sum(axis=1).sort_values(ascending=False)
    top30 = user_totals.head(30).index
    pivot_top = pivot.loc[pivot.index.isin(top30)].sort_index()

    # User says it only has 0/1 possibilities, so we visualize it as binary
    # Ensure it's binary for safety
    pivot_binary = pivot_top.copy()
    pivot_binary[pivot_binary > 0] = 1

    # Binary colormap: 0=Light Gray, 1=Blue
    cmap_bin = matplotlib.colors.ListedColormap(['#F5F5F5', '#4C72B0'])

    im = ax.imshow(pivot_binary.values, aspect='auto', cmap=cmap_bin, interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS_5)))
    ax.set_xticklabels(LABELS_5, fontsize=6.5, rotation=20, ha='right')
    ax.set_yticks(range(len(pivot_top)))
    ax.set_yticklabels(pivot_top.index, fontsize=4.5)
    ax.set_title('c', loc='left', fontsize=11, fontweight='bold')
    
    # Discrete colorbar with 0/1 ticks
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['0', '1'])
    cbar.set_label('Segments', fontsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT_DIR, 'fig2_activity_distribution.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2_activity_distribution.pdf'))
    plt.close(fig)
    print('  -> Saved fig2_activity_distribution')

    # ── Segment duration boxplot ──
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data_box = [gold[gold['category'] == c]['duration_min'].values for c in ACTIVITIES]
    bp = ax.boxplot(data_box, positions=range(len(LABELS_5)), widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=2.5, alpha=0.4))
    for patch, color in zip(bp['boxes'], COLORS_5):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for m in bp['medians']:
        m.set_color('black')
        m.set_linewidth(1.2)

    ax.set_xticks(range(len(LABELS_5)))
    ax.set_xticklabels(LABELS_5, fontsize=8)
    ax.set_ylabel('Segment Duration (min)')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2b_segment_duration_boxplot.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2b_segment_duration_boxplot.pdf'))
    plt.close(fig)
    print('  -> Saved fig2b_segment_duration_boxplot')

    return seg_counts, dur_by_cat


# ═════════════════════════════════════════════════════════════════════════
# 4. Typical Signal Visualization
# ═════════════════════════════════════════════════════════════════════════

def visualize_typical_signals(user_data, gold):
    print('\n[4/5] Typical signal visualization...')

    gold = gold.copy()
    gold['duration_s'] = (gold['end'] - gold['start']) / 1000.0

    ch_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    ch_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#A65628']

    # Select typical segments
    typical = {}
    for cat in ACTIVITIES:
        segs = gold[(gold['category'] == cat) & (gold['user_id'].isin(user_data.keys()))]
        if len(segs) == 0:
            continue
        segs = segs.sort_values('duration_s')
        typical[cat] = segs.iloc[len(segs)//2]

    # Background segment
    bg_seg = None
    for uid, data in user_data.items():
        ulabels = gold[gold['user_id'] == uid].sort_values('start')
        if len(ulabels) > 0:
            first = ulabels.iloc[0]['start']
            ts = data[:, 0]
            bg_mask = ts < first
            bg_data = data[bg_mask]
            if len(bg_data) > 500:
                mid = len(bg_data) // 2
                bg_seg = {'uid': uid, 'data': bg_data[mid-250:mid+250, 1:],
                          'ts': np.arange(500) / 100.0}
                break

    # ── Fig 3: 6-channel signal comparison ──
    n_rows = len(ACTIVITIES) + (1 if bg_seg else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=False)

    panel_letters = list('abcdefghij')
    idx = 0
    if bg_seg:
        ax = axes[idx]
        sig = bg_seg['data'][:500]
        t = bg_seg['ts'][:len(sig)]
        for ch in range(6):
            ax.plot(t, sig[:, ch], color=ch_colors[ch], linewidth=0.5,
                    alpha=0.85, label=ch_names[ch])
        ax.set_ylabel('Background', fontsize=8)
        ax.set_title(panel_letters[idx], loc='left', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', ncol=6, fontsize=6, frameon=True)
        idx += 1

    for cat in ACTIVITIES:
        if cat not in typical:
            continue
        row = typical[cat]
        uid = row['user_id']
        data = user_data[uid]
        ts = data[:, 0]
        mid_t = (row['start'] + row['end']) / 2
        mask = (ts >= mid_t - 2500) & (ts <= mid_t + 2500)
        seg = data[mask]
        if len(seg) < 100:
            mask = (ts >= row['start']) & (ts <= row['end'])
            seg = data[mask][:500]

        ax = axes[idx]
        sig = seg[:, 1:]
        t_rel = (seg[:, 0] - seg[0, 0]) / 1000.0
        for ch in range(6):
            lbl = ch_names[ch] if idx == 0 and bg_seg is None else ''
            ax.plot(t_rel, sig[:, ch], color=ch_colors[ch], linewidth=0.5,
                    alpha=0.85, label=lbl)
        ax.set_ylabel(ACT_EN[cat], fontsize=8)
        ax.set_title(panel_letters[idx], loc='left', fontsize=11, fontweight='bold')
        if idx == 0 and bg_seg is None:
            ax.legend(loc='upper right', ncol=6, fontsize=6, frameon=True)
        idx += 1

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_typical_signals.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3_typical_signals.pdf'))
    plt.close(fig)
    print('  -> Saved fig3_typical_signals')

    # ── Fig 3b: ACC vs GYRO separate ──
    all_items = []
    if bg_seg:
        all_items.append(('Background', bg_seg['uid'], bg_seg['data'][:500], bg_seg['ts'][:500]))

    for cat in ACTIVITIES:
        if cat not in typical:
            continue
        row = typical[cat]
        uid = row['user_id']
        data = user_data[uid]
        ts = data[:, 0]
        mid_t = (row['start'] + row['end']) / 2
        mask = (ts >= mid_t - 2500) & (ts <= mid_t + 2500)
        seg = data[mask]
        if len(seg) < 100:
            mask = (ts >= row['start']) & (ts <= row['end'])
            seg = data[mask][:500]
        all_items.append((ACT_EN[cat], uid, seg[:, 1:], (seg[:, 0] - seg[0, 0]) / 1000.0))

    fig, axes = plt.subplots(len(all_items), 2, figsize=(12, 2.0 * len(all_items)))

    panel_idx = 0
    panel_letters = list('abcdefghijklmnopqrs')
    for i, (name, uid, sig, t_rel) in enumerate(all_items):
        # ACC
        ax = axes[i, 0]
        for ch, nm, c in zip([0,1,2], ['ACC_X','ACC_Y','ACC_Z'],
                             ['#E41A1C','#377EB8','#4DAF4A']):
            ax.plot(t_rel, sig[:, ch], color=c, linewidth=0.4, alpha=0.85, label=nm)
        ax.set_ylabel(f'{name}\n(ACC)', fontsize=7)
        ax.set_title(panel_letters[panel_idx], loc='left', fontsize=10, fontweight='bold')
        panel_idx += 1
        if i == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=5.5)

        # GYRO
        ax = axes[i, 1]
        for ch, nm, c in zip([3,4,5], ['GYRO_X','GYRO_Y','GYRO_Z'],
                             ['#FF7F00','#984EA3','#A65628']):
            ax.plot(t_rel, sig[:, ch], color=c, linewidth=0.4, alpha=0.85, label=nm)
        ax.set_ylabel(f'{name}\n(GYRO)', fontsize=7)
        ax.set_title(panel_letters[panel_idx], loc='left', fontsize=10, fontweight='bold')
        panel_idx += 1
        if i == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=5.5)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(OUT_DIR, 'fig3b_signals_acc_gyro.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3b_signals_acc_gyro.pdf'))
    plt.close(fig)
    print('  -> Saved fig3b_signals_acc_gyro')


# ═════════════════════════════════════════════════════════════════════════
# 5. Signal Statistics
# ═════════════════════════════════════════════════════════════════════════

def analyze_signal_statistics(user_data, gold):
    print('\n[5/5] Signal statistics analysis...')

    gold = gold.copy()
    records = []
    for _, row in gold.iterrows():
        uid = row['user_id']
        if uid not in user_data:
            continue
        data = user_data[uid]
        ts = data[:, 0]
        mask = (ts >= row['start']) & (ts <= row['end'])
        seg = data[mask]
        if len(seg) < 50:
            continue

        acc = seg[:, 1:4]
        gyro = seg[:, 4:7]
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))

        records.append({
            'category': row['category'],
            'acc_mag_mean': np.mean(acc_mag),
            'acc_rms': np.sqrt(np.mean(acc_mag**2)),
            'gyro_rms': np.sqrt(np.mean(gyro_mag**2)),
            'gyro_mag_mean': np.mean(gyro_mag),
            'acc_x_std': np.std(acc[:, 0]), 'acc_y_std': np.std(acc[:, 1]),
            'acc_z_std': np.std(acc[:, 2]), 'gyro_x_std': np.std(gyro[:, 0]),
            'gyro_y_std': np.std(gyro[:, 1]), 'gyro_z_std': np.std(gyro[:, 2]),
        })

    df = pd.DataFrame(records)
    print(f'  Computed features for {len(df)} segments')

    # ── Fig 4: Signal energy boxplots ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = [('acc_mag_mean', 'ACC Magnitude Mean', 'a'),
               ('acc_rms', 'ACC Signal Energy (RMS)', 'b'),
               ('gyro_rms', 'GYRO Signal Energy (RMS)', 'c')]

    for ax, (col, ylabel, panel) in zip(axes, metrics):
        data_box = [df[df['category'] == c][col].values for c in ACTIVITIES]
        bp = ax.boxplot(data_box, labels=LABELS_5, patch_artist=True, widths=0.5,
                        showfliers=True,
                        flierprops=dict(marker='o', markersize=2, alpha=0.4))
        for patch, color in zip(bp['boxes'], COLORS_5):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for m in bp['medians']:
            m.set_color('black')
            m.set_linewidth(1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(panel, loc='left', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT_DIR, 'fig4_signal_energy_boxplot.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig4_signal_energy_boxplot.pdf'))
    plt.close(fig)
    print('  -> Saved fig4_signal_energy_boxplot')

    # ── Fig 4b: Multi-axis features ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) 6-axis variability grouped bar
    ax = axes[0]
    axis_labels = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    axis_cols = ['acc_x_std', 'acc_y_std', 'acc_z_std',
                 'gyro_x_std', 'gyro_y_std', 'gyro_z_std']
    x = np.arange(len(axis_labels))
    width = 0.14
    for i, (cat, label) in enumerate(zip(ACTIVITIES, LABELS_5)):
        sub = df[df['category'] == cat]
        means = [sub[c].mean() for c in axis_cols]
        ax.bar(x + i * width, means, width, label=label, color=COLORS_5[i],
               edgecolor='white', linewidth=0.2)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(axis_labels, fontsize=7.5)
    ax.set_ylabel('Mean Std. Deviation')
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left', ncol=2)

    # (b) ACC vs GYRO scatter
    ax = axes[1]
    for i, (cat, label) in enumerate(zip(ACTIVITIES, LABELS_5)):
        sub = df[df['category'] == cat]
        ax.scatter(sub['acc_mag_mean'], sub['gyro_mag_mean'],
                   c=COLORS_5[i], label=label, alpha=0.55, s=22,
                   edgecolors='white', linewidths=0.3)
    ax.set_xlabel('ACC Magnitude Mean')
    ax.set_ylabel('GYRO Magnitude Mean')
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, 'fig4b_multiaxis_features.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig4b_multiaxis_features.pdf'))
    plt.close(fig)
    print('  -> Saved fig4b_multiaxis_features')

    return df


# ═════════════════════════════════════════════════════════════════════════
# 6. Augmentation Illustration (Moved from sketch_figures)
# ═════════════════════════════════════════════════════════════════════════

def fig6_augmentation_illustration(user_data, gold):
    """Fig 6: Data augmentation illustration - show original vs augmented signals (Real Data)."""
    print('\n[6/5] Augmentation illustration (Real Data)...')
    
    # Local colors to match original figure
    COLORS = {
        'primary': '#2171B5',
        'secondary': '#6BAED6',
        'accent': '#CB181D',
        'green': '#238B45',
        'orange': '#D94801',
        'purple': '#6A51A3',
        'gray': '#969696',
        'light_gray': '#D9D9D9',
    }
    
    # 1. Select a real Badminton segment (Base Signal)
    cat = '羽毛球'
    candidates = gold[(gold['category'] == cat) & (gold['user_id'].isin(user_data.keys()))]
    candidates = candidates.copy()
    candidates['dur'] = candidates['end'] - candidates['start']
    # Filter > 3s
    candidates = candidates[candidates['dur'] > 3500]
    
    signal = None
    if len(candidates) > 0:
        # Calculate RMS for each candidate to find "typical" intensity
        rms_values = []
        valid_indices = []
        
        for idx, row in candidates.iterrows():
            uid = row['user_id']
            data = user_data[uid]
            ts = data[:, 0]
            # Extract segment
            mask = (ts >= row['start']) & (ts <= row['end'])
            seg = data[mask]
            if len(seg) < 100: continue
            
            # ACC magnitude RMS
            acc_mag = np.sqrt(np.sum(seg[:, 1:4]**2, axis=1))
            rms = np.sqrt(np.mean(acc_mag**2))
            rms_values.append(rms)
            valid_indices.append(idx)
            
        if len(rms_values) > 0:
            # Pick median RMS segment
            sorted_pairs = sorted(zip(rms_values, valid_indices))
            median_idx = sorted_pairs[len(sorted_pairs)//2][1]
            row = candidates.loc[median_idx]
            
            uid = row['user_id']
            data = user_data[uid]
            ts = data[:, 0]
            
            # Extract 3s from middle
            mid = (row['start'] + row['end']) / 2
            # Find index closest to mid
            idx_mid = np.searchsorted(ts, mid)
            # Take 150 samples before and 150 after (approx 3s at 100Hz)
            start_idx = max(0, idx_mid - 150)
            end_idx = min(len(data), start_idx + 300)
            
            # Adjust if we hit end
            if end_idx - start_idx < 300:
                start_idx = max(0, end_idx - 300)
                
            seg_data = data[start_idx:end_idx]
            
            # Use ACC_Z (index 3)
            signal = seg_data[:, 3].copy()
            # Normalize to g (assuming input is mg)
            signal = signal / 1000.0


    
    if signal is None:
        # Fallback to simulation if no data (unlikely)
        print("  Warning: No suitable real data found, using simulation fallback.")
        t = np.linspace(0, 3, 300)
        signal = 2 * np.sin(2 * np.pi * 3 * t) * np.exp(-0.5 * ((t - 1.5) / 0.5)**2) + 0.5 * np.sin(2 * np.pi * 8 * t)

    # Ensure length is 300 for consistency with fixed time axis
    if len(signal) > 300: signal = signal[:300]
    if len(signal) < 300:
        signal = np.pad(signal, (0, 300 - len(signal)), 'edge')

    t = np.linspace(0, 3, 300)

    # 2. Prepare mixup signal (Running)
    cat2 = '跑步'
    cand2 = gold[(gold['category'] == cat2) & (gold['user_id'].isin(user_data.keys()))]
    if len(cand2) > 0:
        row2 = cand2.iloc[0]
        data2 = user_data[row2['user_id']]
        # Just take a 300 chunk from middle
        mid2 = len(data2) // 2
        sig2_full = data2[mid2:mid2+300, 3] # ACC_Z
        if len(sig2_full) < 300:
             sig2_full = np.pad(sig2_full, (0, 300-len(sig2_full)), 'edge')
        signal2 = sig2_full / 1000.0
    else:
        signal2 = np.sin(2 * np.pi * 5 * t) # Fallback

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.0), sharex=True, sharey=True)
    fig.suptitle('Fig. 6  Data augmentation strategies applied to Real ACC signal (Badminton)',
                 fontsize=9, fontweight='bold', x=0.01, ha='left')

    titles = ['a  Original (Real)', 'b  Amplitude Scale (0.8x)', 'c  Gaussian Noise (SNR=20dB)',
              'd  Time Shift (+30)', 'e  Time Warp (1.1x)', 'f  Mixup ($\\lambda$=0.7)']

    # Augmented versions
    sig_amp = signal * 0.8
    
    # Noise: SNR = 10log10(P_signal / P_noise) -> P_noise = P_signal / 10^(SNR/10)
    noise = np.random.randn(300) * np.sqrt(np.mean(signal**2) / 100)
    sig_noise = signal + noise
    
    sig_shift = np.roll(signal, 30) # Shift 30 samples (0.3s)
    
    # Time warp
    x_orig = np.linspace(0, 1, 300)
    x_str = np.linspace(0, 1, int(300 * 1.1)) # Stretch
    f_interp = interp1d(x_orig, signal, kind='linear', fill_value='extrapolate')
    sig_warped_full = f_interp(x_str)
    # Resample back to 300 to plot on same axis
    f_back = interp1d(np.linspace(0, 1, len(sig_warped_full)), sig_warped_full, kind='linear', fill_value='extrapolate')
    sig_warp = f_back(x_orig)

    # Mixup
    # Ensure signal2 is same shape
    if signal2.shape != signal.shape:
        signal2 = np.resize(signal2, signal.shape)
    sig_mixup = 0.7 * signal + 0.3 * signal2

    signals = [signal, sig_amp, sig_noise, sig_shift, sig_warp, sig_mixup]

    for ax, sig, title in zip(axes.flat, signals, titles):
        ax.plot(t, sig, color=COLORS['primary'], linewidth=0.5, alpha=0.9)
        ax.set_title(title, fontsize=6.5, loc='left')
        ax.set_xlim(0, 3)
        if ax in axes[1]:
            ax.set_xlabel('Time (s)', fontsize=6.5)
        if ax in [axes[0][0], axes[1][0]]:
            ax.set_ylabel('ACC Z (g)', fontsize=6.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, 'fig6_augmentation.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig6_augmentation.pdf'))
    plt.close(fig)
    print('  -> Saved fig6_augmentation')


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  EDA - Wearable Activity Recognition')
    print('=' * 60)

    user_data, gold, readable, binary = load_all_data()
    df_stats = analyze_completeness(user_data, gold, readable, binary)
    seg_counts, dur_by_cat = analyze_activity_distribution(gold)
    visualize_typical_signals(user_data, gold)
    df_feat = analyze_signal_statistics(user_data, gold)
    fig6_augmentation_illustration(user_data, gold)

    print('\n' + '=' * 60)
    print('  All figures saved to:', OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f'    {f}')
    print('  Done!')


if __name__ == '__main__':
    main()
