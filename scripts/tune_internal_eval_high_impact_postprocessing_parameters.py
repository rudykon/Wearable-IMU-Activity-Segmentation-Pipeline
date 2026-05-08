"""Focused grid search for key temporal post-processing parameters.

Purpose:
    Runs a smaller sweep over high-impact decoding controls such as smoothing,
    cross-activity transition penalties, and median filtering.
Inputs:
    Uses cached or freshly computed model predictions together with internal_eval
    annotations.
Outputs:
    Reports the best-performing parameter settings for local tuning.
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
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

from imu_activity_pipeline.config import *
from imu_activity_pipeline.sensor_data_processing import load_gold_labels, load_sensor_data
from imu_activity_pipeline.inference import (
    load_ensemble_models, predict_multiscale_ensemble,
    extract_segments, merge_same_class_segments,
    filter_short_segments, resolve_overlaps, refine_boundaries,
)
from imu_activity_pipeline.evaluate import calculate_iou
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt

def viterbi_tunable(probs, self_trans=0.97, bg_to_act=0.01,
                    act_to_bg=0.05, cross_act=0.001):
    N, n_states = probs.shape
    trans = np.full((n_states, n_states), 0.001)
    for i in range(n_states): trans[i, i] = self_trans
    for i in range(1, n_states):
        trans[0, i] = bg_to_act; trans[i, 0] = act_to_bg
    for i in range(1, n_states):
        for j in range(1, n_states):
            if i != j: trans[i, j] = cross_act
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
    for t in range(N-2, -1, -1): path[t] = backpointer[t+1, path[t+1]]
    return path

def select_top_k(segments, k=3):
    if len(segments) <= k: return segments
    by_class = {}
    for s in segments:
        c = s['class_idx']
        if c not in by_class or s['confidence'] > by_class[c]['confidence']:
            by_class[c] = s
    sel = list(by_class.values())
    if len(sel) >= k:
        sel.sort(key=lambda s: s['confidence'], reverse=True)
        return sel[:k]
    rem = [s for s in segments if s not in sel]
    rem.sort(key=lambda s: s['confidence'], reverse=True)
    for s in rem:
        if len(sel) >= k: break
        sel.append(s)
    return sorted(sel, key=lambda s: s['start_ts'])

def run_eval(user_data, model_groups, device, gold_df, params, precomputed=None):
    all_results = []
    for uid, data in user_data.items():
        if precomputed and uid in precomputed:
            timestamps, probs = precomputed[uid]
        else:
            timestamps, probs = predict_multiscale_ensemble(data, model_groups, device)
        if len(timestamps) == 0: continue

        sw = params['smooth_window']
        smoothed = np.copy(probs)
        for c in range(probs.shape[1]):
            smoothed[:, c] = uniform_filter1d(probs[:, c], size=sw, mode='nearest')
        mf = params.get('median_filter', 0)
        if mf > 0:
            if mf % 2 == 0: mf += 1
            for c in range(smoothed.shape[1]):
                smoothed[:, c] = medfilt(smoothed[:, c], kernel_size=mf)

        path = viterbi_tunable(smoothed,
            self_trans=params['self_trans'], bg_to_act=params['bg_to_act'],
            act_to_bg=params['act_to_bg'], cross_act=params['cross_act'])
        segments = extract_segments(path, timestamps, smoothed)
        segments = merge_same_class_segments(segments, gap_threshold_sec=params['gap_threshold'])
        segments = refine_boundaries(segments, data, data[:, 0])
        segments = resolve_overlaps(segments)
        segments = filter_short_segments(segments, min_duration_sec=params['min_duration'])
        if params.get('top_k', 0) > 0:
            segments = select_top_k(segments, k=params['top_k'])
        if params.get('conf_min', 0) > 0:
            segments = [s for s in segments if s['confidence'] >= params['conf_min']]
        all_results.append((uid, segments))

    user_f1s = []
    details = {}
    for user_id in gold_df['user_id'].unique():
        g_segs = gold_df[gold_df['user_id'] == user_id]
        G = [{'start': r['start'], 'end': r['end'], 'category': r['category']}
             for _, r in g_segs.iterrows()]
        P = []
        for uid, segs in all_results:
            if uid == user_id:
                for s in segs:
                    P.append({'start': s['start_ts'], 'end': s['end_ts'], 'category': s['class_name']})
        matches = []
        for i, p in enumerate(P):
            for j, g in enumerate(G):
                if p['category'] == g['category']:
                    iou = calculate_iou(p['start'], p['end'], g['start'], g['end'])
                    if iou > 0.5: matches.append({'p': i, 'g': j, 'iou': iou})
        matches.sort(key=lambda x: x['iou'], reverse=True)
        mp, mg = set(), set()
        TP = 0
        for m in matches:
            if m['p'] not in mp and m['g'] not in mg:
                TP += 1; mp.add(m['p']); mg.add(m['g'])
        FP, FN = len(P) - len(mp), len(G) - len(mg)
        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        user_f1s.append(f1)
        details[user_id] = {'TP': TP, 'FP': FP, 'FN': FN, 'F1': f1}
    return np.mean(user_f1s), details, all_results

def main():
    print("Loading models & internal_eval data...")
    model_groups, device = load_ensemble_models()
    from imu_activity_pipeline.signal_file_reader import DataReader
    reader = DataReader(INTERNAL_EVAL_DATA_DIR)
    internal_eval_files = reader.read_data()
    user_data = {}
    for uid in sorted(internal_eval_files.keys()):
        fpath = os.path.join(INTERNAL_EVAL_DATA_DIR, f"{uid}.txt")
        data = load_sensor_data(fpath)
        if data is not None and len(data) >= WINDOW_SIZE:
            user_data[uid] = data
    gold_df = load_gold_labels(INTERNAL_EVAL_GOLD_FILE)
    gold_df['user_id'] = gold_df['user_id'].astype(str)

    # Precompute ensemble predictions (most expensive step)
    print("Precomputing ensemble predictions...")
    precomputed = {}
    for uid, data in user_data.items():
        ts, probs = predict_multiscale_ensemble(data, model_groups, device)
        precomputed[uid] = (ts, probs)
    print(f"Done. {len(precomputed)} users cached.\n")

    # Grid search
    smooth_vals = [5, 7, 9, 11, 13, 15]
    cross_vals = [0.0005, 0.001, 0.002, 0.003]
    median_vals = [0, 3, 5, 7]
    topk_vals = [0, 3, 4]
    self_trans_vals = [0.95, 0.97, 0.98]
    min_dur_vals = [120, 150, 180]
    gap_vals = [60, 90, 120]
    conf_vals = [0.0, 0.3, 0.4]

    # Full grid is too big. Do iterative refinement.
    # Start with Phase 2 best combined params
    best_params = {
        'smooth_window': 9, 'self_trans': 0.97, 'bg_to_act': 0.01,
        'act_to_bg': 0.05, 'cross_act': 0.001, 'gap_threshold': 60,
        'min_duration': 180, 'top_k': 3, 'conf_min': 0.4, 'median_filter': 5,
    }
    best_f1 = 0.9333

    print("=== Iterative refinement from combined best (F1=0.9333) ===\n")

    # Sweep each param around best value
    all_sweeps = [
        ('smooth_window', [7, 9, 11, 13, 15, 17, 19, 21]),
        ('cross_act', [0.0003, 0.0005, 0.001, 0.0015, 0.002]),
        ('median_filter', [0, 3, 5, 7, 9]),
        ('self_trans', [0.95, 0.96, 0.97, 0.98, 0.99]),
        ('top_k', [0, 3, 4, 5]),
        ('min_duration', [100, 120, 150, 180, 200]),
        ('gap_threshold', [30, 60, 90, 120, 180, 240]),
        ('conf_min', [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]),
        ('act_to_bg', [0.02, 0.03, 0.05, 0.07, 0.10]),
        ('bg_to_act', [0.005, 0.01, 0.015, 0.02]),
    ]

    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        print(f"\n--- Iteration {iteration} (current best F1={best_f1:.4f}) ---")
        for param_name, values in all_sweeps:
            local_best_f1 = best_f1
            local_best_val = best_params[param_name]
            for v in values:
                p = best_params.copy()
                p[param_name] = v
                f1, _, _ = run_eval(user_data, model_groups, device, gold_df, p, precomputed)
                if f1 > local_best_f1:
                    local_best_f1 = f1
                    local_best_val = v
            if local_best_f1 > best_f1:
                print(f"  {param_name}: {best_params[param_name]} -> {local_best_val} (F1: {best_f1:.4f} -> {local_best_f1:.4f})")
                best_f1 = local_best_f1
                best_params[param_name] = local_best_val
                improved = True
            else:
                print(f"  {param_name}: no improvement (best={best_params[param_name]})")

    print(f"\n{'='*70}")
    print(f"FINAL BEST F1 = {best_f1:.4f}")
    print(f"BEST PARAMS:")
    for k, v in sorted(best_params.items()):
        print(f"  {k} = {v}")
    print(f"{'='*70}")

    # Final breakdown
    f1, details, results = run_eval(user_data, model_groups, device, gold_df, best_params, precomputed)
    print(f"\nPer-user breakdown (F1={f1:.4f}):")
    for u, d in sorted(details.items()):
        s = "OK" if d['F1'] >= 1.0 else "!!"
        print(f"  [{s}] {u}: TP={d['TP']} FP={d['FP']} FN={d['FN']} F1={d['F1']:.3f}")

if __name__ == '__main__':
    main()
