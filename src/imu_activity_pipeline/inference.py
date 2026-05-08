"""End-to-end inference and temporal post-processing pipeline.

Purpose:
    Loads selected multi-scale checkpoints, converts raw IMU streams into sliding
    windows, runs neural-network inference, fuses scale probabilities, decodes
    temporal labels, refines boundaries, filters segments, and writes results.
Inputs:
    Uses saved model assets from `MODEL_DIR` and signal files from the configured
    split data directory.
Outputs:
    Produces activity-segment rows in `user_id`, `category`, `start`, `end`
    format through the `DataOutput` writer.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from scipy.interpolate import interp1d

from .config import *
from .sensor_data_processing import load_sensor_data, create_windows, normalize_imu
from .neural_network_models import CombinedModel

DEFAULT_FUSION_MODE = 'local_boundary'


def _load_checkpoint(path: str, map_location):
    """Load trusted local checkpoints without triggering torch.load safety warnings."""
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


def _select_best_models_by_scale(ensemble_config: Dict) -> Dict[str, str]:
    """Choose one best model per window scale using explicit config or val_f1 fallback."""
    selected = ensemble_config.get('selected_models')
    if selected:
        return selected

    selected = {}
    val_f1_map = ensemble_config.get('val_f1', {})
    for wc in ensemble_config.get('window_configs', []):
        suffix = wc['suffix']
        candidates = [m for m in ensemble_config.get('models', []) if f'_{suffix}_' in m]
        if not candidates:
            continue
        selected[suffix] = max(candidates, key=lambda name: val_f1_map.get(name, float('-inf')))
    return selected


def load_ensemble_models():
    """Load one best model per scale and prepare multi-scale inference groups.

    Returns:
        model_groups: dict mapping window_suffix -> list of (model, window_size, norm_params)
        device: torch device
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    # Load ensemble config
    config_path = os.path.join(MODEL_DIR, 'ensemble_config.json')
    if not os.path.exists(config_path):
        # Fallback: load single best model
        print("No ensemble config found, falling back to single model")
        return _load_single_model_fallback(device)

    with open(config_path, 'r') as f:
        ensemble_config = json.load(f)

    model_groups = {}
    selected_models = _select_best_models_by_scale(ensemble_config)

    for wc in ensemble_config.get('window_configs', []):
        suffix = wc['suffix']
        ws = wc['window_size']
        ws_step = wc['window_step']
        ws_sec = wc['window_sec']

        # Load norm params for this window size
        norm_file = os.path.join(MODEL_DIR, f'norm_params_{suffix}.pkl')
        if not os.path.exists(norm_file):
            print(f"  Warning: norm params not found for {suffix}, skipping")
            continue
        with open(norm_file, 'rb') as f:
            norm_params = pickle.load(f)

        model_name = selected_models.get(suffix)
        if not model_name:
            print(f"  Warning: no selected model found for {suffix}, skipping")
            continue

        model_path = os.path.join(MODEL_DIR, f'{model_name}.pth')
        if not os.path.exists(model_path):
            print(f"  Warning: selected model {model_name} not found, skipping")
            continue

        model = CombinedModel(input_channels=6, num_classes=6, window_size=ws).to(device)
        checkpoint = _load_checkpoint(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        val_f1 = checkpoint.get('val_f1', ensemble_config.get('val_f1', {}).get(model_name, 'N/A'))
        print(f"  Loaded best {suffix} model: {model_name} (val_f1={val_f1})")

        if model is not None:
            model_groups[suffix] = {
                'models': [model],
                'model_name': model_name,
                'model_val_f1': val_f1,
                'window_size': ws,
                'window_step': ws_step,
                'window_sec': ws_sec,
                'norm_params': norm_params,
            }

    if not model_groups:
        print("No models loaded from ensemble config, falling back to single model")
        return _load_single_model_fallback(device)

    total_models = sum(len(g['models']) for g in model_groups.values())
    print(f"Loaded {total_models} selected models across {len(model_groups)} window scales")
    return model_groups, device


def _load_single_model_fallback(device):
    """Fallback: load the single best model (backward compatible)."""
    with open(os.path.join(MODEL_DIR, 'norm_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)

    model = CombinedModel(input_channels=6, num_classes=6, window_size=WINDOW_SIZE).to(device)
    checkpoint = _load_checkpoint(os.path.join(MODEL_DIR, 'combined_model_best.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded fallback model (val_f1={checkpoint.get('val_f1', 'N/A')})")

    model_groups = {
        '3s': {
            'models': [model],
            'window_size': WINDOW_SIZE,
            'window_step': WINDOW_STEP,
            'window_sec': WINDOW_SIZE_SEC,
            'norm_params': norm_params,
        }
    }
    return model_groups, device


def predict_windows_ensemble(models, windows, device, batch_size=512):
    """Run multiple models on windows and average softmax probabilities.

    Args:
        models: list of torch models
        windows: numpy array (N, window_size, 6)
        device: torch device

    Returns:
        averaged probabilities (N, 6)
    """
    all_model_probs = []

    dataset = torch.FloatTensor(windows)
    n = len(dataset)

    for model in models:
        model.eval()
        model_probs = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = dataset[i:i + batch_size].to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu().numpy())
        all_model_probs.append(np.concatenate(model_probs, axis=0))

    # Average across models
    stacked = np.stack(all_model_probs, axis=0)  # (num_models, N, 6)
    avg_probs = np.mean(stacked, axis=0)  # (N, 6)
    return avg_probs


def predict_windows(model, windows, device, batch_size=512):
    """Backward-compatible single-model prediction wrapper."""
    return predict_windows_ensemble([model], windows, device, batch_size=batch_size)


def _align_scale_probabilities(scale_results: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Align all scale probabilities to the finest timestamp grid."""
    if not scale_results:
        return np.array([]), {}

    ref_idx = max(range(len(scale_results)), key=lambda i: len(scale_results[i]['timestamps']))
    ref_timestamps = scale_results[ref_idx]['timestamps']

    aligned_probs = {}
    for sr in scale_results:
        suffix = sr['suffix']
        if len(sr['timestamps']) == len(ref_timestamps) and np.array_equal(sr['timestamps'], ref_timestamps):
            aligned_probs[suffix] = sr['probs']
            continue

        interp_probs = np.zeros((len(ref_timestamps), 6), dtype=np.float32)
        for c in range(6):
            f = interp1d(
                sr['timestamps'],
                sr['probs'][:, c],
                kind='linear',
                bounds_error=False,
                fill_value=(sr['probs'][0, c], sr['probs'][-1, c]),
            )
            interp_probs[:, c] = f(ref_timestamps)
        aligned_probs[suffix] = interp_probs

    return ref_timestamps, aligned_probs


def _dynamic_boundary_fusion(aligned_probs: Dict[str, np.ndarray]) -> np.ndarray:
    """Fuse scales with stronger 3s emphasis near likely boundaries.

    Heuristic:
    - In stable regions, trust longer windows more.
    - Near boundaries or when scales disagree, shift weight toward 3s.
    """
    suffixes = list(aligned_probs.keys())
    if len(suffixes) == 1:
        return aligned_probs[suffixes[0]]

    ref = next(iter(aligned_probs.values()))
    n_steps = len(ref)
    base_weights = {'3s': 0.20, '5s': 0.35, '8s': 0.45}

    present = {k: v for k, v in base_weights.items() if k in aligned_probs}
    base = np.array([present[k] for k in present], dtype=np.float32)
    base = base / np.sum(base)

    probs_by_suffix = {k: aligned_probs[k] for k in present}
    ordered_suffixes = list(present.keys())

    boundary_score = np.zeros(n_steps, dtype=np.float32)

    if '3s' in probs_by_suffix:
        probs_3s = probs_by_suffix['3s']
        pred_3s = np.argmax(probs_3s, axis=1)
        top2 = np.partition(probs_3s, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]

        left_change = np.zeros(n_steps, dtype=np.float32)
        right_change = np.zeros(n_steps, dtype=np.float32)
        left_change[1:] = (pred_3s[1:] != pred_3s[:-1]).astype(np.float32)
        right_change[:-1] = (pred_3s[:-1] != pred_3s[1:]).astype(np.float32)
        class_change = np.maximum(left_change, right_change)

        ambiguity = np.clip((0.18 - margin) / 0.18, 0.0, 1.0).astype(np.float32)

        delta = np.zeros(n_steps, dtype=np.float32)
        delta[1:] = np.mean(np.abs(probs_3s[1:] - probs_3s[:-1]), axis=1)
        delta = uniform_filter1d(delta, size=5, mode='nearest')
        delta = np.clip(delta / 0.12, 0.0, 1.0).astype(np.float32)

        boundary_score = np.maximum.reduce([class_change, ambiguity, delta])

    if len(ordered_suffixes) > 1:
        preds = np.stack([np.argmax(probs_by_suffix[k], axis=1) for k in ordered_suffixes], axis=0)
        disagreement = 1.0 - np.max(
            np.stack([(preds == cls).mean(axis=0) for cls in range(6)], axis=0),
            axis=0,
        )
        boundary_score = np.maximum(boundary_score, disagreement.astype(np.float32))

    weights = np.tile(base[None, :], (n_steps, 1))

    if '3s' in ordered_suffixes:
        idx_3s = ordered_suffixes.index('3s')
        weights[:, idx_3s] += 0.35 * boundary_score
    if '8s' in ordered_suffixes:
        idx_8s = ordered_suffixes.index('8s')
        weights[:, idx_8s] -= 0.30 * boundary_score
    if '5s' in ordered_suffixes:
        idx_5s = ordered_suffixes.index('5s')
        weights[:, idx_5s] += 0.05 * (1.0 - np.abs(2.0 * boundary_score - 1.0))

    weights = np.clip(weights, 0.05, None)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    fused = np.zeros_like(ref, dtype=np.float32)
    for i, suffix in enumerate(ordered_suffixes):
        fused += weights[:, i:i + 1] * probs_by_suffix[suffix]
    return fused


def _local_boundary_window_fusion(aligned_probs: Dict[str, np.ndarray]) -> np.ndarray:
    """Use average fusion globally, but boost 3s only around explicit local boundaries."""
    suffixes = list(aligned_probs.keys())
    if len(suffixes) == 1 or '3s' not in aligned_probs:
        return next(iter(aligned_probs.values()))

    ordered_suffixes = [k for k in ['3s', '5s', '8s'] if k in aligned_probs]
    probs = {k: aligned_probs[k] for k in ordered_suffixes}
    ref = probs[ordered_suffixes[0]]
    n_steps = len(ref)

    base_weights = {'3s': 0.20, '5s': 0.35, '8s': 0.45}
    base = np.array([base_weights[k] for k in ordered_suffixes], dtype=np.float32)
    base = base / np.sum(base)
    weights = np.tile(base[None, :], (n_steps, 1))

    probs_3s = probs['3s']
    pred_3s = np.argmax(probs_3s, axis=1)
    boundary_mask = np.zeros(n_steps, dtype=np.float32)
    boundary_points = np.where(pred_3s[1:] != pred_3s[:-1])[0] + 1
    radius = 3
    for idx in boundary_points:
        left = max(0, idx - radius)
        right = min(n_steps, idx + radius + 1)
        boundary_mask[left:right] = 1.0
    boundary_mask = uniform_filter1d(boundary_mask, size=3, mode='nearest')
    boundary_mask = np.clip(boundary_mask, 0.0, 1.0)

    idx_3s = ordered_suffixes.index('3s')
    weights[:, idx_3s] += 0.30 * boundary_mask
    if '8s' in ordered_suffixes:
        idx_8s = ordered_suffixes.index('8s')
        weights[:, idx_8s] -= 0.22 * boundary_mask
    if '5s' in ordered_suffixes:
        idx_5s = ordered_suffixes.index('5s')
        weights[:, idx_5s] -= 0.08 * boundary_mask

    weights = np.clip(weights, 0.05, None)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    fused = np.zeros_like(ref, dtype=np.float32)
    for i, suffix in enumerate(ordered_suffixes):
        fused += weights[:, i:i + 1] * probs[suffix]
    return fused


def _confident_conflict_fusion(aligned_probs: Dict[str, np.ndarray]) -> np.ndarray:
    """Boost 3s only when scales disagree and 3s is clearly more confident."""
    suffixes = list(aligned_probs.keys())
    if len(suffixes) == 1 or '3s' not in aligned_probs:
        return next(iter(aligned_probs.values()))

    ordered_suffixes = [k for k in ['3s', '5s', '8s'] if k in aligned_probs]
    probs = {k: aligned_probs[k] for k in ordered_suffixes}
    ref = probs[ordered_suffixes[0]]
    n_steps = len(ref)

    base_weights = {'3s': 0.20, '5s': 0.35, '8s': 0.45}
    base = np.array([base_weights[k] for k in ordered_suffixes], dtype=np.float32)
    base = base / np.sum(base)
    weights = np.tile(base[None, :], (n_steps, 1))

    probs_3s = probs['3s']
    pred_3s = np.argmax(probs_3s, axis=1)
    top2_3s = np.partition(probs_3s, -2, axis=1)[:, -2:]
    margin_3s = top2_3s[:, 1] - top2_3s[:, 0]
    top1_3s = np.max(probs_3s, axis=1)

    disagreement = np.zeros(n_steps, dtype=np.float32)
    if '5s' in probs:
        disagreement = np.maximum(disagreement, (pred_3s != np.argmax(probs['5s'], axis=1)).astype(np.float32))
    if '8s' in probs:
        disagreement = np.maximum(disagreement, (pred_3s != np.argmax(probs['8s'], axis=1)).astype(np.float32))

    margin_adv = np.zeros(n_steps, dtype=np.float32)
    top1_adv = np.zeros(n_steps, dtype=np.float32)
    for suffix in ['5s', '8s']:
        if suffix not in probs:
            continue
        other = probs[suffix]
        top2_other = np.partition(other, -2, axis=1)[:, -2:]
        margin_other = top2_other[:, 1] - top2_other[:, 0]
        top1_other = np.max(other, axis=1)
        margin_adv = np.maximum(margin_adv, (margin_3s - margin_other).astype(np.float32))
        top1_adv = np.maximum(top1_adv, (top1_3s - top1_other).astype(np.float32))

    confidence_gate = (
        (disagreement > 0.0)
        & (margin_3s >= 0.12)
        & (margin_adv >= 0.05)
        & (top1_adv >= 0.03)
    ).astype(np.float32)
    confidence_gate = uniform_filter1d(confidence_gate, size=3, mode='nearest')
    confidence_gate = np.clip(confidence_gate, 0.0, 1.0)

    idx_3s = ordered_suffixes.index('3s')
    weights[:, idx_3s] += 0.28 * confidence_gate
    if '8s' in ordered_suffixes:
        idx_8s = ordered_suffixes.index('8s')
        weights[:, idx_8s] -= 0.20 * confidence_gate
    if '5s' in ordered_suffixes:
        idx_5s = ordered_suffixes.index('5s')
        weights[:, idx_5s] -= 0.08 * confidence_gate

    weights = np.clip(weights, 0.05, None)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    fused = np.zeros_like(ref, dtype=np.float32)
    for i, suffix in enumerate(ordered_suffixes):
        fused += weights[:, i:i + 1] * probs[suffix]
    return fused


def _fixed_weight_fusion(aligned_probs: Dict[str, np.ndarray], weight_map: Dict[str, float]) -> np.ndarray:
    """Fuse scales with fixed per-scale weights."""
    suffixes = [k for k in ['3s', '5s', '8s'] if k in aligned_probs]
    if len(suffixes) == 1:
        return aligned_probs[suffixes[0]]

    ref = aligned_probs[suffixes[0]]
    weights = np.array([weight_map[k] for k in suffixes], dtype=np.float32)
    weights = weights / np.sum(weights)

    fused = np.zeros_like(ref, dtype=np.float32)
    for i, suffix in enumerate(suffixes):
        fused += weights[i] * aligned_probs[suffix]
    return fused


def predict_multiscale_ensemble(data, model_groups, device, fusion_mode: str = DEFAULT_FUSION_MODE):
    """Run multi-scale ensemble with configurable cross-scale fusion.

    Args:
        data: raw sensor data (N, 7)
        model_groups: dict from load_ensemble_models()
        device: torch device
        fusion_mode: default is 'local_boundary'; supported: 'average', 'dynamic_boundary', 'local_boundary',
            'confident_conflict', 'weighted_long', or 'weighted_balanced'

    Returns:
        timestamps: aligned timestamps from the finest resolution
        fused_probs: (T, 6) fused probability matrix
    """
    scale_results = []

    for suffix, group in model_groups.items():
        ws = group['window_size']
        ws_step = group['window_step']
        norm_params = group['norm_params']
        models = group['models']

        # Create windows at this scale
        timestamps, windows = create_windows(data, ws, ws_step)
        if len(windows) == 0:
            continue

        # Normalize
        norm_windows, _, _ = normalize_imu(windows, norm_params['mean'], norm_params['std'])

        # Predict with ensemble
        probs = predict_windows_ensemble(models, norm_windows, device)

        scale_results.append({
            'suffix': suffix,
            'timestamps': timestamps,
            'probs': probs,
        })

    if not scale_results:
        return np.array([]), np.array([])

    if len(scale_results) == 1:
        return scale_results[0]['timestamps'], scale_results[0]['probs']

    ref_timestamps, aligned_probs = _align_scale_probabilities(scale_results)
    if fusion_mode == 'dynamic_boundary':
        return ref_timestamps, _dynamic_boundary_fusion(aligned_probs)
    if fusion_mode == 'local_boundary':
        return ref_timestamps, _local_boundary_window_fusion(aligned_probs)
    if fusion_mode == 'confident_conflict':
        return ref_timestamps, _confident_conflict_fusion(aligned_probs)
    if fusion_mode == 'weighted_long':
        return ref_timestamps, _fixed_weight_fusion(
            aligned_probs,
            {'3s': 0.15, '5s': 0.30, '8s': 0.55},
        )
    if fusion_mode == 'weighted_balanced':
        return ref_timestamps, _fixed_weight_fusion(
            aligned_probs,
            {'3s': 0.25, '5s': 0.35, '8s': 0.40},
        )

    stacked = np.stack(list(aligned_probs.values()), axis=0)
    avg_probs = np.mean(stacked, axis=0)
    return ref_timestamps, avg_probs


def smooth_predictions(probs: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Apply temporal smoothing to prediction probabilities."""
    smooth_size = 7  # tuned from STAGE1_SMOOTH_WINDOW(5)
    smoothed = np.copy(probs)
    for c in range(probs.shape[1]):
        smoothed[:, c] = uniform_filter1d(probs[:, c], size=smooth_size, mode='nearest')
    # Median filter to remove impulse noise (tuned: kernel=5)
    for c in range(smoothed.shape[1]):
        smoothed[:, c] = medfilt(smoothed[:, c], kernel_size=5)
    return smoothed


def viterbi_decode(probs: np.ndarray, n_states: int = 6) -> np.ndarray:
    """Apply Viterbi decoding to enforce temporal consistency.

    States: 0=background, 1-5=activities
    """
    N = len(probs)
    if N == 0:
        return np.array([])

    # Transition matrix
    trans = np.full((n_states, n_states), 0.001)

    # Self-transition (tuned from 0.90)
    for i in range(n_states):
        trans[i, i] = 0.97

    # Background <-> Activity transitions
    for i in range(1, n_states):
        trans[0, i] = 0.01
        trans[i, 0] = 0.05

    # Activity <-> Activity (tuned from 0.005)
    for i in range(1, n_states):
        for j in range(1, n_states):
            if i != j:
                trans[i, j] = 0.001

    # Normalize rows
    trans = trans / trans.sum(axis=1, keepdims=True)

    # Log probabilities for numerical stability
    log_trans = np.log(trans + 1e-10)
    log_probs = np.log(probs + 1e-10)

    # Initial probability (uniform)
    log_init = np.log(np.ones(n_states) / n_states)

    # Viterbi
    V = np.zeros((N, n_states))
    backpointer = np.zeros((N, n_states), dtype=int)

    V[0] = log_init + log_probs[0]

    for t in range(1, N):
        for s in range(n_states):
            transitions = V[t - 1] + log_trans[:, s]
            best_prev = np.argmax(transitions)
            V[t, s] = transitions[best_prev] + log_probs[t, s]
            backpointer[t, s] = best_prev

    # Backtrace
    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(V[-1])
    for t in range(N - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path


def extract_segments(path: np.ndarray, timestamps: np.ndarray,
                     probs: np.ndarray, window_sec: int = WINDOW_SIZE_SEC) -> List[Dict]:
    """Extract continuous segments from decoded path."""
    if len(path) == 0:
        return []

    segments = []
    current_class = path[0]
    start_idx = 0

    for i in range(1, len(path)):
        if path[i] != current_class:
            if current_class > 0:
                start_ts = timestamps[start_idx]
                end_ts = timestamps[i - 1]
                start_ts -= (window_sec * 500)
                end_ts += (window_sec * 500)
                confidence = np.mean(probs[start_idx:i, current_class])
                duration = (end_ts - start_ts) / 1000

                segments.append({
                    'class_idx': current_class - 1,
                    'class_name': IDX_TO_ACTIVITY[current_class - 1],
                    'start_ts': int(start_ts),
                    'end_ts': int(end_ts),
                    'confidence': float(confidence),
                    'duration': duration,
                    'start_window_idx': start_idx,
                    'end_window_idx': i - 1,
                })

            current_class = path[i]
            start_idx = i

    # Handle last segment
    if current_class > 0:
        start_ts = timestamps[start_idx]
        end_ts = timestamps[-1]
        start_ts -= (window_sec * 500)
        end_ts += (window_sec * 500)
        confidence = np.mean(probs[start_idx:, current_class])
        duration = (end_ts - start_ts) / 1000

        segments.append({
            'class_idx': current_class - 1,
            'class_name': IDX_TO_ACTIVITY[current_class - 1],
            'start_ts': int(start_ts),
            'end_ts': int(end_ts),
            'confidence': float(confidence),
            'duration': duration,
            'start_window_idx': start_idx,
            'end_window_idx': len(path) - 1,
        })

    return segments


def merge_same_class_segments(segments: List[Dict],
                              gap_threshold_sec: float = SHORT_GAP_THRESHOLD) -> List[Dict]:
    """Merge adjacent segments of the same class with short gaps between them."""
    if len(segments) <= 1:
        return segments

    segments = sorted(segments, key=lambda s: s['start_ts'])

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = (seg['start_ts'] - prev['end_ts']) / 1000

        if seg['class_idx'] == prev['class_idx'] and gap < gap_threshold_sec:
            prev['end_ts'] = seg['end_ts']
            prev['duration'] = (prev['end_ts'] - prev['start_ts']) / 1000
            prev['confidence'] = (prev['confidence'] + seg['confidence']) / 2
        else:
            merged.append(seg.copy())

    return merged


def filter_short_segments(segments: List[Dict],
                          min_duration_sec: float = MIN_SEGMENT_FOR_OUTPUT) -> List[Dict]:
    """Remove segments shorter than minimum duration."""
    return [s for s in segments if s['duration'] >= min_duration_sec]


def resolve_overlaps(segments: List[Dict]) -> List[Dict]:
    """Resolve overlapping segments by keeping the one with higher confidence."""
    if len(segments) <= 1:
        return segments

    segments = sorted(segments, key=lambda s: s['start_ts'])
    resolved = [segments[0]]

    for seg in segments[1:]:
        prev = resolved[-1]
        if seg['start_ts'] < prev['end_ts']:
            if seg['class_idx'] == prev['class_idx']:
                prev['end_ts'] = max(prev['end_ts'], seg['end_ts'])
                prev['duration'] = (prev['end_ts'] - prev['start_ts']) / 1000
            else:
                if seg['confidence'] > prev['confidence']:
                    mid = (seg['start_ts'] + prev['end_ts']) // 2
                    prev['end_ts'] = mid
                    prev['duration'] = (prev['end_ts'] - prev['start_ts']) / 1000
                    seg['start_ts'] = mid
                    seg['duration'] = (seg['end_ts'] - seg['start_ts']) / 1000
                    resolved.append(seg)
                else:
                    seg['start_ts'] = prev['end_ts']
                    seg['duration'] = (seg['end_ts'] - seg['start_ts']) / 1000
                    if seg['duration'] > 0:
                        resolved.append(seg)
        else:
            resolved.append(seg)

    return resolved


def refine_boundaries(segments: List[Dict], data: np.ndarray,
                      timestamps_all: np.ndarray) -> List[Dict]:
    """Refine segment boundaries using energy change point detection."""
    if len(data) == 0 or len(segments) == 0:
        return segments

    acc = data[:, 1:4]
    energy = np.sqrt(np.sum(acc ** 2, axis=1))
    smooth_energy = uniform_filter1d(energy, size=200, mode='nearest')

    refined = []
    for seg in segments:
        start_ts = seg['start_ts']
        end_ts = seg['end_ts']

        search_range_ms = 15000

        # Refine start
        start_search_begin = start_ts - search_range_ms
        start_search_end = start_ts + search_range_ms
        start_mask = (timestamps_all >= start_search_begin) & (timestamps_all <= start_search_end)
        start_indices = np.where(start_mask)[0]

        if len(start_indices) > 100:
            local_energy = smooth_energy[start_indices]
            gradient = np.gradient(local_energy)
            best_idx = start_indices[np.argmax(np.abs(gradient))]
            new_start = int(timestamps_all[best_idx])
            if abs(new_start - start_ts) < search_range_ms:
                seg['start_ts'] = new_start

        # Refine end
        end_search_begin = end_ts - search_range_ms
        end_search_end = end_ts + search_range_ms
        end_mask = (timestamps_all >= end_search_begin) & (timestamps_all <= end_search_end)
        end_indices = np.where(end_mask)[0]

        if len(end_indices) > 100:
            local_energy = smooth_energy[end_indices]
            gradient = np.gradient(local_energy)
            best_idx = end_indices[np.argmin(gradient)]
            new_end = int(timestamps_all[best_idx])
            if abs(new_end - end_ts) < search_range_ms:
                seg['end_ts'] = new_end

        if seg['start_ts'] >= seg['end_ts']:
            seg['start_ts'] = start_ts
            seg['end_ts'] = end_ts

        seg['duration'] = (seg['end_ts'] - seg['start_ts']) / 1000
        refined.append(seg)

    return refined


def _select_top_k(segments: List[Dict], k: int = 3) -> List[Dict]:
    """Keep top-k segments by confidence, preferring one per class."""
    if len(segments) <= k:
        return segments
    by_class = {}
    for s in segments:
        c = s['class_idx']
        if c not in by_class or s['confidence'] > by_class[c]['confidence']:
            by_class[c] = s
    selected = list(by_class.values())
    if len(selected) >= k:
        selected.sort(key=lambda s: s['confidence'], reverse=True)
        return sorted(selected[:k], key=lambda s: s['start_ts'])
    remaining = [s for s in segments if s not in selected]
    remaining.sort(key=lambda s: s['confidence'], reverse=True)
    for s in remaining:
        if len(selected) >= k:
            break
        selected.append(s)
    return sorted(selected, key=lambda s: s['start_ts'])


def process_single_user_with_options(
    user_id: str,
    data: np.ndarray,
    model_groups,
    device,
    fusion_mode: str = DEFAULT_FUSION_MODE,
    min_duration_sec: float = 180,
    top_k: int = 3,
    conf_min: float = 0.45,
    verbose: bool = True,
) -> List[Dict]:
    """Process a single user's data through the full ensemble pipeline.

    Args:
        user_id: user identifier
        data: raw sensor data (N, 7)
        model_groups: dict from load_ensemble_models()
        device: torch device
        fusion_mode: cross-scale fusion mode passed to predict_multiscale_ensemble
        min_duration_sec: minimum duration filter after overlap resolution
        top_k: keep best-k segments per user; 0 disables top-k pruning
        conf_min: minimum confidence threshold after pruning
        verbose: whether to print per-user pipeline diagnostics

    Returns:
        List of detected segments
    """
    if verbose:
        print(f"\n  Processing {user_id}...")

    # Step 1-3: Multi-scale ensemble prediction
    timestamps, probs = predict_multiscale_ensemble(
        data,
        model_groups,
        device,
        fusion_mode=fusion_mode,
    )
    if len(timestamps) == 0:
        if verbose:
            print(f"    No windows created for {user_id}")
        return []

    if verbose:
        print(f"    Windows: {len(timestamps)}, Duration: {(timestamps[-1]-timestamps[0])/60000:.1f} min")
        print(f"    Ensemble prediction shape: {probs.shape}")

    # Step 4: Smooth predictions
    smoothed_probs = smooth_predictions(probs, timestamps)

    # Step 5: Viterbi decode
    path = viterbi_decode(smoothed_probs)

    # Count classes in path
    unique, counts = np.unique(path, return_counts=True)
    class_names = ['BG'] + [a[:2] for a in ACTIVITIES]
    path_summary = ", ".join([f"{class_names[u]}:{c}" for u, c in zip(unique, counts)])
    if verbose:
        print(f"    Viterbi path: {path_summary}")

    # Step 6: Extract segments
    segments = extract_segments(path, timestamps, smoothed_probs)
    if verbose:
        print(f"    Raw segments: {len(segments)}")

    # Step 7: Merge same-class segments with short gaps
    segments = merge_same_class_segments(segments, gap_threshold_sec=SHORT_GAP_THRESHOLD)
    if verbose:
        print(f"    After merge: {len(segments)}")

    # Step 8: Refine boundaries
    all_timestamps = data[:, 0]
    segments = refine_boundaries(segments, data, all_timestamps)

    # Step 9: Resolve overlaps
    segments = resolve_overlaps(segments)

    # Step 10: Filter short segments (tuned from 120s)
    segments = filter_short_segments(segments, min_duration_sec=min_duration_sec)
    if verbose:
        print(f"    After filter: {len(segments)}")

    # Step 11: Top-K segment selection (tuned: keep best 3 per user)
    if top_k > 0 and len(segments) > top_k:
        segments = _select_top_k(segments, k=top_k)
        if verbose:
            print(f"    After top-{top_k}: {len(segments)}")

    # Step 12: Confidence filter (tuned: min 0.45)
    if conf_min > 0:
        segments = [s for s in segments if s['confidence'] >= conf_min]

    if verbose:
        for seg in segments:
            print(f"      {seg['class_name']}: {seg['duration']:.0f}s "
                  f"(conf={seg['confidence']:.3f})")

    return segments


def process_single_user(user_id: str, data: np.ndarray,
                        model_groups, device) -> List[Dict]:
    """Backward-compatible wrapper for the default final reporting pipeline."""
    return process_single_user_with_options(
        user_id=user_id,
        data=data,
        model_groups=model_groups,
        device=device,
        fusion_mode=DEFAULT_FUSION_MODE,
        min_duration_sec=180,
        top_k=3,
        conf_min=0.45,
        verbose=True,
    )


def run_inference(data_dir: str, output_file: str = "./predictions_external_test.xlsx"):
    """Run inference on all signal files in one split using multi-model multi-scale ensemble."""
    print("=" * 60)
    print("INFERENCE PIPELINE (Multi-Scale Ensemble)")
    print("=" * 60)

    # Load all ensemble models
    model_groups, device = load_ensemble_models()

    # Read split data
    from .signal_file_reader import DataReader
    reader = DataReader(data_dir)
    split_data_dict = reader.read_data()
    print(f"\nLoaded {len(split_data_dict)} signal files")

    all_results = []

    for user_id, text_content in sorted(split_data_dict.items()):
        file_path = os.path.join(data_dir, f"{user_id}.txt")
        data = load_sensor_data(file_path)

        if data is None or len(data) < WINDOW_SIZE:
            print(f"  Skipping {user_id}: insufficient data")
            continue

        # Run ensemble pipeline
        segments = process_single_user(user_id, data, model_groups, device)

        for seg in segments:
            all_results.append([
                user_id,
                seg['class_name'],
                seg['start_ts'],
                seg['end_ts'],
            ])

    print(f"\n{'='*60}")
    print(f"Total results: {len(all_results)} segments")

    # Save using standard output interface
    from .prediction_writer import DataOutput
    if all_results:
        data_output = DataOutput(all_results, output_file=output_file)
        data_output.save_predictions()
    else:
        print("WARNING: No results to save!")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=EXTERNAL_TEST_DATA_DIR)
    parser.add_argument("--output", default="./predictions_external_test.xlsx")
    args = parser.parse_args()

    run_inference(args.data_dir, args.output)
