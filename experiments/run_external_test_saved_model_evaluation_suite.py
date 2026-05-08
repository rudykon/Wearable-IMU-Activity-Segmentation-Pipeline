"""External-test saved-model evaluation suite for the IMU activity pipeline.

Purpose:
    Evaluates model variants, decoding policies, calibration, bootstrap
    confidence intervals, effect sizes, and error summaries for the saved
    inference pipeline.
Inputs:
    Reads selected checkpoints, normalization files, split signals, and reference
    annotations from the configured repository layout.
Outputs:
    Writes experiment tables and diagnostic artifacts under `experiments/results/`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import copy
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from imu_activity_pipeline.config import (
    DEVICE,
    EXTERNAL_TEST_DATA_DIR,
    EXTERNAL_TEST_GOLD_FILE,
    IDX_TO_ACTIVITY,
    MODEL_DIR,
)
from imu_activity_pipeline.sensor_data_processing import load_gold_labels, load_sensor_data
from imu_activity_pipeline.evaluate import calculate_iou
from imu_activity_pipeline.inference import (
    extract_segments,
    filter_short_segments,
    merge_same_class_segments,
    predict_multiscale_ensemble,
    refine_boundaries,
    resolve_overlaps,
)
from imu_activity_pipeline.neural_network_models import CombinedModel


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def viterbi_tunable(
    probs: np.ndarray,
    self_trans: float = 0.97,
    bg_to_act: float = 0.01,
    act_to_bg: float = 0.05,
    cross_act: float = 0.001,
) -> np.ndarray:
    n_steps, n_states = probs.shape
    if n_steps == 0:
        return np.array([], dtype=np.int64)

    trans = np.full((n_states, n_states), 0.001, dtype=np.float64)
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

    log_trans = np.log(trans + 1e-12)
    log_probs = np.log(probs + 1e-12)

    v = np.full((n_steps, n_states), -np.inf, dtype=np.float64)
    backpointer = np.zeros((n_steps, n_states), dtype=np.int64)
    v[0] = log_probs[0]

    for t in range(1, n_steps):
        transitions = v[t - 1][:, None] + log_trans
        v[t] = np.max(transitions, axis=0) + log_probs[t]
        backpointer[t] = np.argmax(transitions, axis=0)

    path = np.zeros(n_steps, dtype=np.int64)
    path[-1] = np.argmax(v[-1])
    for t in range(n_steps - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path


def select_top_k(segments: List[Dict], k: int = 3) -> List[Dict]:
    if len(segments) <= k:
        return segments
    by_class = {}
    for seg in segments:
        cls = seg["class_idx"]
        if cls not in by_class or seg["confidence"] > by_class[cls]["confidence"]:
            by_class[cls] = seg
    selected = list(by_class.values())
    if len(selected) >= k:
        selected.sort(key=lambda x: x["confidence"], reverse=True)
        return sorted(selected[:k], key=lambda x: x["start_ts"])
    remaining = [s for s in segments if s not in selected]
    remaining.sort(key=lambda x: x["confidence"], reverse=True)
    for seg in remaining:
        if len(selected) >= k:
            break
        selected.append(seg)
    return sorted(selected, key=lambda x: x["start_ts"])


def bootstrap_ci(values: List[float], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_effect_size(a: List[float], b: List[float]) -> float:
    diff = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def load_gold(gold_file: str = EXTERNAL_TEST_GOLD_FILE) -> pd.DataFrame:
    df = load_gold_labels(gold_file)
    df["user_id"] = df["user_id"].astype(str)
    return df


def load_split_users(data_dir: str = EXTERNAL_TEST_DATA_DIR) -> Dict[str, np.ndarray]:
    users = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".txt"):
            continue
        uid = name.replace(".txt", "")
        arr = load_sensor_data(os.path.join(data_dir, name), apply_filter=True)
        if arr is not None and len(arr) > 0:
            users[uid] = arr
    return users


def load_model(ckpt_path: str, window_size: int, device: torch.device) -> torch.nn.Module:
    model = CombinedModel(input_channels=6, num_classes=6, window_size=window_size).to(device)
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_model_groups(device: torch.device):
    with open(os.path.join(MODEL_DIR, "norm_params_3s.pkl"), "rb") as f:
        norm_3s = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "norm_params_5s.pkl"), "rb") as f:
        norm_5s = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "norm_params_8s.pkl"), "rb") as f:
        norm_8s = pickle.load(f)

    m_best_3s = load_model(os.path.join(MODEL_DIR, "combined_model_best.pth"), 300, device)

    m_3s = [
        load_model(os.path.join(MODEL_DIR, "combined_model_3s_seed42.pth"), 300, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_3s_seed123.pth"), 300, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_3s_seed456.pth"), 300, device),
    ]
    m_5s = [
        load_model(os.path.join(MODEL_DIR, "combined_model_5s_seed42.pth"), 500, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_5s_seed123.pth"), 500, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_5s_seed456.pth"), 500, device),
    ]
    m_8s = [
        load_model(os.path.join(MODEL_DIR, "combined_model_8s_seed42.pth"), 800, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_8s_seed123.pth"), 800, device),
        load_model(os.path.join(MODEL_DIR, "combined_model_8s_seed456.pth"), 800, device),
    ]

    def group(models_3s=None, models_5s=None, models_8s=None):
        g = {}
        if models_3s:
            g["3s"] = {
                "models": models_3s,
                "window_size": 300,
                "window_step": 100,
                "window_sec": 3,
                "norm_params": norm_3s,
            }
        if models_5s:
            g["5s"] = {
                "models": models_5s,
                "window_size": 500,
                "window_step": 100,
                "window_sec": 5,
                "norm_params": norm_5s,
            }
        if models_8s:
            g["8s"] = {
                "models": models_8s,
                "window_size": 800,
                "window_step": 100,
                "window_sec": 8,
                "norm_params": norm_8s,
            }
        return g

    groups = {
        "single_3s_best": {"group": group(models_3s=[m_best_3s]), "window_sec": 3},
        "ensemble_3s": {"group": group(models_3s=m_3s), "window_sec": 3},
        "ensemble_5s": {"group": group(models_5s=m_5s), "window_sec": 5},
        "ensemble_6": {"group": group(models_3s=m_3s, models_5s=m_5s), "window_sec": 3},
        "ensemble_9": {"group": group(models_3s=m_3s, models_5s=m_5s, models_8s=m_8s), "window_sec": 3},
    }
    return groups


def precompute_probs(
    user_data: Dict[str, np.ndarray], model_group: Dict, device: torch.device
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    cache = {}
    for uid, data in user_data.items():
        ts, probs = predict_multiscale_ensemble(data, model_group, device)
        cache[uid] = (ts, probs)
    return cache


@dataclass
class PPConfig:
    smooth_window: int
    median_filter: int
    use_viterbi: bool
    self_trans: float
    bg_to_act: float
    act_to_bg: float
    cross_act: float
    merge_gap: float
    refine_boundary: bool
    resolve_overlap: bool
    min_duration: float
    top_k: int
    conf_min: float


PP_RAW = PPConfig(
    smooth_window=0,
    median_filter=0,
    use_viterbi=False,
    self_trans=0.90,
    bg_to_act=0.01,
    act_to_bg=0.05,
    cross_act=0.005,
    merge_gap=0,
    refine_boundary=False,
    resolve_overlap=False,
    min_duration=0,
    top_k=0,
    conf_min=0.0,
)

PP_BASELINE = PPConfig(
    smooth_window=5,
    median_filter=0,
    use_viterbi=True,
    self_trans=0.90,
    bg_to_act=0.01,
    act_to_bg=0.05,
    cross_act=0.005,
    merge_gap=60,
    refine_boundary=True,
    resolve_overlap=True,
    min_duration=120,
    top_k=0,
    conf_min=0.0,
)

PP_FULL = PPConfig(
    smooth_window=7,
    median_filter=5,
    use_viterbi=True,
    self_trans=0.97,
    bg_to_act=0.01,
    act_to_bg=0.05,
    cross_act=0.001,
    merge_gap=60,
    refine_boundary=True,
    resolve_overlap=True,
    min_duration=180,
    top_k=3,
    conf_min=0.45,
)


def postprocess(
    timestamps: np.ndarray,
    probs: np.ndarray,
    raw_data: np.ndarray,
    cfg: PPConfig,
    window_sec: int,
) -> List[Dict]:
    if len(timestamps) == 0:
        return []

    cur_probs = np.copy(probs)
    if cfg.smooth_window > 1:
        for c in range(cur_probs.shape[1]):
            cur_probs[:, c] = uniform_filter1d(cur_probs[:, c], size=cfg.smooth_window, mode="nearest")
    if cfg.median_filter > 0:
        k = cfg.median_filter if cfg.median_filter % 2 == 1 else cfg.median_filter + 1
        for c in range(cur_probs.shape[1]):
            cur_probs[:, c] = medfilt(cur_probs[:, c], kernel_size=k)

    if cfg.use_viterbi:
        path = viterbi_tunable(
            cur_probs,
            self_trans=cfg.self_trans,
            bg_to_act=cfg.bg_to_act,
            act_to_bg=cfg.act_to_bg,
            cross_act=cfg.cross_act,
        )
    else:
        path = np.argmax(cur_probs, axis=1)

    segments = extract_segments(path, timestamps, cur_probs, window_sec=window_sec)

    if cfg.merge_gap > 0:
        segments = merge_same_class_segments(segments, gap_threshold_sec=cfg.merge_gap)
    if cfg.refine_boundary:
        segments = refine_boundaries(segments, raw_data, raw_data[:, 0])
    if cfg.resolve_overlap:
        segments = resolve_overlaps(segments)
    if cfg.min_duration > 0:
        segments = filter_short_segments(segments, min_duration_sec=cfg.min_duration)
    if cfg.top_k > 0:
        segments = select_top_k(segments, k=cfg.top_k)
    if cfg.conf_min > 0:
        segments = [s for s in segments if s["confidence"] >= cfg.conf_min]
    return segments


def evaluate_segments(pred_by_user: Dict[str, List[Dict]], gold_df: pd.DataFrame) -> Dict:
    users = sorted(gold_df["user_id"].unique())
    per_user = {}
    per_class = {}
    class_names = sorted(gold_df["category"].unique())
    for cls in class_names:
        per_class[cls] = {"TP": 0, "FP": 0, "FN": 0}

    tp_total = fp_total = fn_total = 0
    user_f1_list = []

    for uid in users:
        g_rows = gold_df[gold_df["user_id"] == uid]
        gt = [{"start": r["start"], "end": r["end"], "category": r["category"]} for _, r in g_rows.iterrows()]
        pred = []
        for seg in pred_by_user.get(uid, []):
            pred.append(
                {
                    "start": int(seg["start_ts"]),
                    "end": int(seg["end_ts"]),
                    "category": seg["class_name"],
                }
            )

        matches = []
        for i, p in enumerate(pred):
            for j, g in enumerate(gt):
                if p["category"] != g["category"]:
                    continue
                iou = calculate_iou(p["start"], p["end"], g["start"], g["end"])
                if iou > 0.5:
                    matches.append((iou, i, j))
        matches.sort(reverse=True, key=lambda x: x[0])

        matched_p = set()
        matched_g = set()
        accepted_pairs = []
        for _, i, j in matches:
            if i in matched_p or j in matched_g:
                continue
            matched_p.add(i)
            matched_g.add(j)
            accepted_pairs.append((i, j))

        tp = len(matched_p)
        fp = len(pred) - tp
        fn = len(gt) - len(matched_g)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        for i, p in enumerate(pred):
            if i not in matched_p:
                per_class[p["category"]]["FP"] += 1
        for j, g in enumerate(gt):
            if j not in matched_g:
                per_class[g["category"]]["FN"] += 1
        for i, _ in accepted_pairs:
            cls = pred[i]["category"]
            per_class[cls]["TP"] += 1

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        user_f1_list.append(float(f1))
        per_user[uid] = {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f1}

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    mean_f1 = float(np.mean(user_f1_list)) if user_f1_list else 0.0
    ci_low, ci_high = bootstrap_ci(user_f1_list, n_boot=10000, seed=42)

    for cls in class_names:
        c = per_class[cls]
        p = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else 0.0
        r = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        c["precision"] = p
        c["recall"] = r
        c["f1"] = f1

    return {
        "overall": {
            "mean_user_f1": mean_f1,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
        },
        "per_user": per_user,
        "per_class": per_class,
    }


def run_method(
    user_data: Dict[str, np.ndarray],
    probs_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pp_cfg: PPConfig,
    window_sec: int,
) -> Tuple[Dict[str, List[Dict]], float]:
    pred_by_user = {}
    t0 = time.perf_counter()
    for uid, raw in user_data.items():
        ts, probs = probs_cache[uid]
        pred_by_user[uid] = postprocess(ts, probs, raw, pp_cfg, window_sec)
    elapsed = time.perf_counter() - t0
    return pred_by_user, elapsed


def add_gaussian_noise(data: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    sig = out[:, 1:].astype(np.float64)
    power = np.mean(sig ** 2, axis=0)
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power + 1e-12), size=sig.shape)
    out[:, 1:] = (sig + noise).astype(np.float32)
    return out


def add_random_dropout(data: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    sig = out[:, 1:].astype(np.float64)
    n = sig.shape[0]
    mask = rng.random(n) < ratio
    if np.all(mask):
        mask[0] = False
        mask[-1] = False
    x = np.arange(n)
    for c in range(sig.shape[1]):
        y = sig[:, c].copy()
        y[mask] = np.nan
        valid = ~np.isnan(y)
        y_interp = np.interp(x, x[valid], y[valid])
        sig[:, c] = y_interp
    out[:, 1:] = sig.astype(np.float32)
    return out


def add_timestamp_jitter(data: np.ndarray, std_ms: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    ts = out[:, 0].astype(np.float64)
    if len(ts) < 3:
        return out

    diffs = np.diff(ts)
    jittered = diffs + rng.normal(loc=0.0, scale=std_ms, size=len(diffs))
    floor_ms = max(1.0, float(np.median(diffs) * 0.2))
    jittered = np.clip(jittered, floor_ms, None)

    new_ts = np.concatenate([[ts[0]], ts[0] + np.cumsum(jittered)])
    span = float(ts[-1] - ts[0])
    new_span = float(new_ts[-1] - new_ts[0])
    if span > 0 and new_span > 0:
        new_ts = ts[0] + (new_ts - new_ts[0]) * (span / new_span)

    new_ts = np.round(new_ts).astype(np.int64)
    for i in range(1, len(new_ts)):
        if new_ts[i] <= new_ts[i - 1]:
            new_ts[i] = new_ts[i - 1] + 1
    out[:, 0] = new_ts
    return out


def add_burst_missing(
    data: np.ndarray,
    missing_ratio: float,
    burst_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = data.copy()
    sig = out[:, 1:].astype(np.float64)
    n = sig.shape[0]
    if n < 4:
        return out

    ts = out[:, 0].astype(np.float64)
    median_step_ms = float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0
    burst_len = max(2, int(round((burst_sec * 1000.0) / max(median_step_ms, 1.0))))
    target_missing = max(1, int(round(n * missing_ratio)))
    mask = np.zeros(n, dtype=bool)

    attempts = 0
    while int(mask.sum()) < target_missing and attempts < max(32, n // max(1, burst_len)):
        start = int(rng.integers(0, max(1, n - burst_len)))
        end = min(n, start + burst_len)
        mask[start:end] = True
        attempts += 1

    mask[0] = False
    mask[-1] = False
    x = np.arange(n)
    for c in range(sig.shape[1]):
        y = sig[:, c].copy()
        y[mask] = np.nan
        valid = ~np.isnan(y)
        y_interp = np.interp(x, x[valid], y[valid])
        sig[:, c] = y_interp
    out[:, 1:] = sig.astype(np.float32)
    return out


def add_axis_saturation(data: np.ndarray, clip_quantile: float) -> np.ndarray:
    out = data.copy()
    sig = out[:, 1:].astype(np.float64)
    thresholds = np.quantile(np.abs(sig), clip_quantile, axis=0)
    thresholds = np.maximum(thresholds, 1e-6)
    out[:, 1:] = np.clip(sig, -thresholds, thresholds).astype(np.float32)
    return out


def add_bias_drift(data: np.ndarray, drift_ratio: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    sig = out[:, 1:].astype(np.float64)
    n = sig.shape[0]
    if n < 4:
        return out

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    for c in range(sig.shape[1]):
        std = float(np.std(sig[:, c]))
        amp = drift_ratio * std
        slope = rng.uniform(-1.0, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        drift = amp * (slope * (t - 0.5) + 0.5 * np.sin(2.0 * np.pi * t + phase))
        sig[:, c] = sig[:, c] + drift
    out[:, 1:] = sig.astype(np.float32)
    return out


def add_low_rate_distortion(data: np.ndarray, target_hz: float) -> np.ndarray:
    out = data.copy()
    ts = out[:, 0].astype(np.float64)
    sig = out[:, 1:].astype(np.float64)
    if len(ts) < 4:
        return out

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    median_step_ms = float(np.median(diffs)) if len(diffs) else 10.0
    if median_step_ms <= 0:
        return out
    native_hz = 1000.0 / median_step_ms
    if target_hz >= native_hz:
        return out

    stride = max(2, int(round(native_hz / target_hz)))
    keep_idx = np.arange(0, len(ts), stride, dtype=np.int64)
    if keep_idx[-1] != len(ts) - 1:
        keep_idx = np.append(keep_idx, len(ts) - 1)

    hold_lengths = np.diff(np.append(keep_idx, len(ts)))
    for c in range(sig.shape[1]):
        held = np.repeat(sig[keep_idx, c], hold_lengths)
        sig[:, c] = held[: len(ts)]
    out[:, 1:] = sig.astype(np.float32)
    return out


def perturb_dataset(
    user_data: Dict[str, np.ndarray], fn: Optional[Callable[[np.ndarray], np.ndarray]]
) -> Dict[str, np.ndarray]:
    if fn is None:
        return {k: v.copy() for k, v in user_data.items()}
    return {k: fn(v.copy()) for k, v in user_data.items()}


def run_efficiency(
    user_data: Dict[str, np.ndarray],
    model_group: Dict,
    pp_cfg: PPConfig,
    window_sec: int,
    device: torch.device,
) -> Dict:
    per_user_seconds = []
    for uid, raw in user_data.items():
        t0 = time.perf_counter()
        ts, probs = predict_multiscale_ensemble(raw, model_group, device)
        _ = postprocess(ts, probs, raw, pp_cfg, window_sec)
        per_user_seconds.append(time.perf_counter() - t0)
    avg_sec = float(np.mean(per_user_seconds))
    med_sec = float(np.median(per_user_seconds))
    return {
        "avg_sec_per_user_file": avg_sec,
        "median_sec_per_user_file": med_sec,
        "users": len(per_user_seconds),
    }


def main():
    set_seed(42)
    print("Running supplementary experiments...")
    print(f"Output dir: {OUT_DIR}")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Main device: {device}")

    gold_df = load_gold()
    user_data = load_split_users()
    print(f"Loaded {len(user_data)} users.")

    model_groups = build_model_groups(device)
    for k in model_groups:
        print(f"Loaded model group: {k}")

    print("Precomputing clean probabilities...")
    clean_cache = {}
    for k, v in model_groups.items():
        t0 = time.perf_counter()
        clean_cache[k] = precompute_probs(user_data, v["group"], device)
        dt = time.perf_counter() - t0
        print(f"  {k}: {dt:.1f}s")

    methods = [
        ("M0_RawArgmax_Single3s", "single_3s_best", PP_RAW),
        ("M1_BaselinePP_Single3s", "single_3s_best", PP_BASELINE),
        ("M2_FullPP_Single3s", "single_3s_best", PP_FULL),
        ("M3_FullPP_Ens3s", "ensemble_3s", PP_FULL),
        ("M4_BaselinePP_Ens9", "ensemble_9", PP_BASELINE),
        ("M5_Proposed_Ens9_FullPP", "ensemble_9", PP_FULL),
    ]

    comparison_rows = []
    method_results = {}
    for name, gname, pp in methods:
        print(f"Evaluating {name} ...")
        pred_by_user, post_sec = run_method(
            user_data=user_data,
            probs_cache=clean_cache[gname],
            pp_cfg=pp,
            window_sec=model_groups[gname]["window_sec"],
        )
        metrics = evaluate_segments(pred_by_user, gold_df)
        method_results[name] = {
            "metrics": metrics,
            "postprocess_seconds": post_sec,
            "pred_by_user": pred_by_user,
        }
        comparison_rows.append(
            {
                "method": name,
                "mean_user_f1": metrics["overall"]["mean_user_f1"],
                "ci95_low": metrics["overall"]["ci95_low"],
                "ci95_high": metrics["overall"]["ci95_high"],
                "micro_f1": metrics["overall"]["micro_f1"],
                "TP": metrics["overall"]["TP"],
                "FP": metrics["overall"]["FP"],
                "FN": metrics["overall"]["FN"],
            }
        )

    df_comparison = pd.DataFrame(comparison_rows).sort_values("mean_user_f1", ascending=False)
    df_comparison.to_csv(os.path.join(OUT_DIR, "main_comparison.csv"), index=False, encoding="utf-8-sig")

    # Ablation on proposed setup
    print("Running ablation...")
    ablations = [
        ("A0_Full", "ensemble_9", PP_FULL),
        ("A1_NoMedian", "ensemble_9", replace(PP_FULL, median_filter=0)),
        ("A2_NoTopKConf", "ensemble_9", replace(PP_FULL, top_k=0, conf_min=0.0)),
        ("A3_NoBoundaryRefine", "ensemble_9", replace(PP_FULL, refine_boundary=False)),
        ("A4_NoViterbi", "ensemble_9", replace(PP_FULL, use_viterbi=False)),
        ("A5_Only3s", "ensemble_3s", PP_FULL),
    ]
    ablation_rows = []
    for name, gname, pp in ablations:
        pred_by_user, _ = run_method(
            user_data=user_data,
            probs_cache=clean_cache[gname],
            pp_cfg=pp,
            window_sec=model_groups[gname]["window_sec"],
        )
        metrics = evaluate_segments(pred_by_user, gold_df)
        ablation_rows.append(
            {
                "ablation": name,
                "mean_user_f1": metrics["overall"]["mean_user_f1"],
                "micro_f1": metrics["overall"]["micro_f1"],
                "TP": metrics["overall"]["TP"],
                "FP": metrics["overall"]["FP"],
                "FN": metrics["overall"]["FN"],
            }
        )
    df_ablation = pd.DataFrame(ablation_rows).sort_values("mean_user_f1", ascending=False)
    df_ablation.to_csv(os.path.join(OUT_DIR, "ablation.csv"), index=False, encoding="utf-8-sig")

    # Robustness on proposed method
    print("Running robustness...")
    rng = np.random.default_rng(42)
    robustness_conditions = [
        ("Clean", None),
        ("Noise_SNR20dB", lambda x: add_gaussian_noise(x, 20.0, rng)),
        ("Noise_SNR10dB", lambda x: add_gaussian_noise(x, 10.0, rng)),
        ("Noise_SNR5dB", lambda x: add_gaussian_noise(x, 5.0, rng)),
        ("Dropout_5pct", lambda x: add_random_dropout(x, 0.05, rng)),
        ("Dropout_10pct", lambda x: add_random_dropout(x, 0.10, rng)),
        ("Dropout_20pct", lambda x: add_random_dropout(x, 0.20, rng)),
    ]
    robust_rows = []
    proposed_group = model_groups["ensemble_9"]["group"]
    for cname, fn in robustness_conditions:
        print(f"  Condition: {cname}")
        perturbed = perturb_dataset(user_data, fn)
        probs_cache = precompute_probs(perturbed, proposed_group, device)
        pred_by_user, _ = run_method(
            user_data=perturbed,
            probs_cache=probs_cache,
            pp_cfg=PP_FULL,
            window_sec=model_groups["ensemble_9"]["window_sec"],
        )
        metrics = evaluate_segments(pred_by_user, gold_df)
        robust_rows.append(
            {
                "condition": cname,
                "mean_user_f1": metrics["overall"]["mean_user_f1"],
                "ci95_low": metrics["overall"]["ci95_low"],
                "ci95_high": metrics["overall"]["ci95_high"],
                "micro_f1": metrics["overall"]["micro_f1"],
                "TP": metrics["overall"]["TP"],
                "FP": metrics["overall"]["FP"],
                "FN": metrics["overall"]["FN"],
            }
        )
    df_robust = pd.DataFrame(robust_rows)
    df_robust.to_csv(os.path.join(OUT_DIR, "robustness.csv"), index=False, encoding="utf-8-sig")

    # Significance: Proposed vs strongest non-proposed baseline
    print("Computing significance...")
    proposed = method_results["M5_Proposed_Ens9_FullPP"]["metrics"]["per_user"]
    baseline = method_results["M3_FullPP_Ens3s"]["metrics"]["per_user"]
    users = sorted(proposed.keys())
    x = [proposed[u]["f1"] for u in users]
    y = [baseline[u]["f1"] for u in users]
    try:
        wilcoxon = stats.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        p_val = float(wilcoxon.pvalue)
        stat_val = float(wilcoxon.statistic)
    except Exception:
        p_val = 1.0
        stat_val = 0.0
    effect = paired_effect_size(x, y)
    sig = {
        "comparison": "M5_Proposed_Ens9_FullPP vs M3_FullPP_Ens3s",
        "wilcoxon_statistic": stat_val,
        "wilcoxon_p_one_sided": p_val,
        "paired_cohens_d": effect,
        "mean_diff": float(np.mean(np.array(x) - np.array(y))),
    }
    with open(os.path.join(OUT_DIR, "significance.json"), "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2, ensure_ascii=False)

    # Efficiency (GPU + CPU for proposed and single)
    print("Measuring efficiency...")
    efficiency = {}
    # GPU/active device
    efficiency["active_device_single3s"] = run_efficiency(
        user_data, model_groups["single_3s_best"]["group"], PP_BASELINE, 3, device
    )
    efficiency["active_device_proposed"] = run_efficiency(
        user_data, model_groups["ensemble_9"]["group"], PP_FULL, 3, device
    )

    # CPU
    cpu_device = torch.device("cpu")
    cpu_groups = build_model_groups(cpu_device)
    efficiency["cpu_single3s"] = run_efficiency(
        user_data, cpu_groups["single_3s_best"]["group"], PP_BASELINE, 3, cpu_device
    )
    efficiency["cpu_proposed"] = run_efficiency(
        user_data, cpu_groups["ensemble_9"]["group"], PP_FULL, 3, cpu_device
    )
    with open(os.path.join(OUT_DIR, "efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2, ensure_ascii=False)

    # Save full metrics
    full = {}
    for name in method_results:
        full[name] = method_results[name]["metrics"]
    with open(os.path.join(OUT_DIR, "main_metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    if HAS_MATPLOTLIB:
        # Plot: main comparison
        plt.figure(figsize=(12, 5))
        order = df_comparison.sort_values("mean_user_f1", ascending=True)
        plt.barh(order["method"], order["mean_user_f1"], color="#2a9d8f")
        plt.xlim(0.6, 1.0)
        plt.xlabel("Mean User F1")
        plt.title("Main Comparison (Segment-Level IoU>=0.5)")
        for i, v in enumerate(order["mean_user_f1"]):
            plt.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "main_comparison.png"), dpi=220)
        plt.close()

        # Plot: robustness curve
        plt.figure(figsize=(10, 4))
        plt.plot(df_robust["condition"], df_robust["mean_user_f1"], marker="o", color="#264653")
        plt.ylim(0.6, 1.0)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Mean User F1")
        plt.title("Robustness of Proposed Method")
        for i, v in enumerate(df_robust["mean_user_f1"]):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "robustness.png"), dpi=220)
        plt.close()
    else:
        print(
            "WARNING: matplotlib not found; skipping plot generation in "
            "run_external_test_saved_model_evaluation_suite.py"
        )

    summary = {
        "main_comparison_top": df_comparison.iloc[0].to_dict(),
        "proposed_main": method_results["M5_Proposed_Ens9_FullPP"]["metrics"]["overall"],
        "significance": sig,
        "efficiency": efficiency,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
