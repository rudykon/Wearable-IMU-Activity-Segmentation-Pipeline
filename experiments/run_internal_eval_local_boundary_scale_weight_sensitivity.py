"""Sensitivity analysis for local-boundary scale-arbitration weights.

Purpose:
    Tests how base scale weights and boundary-boost amplitudes affect fused
    probabilities and segment-level metrics.
Inputs:
    Loads saved models, internal evaluation signals, and reference annotations.
Outputs:
    Writes sensitivity tables under `experiments/results/` for comparison.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import experiments.run_external_test_saved_model_evaluation_suite as ex
from imu_activity_pipeline import inference
from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE
from imu_activity_pipeline.sensor_data_processing import create_windows, normalize_imu


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


BASE_WEIGHT_GRID = [
    {"tag": "long055", "weights": {"3s": 0.15, "5s": 0.30, "8s": 0.55}},
    {"tag": "current", "weights": {"3s": 0.20, "5s": 0.35, "8s": 0.45}},
    {"tag": "mid050", "weights": {"3s": 0.20, "5s": 0.30, "8s": 0.50}},
    {"tag": "alt045", "weights": {"3s": 0.25, "5s": 0.30, "8s": 0.45}},
    {"tag": "balanced040", "weights": {"3s": 0.25, "5s": 0.35, "8s": 0.40}},
]

BOOST_GRID = [
    {"tag": "boost000", "boost_3s": 0.00},
    {"tag": "boost020", "boost_3s": 0.20},
    {"tag": "boost030", "boost_3s": 0.30},
    {"tag": "boost040", "boost_3s": 0.40},
]

# Preserve the original 5s/8s reduction ratio from the reported LBSA setting:
# 0.08 : 0.22 = 4 : 11
REDUCE_5S_RATIO = 4.0 / 15.0
REDUCE_8S_RATIO = 11.0 / 15.0


def precompute_aligned_probs(
    user_data: Dict[str, np.ndarray], model_group: Dict, device: torch.device
) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    cache = {}
    for uid, data in user_data.items():
        scale_results: List[Dict] = []
        for suffix, group in model_group.items():
            timestamps, windows = create_windows(data, group["window_size"], group["window_step"])
            if len(windows) == 0:
                continue
            norm = group["norm_params"]
            norm_windows, _, _ = normalize_imu(windows, norm["mean"], norm["std"])
            probs = inference.predict_windows_ensemble(group["models"], norm_windows, device)
            scale_results.append({"suffix": suffix, "timestamps": timestamps, "probs": probs})

        ref_timestamps, aligned_probs = inference._align_scale_probabilities(scale_results)
        cache[uid] = (ref_timestamps, aligned_probs)
    return cache


def fuse_local_boundary_parametric(
    aligned_probs: Dict[str, np.ndarray],
    base_weights: Dict[str, float],
    boost_3s: float,
) -> np.ndarray:
    suffixes = [k for k in ["3s", "5s", "8s"] if k in aligned_probs]
    if len(suffixes) == 1 or "3s" not in aligned_probs:
        return aligned_probs[suffixes[0]]

    probs = {k: aligned_probs[k] for k in suffixes}
    ref = probs[suffixes[0]]
    n_steps = len(ref)

    base = np.array([base_weights[k] for k in suffixes], dtype=np.float32)
    base = base / np.sum(base)
    weights = np.tile(base[None, :], (n_steps, 1))

    probs_3s = probs["3s"]
    pred_3s = np.argmax(probs_3s, axis=1)
    boundary_mask = np.zeros(n_steps, dtype=np.float32)
    boundary_points = np.where(pred_3s[1:] != pred_3s[:-1])[0] + 1
    for idx in boundary_points:
        left = max(0, idx - 3)
        right = min(n_steps, idx + 4)
        boundary_mask[left:right] = 1.0
    boundary_mask = uniform_filter1d(boundary_mask, size=3, mode="nearest")
    boundary_mask = np.clip(boundary_mask, 0.0, 1.0)

    idx_3s = suffixes.index("3s")
    weights[:, idx_3s] += boost_3s * boundary_mask

    if "5s" in suffixes:
        idx_5s = suffixes.index("5s")
        weights[:, idx_5s] -= (boost_3s * REDUCE_5S_RATIO) * boundary_mask
    if "8s" in suffixes:
        idx_8s = suffixes.index("8s")
        weights[:, idx_8s] -= (boost_3s * REDUCE_8S_RATIO) * boundary_mask

    weights = np.clip(weights, 0.05, None)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    fused = np.zeros_like(ref, dtype=np.float32)
    for i, suffix in enumerate(suffixes):
        fused += weights[:, i : i + 1] * probs[suffix]
    return fused


def summarize_row(
    base_tag: str,
    base_weights: Dict[str, float],
    boost_tag: str,
    boost_3s: float,
    metrics: Dict,
    elapsed_sec: float,
) -> Dict:
    overall = metrics["overall"]
    return {
        "base_tag": base_tag,
        "weights_3s": float(base_weights["3s"]),
        "weights_5s": float(base_weights["5s"]),
        "weights_8s": float(base_weights["8s"]),
        "boost_tag": boost_tag,
        "boost_3s": float(boost_3s),
        "reduce_5s": float(boost_3s * REDUCE_5S_RATIO),
        "reduce_8s": float(boost_3s * REDUCE_8S_RATIO),
        "mean_user_f1": float(overall["mean_user_f1"]),
        "ci95_low": float(overall["ci95_low"]),
        "ci95_high": float(overall["ci95_high"]),
        "micro_f1": float(overall["micro_f1"]),
        "TP": int(overall["TP"]),
        "FP": int(overall["FP"]),
        "FN": int(overall["FN"]),
        "elapsed_sec": float(elapsed_sec),
    }


def main() -> None:
    ex.set_seed(42)
    device = torch.device(ex.DEVICE if torch.cuda.is_available() else "cpu")
    gold_df = ex.load_gold(INTERNAL_EVAL_GOLD_FILE)
    user_data = ex.load_split_users(INTERNAL_EVAL_DATA_DIR)
    model_groups, _ = inference.load_ensemble_models()

    aligned_cache = precompute_aligned_probs(user_data, model_groups, device)
    rows = []

    for base_cfg in BASE_WEIGHT_GRID:
        for boost_cfg in BOOST_GRID:
            probs_cache = {}
            for uid, (ts, aligned_probs) in aligned_cache.items():
                fused = fuse_local_boundary_parametric(
                    aligned_probs=aligned_probs,
                    base_weights=base_cfg["weights"],
                    boost_3s=boost_cfg["boost_3s"],
                )
                probs_cache[uid] = (ts, fused)

            pred_by_user, elapsed = ex.run_method(
                user_data=user_data,
                probs_cache=probs_cache,
                pp_cfg=ex.PP_FULL,
                window_sec=3,
            )
            metrics = ex.evaluate_segments(pred_by_user, gold_df)
            rows.append(
                summarize_row(
                    base_tag=base_cfg["tag"],
                    base_weights=base_cfg["weights"],
                    boost_tag=boost_cfg["tag"],
                    boost_3s=boost_cfg["boost_3s"],
                    metrics=metrics,
                    elapsed_sec=elapsed,
                )
            )

    df = pd.DataFrame(rows).sort_values(
        ["mean_user_f1", "micro_f1", "FP", "FN"],
        ascending=[False, False, True, True],
    )

    best = df.iloc[0].to_dict()
    current_mask = (df["base_tag"] == "current") & (df["boost_tag"] == "boost030")
    current = df[current_mask].iloc[0].to_dict()

    csv_path = os.path.join(OUT_DIR, "lbsa_weight_sensitivity_20260427.csv")
    json_path = os.path.join(OUT_DIR, "lbsa_weight_sensitivity_20260427.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "dataset": "internal 20-user evaluation set only (data/signals/internal_eval + internal_eval_annotations.csv)",
        "restriction": "Sensitivity analysis performed only on the internal 20-user set with the reported best-per-scale 3-model stack and PP_FULL post-processing.",
        "current_setting": current,
        "best_setting": best,
        "n_settings": int(len(df)),
        "artifacts": {
            "csv": os.path.relpath(csv_path, BASE_DIR),
            "json": os.path.relpath(json_path, BASE_DIR),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
