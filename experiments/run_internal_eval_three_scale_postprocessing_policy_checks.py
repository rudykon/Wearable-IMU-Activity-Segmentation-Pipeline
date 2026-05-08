"""Internal-evaluation post-processing policy checks for the 3-scale model stack.

Purpose:
    Re-evaluates fixed post-processing policies on the internal evaluation split
    to select a stable decoding configuration for the current saved models.
Inputs:
    Loads `ensemble_config.json`, selected checkpoints, internal evaluation
    signals, and reference annotations.
Outputs:
    Writes policy metrics and selected-policy summaries under
    `experiments/results/`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import os
import sys
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import experiments.run_external_test_saved_model_evaluation_suite as ex
import experiments.run_internal_eval_split_protocol_policy_selection_checks as pr
from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE
from imu_activity_pipeline.inference import load_ensemble_models


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_selected_models() -> Dict[str, str]:
    with open(os.path.join(BASE_DIR, "saved_models", "ensemble_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("selected_models", {})


def choose_recommended_fixed_policy(
    fixed_policy_outer_stats: Dict[str, Dict],
    preferred_order: List[str],
) -> str:
    def key(name: str):
        stats = fixed_policy_outer_stats[name]
        order_rank = preferred_order.index(name) if name in preferred_order else len(preferred_order)
        return (
            float(stats["mean"]),
            -float(stats["std"]),
            -order_rank,
        )

    return max(fixed_policy_outer_stats.keys(), key=key)


def main():
    ex.set_seed(42)
    device = torch.device(ex.DEVICE if torch.cuda.is_available() else "cpu")
    selected_models = load_selected_models()

    gold_df = ex.load_gold(INTERNAL_EVAL_GOLD_FILE)
    user_data = ex.load_split_users(INTERNAL_EVAL_DATA_DIR)
    model_group, _ = load_ensemble_models()
    probs_cache = ex.precompute_probs(user_data, model_group, device)

    policy_suite = [
        ("S0_Full", ex.PP_FULL),
        ("S6_NoTopK_KeepConf", replace(ex.PP_FULL, top_k=0)),
        ("S8_Top4_Conf50", replace(ex.PP_FULL, top_k=4, conf_min=0.50)),
        ("S9_Top4_Conf55", replace(ex.PP_FULL, top_k=4, conf_min=0.55)),
        ("S10_Top4_Conf50_Min210", replace(ex.PP_FULL, top_k=4, conf_min=0.50, min_duration=210)),
    ]
    candidate_policies = [name for name, _ in policy_suite]
    fixed_reference_policies = [name for name, _ in policy_suite]

    policy_predictions = {}
    full_internal_eval_rows = []
    for name, cfg in policy_suite:
        pred_by_user, elapsed = ex.run_method(user_data, probs_cache, cfg, window_sec=3)
        policy_predictions[name] = pred_by_user
        overall = ex.evaluate_segments(pred_by_user, gold_df)["overall"]
        full_internal_eval_rows.append(
            {
                "policy": name,
                "mean_user_f1": float(overall["mean_user_f1"]),
                "micro_f1": float(overall["micro_f1"]),
                "TP": int(overall["TP"]),
                "FP": int(overall["FP"]),
                "FN": int(overall["FN"]),
                "elapsed_sec": float(elapsed),
            }
        )

    full_internal_eval_df = pd.DataFrame(full_internal_eval_rows).sort_values(
        ["mean_user_f1", "micro_f1"], ascending=False
    )
    full_internal_eval_path = os.path.join(
        OUT_DIR, "internal_eval_three_scale_postprocessing_policy_full_split_metrics.csv"
    )
    full_internal_eval_df.to_csv(full_internal_eval_path, index=False, encoding="utf-8-sig")

    splits_df, split_summary, _ = pr.run_split_separated_selection(
        gold_df=gold_df,
        policy_predictions=policy_predictions,
        candidate_policies=candidate_policies,
        fixed_reference_policies=fixed_reference_policies,
        n_repeats=50,
    )
    splits_path = os.path.join(
        OUT_DIR, "internal_eval_three_scale_postprocessing_policy_repeated_split_metrics.csv"
    )
    splits_df.to_csv(splits_path, index=False, encoding="utf-8-sig")

    recommended_policy = choose_recommended_fixed_policy(
        split_summary["fixed_policy_outer_stats"],
        preferred_order=["S0_Full", "S6_NoTopK_KeepConf", "S8_Top4_Conf50", "S9_Top4_Conf55", "S10_Top4_Conf50_Min210"],
    )

    summary = {
        "dataset": "internal 20-user evaluation set only (data/signals/internal_eval + internal_eval_annotations.csv)",
        "restriction": "Policy search performed only on the internal 20-user set. external_test labels were not used.",
        "setting": "best-per-scale 3-model + small-range post-processing policy checks",
        "selected_models": selected_models,
        "candidate_policies": candidate_policies,
        "full_internal_eval_policy_metrics": {row["policy"]: row for row in full_internal_eval_rows},
        "split_separated_summary": split_summary,
        "recommended_fixed_policy": recommended_policy,
        "selection_rule": "Choose the fixed policy with highest split-separated outer mean user F1; break ties by lower std, then conservative preference order.",
        "artifacts": {
            "full_internal_eval_csv": os.path.relpath(full_internal_eval_path, BASE_DIR),
            "splits_csv": os.path.relpath(splits_path, BASE_DIR),
        },
    }

    summary_path = os.path.join(OUT_DIR, "internal_eval_three_scale_postprocessing_policy_selection.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
