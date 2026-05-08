"""External unlabeled-cohort stress-test script.

Purpose:
    Applies the saved inference pipeline to external signal files without using
    labels, then summarizes segment counts, duration distributions, confidence,
    and policy sensitivity.
Inputs:
    Reads raw-index metadata, archived raw signal files, and saved model assets.
Outputs:
    Writes unlabeled-cohort prediction summaries under `experiments/results/`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import os
import pickle
import re
import sys
import zipfile
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from imu_activity_pipeline.config import (  # noqa: E402
    ACTIVITIES,
    DEVICE,
    EXTERNAL_TEST_DATA_DIR,
    MODEL_DIR,
    RAW_INDEX_FILE,
    RAW_ZIP_DIR,
    TRAIN_ANNOTATIONS_FILE,
)
from imu_activity_pipeline.sensor_data_processing import butterworth_filter, load_gold_labels, load_sensor_data  # noqa: E402
import experiments.run_external_test_saved_model_evaluation_suite as ex  # noqa: E402


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
RAW_INDEX = RAW_INDEX_FILE
GOLD_PATH = TRAIN_ANNOTATIONS_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external unlabeled cohort stress test.")
    parser.add_argument("--max-ids", type=int, default=8, help="Maximum number of external IDs.")
    parser.add_argument(
        "--max-hours-per-id",
        type=float,
        default=6.0,
        help="Approximate cap of indexed hours per external ID from optional raw index metadata.",
    )
    parser.add_argument("--min-files-per-id", type=int, default=3, help="Minimum file count per external ID.")
    parser.add_argument(
        "--include-generic-ids",
        action="store_true",
        help="Include generic IDs like testDB3/tes_68A/48E/test_jzq.",
    )
    return parser.parse_args()


def is_user_like_external_id(external_id: str) -> bool:
    return bool(re.match(r"^(HNU|HAU|HDU|HN)\d+$", external_id))


def load_raw_index() -> pd.DataFrame:
    df = pd.read_csv(RAW_INDEX)
    df["externalid"] = df["externalid"].astype(str)
    df["sensorData"] = df["sensorData"].astype(str)
    df["start_dt"] = pd.to_datetime(df["timeStamp.startTime"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["timeStamp.endTime"], errors="coerce")
    df["duration_h_meta"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    df["duration_h_meta"] = df["duration_h_meta"].fillna(0.0).clip(lower=0.0)
    return df


def load_gold_ids() -> List[str]:
    gold = load_gold_labels(GOLD_PATH)
    return sorted(gold["user_id"].astype(str).unique().tolist())


def select_external_rows(
    raw_df: pd.DataFrame,
    gold_ids: List[str],
    max_ids: int,
    min_files_per_id: int,
    max_hours_per_id: float,
    include_generic_ids: bool,
) -> pd.DataFrame:
    ext = raw_df[~raw_df["externalid"].isin(gold_ids)].copy()
    if not include_generic_ids:
        ext = ext[ext["externalid"].map(is_user_like_external_id)].copy()

    counts = ext["externalid"].value_counts()
    kept_ids = counts[counts >= min_files_per_id].sort_values(ascending=False).head(max_ids).index.tolist()
    if not kept_ids:
        return ext.iloc[0:0].copy()

    selected_rows = []
    for eid in kept_ids:
        part = ext[ext["externalid"] == eid].copy()
        part = part.sort_values(["start_dt", "end_dt", "sensorData"], kind="stable")
        if max_hours_per_id <= 0:
            selected_rows.append(part)
            continue

        cur_hours = 0.0
        keep_mask = []
        for _, row in part.iterrows():
            keep = cur_hours < max_hours_per_id
            keep_mask.append(keep)
            if keep:
                cur_hours += float(row["duration_h_meta"])
        chosen = part.loc[keep_mask].copy()
        if chosen.empty:
            chosen = part.head(1).copy()
        selected_rows.append(chosen)

    out = pd.concat(selected_rows, axis=0, ignore_index=True)
    out["zip_name"] = out["sensorData"].map(lambda x: os.path.basename(x))
    out["zip_path"] = out["zip_name"].map(lambda x: os.path.join(RAW_ZIP_DIR, x))
    out["zip_exists"] = out["zip_path"].map(os.path.exists)
    return out


def load_sensor_from_zip(zip_path: str) -> Optional[np.ndarray]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txt_candidates:
                return None
            txt_name = txt_candidates[0]
            with zf.open(txt_name, "r") as f:
                df = pd.read_csv(f, sep="\t", on_bad_lines="skip", low_memory=False)
    except Exception:
        return None

    if "GYRO_Z" not in df.columns and "GYRO_" in df.columns:
        df = df.rename(columns={"GYRO_": "GYRO_Z"})

    needed = ["ACC_TIME", "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    if any(col not in df.columns for col in needed):
        return None

    try:
        df = df[needed].apply(pd.to_numeric, errors="coerce")
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return None

        df["ACC_TIME"] = df["ACC_TIME"].astype(np.int64)
        for col in needed[1:]:
            df[col] = df[col].astype(np.float32)

        df = df[df["ACC_TIME"] > 0].reset_index(drop=True)
        if df.empty:
            return None

        df = df.sort_values("ACC_TIME").reset_index(drop=True)
        arr = df.values
        if len(arr) > 100:
            arr[:, 1:] = butterworth_filter(arr[:, 1:])
        return arr
    except Exception:
        return None


def build_external_user_data(selection_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    user_data = {}
    for eid, part in selection_df.groupby("externalid"):
        part = part.sort_values(["start_dt", "end_dt", "sensorData"], kind="stable")
        arrays = []
        for _, row in part.iterrows():
            zip_path = row["zip_path"]
            if not os.path.exists(zip_path):
                continue
            arr = load_sensor_from_zip(zip_path)
            if arr is None or len(arr) < 300:
                continue
            arrays.append(arr)

        if not arrays:
            continue

        merged = np.concatenate(arrays, axis=0)
        order = np.argsort(merged[:, 0], kind="stable")
        merged = merged[order]
        _, uniq_idx = np.unique(merged[:, 0], return_index=True)
        merged = merged[np.sort(uniq_idx)]
        if len(merged) >= 300:
            user_data[str(eid)] = merged
    return user_data


def build_single3s_group(device: torch.device) -> Dict:
    with open(os.path.join(MODEL_DIR, "norm_params_3s.pkl"), "rb") as f:
        norm_3s = pickle.load(f)
    model = ex.load_model(os.path.join(MODEL_DIR, "combined_model_best.pth"), 300, device)
    return {
        "3s": {
            "models": [model],
            "window_size": 300,
            "window_step": 100,
            "window_sec": 3,
            "norm_params": norm_3s,
        }
    }


def infer_dataset(
    cohort_name: str,
    user_data: Dict[str, np.ndarray],
    model_group: Dict,
    policies: Dict[str, ex.PPConfig],
    device: torch.device,
) -> (pd.DataFrame, pd.DataFrame):
    seg_rows = []
    summary_rows = []

    for uid, raw in sorted(user_data.items()):
        ts, probs = ex.predict_multiscale_ensemble(raw, model_group, device)
        if len(ts) == 0:
            continue

        span_h = max(1e-9, float((raw[-1, 0] - raw[0, 0]) / 1000.0 / 3600.0))
        for policy_name, cfg in policies.items():
            segs = ex.postprocess(ts, probs, raw, cfg, window_sec=3)

            confs = [float(s["confidence"]) for s in segs]
            durs = [float(max(0.0, (int(s["end_ts"]) - int(s["start_ts"])) / 1000.0)) for s in segs]
            active_seconds = float(np.sum(durs)) if durs else 0.0

            for s, dur in zip(segs, durs):
                seg_rows.append(
                    {
                        "cohort": cohort_name,
                        "policy": policy_name,
                        "user_id": uid,
                        "category": s["class_name"],
                        "confidence": float(s["confidence"]),
                        "duration_sec": dur,
                        "start_ts": int(s["start_ts"]),
                        "end_ts": int(s["end_ts"]),
                    }
                )

            summary_rows.append(
                {
                    "cohort": cohort_name,
                    "policy": policy_name,
                    "user_id": uid,
                    "record_hours": span_h,
                    "n_segments": int(len(segs)),
                    "segments_per_hour": float(len(segs) / span_h),
                    "active_seconds": active_seconds,
                    "active_ratio": float(active_seconds / (span_h * 3600.0)),
                    "mean_confidence": float(np.mean(confs)) if confs else 0.0,
                    "median_confidence": float(np.median(confs)) if confs else 0.0,
                    "p10_confidence": float(np.quantile(confs, 0.10)) if confs else 0.0,
                    "p90_confidence": float(np.quantile(confs, 0.90)) if confs else 0.0,
                    "mean_duration_sec": float(np.mean(durs)) if durs else 0.0,
                    "median_duration_sec": float(np.median(durs)) if durs else 0.0,
                    "p90_duration_sec": float(np.quantile(durs, 0.90)) if durs else 0.0,
                }
            )

    return pd.DataFrame(seg_rows), pd.DataFrame(summary_rows)


def summarize_policies(seg_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, policy), part in user_df.groupby(["cohort", "policy"], dropna=False):
        seg_part = seg_df[(seg_df["cohort"] == cohort) & (seg_df["policy"] == policy)]
        total_hours = float(part["record_hours"].sum())
        total_segments = int(part["n_segments"].sum())
        weighted_seg_h = float(total_segments / total_hours) if total_hours > 0 else 0.0

        class_counts = seg_part["category"].value_counts().to_dict()
        total_cls = sum(int(v) for v in class_counts.values())
        class_share = {}
        for cls in ACTIVITIES:
            c = int(class_counts.get(cls, 0))
            class_share[f"class_count_{cls}"] = c
            class_share[f"class_share_{cls}"] = float(c / total_cls) if total_cls > 0 else 0.0

        rows.append(
            {
                "cohort": cohort,
                "policy": policy,
                "users": int(part["user_id"].nunique()),
                "total_hours": total_hours,
                "total_segments": total_segments,
                "segments_per_hour_weighted": weighted_seg_h,
                "segments_per_hour_user_mean": float(part["segments_per_hour"].mean()),
                "segments_per_hour_user_median": float(part["segments_per_hour"].median()),
                "active_ratio_user_mean": float(part["active_ratio"].mean()),
                "segment_confidence_mean": float(seg_part["confidence"].mean()) if not seg_part.empty else 0.0,
                "segment_confidence_median": float(seg_part["confidence"].median()) if not seg_part.empty else 0.0,
                "segment_duration_median_sec": float(seg_part["duration_sec"].median()) if not seg_part.empty else 0.0,
                **class_share,
            }
        )
    return pd.DataFrame(rows)


def build_shift_table(seg_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in sorted(policy_df["policy"].unique()):
        ext_row = policy_df[(policy_df["cohort"] == "external_unlabeled") & (policy_df["policy"] == policy)]
        external_test_row = policy_df[
            (policy_df["cohort"] == "external_test_labeled") & (policy_df["policy"] == policy)
        ]
        if ext_row.empty or external_test_row.empty:
            continue

        ext_row = ext_row.iloc[0]
        external_test_row = external_test_row.iloc[0]

        ext_seg = seg_df[(seg_df["cohort"] == "external_unlabeled") & (seg_df["policy"] == policy)]
        external_test_seg = seg_df[
            (seg_df["cohort"] == "external_test_labeled") & (seg_df["policy"] == policy)
        ]

        ext_dist = np.array([float(ext_row.get(f"class_share_{cls}", 0.0)) for cls in ACTIVITIES], dtype=np.float64)
        external_test_dist = np.array(
            [float(external_test_row.get(f"class_share_{cls}", 0.0)) for cls in ACTIVITIES],
            dtype=np.float64,
        )
        if ext_dist.sum() > 0:
            ext_dist = ext_dist / ext_dist.sum()
        if external_test_dist.sum() > 0:
            external_test_dist = external_test_dist / external_test_dist.sum()
        js = float(jensenshannon(ext_dist + 1e-12, external_test_dist + 1e-12, base=2.0))

        if not ext_seg.empty and not external_test_seg.empty:
            ks_conf = ks_2samp(ext_seg["confidence"].values, external_test_seg["confidence"].values)
            ks_dur = ks_2samp(ext_seg["duration_sec"].values, external_test_seg["duration_sec"].values)
            ks_conf_stat, ks_conf_p = float(ks_conf.statistic), float(ks_conf.pvalue)
            ks_dur_stat, ks_dur_p = float(ks_dur.statistic), float(ks_dur.pvalue)
        else:
            ks_conf_stat, ks_conf_p = 0.0, 1.0
            ks_dur_stat, ks_dur_p = 0.0, 1.0

        rows.append(
            {
                "policy": policy,
                "ext_users": int(ext_row["users"]),
                "external_test_users": int(external_test_row["users"]),
                "delta_segments_per_hour_weighted": float(
                    ext_row["segments_per_hour_weighted"] - external_test_row["segments_per_hour_weighted"]
                ),
                "delta_segment_confidence_mean": float(
                    ext_row["segment_confidence_mean"] - external_test_row["segment_confidence_mean"]
                ),
                "delta_segment_duration_median_sec": float(
                    ext_row["segment_duration_median_sec"] - external_test_row["segment_duration_median_sec"]
                ),
                "class_js_distance": js,
                "ks_conf_stat": ks_conf_stat,
                "ks_conf_pvalue": ks_conf_p,
                "ks_duration_stat": ks_dur_stat,
                "ks_duration_pvalue": ks_dur_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_df = load_raw_index()
    gold_ids = load_gold_ids()
    selection_df = select_external_rows(
        raw_df=raw_df,
        gold_ids=gold_ids,
        max_ids=args.max_ids,
        min_files_per_id=args.min_files_per_id,
        max_hours_per_id=args.max_hours_per_id,
        include_generic_ids=args.include_generic_ids,
    )
    selection_df.to_csv(os.path.join(OUT_DIR, "external_unlabeled_selection.csv"), index=False, encoding="utf-8-sig")

    external_user_data = build_external_user_data(selection_df)
    if not external_user_data:
        raise RuntimeError("No external user data could be built. Check selection and zip files.")

    external_test_user_data = {}
    for name in sorted(os.listdir(EXTERNAL_TEST_DATA_DIR)):
        if not name.endswith(".txt"):
            continue
        uid = name.replace(".txt", "")
        arr = load_sensor_data(os.path.join(EXTERNAL_TEST_DATA_DIR, name), apply_filter=True)
        if arr is not None and len(arr) > 0:
            external_test_user_data[uid] = arr

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model_group = build_single3s_group(device)

    policies = {
        "S5_BaselinePP": ex.PP_BASELINE,
        "S0_Full": ex.PP_FULL,
        "S2_NoTopKConf": replace(ex.PP_FULL, top_k=0, conf_min=0.0),
    }

    ext_seg_df, ext_user_df = infer_dataset(
        cohort_name="external_unlabeled",
        user_data=external_user_data,
        model_group=model_group,
        policies=policies,
        device=device,
    )
    external_test_seg_df, external_test_user_df = infer_dataset(
        cohort_name="external_test_labeled",
        user_data=external_test_user_data,
        model_group=model_group,
        policies=policies,
        device=device,
    )

    seg_df = pd.concat([ext_seg_df, external_test_seg_df], axis=0, ignore_index=True)
    user_df = pd.concat([ext_user_df, external_test_user_df], axis=0, ignore_index=True)
    policy_df = summarize_policies(seg_df, user_df)
    shift_df = build_shift_table(seg_df, policy_df)

    seg_df.to_csv(os.path.join(OUT_DIR, "external_unlabeled_segments.csv"), index=False, encoding="utf-8-sig")
    user_df.to_csv(os.path.join(OUT_DIR, "external_unlabeled_user_summary.csv"), index=False, encoding="utf-8-sig")
    policy_df.to_csv(os.path.join(OUT_DIR, "external_unlabeled_policy_summary.csv"), index=False, encoding="utf-8-sig")
    shift_df.to_csv(os.path.join(OUT_DIR, "external_vs_external_test_shift.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "selection": {
            "selected_ids": sorted(selection_df["externalid"].unique().tolist()),
            "selected_id_count": int(selection_df["externalid"].nunique()),
            "selected_rows": int(len(selection_df)),
            "selected_existing_zip_rows": int(selection_df["zip_exists"].sum()),
        },
        "external_users_loaded": int(len(external_user_data)),
        "external_test_users_loaded": int(len(external_test_user_data)),
        "policies": list(policies.keys()),
    }
    if not shift_df.empty:
        summary["shift_table"] = shift_df.to_dict(orient="records")

    with open(os.path.join(OUT_DIR, "external_unlabeled_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"External users loaded: {len(external_user_data)}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
