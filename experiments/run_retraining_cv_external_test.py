"""Retraining-based cross-validation and external_test experiment.

Purpose:
    Trains fold-specific models and an external_test evaluation model to assess
    robustness, transfer behavior, and variance beyond the fixed saved ensemble.
Inputs:
    Reads labeled training/evaluation signals and annotations from the configured
    data layout.
Outputs:
    Writes retrained checkpoints, prediction workbooks, and evaluation summaries.
"""
from __future__ import annotations
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
import sys
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from imu_activity_pipeline import train as train_mod
from imu_activity_pipeline.config import (
    ALL_ANNOTATIONS_FILE,
    EXTERNAL_TEST_DATA_DIR,
    EXTERNAL_TEST_GOLD_FILE,
    INTERNAL_EVAL_DATA_DIR,
    MODEL_DIR,
    TRAIN_DATA_DIR,
)
from imu_activity_pipeline.sensor_data_processing import assign_window_labels, create_windows, load_gold_labels, load_sensor_data, normalize_imu
from imu_activity_pipeline.inference import predict_multiscale_ensemble
from imu_activity_pipeline.neural_network_models import CombinedModel, FocalLoss, TripletLoss

import experiments.run_external_test_saved_model_evaluation_suite as ex


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def build_user_file_map() -> Dict[str, str]:
    user_to_file: Dict[str, str] = {}
    for dpath in (TRAIN_DATA_DIR, INTERNAL_EVAL_DATA_DIR, EXTERNAL_TEST_DATA_DIR):
        if not os.path.isdir(dpath):
            continue
        for name in sorted(os.listdir(dpath)):
            if not name.endswith(".txt"):
                continue
            uid = name[:-4]
            user_to_file[uid] = os.path.join(dpath, name)
    return user_to_file


def build_window_dataset(
    users: List[str],
    gold_df: pd.DataFrame,
    user_raw: Dict[str, np.ndarray],
    window_size: int,
    window_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    g_list: List[np.ndarray] = []

    for i, uid in enumerate(users, start=1):
        raw = user_raw[uid]
        ts, windows = create_windows(raw, window_size, window_step)
        if len(windows) == 0:
            continue
        _, cls = assign_window_labels(ts, gold_df, uid)
        labels = np.where(cls >= 0, cls + 1, 0).astype(np.int64)
        x_list.append(windows.astype(np.float32))
        y_list.append(labels)
        g_list.append(np.array([uid] * len(labels), dtype=object))
        if i % 10 == 0 or i == len(users):
            print(f"  built windows for {i}/{len(users)} users")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return x_all, y_all, groups


def make_single_model_group(
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    window_size: int,
    window_step: int,
    window_sec: int,
) -> Dict:
    return {
        "3s": {
            "models": [model],
            "window_size": window_size,
            "window_step": window_step,
            "window_sec": window_sec,
            "norm_params": {"mean": mean, "std": std},
        }
    }


def infer_users(
    users: List[str],
    user_raw: Dict[str, np.ndarray],
    model_group: Dict,
    device: torch.device,
    policy_cfg: ex.PPConfig,
    window_sec: int,
) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for uid in users:
        raw = user_raw[uid]
        ts, probs = predict_multiscale_ensemble(raw, model_group, device)
        out[uid] = ex.postprocess(ts, probs, raw, policy_cfg, window_sec=window_sec)
    return out


def evaluate_policy(
    policy_name: str,
    users: List[str],
    gold_df: pd.DataFrame,
    user_raw: Dict[str, np.ndarray],
    model_group: Dict,
    device: torch.device,
    policy_cfg: ex.PPConfig,
    window_sec: int,
) -> Tuple[Dict, Dict[str, List[Dict]]]:
    pred_by_user = infer_users(users, user_raw, model_group, device, policy_cfg, window_sec=window_sec)
    eval_payload = ex.evaluate_segments(pred_by_user, gold_df[gold_df["user_id"].isin(users)].copy())
    overall = eval_payload["overall"]
    return {
        "policy": policy_name,
        "mean_user_f1": float(overall["mean_user_f1"]),
        "ci95_low": float(overall["ci95_low"]),
        "ci95_high": float(overall["ci95_high"]),
        "micro_f1": float(overall["micro_f1"]),
        "TP": int(overall["TP"]),
        "FP": int(overall["FP"]),
        "FN": int(overall["FN"]),
    }, pred_by_user


def train_one_split(
    x_all: np.ndarray,
    y_all: np.ndarray,
    groups: np.ndarray,
    train_users: List[str],
    val_users: List[str],
    seed: int,
    model_name: str,
    window_size: int,
    epochs: int,
    batch_size: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    train_mask = np.isin(groups, train_users)
    val_mask = np.isin(groups, val_users)

    x_train_raw = x_all[train_mask]
    y_train = y_all[train_mask]
    x_val_raw = x_all[val_mask]
    y_val = y_all[val_mask]

    x_train, mean, std = normalize_imu(x_train_raw)
    x_val, _, _ = normalize_imu(x_val_raw, mean, std)

    device = torch.device(train_mod.DEVICE if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).long(),
    )

    class_counts = np.bincount(y_train, minlength=6).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_weights = (len(y_train) / (6.0 * class_counts)).astype(np.float32)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=True,
    )

    model = CombinedModel(input_channels=6, num_classes=6, window_size=window_size).to(device)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    focal = FocalLoss(alpha=weight_tensor, gamma=2.0)
    triplet = TripletLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    best_f1 = -1.0
    best_state = None
    patience = 2
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, emb = model(xb, return_embedding=True)
            loss = ce(logits, yb) + 0.2 * focal(logits, yb) + 0.1 * triplet(emb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(yb.numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  {model_name} epoch {epoch}/{epochs} val_macro_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    ckpt_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "epoch": epoch,
            "val_f1": best_f1,
            "seed": seed,
            "window_size": window_size,
        },
        ckpt_path,
    )
    model = ex.load_model(ckpt_path, window_size, device)
    return mean, std, model


def save_external_test_predictions(
    out_csv: str,
    pred_s0: Dict[str, List[Dict]],
    pred_s2: Dict[str, List[Dict]],
) -> None:
    rows: List[Dict] = []
    for policy, payload in (("S0_Full", pred_s0), ("S2_NoTopKConf", pred_s2)):
        for uid, segs in payload.items():
            for seg in segs:
                rows.append(
                    {
                        "policy": policy,
                        "user_id": uid,
                        "category": seg["class_name"],
                        "confidence": float(seg["confidence"]),
                        "duration_sec": float(seg["duration"]),
                        "start": int(seg["start_ts"]),
                        "end": int(seg["end_ts"]),
                    }
                )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(
            columns=["policy", "user_id", "category", "confidence", "duration_sec", "start", "end"]
        )
    df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-step", type=int, default=100)
    parser.add_argument("--window-sec", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=6)
    parser.add_argument("--external-test-epochs", "--external-epochs", dest="external_test_epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gold_all = load_gold_labels(ALL_ANNOTATIONS_FILE)
    gold_external_test = load_gold_labels(EXTERNAL_TEST_GOLD_FILE)
    gold_external_test["user_id"] = gold_external_test["user_id"].astype(str)
    gold_all["user_id"] = gold_all["user_id"].astype(str)

    user_to_file = build_user_file_map()
    external_test_users = sorted(set(gold_external_test["user_id"].unique().tolist()))
    all_users = sorted(set(gold_all["user_id"].unique().tolist()))
    internal_users = sorted([u for u in all_users if u not in set(external_test_users)])

    print("==== Dataset split summary ====")
    print(f"all labeled users: {len(all_users)}")
    print(f"internal users (for retrain CV): {len(internal_users)}")
    print(f"external_test labeled users: {len(external_test_users)}")

    user_raw: Dict[str, np.ndarray] = {}
    for uid in all_users:
        if uid not in user_to_file:
            continue
        raw = load_sensor_data(user_to_file[uid], apply_filter=True)
        if raw is None or len(raw) < args.window_size:
            continue
        user_raw[uid] = raw

    internal_users = [u for u in internal_users if u in user_raw]
    external_test_users = [u for u in external_test_users if u in user_raw]

    print("==== Building internal windows ====")
    x_all, y_all, groups = build_window_dataset(
        internal_users,
        gold_all,
        user_raw,
        window_size=args.window_size,
        window_step=args.window_step,
    )
    print(f"internal windows: {len(y_all)}")

    device = torch.device(train_mod.DEVICE if torch.cuda.is_available() else "cpu")
    policy_s0 = ex.PP_FULL
    policy_s2 = replace(ex.PP_FULL, top_k=0, conf_min=0.0)

    # 1) Retraining CV
    cv_records: List[Dict] = []
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    internal_users_arr = np.array(internal_users)

    print("==== Retraining CV ====")
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(internal_users_arr), start=1):
        tr_users = internal_users_arr[tr_idx].tolist()
        va_users = internal_users_arr[va_idx].tolist()
        fold_seed = args.seed + fold_id
        model_name = f"retrain_cv_fold{fold_id}_seed{fold_seed}"
        print(f"\n[CV fold {fold_id}/{args.cv_folds}] train_users={len(tr_users)} val_users={len(va_users)}")

        mean, std, model = train_one_split(
            x_all,
            y_all,
            groups,
            tr_users,
            va_users,
            seed=fold_seed,
            model_name=model_name,
            window_size=args.window_size,
            epochs=args.cv_epochs,
            batch_size=args.batch_size,
            workers=args.workers,
        )

        model_group = make_single_model_group(
            model,
            mean,
            std,
            window_size=args.window_size,
            window_step=args.window_step,
            window_sec=args.window_sec,
        )

        rec_s0, _ = evaluate_policy(
            "S0_Full",
            va_users,
            gold_all,
            user_raw,
            model_group,
            device,
            policy_s0,
            window_sec=args.window_sec,
        )
        rec_s0.update({"fold": fold_id, "n_val_users": len(va_users)})
        cv_records.append(rec_s0)

        rec_s2, _ = evaluate_policy(
            "S2_NoTopKConf",
            va_users,
            gold_all,
            user_raw,
            model_group,
            device,
            policy_s2,
            window_sec=args.window_sec,
        )
        rec_s2.update({"fold": fold_id, "n_val_users": len(va_users)})
        cv_records.append(rec_s2)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_df = pd.DataFrame(cv_records)
    cv_csv = os.path.join(OUT_DIR, "retrain_cv_fold_metrics.csv")
    cv_df.to_csv(cv_csv, index=False)

    # 2) Labeled external_test evaluation
    print("\n==== Labeled external_test evaluation ====")
    tr_users_external_test, va_users_external_test = train_test_split(
        internal_users, test_size=0.12, random_state=args.seed, shuffle=True
    )
    model_name_external_test = f"retrain_external_test_seed{args.seed}"
    mean_external_test, std_external_test, model_external_test = train_one_split(
        x_all,
        y_all,
        groups,
        tr_users_external_test,
        va_users_external_test,
        seed=args.seed,
        model_name=model_name_external_test,
        window_size=args.window_size,
        epochs=args.external_test_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    model_group_external_test = make_single_model_group(
        model_external_test,
        mean_external_test,
        std_external_test,
        window_size=args.window_size,
        window_step=args.window_step,
        window_sec=args.window_sec,
    )

    external_test_s0, pred_s0 = evaluate_policy(
        "S0_Full",
        external_test_users,
        gold_external_test,
        user_raw,
        model_group_external_test,
        device,
        policy_s0,
        window_sec=args.window_sec,
    )
    external_test_s2, pred_s2 = evaluate_policy(
        "S2_NoTopKConf",
        external_test_users,
        gold_external_test,
        user_raw,
        model_group_external_test,
        device,
        policy_s2,
        window_sec=args.window_sec,
    )
    external_test_df = pd.DataFrame([external_test_s0, external_test_s2])
    external_test_df["n_external_test_users"] = len(external_test_users)
    external_test_csv = os.path.join(OUT_DIR, "external_test_retrain_eval.csv")
    external_test_df.to_csv(external_test_csv, index=False)

    pred_csv = os.path.join(OUT_DIR, "external_test_retrain_predictions.csv")
    save_external_test_predictions(pred_csv, pred_s0, pred_s2)

    # 3) Summary payload
    summary = {
        "config": {
            "window_size": args.window_size,
            "window_step": args.window_step,
            "window_sec": args.window_sec,
            "cv_folds": args.cv_folds,
            "cv_epochs": args.cv_epochs,
            "external_test_epochs": args.external_test_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "dataset": {
            "all_labeled_users": len(all_users),
            "internal_users": len(internal_users),
            "external_test_users": len(external_test_users),
            "internal_windows": int(len(y_all)),
        },
        "cv_summary": {},
        "external_test_eval": external_test_df.to_dict(orient="records"),
        "artifacts": {
            "cv_fold_metrics_csv": cv_csv,
            "external_test_eval_csv": external_test_csv,
            "external_test_predictions_csv": pred_csv,
        },
    }

    for policy, grp in cv_df.groupby("policy"):
        summary["cv_summary"][policy] = {
            "mean_user_f1_mean": float(grp["mean_user_f1"].mean()),
            "mean_user_f1_std": float(grp["mean_user_f1"].std(ddof=1) if len(grp) > 1 else 0.0),
            "micro_f1_mean": float(grp["micro_f1"].mean()),
            "micro_f1_std": float(grp["micro_f1"].std(ddof=1) if len(grp) > 1 else 0.0),
        }

    summary_json = os.path.join(OUT_DIR, "retrain_cv_external_test_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Done ====")
    print(f"- {cv_csv}")
    print(f"- {external_test_csv}")
    print(f"- {pred_csv}")
    print(f"- {summary_json}")


if __name__ == "__main__":
    main()
