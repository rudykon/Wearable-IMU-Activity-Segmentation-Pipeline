"""Feature-MLP utilities for public-dataset TRL portability experiments.

Purpose:
    Implements normalization, dataloaders, a compact feature MLP, training loops,
    probability prediction, and summary printing for feature-based portability
    checks.
Inputs:
    Consumes fixed-size hand-crafted feature matrices built by public dataset
    loaders.
Outputs:
    Produces per-window probabilities, decoded subject-level predictions, and
    compact metrics for result tables.
"""
from __future__ import annotations


import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from temporal_record_layer_common import (
    SubjectSequence,
    build_subject_windows,
    evaluate_subjects,
    select_trl_params,
    write_dataset_outputs,
)


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32, copy=False)


class TinyFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, width: int = 128, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    requested = os.environ.get("IMU_PUBLIC_MLP_DEVICE") or os.environ.get("IMU_PUBLIC_TCN_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_features(subjects: list) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([subject.features for subject in subjects]).astype(np.float32, copy=False)
    y = np.concatenate([subject.labels for subject in subjects]).astype(np.int32, copy=False)
    return x, y


def fit_normalizer(x_train: np.ndarray) -> FeatureNormalizer:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureNormalizer(mean=mean, std=std)


def make_loader(x: np.ndarray, y_idx: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_idx.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_proba_array(
    model: nn.Module,
    x: np.ndarray,
    *,
    normalizer: FeatureNormalizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    x_norm = normalizer.transform(x)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_norm)), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for (xb,) in loader:
        logits = model(xb.to(device, non_blocking=True))
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def predict_subjects(
    model: nn.Module,
    subjects: list,
    *,
    normalizer: FeatureNormalizer,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    return {
        subject.subject: predict_proba_array(
            model,
            subject.features,
            normalizer=normalizer,
            device=device,
            batch_size=batch_size,
        )
        for subject in subjects
    }


def train_feature_mlp(
    train_subjects: list,
    dev_subjects: list,
    *,
    seed: int = 2026,
) -> tuple[nn.Module, np.ndarray, FeatureNormalizer, list[dict], torch.device]:
    set_seed(seed)
    device = resolve_device()
    max_epochs = int(os.environ.get("IMU_PUBLIC_MLP_EPOCHS", 80))
    patience = int(os.environ.get("IMU_PUBLIC_MLP_PATIENCE", 12))
    batch_size = int(os.environ.get("IMU_PUBLIC_MLP_BATCH_SIZE", 512))
    learning_rate = float(os.environ.get("IMU_PUBLIC_MLP_LR", 1e-3))
    weight_decay = float(os.environ.get("IMU_PUBLIC_MLP_WEIGHT_DECAY", 1e-4))
    width = int(os.environ.get("IMU_PUBLIC_MLP_WIDTH", 128))
    dropout = float(os.environ.get("IMU_PUBLIC_MLP_DROPOUT", 0.25))

    x_train, y_train = stack_features(train_subjects)
    x_dev, y_dev = stack_features(dev_subjects)
    classes = np.asarray(sorted(np.unique(y_train).tolist()), dtype=np.int32)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    y_train_idx = np.asarray([class_to_idx[int(label)] for label in y_train], dtype=np.int64)

    normalizer = fit_normalizer(x_train)
    x_train_norm = normalizer.transform(x_train)
    train_loader = make_loader(x_train_norm, y_train_idx, batch_size=batch_size, shuffle=True)

    model = TinyFeatureMLP(input_dim=x_train.shape[1], n_classes=len(classes), width=width, dropout=dropout).to(device)
    counts = np.bincount(y_train_idx, minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        dev_proba = predict_proba_array(model, x_dev, normalizer=normalizer, device=device, batch_size=batch_size)
        dev_pred = classes[np.argmax(dev_proba, axis=1)]
        dev_acc = accuracy_score(y_dev, dev_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "dev_window_accuracy": float(dev_acc),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )

        if dev_acc > best_score + 1e-5:
            best_score = float(dev_acc)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.training_config = {"width": width, "dropout": dropout}
    return model, classes, normalizer, history, device


def run_mlp_portability_experiment(
    *,
    dataset_name: str,
    source: str,
    sequences: dict[str, SubjectSequence],
    split: dict[str, list[str]],
    label_names: dict[int, str],
    window_s: float,
    step_s: float,
    fs_hz: float,
    result_dir: Path,
    prefix: str,
    caption: str,
    seed: int = 2026,
    ignore_labels: set[int] | None = None,
) -> dict:
    all_subject_ids = split["train_subjects"] + split["dev_subjects"] + split["test_subjects"]
    missing = [subject for subject in all_subject_ids if subject not in sequences]
    if missing:
        raise FileNotFoundError(f"Missing subjects for {dataset_name}: {missing}")

    windows = {
        subject: build_subject_windows(sequences[subject], window_s=window_s, step_s=step_s, fs_hz=fs_hz)
        for subject in all_subject_ids
    }
    train = [windows[s] for s in split["train_subjects"]]
    dev = [windows[s] for s in split["dev_subjects"]]
    test = [windows[s] for s in split["test_subjects"]]

    model, classes, normalizer, history, device = train_feature_mlp(train, dev, seed=seed)
    batch_size = int(os.environ.get("IMU_PUBLIC_MLP_BATCH_SIZE", 512))
    dev_proba = predict_subjects(model, dev, normalizer=normalizer, device=device, batch_size=batch_size)
    test_proba = predict_subjects(model, test, normalizer=normalizer, device=device, batch_size=batch_size)

    selected_params, dev_grid = select_trl_params(
        dev,
        dev_proba,
        classes,
        min_segment_s=window_s,
        merge_gaps_s=(0.0, step_s, window_s, 2.0 * window_s),
        min_segments_s=(0.0, window_s, 1.5 * window_s, 2.0 * window_s),
        ignore_labels=ignore_labels,
    )
    test_baseline = evaluate_subjects(
        test,
        test_proba,
        classes,
        min_segment_s=window_s,
        mode="argmax",
        ignore_labels=ignore_labels,
    )
    test_trl = evaluate_subjects(
        test,
        test_proba,
        classes,
        min_segment_s=window_s,
        mode="trl",
        params=selected_params,
        ignore_labels=ignore_labels,
    )

    summary = {
        "dataset": dataset_name,
        "source": source,
        "split": split,
        "window": {"length_s": window_s, "step_s": step_s, "fs_hz": fs_hz},
        "evaluation": {
            "matching": "class-consistent one-to-one IoU > 0.5",
            "minimum_episode_s": window_s,
            "ignored_labels": sorted(ignore_labels) if ignore_labels else [],
            "note": "Record-level matching excludes episodes shorter than one analysis window.",
        },
        "classifier": "TinyFeatureMLP(hand-crafted robust window statistics, class-balanced cross entropy)",
        "training": {
            "device": str(device),
            "seed": seed,
            **getattr(model, "training_config", {}),
            "epochs_run": len(history),
            "best_dev_window_accuracy": max((item["dev_window_accuracy"] for item in history), default=None),
            "history": history,
        },
        "labels": label_names,
        "selected_trl_params": selected_params,
        "dev_grid": dev_grid[:250],
        "test_metrics": {
            "Window argmax + merge": test_baseline,
            "Window argmax + dev-selected TRL-style record layer": test_trl,
        },
    }
    summary["dev_grid_csv"] = str(result_dir / f"{prefix}_trl_dev_grid.csv")
    write_dataset_outputs(summary, result_dir, prefix, caption)
    return summary


def print_summary(summary: dict, result_dir: Path, prefix: str) -> None:
    print("Selected TRL params:")
    print(pd.Series(summary["selected_trl_params"]).to_string())
    print("Training:")
    print(pd.Series({k: v for k, v in summary["training"].items() if k != "history"}).to_string())
    print(
        pd.DataFrame(
            [{"setting": key, **value} for key, value in summary["test_metrics"].items()]
        ).to_string(index=False)
    )
    print(f"Wrote {result_dir / f'{prefix}_trl_summary.json'}")
