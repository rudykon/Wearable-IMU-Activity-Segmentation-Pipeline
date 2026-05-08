"""Temporal neural-network utilities for public-dataset TRL checks.

Purpose:
    Provides raw-window normalization, temporal CNN/GRU model builders, training
    loops, prediction helpers, and result summarization for public-corpus
    portability experiments.
Inputs:
    Consumes fixed-length raw sensor windows and labels from dataset-specific
    loaders.
Outputs:
    Produces per-window probabilities, decoded subject-level segments, and
    summary metrics.
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
    decode_subjects,
    evaluate_subjects,
    evaluate_decoded_subjects,
    labels_to_segments,
    majority_label,
    write_dataset_outputs,
)


@dataclass
class TemporalSubjectWindows:
    subject: str
    windows: np.ndarray
    labels: np.ndarray
    centers_s: np.ndarray
    duration_s: float
    gt_segments: list[dict]


@dataclass
class WindowNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - self.mean) / self.std).astype(np.float32, copy=False)


class TinyTemporalCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, width: int = 64, dropout: float = 0.20):
        super().__init__()
        wider = width * 2
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, wider, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(wider),
            nn.GELU(),
            nn.Conv1d(wider, wider, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(wider),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(wider, wider, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(wider),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(wider, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class TinyTemporalGRU(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden_size: int = 96, dropout: float = 0.20):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).transpose(1, 2)
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))


def build_temporal_model(model_name: str, in_channels: int, n_classes: int) -> nn.Module:
    if model_name == "tcn":
        return TinyTemporalCNN(in_channels=in_channels, n_classes=n_classes)
    if model_name == "gru":
        return TinyTemporalGRU(in_channels=in_channels, n_classes=n_classes)
    raise ValueError(f"Unknown temporal model: {model_name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    requested = os.environ.get("IMU_PUBLIC_TCN_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_temporal_subject_windows(
    sequence: SubjectSequence,
    *,
    window_s: float,
    step_s: float,
    fs_hz: float,
) -> TemporalSubjectWindows:
    window_n = int(round(window_s * fs_hz))
    step_n = int(round(step_s * fs_hz))
    if len(sequence.signal) < window_n:
        raise ValueError(f"{sequence.subject} is shorter than one analysis window")

    windows = []
    window_labels = []
    centers = []
    for start in range(0, len(sequence.signal) - window_n + 1, step_n):
        end = start + window_n
        windows.append(sequence.signal[start:end].T.astype(np.float32, copy=False))
        window_labels.append(majority_label(sequence.labels[start:end]))
        centers.append(float((sequence.timestamps_s[start] + sequence.timestamps_s[end - 1]) / 2.0))

    gt_segments, duration_s = labels_to_segments(sequence.labels, sequence.timestamps_s, fs_hz)
    return TemporalSubjectWindows(
        subject=sequence.subject,
        windows=np.stack(windows).astype(np.float32, copy=False),
        labels=np.asarray(window_labels, dtype=np.int32),
        centers_s=np.asarray(centers, dtype=np.float64),
        duration_s=duration_s,
        gt_segments=gt_segments,
    )


def stack_windows(subjects: list[TemporalSubjectWindows]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([subject.windows for subject in subjects], axis=0)
    y = np.concatenate([subject.labels for subject in subjects], axis=0)
    return x.astype(np.float32, copy=False), y.astype(np.int32, copy=False)


def fit_normalizer(x_train: np.ndarray) -> WindowNormalizer:
    mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return WindowNormalizer(mean=mean, std=std)


def make_loader(
    x: np.ndarray,
    y_idx: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_idx.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_proba_array(
    model: nn.Module,
    x: np.ndarray,
    *,
    normalizer: WindowNormalizer,
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
    subjects: list[TemporalSubjectWindows],
    *,
    normalizer: WindowNormalizer,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    return {
        subject.subject: predict_proba_array(
            model,
            subject.windows,
            normalizer=normalizer,
            device=device,
            batch_size=batch_size,
        )
        for subject in subjects
    }


def train_temporal_cnn(
    train_subjects: list[TemporalSubjectWindows],
    dev_subjects: list[TemporalSubjectWindows],
    *,
    model_name: str = "tcn",
    seed: int = 2026,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> tuple[nn.Module, np.ndarray, WindowNormalizer, list[dict], torch.device]:
    set_seed(seed)
    device = resolve_device()
    max_epochs = int(os.environ.get("IMU_PUBLIC_TCN_EPOCHS", max_epochs or 35))
    patience = int(os.environ.get("IMU_PUBLIC_TCN_PATIENCE", patience or 8))
    batch_size = int(os.environ.get("IMU_PUBLIC_TCN_BATCH_SIZE", batch_size or 256))
    learning_rate = float(os.environ.get("IMU_PUBLIC_TCN_LR", learning_rate or 1e-3))
    weight_decay = float(os.environ.get("IMU_PUBLIC_TCN_WEIGHT_DECAY", weight_decay or 1e-4))

    x_train, y_train = stack_windows(train_subjects)
    x_dev, y_dev = stack_windows(dev_subjects)
    classes = np.asarray(sorted(np.unique(y_train).tolist()), dtype=np.int32)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    y_train_idx = np.asarray([class_to_idx[int(label)] for label in y_train], dtype=np.int64)

    normalizer = fit_normalizer(x_train)
    x_train_norm = normalizer.transform(x_train)
    train_loader = make_loader(x_train_norm, y_train_idx, batch_size=batch_size, shuffle=True)

    model = build_temporal_model(model_name, in_channels=x_train.shape[1], n_classes=len(classes)).to(device)
    counts = np.bincount(y_train_idx, minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

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
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        dev_proba = predict_proba_array(model, x_dev, normalizer=normalizer, device=device, batch_size=batch_size)
        dev_pred = classes[np.argmax(dev_proba, axis=1)]
        dev_acc = accuracy_score(y_dev, dev_pred)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_window_accuracy": float(dev_acc),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )

        score = float(dev_acc)
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, classes, normalizer, history, device


def select_sequence_trl_params(
    dev_subjects: list[TemporalSubjectWindows],
    proba_by_subject: dict[str, np.ndarray],
    classes: np.ndarray,
    *,
    min_segment_s: float,
    ignore_labels: set[int] | None = None,
) -> tuple[dict, list[dict]]:
    """Select only sequence-stabilization TRL constants for stronger CNN bases.

    The linear public baseline benefits from aggressive record filtering, but
    with a better temporal CNN that can over-prune valid short activities. This
    selector keeps the TRL layer focused on probability smoothing and transition
    regularization, then leaves duration filtering to the fixed evaluator.
    """

    evaluated = []
    for smooth_width in (1, 3, 5, 7, 9):
        for median_width in (1, 3, 5):
            for self_prob in (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97):
                for min_run in (1, 2, 3, 5):
                    params = {
                        "smooth_width": smooth_width,
                        "median_width": median_width,
                        "self_prob": self_prob,
                        "min_run": min_run,
                        "merge_gap_s": 0.0,
                        "min_segment_s": 0.0,
                        "conf_threshold": 0.0,
                    }
                    decoded = decode_subjects(dev_subjects, proba_by_subject, "trl", params)
                    metrics = evaluate_decoded_subjects(
                        dev_subjects,
                        decoded,
                        classes,
                        min_segment_s=min_segment_s,
                        params=params,
                        ignore_labels=ignore_labels,
                    )
                    evaluated.append({**params, **metrics, "stage": "sequence_only"})

    evaluated.sort(key=lambda item: (item["segment_f1"], item["miou"], -item["records_per_hour"]), reverse=True)
    param_keys = ("smooth_width", "median_width", "self_prob", "min_run", "merge_gap_s", "min_segment_s", "conf_threshold")
    return {key: evaluated[0][key] for key in param_keys}, evaluated


def run_tcn_portability_experiment(
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
    model_name: str = "tcn",
    seed: int = 2026,
    ignore_labels: set[int] | None = None,
) -> dict:
    all_subject_ids = split["train_subjects"] + split["dev_subjects"] + split["test_subjects"]
    missing = [subject for subject in all_subject_ids if subject not in sequences]
    if missing:
        raise FileNotFoundError(f"Missing subjects for {dataset_name}: {missing}")

    windows = {
        subject: build_temporal_subject_windows(sequences[subject], window_s=window_s, step_s=step_s, fs_hz=fs_hz)
        for subject in all_subject_ids
    }
    train = [windows[s] for s in split["train_subjects"]]
    dev = [windows[s] for s in split["dev_subjects"]]
    test = [windows[s] for s in split["test_subjects"]]

    model, classes, normalizer, history, device = train_temporal_cnn(train, dev, model_name=model_name, seed=seed)
    batch_size = int(os.environ.get("IMU_PUBLIC_TCN_BATCH_SIZE", 256))
    dev_proba = predict_subjects(model, dev, normalizer=normalizer, device=device, batch_size=batch_size)
    test_proba = predict_subjects(model, test, normalizer=normalizer, device=device, batch_size=batch_size)

    selected_params, dev_grid = select_sequence_trl_params(
        dev,
        dev_proba,
        classes,
        min_segment_s=window_s,
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
        "classifier": f"{model_name.upper()} raw-window neural classifier (class-balanced cross entropy)",
        "trl_selection": "sequence_only_smoothing_viterbi_short_run_repair",
        "training": {
            "device": str(device),
            "seed": seed,
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
