"""Experiment summary figure generator.

Purpose:
    Recreates aggregate figures from saved experiment tables, including main
    comparisons, ablations, split summaries, boundary analysis, and robustness
    views.
Inputs:
    Reads CSV/JSON result files produced by scripts in `experiments/`.
Outputs:
    Saves static figures to `experiments/figures/` for documentation and review.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"


POLICY_LABEL = {
    "S5_BaselinePP": "S5 Baseline",
    "S0_Full": "S0 Full",
    "S2_NoTopKConf": "S2 NoTopK/Conf",
    "S6_NoTopK_KeepConf": "S6 NoTopK + Conf",
    "S7_KeepTopK_NoConf": "S7 KeepTopK / NoConf",
    "SelectedPolicy": "Selected Policy",
}

CLASS_LABEL = {
    "乒乓球": "Table tennis",
    "羽毛球": "Badminton",
    "跑步": "Running",
    "跳绳": "Rope skipping",
    "飞鸟": "Dumbbell fly",
}

METHOD_LABEL = {
    "M0_RawArgmax_Single3s": "M0 RawArgmax (3s)",
    "M1_BaselinePP_Single3s": "M1 BaselinePP (3s)",
    "M2_FullPP_Single3s": "M2 FullPP (3s)",
    "M3_FullPP_Ens3s": "M3 FullPP (3x3s)",
    "M4_BaselinePP_Ens9": "M4 BaselinePP (3s+5s+8s)",
    "M5_Proposed_Ens9_FullPP": "M5 FullPP (3s+5s+8s)",
}

PALETTE = {
    "accent": "#a9dcd5",
    "accent2": "#c3dfab",
    "warm": "#f3c8a7",
    "danger": "#efb2b8",
    "neutral": "#c8c4c4",
    "single": "#9cc6e7",
    "ensemble": "#cfc4ea",
    "baseline": "#a9b7cf",
    "ink": "#222222",
    "guide": "#d9dde5",
    "highlight": "#f8f3de",
    "green_line": "#9ec97a",
    "pink_line": "#ee95a1",
    "blue_line": "#86a8d8",
    "green_edge": "#8abd70",
    "pink_edge": "#e28f9c",
    "blue_edge": "#7c9fd2",
    "peach_edge": "#e1b188",
    "gray_edge": "#afa9a9",
}

RNG = np.random.default_rng(42)

ABLATION_LABEL = {
    "S6_NoTopK_KeepConf": "No Top-K +\nkeep conf.",
    "S0_Full": "Full\ndefault",
    "S1_NoMedian": "No median\nfilter",
    "S3_NoBoundaryRefine": "No boundary\nrefine",
    "S2_NoTopKConf": "No Top-K +\nno conf.",
    "S5_BaselinePP": "Baseline\npost-proc.",
    "S4_NoViterbi": "No\nViterbi",
}

HELDOUT_VARIANT_SPECS = [
    {
        "key": "best_5s8s",
        "label": "Best 5s + 8s + TRL",
        "short_label": "5s+8s\n+ TRL",
        "path": RESULT_DIR / "heldout_eval_best2scale_5s8s_S0_Full_20260424.json",
        "color": "#9aa8bf",
    },
    {
        "key": "average",
        "label": "Best-per-scale 3-model + average fusion + TRL",
        "short_label": "Average\n+ TRL",
        "path": RESULT_DIR / "heldout_eval_best3scale_average_S0_20260424.json",
        "color": "#1648c8",
    },
    {
        "key": "weighted_long",
        "label": "Best-per-scale 3-model + weighted-long fusion + TRL",
        "short_label": "Weighted-long\n+ TRL",
        "path": RESULT_DIR / "heldout_eval_best3scale_weighted_long_S0_20260424.json",
        "color": "#ff6a00",
    },
    {
        "key": "lbsa_full",
        "label": "Best-per-scale 3-model + LBSA + TRL",
        "short_label": "LBSA\n+ TRL",
        "path": RESULT_DIR / "heldout_eval_best3scale_local_boundary_S0_20260424.json",
        "color": "#148f24",
    },
    {
        "key": "lbsa_relaxed",
        "label": "Best-per-scale 3-model + LBSA + relaxed Top-K policy",
        "short_label": "LBSA\n+ relaxed",
        "policy_key": "S6_NoTopK_KeepConf",
        "color": "#e31a1c",
    },
]


def load_main_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / "main_comparison.csv", encoding="utf-8-sig")


def load_single3s_boundary() -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / "boundary_metrics_single3s.csv", encoding="utf-8-sig")


def load_single3s_ablation() -> pd.DataFrame:
    df = pd.read_csv(RESULT_DIR / "single3s_ablation.csv", encoding="utf-8-sig")
    if "S6_NoTopK_KeepConf" not in set(df["ablation"]):
        boundary_df = load_single3s_boundary().set_index("policy")
        s6 = boundary_df.loc["S6_NoTopK_KeepConf"]
        df = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "ablation": "S6_NoTopK_KeepConf",
                            "mean_user_f1": s6["mean_user_f1"],
                            "micro_f1": s6["micro_f1"],
                            "TP": int(s6["TP"]),
                            "FP": int(s6["FP"]),
                            "FN": int(s6["FN"]),
                        }
                    ]
                ),
                df,
            ],
            ignore_index=True,
        )
    return df


def load_split_summary() -> Dict[str, float]:
    df = pd.read_csv(
        RESULT_DIR / "internal_eval_split_protocol_policy_selection_distribution.csv",
        encoding="utf-8-sig",
    )
    selected = df[df["setting"] == "SelectedPolicy"]["outer_mean_user_f1"].to_numpy(dtype=float)
    splits = pd.read_csv(
        RESULT_DIR / "internal_eval_split_protocol_policy_selection_splits.csv",
        encoding="utf-8-sig",
    )
    return {
        "selected_mean_user_f1": float(selected.mean()),
        "selected_micro_f1": float(splits["test_micro_f1"].mean()),
    }


def load_incremental_outer_summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / "incremental_outer_single3s_summary.csv", encoding="utf-8-sig")


def load_heldout_main() -> Dict[str, float]:
    result_path = RESULT_DIR / "heldout_eval_best3scale_local_boundary_S0_20260424.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        with open(RESULT_DIR / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        payload = summary.get("heldout_main", {})

    required = ["mean_user_f1", "micro_f1", "TP", "FP", "FN"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"held-out summary missing keys: {missing}")
    return payload


def load_heldout_variant_summary() -> pd.DataFrame:
    rows: List[Dict] = []
    policy_path = RESULT_DIR / "heldout_eval_best3scale_policy_check_20260424.json"
    policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_metrics = policy_data.get("policies", {})

    for spec in HELDOUT_VARIANT_SPECS:
        if "path" in spec:
            payload = json.loads(spec["path"].read_text(encoding="utf-8"))
        else:
            payload = policy_metrics[spec["policy_key"]]
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "short_label": spec["short_label"],
                "color": spec["color"],
                "mean_user_f1": float(payload["mean_user_f1"]),
                "ci95_low": float(payload["ci95_low"]),
                "ci95_high": float(payload["ci95_high"]),
                "micro_f1": float(payload["micro_f1"]),
                "TP": int(payload["TP"]),
                "FP": int(payload["FP"]),
                "FN": int(payload["FN"]),
            }
        )
    return pd.DataFrame(rows)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["ink"],
            "axes.grid": False,
            "grid.color": PALETTE["guide"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / name
    fig.savefig(out_png)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color(PALETTE["ink"])
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", colors=PALETTE["ink"], width=0.55, length=3.0)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.45, color=PALETTE["guide"])


def add_vertical_guides(ax: plt.Axes, positions) -> None:
    for pos in positions:
        ax.axvline(pos, color=PALETTE["guide"], linestyle=(0, (2, 3)), linewidth=1.0, zorder=0)


def add_highlight(ax: plt.Axes, start: float, end: float) -> None:
    ax.axvspan(start, end, color=PALETTE["highlight"], alpha=0.45, zorder=0)


def add_callout(ax: plt.Axes, text: str, xy, xytext, *, ha: str = "center", va: str = "center") -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha=ha,
        va=va,
        fontsize=8.3,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.18",
            "fc": "white",
            "ec": PALETTE["ink"],
            "lw": 1.0,
        },
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.0,
            "color": PALETTE["ink"],
            "shrinkA": 2,
            "shrinkB": 5,
        },
        annotation_clip=False,
    )


def plot_window_to_record_gap() -> None:
    from matplotlib.patches import FancyBboxPatch

    time_axis = np.linspace(0, 360, 721)

    running = 0.05 + 0.84 * np.exp(-((time_axis - 96) / 33) ** 2) - 0.30 * np.exp(-((time_axis - 140) / 7.5) ** 2)
    rope = 0.06 + 0.86 * np.exp(-((time_axis - 203) / 36) ** 2)
    table = 0.05 + 0.84 * np.exp(-((time_axis - 288) / 34) ** 2) - 0.22 * np.exp(-((time_axis - 320) / 8.0) ** 2)
    null = (
        0.12
        + 0.56 * np.exp(-((time_axis - 18) / 24) ** 2)
        + 0.20 * np.exp(-((time_axis - 150) / 14) ** 2)
        + 0.46 * np.exp(-((time_axis - 348) / 20) ** 2)
    )

    curves = {
        "Running": np.clip(running, 0.02, 0.98),
        "Rope skipping": np.clip(rope, 0.02, 0.98),
        "Table tennis": np.clip(table, 0.02, 0.98),
        "Null / transition": np.clip(null, 0.02, 0.98),
    }
    colors = {
        "Running": "#4c78a8",
        "Rope skipping": "#1b9e77",
        "Table tennis": "#d95f02",
        "Null / transition": "#98a2b3",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0], "hspace": 0.18},
    )

    boundary_zones = [(132, 28), (312, 24)]
    for ax in axes:
        for start, width in boundary_zones:
            ax.axvspan(start, start + width, color="#fff4d6", alpha=0.85, zorder=0)
        ax.set_xlim(0, 360)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.grid(axis="x", linestyle="--", alpha=0.25)

    for label, values in curves.items():
        axes[0].plot(time_axis, values, linewidth=2.2, color=colors[label], label=label)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Posterior")
    axes[0].set_title("(a) Window posterior trajectory")
    axes[0].legend(loc="upper right", ncol=2, fontsize=7.8)
    axes[0].annotate(
        "local posterior dip\ninduces a false split",
        xy=(141, 0.56),
        xytext=(86, 0.87),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#475467"},
        fontsize=8.3,
        color="#344054",
        ha="center",
    )
    axes[0].annotate(
        "transition uncertainty\nshifts record boundaries",
        xy=(319, 0.51),
        xytext=(282, 0.87),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#475467"},
        fontsize=8.3,
        color="#344054",
        ha="center",
    )

    def draw_segments(ax: plt.Axes, segments: List[tuple], title: str, subtitle: str) -> None:
        for label, start, end in segments:
            patch = FancyBboxPatch(
                (start, 0.20),
                end - start,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                linewidth=1.1,
                edgecolor=colors[label],
                facecolor=mpl.colors.to_rgba(colors[label], 0.24),
            )
            ax.add_patch(patch)
            ax.text((start + end) / 2, 0.47, label, ha="center", va="center", fontsize=8.1, color="#111827")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(title)
        ax.text(
            0.01,
            0.96,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            color="#475467",
        )

    draw_segments(
        axes[1],
        [
            ("Running", 58, 132),
            ("Running", 144, 160),
            ("Rope skipping", 162, 232),
            ("Table tennis", 248, 319),
            ("Table tennis", 324, 336),
        ],
        "(b) Naive record extraction",
        "fragmented list, short false split, early/late boundaries",
    )
    draw_segments(
        axes[2],
        [
            ("Running", 58, 158),
            ("Rope skipping", 170, 232),
            ("Table tennis", 248, 332),
        ],
        "(c) TRL output record list",
        "merge short gaps, suppress tiny fragments, refine boundaries",
    )

    axes[1].annotate(
        "false split",
        xy=(148, 0.77),
        xytext=(114, 0.95),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#7c2d12"},
        fontsize=8.0,
        color="#7c2d12",
    )
    axes[2].annotate(
        "stable final record",
        xy=(110, 0.77),
        xytext=(72, 0.95),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#14532d"},
        fontsize=8.0,
        color="#14532d",
    )

    axes[2].set_xlabel("Time in long recording (s)")
    axes[2].set_xticks(np.arange(0, 361, 60))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.09)
    save(fig, "fig01_window_to_record_gap.png")


def plot_heldout_lbsa_summary() -> None:
    df = load_heldout_variant_summary()
    x = np.arange(len(df))
    mean_user_f1 = df["mean_user_f1"].to_numpy(dtype=float)
    ci_low = mean_user_f1 - df["ci95_low"].to_numpy(dtype=float)
    ci_high = df["ci95_high"].to_numpy(dtype=float) - mean_user_f1
    micro_f1 = df["micro_f1"].to_numpy(dtype=float)
    best_idx = int(df.index[df["key"] == "lbsa_full"][0])
    relaxed_idx = int(df.index[df["key"] == "lbsa_relaxed"][0])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.85, 5.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1.42, 1.0], "hspace": 0.08},
    )
    style_axes(ax1, "y")
    style_axes(ax2, "y")
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax1.plot(
        x,
        mean_user_f1,
        color=PALETTE["blue_edge"],
        linewidth=1.05,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        zorder=4,
        label="Mean user F1",
    )
    ax1.errorbar(
        x,
        mean_user_f1,
        yerr=np.vstack([ci_low, ci_high]),
        fmt="o",
        markersize=0,
        ecolor=(0.68, 0.72, 0.78, 0.45),
        elinewidth=0.48,
        capsize=1.8,
        capthick=0.48,
        zorder=3,
    )
    ax1.plot(
        x,
        micro_f1,
        color="#d8b87b",
        linewidth=1.05,
        marker="o",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.75,
        zorder=5,
        label="Micro-F1",
    )
    ax1.scatter(
        [x[best_idx]],
        [mean_user_f1[best_idx]],
        s=34,
        color=PALETTE["accent2"],
        edgecolor=PALETTE["green_edge"],
        linewidth=0.8,
        zorder=6,
    )
    ax1.scatter(
        [x[relaxed_idx]],
        [mean_user_f1[relaxed_idx]],
        s=34,
        color=PALETTE["danger"],
        edgecolor=PALETTE["pink_edge"],
        linewidth=0.8,
        zorder=6,
    )

    for xpos, value in zip(x, mean_user_f1):
        ax1.text(
            xpos,
            value + 0.0010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#4b5563",
        )
    for xpos, value in zip(x, micro_f1):
        ax1.text(
            xpos,
            value + 0.0012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#8a6b39",
        )

    ax1.set_ylabel("F1")
    ax1.set_ylim(0.870, 0.907)
    ax1.set_title("(a) Held-out evaluation F1 with 95% CI on mean user F1", loc="left", pad=4)
    ax1.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.9, fontsize=7.7)

    width = 0.28
    bars_fp = ax2.bar(
        x - width / 2,
        df["FP"],
        width=width,
        color=PALETTE["danger"],
        edgecolor=PALETTE["pink_edge"],
        linewidth=0.7,
        label="FP",
    )
    bars_fn = ax2.bar(
        x + width / 2,
        df["FN"],
        width=width,
        color=PALETTE["neutral"],
        edgecolor=PALETTE["gray_edge"],
        linewidth=0.7,
        label="FN",
    )
    tp_labels = [f"{lbl}\nTP {tp}" for lbl, tp in zip(df["short_label"], df["TP"])]
    for bars in [bars_fp, bars_fn]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.16,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color="#4b5563",
            )
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, 19.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tp_labels)
    ax2.set_title("(b) Error counts at the same operating points", loc="left", pad=3)
    ax2.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=1.0, fontsize=7.9)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.11)
    save(fig, "fig04_heldout_variant_comparison.png")


def plot_trl_ablation_boundary_combo() -> None:
    df = load_incremental_outer_summary().sort_values("order_id").reset_index(drop=True)
    split_df = pd.read_csv(
        RESULT_DIR / "internal_eval_split_protocol_policy_selection_distribution.csv",
        encoding="utf-8-sig",
    )
    label_map = {
        "I0_Raw": "Raw",
        "I1_BaseTemporal": "+ Base",
        "I2_WiderSmooth180": "+ Smooth180",
        "I3_StrongerPrior": "+ Prior",
        "I4_Median": "+ Median",
        "I5_ConfClip": "+ Clip",
        "I6_Full": "+ Top-K",
    }
    color_map = {
        "I0_Raw": PALETTE["neutral"],
        "I1_BaseTemporal": PALETTE["baseline"],
        "I2_WiderSmooth180": PALETTE["warm"],
        "I3_StrongerPrior": PALETTE["accent"],
        "I4_Median": "#c9d3df",
        "I5_ConfClip": PALETTE["accent2"],
        "I6_Full": "#b7c8df",
    }
    edge_map = {
        "I5_ConfClip": PALETTE["green_edge"],
        "I6_Full": PALETTE["blue_edge"],
    }

    full_x = np.arange(len(df))
    selected = df[df["policy"].isin(["I1_BaseTemporal", "I2_WiderSmooth180", "I3_StrongerPrior", "I5_ConfClip", "I6_Full"])].copy()
    selected_x = np.arange(len(selected))
    selected_labels = [label_map[k] for k in selected["policy"]]
    selected_colors = [color_map[k] for k in selected["policy"]]
    selected_edges = [edge_map.get(k, PALETTE["gray_edge"]) for k in selected["policy"]]

    split_order = [
        "SelectedPolicy",
        "S0_Full",
        "S2_NoTopKConf",
        "S6_NoTopK_KeepConf",
        "S7_KeepTopK_NoConf",
        "S5_BaselinePP",
    ]
    split_labels = {
        "SelectedPolicy": "Selected",
        "S0_Full": "S0",
        "S2_NoTopKConf": "S2",
        "S6_NoTopK_KeepConf": "NoTopK",
        "S7_KeepTopK_NoConf": "NoConf",
        "S5_BaselinePP": "S5",
    }
    split_colors = {
        "SelectedPolicy": PALETTE["accent"],
        "S0_Full": PALETTE["accent2"],
        "S2_NoTopKConf": PALETTE["warm"],
        "S6_NoTopK_KeepConf": PALETTE["single"],
        "S7_KeepTopK_NoConf": PALETTE["ensemble"],
        "S5_BaselinePP": PALETTE["baseline"],
    }
    split_order = [key for key in split_order if key in set(split_df["setting"].unique())]
    split_data = [
        split_df.loc[split_df["setting"] == key, "outer_mean_user_f1"].to_numpy(dtype=np.float64)
        for key in split_order
    ]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.55),
        gridspec_kw={"wspace": 0.22, "width_ratios": [1.08, 1.0]},
    )
    style_axes(ax1, "y")
    style_axes(ax2, "y")

    bp = ax1.boxplot(
        split_data,
        positions=np.arange(len(split_order)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch, key in zip(bp["boxes"], split_order):
        color = split_colors[key]
        patch.set_facecolor(mpl.colors.to_rgba(color, 0.72))
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for median in bp["medians"]:
        median.set_color("#f08c2b")
        median.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color(PALETTE["gray_edge"])
        whisker.set_linewidth(0.85)
    for cap in bp["caps"]:
        cap.set_color(PALETTE["gray_edge"])
        cap.set_linewidth(0.85)

    ax1.set_xticks(np.arange(len(split_order)))
    ax1.set_xticklabels([split_labels[key] for key in split_order], fontsize=6.9)
    ax1.set_ylim(0.68, 1.01)
    ax1.set_ylabel("Outer mean user F1")
    ax1.set_title("(a) Outer-split F1 distributions", loc="left", pad=3)
    ax1.text(
        0.985,
        0.98,
        "50 random 10/10 splits",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#4b5563",
    )

    ax2.plot(
        selected_x,
        selected["matched_iou_mean"],
        color=PALETTE["green_line"],
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.3,
        zorder=4,
        label="Matched IoU",
    )
    ax2.set_xticks(selected_x)
    ax2.set_xticklabels(selected_labels)
    ax2.set_ylabel("Matched IoU")
    ax2.set_ylim(0.82, 0.86)
    ax2.set_title("(b) Boundary quality and FP cost", loc="left", pad=3)

    ax2b = ax2.twinx()
    ax2b.bar(
        selected_x,
        selected["fp_per_recording_hour_mean"],
        color=[mpl.colors.to_rgba(c, 0.65) for c in selected_colors],
        edgecolor=selected_edges,
        linewidth=0.7,
        width=0.58,
        zorder=2,
    )
    ax2b.set_ylim(0.0, 0.95)
    ax2b.set_ylabel("FP / hour")
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_linewidth(0.55)
    ax2b.spines["right"].set_color(PALETTE["ink"])
    ax2b.tick_params(axis="y", colors=PALETTE["ink"], width=0.55, length=3.0)
    ax2.set_zorder(ax2b.get_zorder() + 1)
    ax2.patch.set_alpha(0.0)

    for xpos, value in zip(selected_x, selected["matched_iou_mean"]):
        ax2.text(
            xpos,
            value + 0.0014,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4b5563",
            zorder=6,
        )
    for xpos, value in zip(selected_x, selected["fp_per_recording_hour_mean"]):
        ax2b.text(
            xpos,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4b5563",
            zorder=4,
        )

    legend_handles = [
        plt.Line2D([0], [0], color=PALETTE["green_line"], marker="o", markerfacecolor="white", markeredgewidth=0.9, linewidth=1.3, label="Matched IoU"),
        mpl.patches.Patch(facecolor=mpl.colors.to_rgba(PALETTE["baseline"], 0.65), edgecolor=PALETTE["gray_edge"], label="FP / hour"),
    ]
    ax2.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=7.2, handlelength=1.25)

    fig.subplots_adjust(left=0.07, right=0.965, top=0.92, bottom=0.19)
    save(fig, "fig05_outer_split_boundary_summary.png")


def plot_graphical_abstract() -> None:
    boundary_df = load_single3s_boundary().set_index("policy")
    main_df = load_main_comparison().set_index("method")
    split_summary = load_split_summary()
    heldout = load_heldout_main()
    best_single = boundary_df.loc["S6_NoTopK_KeepConf"]
    best_main = main_df.loc["M5_Proposed_Ens9_FullPP"]
    baseline_single = boundary_df.loc["S5_BaselinePP"]

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, h, title, body, fc, ec="#475467"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=11, fontweight="bold", color="#0f172a")
        ax.text(x + 0.02, y + h - 0.14, body, fontsize=9.2, color="#1f2937", va="top")

    box(
        0.03,
        0.55,
        0.24,
        0.34,
        "IMU Stream Input",
        "ACC/GYRO 6-axis\n100 Hz long recordings\nUser-disjoint evaluation",
        "#e8f3ff",
    )
    box(
        0.33,
        0.55,
        0.24,
        0.34,
        "Compact Backbone",
        "1D-CNN + BiLSTM\nSingle 3-second model\nCE + Focal + Triplet loss",
        "#e9f9f2",
    )
    box(
        0.63,
        0.55,
        0.24,
        0.34,
        "Temporal Decoding",
        "Smoothing + Viterbi\nMerge + overlap resolve\nDuration constraints",
        "#fff2e8",
    )

    arrow = dict(arrowstyle="->", lw=1.6, color="#334155", shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0.33, 0.72), xytext=(0.27, 0.72), arrowprops=arrow)
    ax.annotate("", xy=(0.63, 0.72), xytext=(0.57, 0.72), arrowprops=arrow)

    metrics_box = FancyBboxPatch(
        (0.03, 0.08),
        0.84,
        0.34,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#475467",
        facecolor="#f8fafc",
    )
    ax.add_patch(metrics_box)
    ax.text(0.05, 0.37, "Key Findings", fontsize=11, fontweight="bold", color="#111827")
    ax.text(
        0.05,
        0.30,
        f"• Held-out evaluation (37 files, best-per-scale + LBSA): Mean User F1 = {heldout['mean_user_f1']:.4f}",
        fontsize=9.6,
        color="#1f2937",
    )
    ax.text(
        0.05,
        0.24,
        f"• Held-out evaluation micro-F1 = {heldout['micro_f1']:.4f}; TP/FP/FN = {heldout['TP']}/{heldout['FP']}/{heldout['FN']}",
        fontsize=9.6,
        color="#1f2937",
    )
    ax.text(
        0.05,
        0.18,
        f"• Internal tuning evidence only: 20-user best Ens9 = {best_main['mean_user_f1']:.4f}; best single-model S6 = {best_single['mean_user_f1']:.4f}",
        fontsize=9.6,
        color="#1f2937",
    )
    ax.text(
        0.05,
        0.12,
        f"• Split-separated tuning estimate on internal users: {split_summary['selected_mean_user_f1']:.4f}",
        fontsize=9.6,
        color="#1f2937",
    )

    badge = FancyBboxPatch(
        (0.76, 0.11),
        0.16,
        0.26,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor="#9a3412",
        facecolor="#ffedd5",
    )
    ax.add_patch(badge)
    ax.text(0.84, 0.31, "FP/hour", ha="center", fontsize=10, fontweight="bold", color="#7c2d12")
    ax.text(
        0.84,
        0.24,
        f"{baseline_single['fp_per_recording_hour']:.3f} → {best_single['fp_per_recording_hour']:.3f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#b91c1c",
    )
    ax.text(0.84, 0.16, "Temporal policy\nsuppresses\nfalse alarms", ha="center", fontsize=8.8, color="#7c2d12")

    ax.set_title("Graphical Abstract: Temporal-Constraint Pipeline and Evidence Summary", fontsize=13, fontweight="bold")
    save(fig, "graphical_abstract_pipeline.png")


def plot_heldout_vs_dev_summary() -> None:
    boundary_df = load_single3s_boundary().set_index("policy")
    split_summary = load_split_summary()
    heldout = load_heldout_main()
    best_single = boundary_df.loc["S6_NoTopK_KeepConf"]

    labels = [
        "Held-out\n(37 files)",
        "Dev holdout\nbest S6",
        "Dev split-separated\nselected",
    ]
    mean_user_f1 = np.array([heldout["mean_user_f1"], best_single["mean_user_f1"], split_summary["selected_mean_user_f1"]])
    micro_f1 = np.array([heldout["micro_f1"], best_single["micro_f1"], split_summary["selected_micro_f1"]])
    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width / 2, mean_user_f1, width=width, color=PALETTE["accent"], label="Mean User F1")
    ax.bar(x + width / 2, micro_f1, width=width, color=PALETTE["warm"], label="Micro-F1")

    for xpos, val in zip(x - width / 2, mean_user_f1):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#111827")
    for xpos, val in zip(x + width / 2, micro_f1):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#111827")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Held-out Evaluation vs Development Diagnostics")
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.02,
        "Held-out evaluation uses data/signals/external_test\nintersected with data/annotations/external_test_annotations.csv.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#d0d5dd"),
    )
    save(fig, "heldout_vs_dev_summary.png")


def plot_main_comparison() -> None:
    df = pd.read_csv(RESULT_DIR / "main_comparison.csv", encoding="utf-8-sig")
    df["label"] = df["method"].map(METHOD_LABEL).fillna(df["method"])
    df = df.sort_values("mean_user_f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        is_single = "Single3s" in row["method"]
        color = PALETTE["single"] if is_single else PALETTE["ensemble"]
        ax.hlines(i, row["ci95_low"], row["ci95_high"], color="#c9d3e7", linewidth=8, zorder=1)
        ax.scatter(row["mean_user_f1"], i, s=95, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(
            row["mean_user_f1"] + 0.012,
            i,
            f"micro={row['micro_f1']:.3f}",
            va="center",
            fontsize=8,
            color="#344054",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Mean User F1 with 95% CI")
    ax.set_xlim(0.0, 1.03)
    ax.set_title("Main Comparison: Mean User F1 + Confidence Intervals")
    ax.grid(axis="x")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    save(fig, "main_comparison.png")


def plot_outer_split_distribution() -> None:
    df = pd.read_csv(
        RESULT_DIR / "internal_eval_split_protocol_policy_selection_distribution.csv",
        encoding="utf-8-sig",
    )
    keep = ["SelectedPolicy", "S6_NoTopK_KeepConf", "S0_Full", "S5_BaselinePP"]
    df = df[df["setting"].isin(keep)].copy()
    df["label"] = df["setting"].map(POLICY_LABEL)
    order = ["Selected Policy", "S6 NoTopK + Conf", "S0 Full", "S5 Baseline"]
    grouped = [df.loc[df["label"] == k, "outer_mean_user_f1"].to_numpy() for k in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    vp = ax.violinplot(grouped, positions=np.arange(1, len(order) + 1), showmeans=False, showextrema=False, widths=0.85)
    colors = [PALETTE["accent"], PALETTE["warm"], PALETTE["accent2"], PALETTE["baseline"]]
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.22)
        body.set_edgecolor(c)
        body.set_linewidth(1.0)

    for i, vals in enumerate(grouped, start=1):
        x = RNG.normal(i, 0.045, len(vals))
        ax.scatter(x, vals, s=12, color=colors[i - 1], alpha=0.55, edgecolor="none", zorder=3)
        mean_v = float(np.mean(vals))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.errorbar(i, mean_v, yerr=[[mean_v - lo], [hi - mean_v]], fmt="D", color=colors[i - 1], capsize=4, ms=6, zorder=5)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Outer Mean User F1")
    ax.set_ylim(0.84, 1.01)
    ax.set_title("Split-Separated Outer F1 Distribution (50 Repeats)")
    save(fig, "outer_split_distribution.png")


def plot_boundary_distribution() -> None:
    df = pd.read_csv(RESULT_DIR / "boundary_matches_single3s.csv", encoding="utf-8-sig")
    keep = ["S5_BaselinePP", "S0_Full", "S6_NoTopK_KeepConf"]
    df = df[df["policy"].isin(keep)].copy()
    readable_label = {
        "S5_BaselinePP": "Baseline\nrecord list",
        "S0_Full": "Full\ntemporal decoding",
        "S6_NoTopK_KeepConf": "Permissive\ntemporal decoding",
    }
    df["label"] = df["policy"].map(readable_label)
    order = ["Baseline\nrecord list", "Full\ntemporal decoding", "Permissive\ntemporal decoding"]
    colors = [PALETTE["baseline"], PALETTE["accent2"], PALETTE["warm"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    metrics = [("matched_iou", "(a) Matched IoU"), ("duration_delta_sec", "(b) Signed duration bias (s)")]
    for ax, (col, title) in zip(axes, metrics):
        groups = [df.loc[df["label"] == k, col].to_numpy() for k in order]
        vp = ax.violinplot(groups, positions=np.arange(1, len(order) + 1), widths=0.82, showmeans=False, showextrema=False)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.2)
            body.set_edgecolor(c)
        bp = ax.boxplot(groups, positions=np.arange(1, len(order) + 1), widths=0.24, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
        for i, vals in enumerate(groups, start=1):
            x = RNG.normal(i, 0.045, len(vals))
            ax.scatter(x, vals, s=9, color=colors[i - 1], alpha=0.30, edgecolor="none")
        if col == "duration_delta_sec":
            ax.axhline(0, color="#667085", linewidth=1.0)
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order)
        ax.set_title(title)

    axes[0].set_ylim(0.45, 1.02)
    axes[1].set_ylim(-500, 500)
    save(fig, "boundary_distribution_summary.png")


def plot_segment_reliability() -> None:
    df = pd.read_csv(RESULT_DIR / "segment_calibration_bins.csv", encoding="utf-8-sig")
    keep = ["S5_BaselinePP", "S0_Full", "S6_NoTopK_KeepConf"]
    df = df[df["policy"].isin(keep)].copy()
    df["label"] = df["policy"].map(POLICY_LABEL)
    colors = {"S5 Baseline": PALETTE["baseline"], "S0 Full": PALETTE["accent2"], "S6 NoTopK + Conf": PALETTE["warm"]}

    fig = plt.figure(figsize=(9.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.4], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    ax.plot([0.4, 1.0], [0.4, 1.0], linestyle="--", color="#6b7280", linewidth=1.2, label="Perfect calibration")
    for label in ["S5 Baseline", "S0 Full", "S6 NoTopK + Conf"]:
        sub = df[df["label"] == label].sort_values("mean_conf")
        sizes = 30 + sub["count"].to_numpy() * 6
        ax.plot(sub["mean_conf"], sub["empirical_acc"], color=colors[label], linewidth=1.8, alpha=0.95)
        ax.scatter(sub["mean_conf"], sub["empirical_acc"], s=sizes, color=colors[label], alpha=0.75, edgecolor="white", linewidth=0.8, label=label)

    ax.set_ylabel("Empirical Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Segment Reliability Diagram (Bubble Size = Bin Count)")
    ax.legend(loc="lower right", fontsize=8)

    centers = ((df["bin_left"] + df["bin_right"]) / 2).round(2)
    df["center"] = centers
    x_bins = sorted(df["center"].unique())
    width = 0.018
    offsets = {"S5 Baseline": -width, "S0 Full": 0.0, "S6 NoTopK + Conf": width}
    for label in ["S5 Baseline", "S0 Full", "S6 NoTopK + Conf"]:
        sub = df[df["label"] == label][["center", "count"]].set_index("center").reindex(x_bins)
        counts = sub["count"].fillna(0)
        ax2.bar(np.array(x_bins) + offsets[label], counts, width=width, color=colors[label], alpha=0.75, label=label)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Bin Count")
    ax2.legend(loc="upper left", ncol=3, fontsize=8)
    save(fig, "segment_reliability.png")


def plot_robustness() -> None:
    df = pd.read_csv(RESULT_DIR / "single3s_robustness.csv", encoding="utf-8-sig")
    order = [
        "Clean",
        "Timestamp jitter",
        "Burst missing",
        "Axis saturation",
        "Bias drift",
        "Low-rate distortion",
    ]
    df = df.set_index("display_label").reindex(order).reset_index()

    x = np.arange(len(df))
    labels = ["Clean", "Timestamp\njitter", "Burst\nmissing", "Axis\nsaturation", "Bias\ndrift", "Low-rate\ndistortion"]
    severity_colors = [
        "#94a3b8",
        "#7aa6d1",
        "#c96a50",
        "#6fa7a1",
        "#b6546c",
        "#7f8fc9",
    ]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.2], "hspace": 0.06},
    )

    mean_f1 = df["mean_user_f1"].to_numpy(dtype=float)
    ci_low = df["ci95_low"].to_numpy(dtype=float)
    ci_high = df["ci95_high"].to_numpy(dtype=float)
    fp_hour = df["fp_per_recording_hour"].to_numpy(dtype=float)
    baseline_f1 = float(mean_f1[0])

    ax1.axhline(baseline_f1, color="#94a3b8", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax1.text(
        x[-1] + 0.04,
        baseline_f1 + 0.001,
        "clean baseline",
        color="#64748b",
        fontsize=9,
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.85},
    )

    ax1.plot(x, mean_f1, color="#1f5fa9", linewidth=2.4, zorder=3)
    for xi, yi, lo, hi, color, delta in zip(x, mean_f1, ci_low, ci_high, severity_colors, df["delta_vs_clean"]):
        ax1.vlines(xi, lo, hi, color=color, linewidth=1.8, alpha=0.9, zorder=2)
        ax1.scatter(xi, yi, s=78, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        if xi != 0:
            y_offset = 0.010 if delta > -0.05 else 0.014
            ax1.text(
                xi,
                yi + y_offset,
                f"{delta:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#344054",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12, "alpha": 0.78},
            )

    bars = ax2.bar(x, fp_hour, color=severity_colors, width=0.58, alpha=0.88, edgecolor="none", zorder=3)
    for rect, value in zip(bars, fp_hour):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#344054",
        )

    ax1.set_ylabel("Mean User F1")
    ax2.set_ylabel("FP/hour")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax1.set_ylim(0.79, 0.94)
    ax1.set_yticks([0.80, 0.84, 0.88, 0.92])
    ax2.set_ylim(0.0, 0.92)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax1.grid(axis="x", visible=False)
    ax2.grid(axis="x", visible=False)
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.tick_params(axis="x", pad=6)
    ax1.set_xlim(-0.45, len(df) - 0.55)
    ax1.set_title("Performance under perturbation", loc="left", pad=10, fontsize=11, color="#344054")
    ax2.set_title("Alarm cost", loc="left", pad=4, fontsize=11, color="#344054")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.12)
    save(fig, "single3s_robustness.png")


def plot_robustness_delta() -> None:
    df = pd.read_csv(RESULT_DIR / "single3s_robustness.csv", encoding="utf-8-sig")
    df = df[df["display_label"] != "Clean"].copy()
    df = df.sort_values("delta_vs_clean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    colors = [PALETTE["danger"] if v < 0 else PALETTE["accent2"] for v in df["delta_vs_clean"]]
    ax.barh(df["display_label"], df["delta_vs_clean"], color=colors, alpha=0.85)
    ax.axvline(0, color="#475467", linewidth=1.0)
    for i, v in enumerate(df["delta_vs_clean"]):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Delta Mean User F1 vs Clean")
    ax.set_title("Robustness Delta by Perturbation Condition")
    save(fig, "single3s_robustness_delta.png")


def load_single3s_user_class() -> Dict[str, Dict]:
    with open(RESULT_DIR / "single3s_detailed_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_df(metrics: Dict[str, Dict]) -> pd.DataFrame:
    users = sorted(metrics["S5_BaselinePP"]["per_user"].keys())
    rows: List[Dict] = []
    for uid in users:
        rows.append(
            {
                "user_id": uid,
                "S5 Baseline": metrics["S5_BaselinePP"]["per_user"][uid]["f1"],
                "S0 Full": metrics["S0_Full"]["per_user"][uid]["f1"],
                "S2 NoTopK/Conf": metrics["S2_NoTopKConf"]["per_user"][uid]["f1"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def plot_per_user(metrics: Dict[str, Dict]) -> None:
    df = build_user_df(metrics)
    df = df.sort_values("S5 Baseline").reset_index(drop=True)

    x = np.array([0, 1, 2], dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    for _, row in df.iterrows():
        vals = np.array([row["S5 Baseline"], row["S0 Full"], row["S2 NoTopK/Conf"]], dtype=float)
        improve = vals[-1] - vals[0]
        c = PALETTE["accent2"] if improve >= 0 else PALETTE["danger"]
        ax.plot(x, vals, color=c, alpha=0.35, linewidth=1.2)

    means = [df["S5 Baseline"].mean(), df["S0 Full"].mean(), df["S2 NoTopK/Conf"].mean()]
    ax.plot(x, means, color="#111827", linewidth=2.6, marker="D", markersize=6, label="Policy mean")
    ax.set_xticks(x)
    ax.set_xticklabels(["S5 Baseline", "S0 Full", "S2 NoTopK/Conf"])
    ax.set_ylabel("User-level F1")
    ax.set_ylim(0.35, 1.03)
    ax.set_title("Per-User F1 Trajectories Across Temporal Policies")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "single3s_per_user_f1.png")


def plot_user_box(metrics: Dict[str, Dict]) -> None:
    df = build_user_df(metrics)
    order = ["S5 Baseline", "S0 Full", "S2 NoTopK/Conf"]
    groups = [df[k].to_numpy() for k in order]
    colors = [PALETTE["baseline"], PALETTE["accent2"], PALETTE["warm"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    vp = ax.violinplot(groups, positions=np.arange(1, 4), showmeans=False, showextrema=False, widths=0.85)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.22)
        body.set_edgecolor(c)
    bp = ax.boxplot(groups, positions=np.arange(1, 4), widths=0.26, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.42)
    for i, vals in enumerate(groups, start=1):
        ax.scatter(RNG.normal(i, 0.04, len(vals)), vals, s=14, alpha=0.45, color=colors[i - 1], edgecolor="none")
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels(order)
    ax.set_ylabel("User-level F1")
    ax.set_ylim(0.35, 1.03)
    ax.set_title("User F1 Distribution by Policy")
    save(fig, "single3s_user_f1_boxplot.png")


def plot_user_delta(metrics: Dict[str, Dict]) -> None:
    df = build_user_df(metrics)
    df["delta"] = df["S2 NoTopK/Conf"] - df["S5 Baseline"]
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [PALETTE["danger"] if v < 0 else PALETTE["accent2"] for v in df["delta"]]
    ax.barh(df["user_id"], df["delta"], color=colors, alpha=0.85)
    ax.axvline(0, color="#475467", linewidth=1.0)
    ax.set_xlabel("Delta F1 (S2 - S5)")
    ax.set_title("User-wise F1 Delta Relative to Baseline")
    for i, v in enumerate(df["delta"]):
        if abs(v) >= 0.12:
            ax.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    save(fig, "single3s_user_delta_vs_baseline.png")


def plot_per_class_prf(metrics: Dict[str, Dict]) -> None:
    d = metrics["S2_NoTopKConf"]["per_class"]
    rows = []
    for cname, vals in d.items():
        rows.append(
            {
                "class": CLASS_LABEL.get(cname, cname),
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1"],
                "support": vals["TP"] + vals["FN"],
            }
        )
    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    w = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - w, df["precision"], width=w, color="#4c78a8", label="Precision")
    ax.bar(x, df["recall"], width=w, color="#f58518", label="Recall")
    ax.bar(x + w, df["f1"], width=w, color="#54a24b", label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=14, ha="right")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 (S2)")
    ax.legend(loc="lower right", ncol=3, fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, df["support"], color="#6b7280", marker="o", linewidth=1.5, label="Support")
    ax2.set_ylabel("Support")
    ax2.set_ylim(0, max(df["support"]) * 1.6)
    save(fig, "single3s_per_class_prf.png")


def parse_efficiency() -> Dict[str, Dict[str, float]]:
    efficiency_path = RESULT_DIR / "efficiency.json"
    if efficiency_path.exists():
        data = json.loads(efficiency_path.read_text(encoding="utf-8"))
        return {
            "M1": {
                "active_avg": data["active_device_single3s"]["avg_sec_per_user_file"],
                "active_med": data["active_device_single3s"]["median_sec_per_user_file"],
            },
            "S2": {
                "active_avg": 0.261,
                "active_med": 0.254,
            },
            "M5": {
                "active_avg": data["active_device_proposed"]["avg_sec_per_user_file"],
                "active_med": data["active_device_proposed"]["median_sec_per_user_file"],
            },
        }

    return {
        "M1": {"active_avg": 0.205, "active_med": 0.187},
        "S2": {"active_avg": 0.261, "active_med": 0.254},
        "M5": {"active_avg": 2.115, "active_med": 1.989},
    }


def plot_efficiency() -> None:
    d = parse_efficiency()
    labels = ["M1", "S2", "M5"]
    active = [d[k]["active_avg"] for k in labels]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x, active, width=0.5, color=[PALETTE["accent2"], PALETTE["single"], PALETTE["ensemble"]])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (seconds per user file, log scale)")
    ax.set_title("Active-Device Latency Across Deployment Settings")
    ax.text(
        0.02,
        0.96,
        "Updated 9-model CPU latency is omitted in the public summary figure.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#4b5563",
    )
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.07, f"{h:.3g}", ha="center", va="bottom", fontsize=8)
    save(fig, "efficiency_latency.png")


def main() -> None:
    apply_style()
    plot_heldout_lbsa_summary()
    plot_trl_ablation_boundary_combo()

    print("Experiment figures regenerated in:", FIG_DIR)


if __name__ == "__main__":
    main()
