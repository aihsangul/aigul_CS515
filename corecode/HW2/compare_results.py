"""Compare test results across all HW2 experiments.

Loads the "*_test_results.json" files produced by each experiment and
outputs:

1. A console table summarizing accuracy, MACs, and parameter counts.
2. A bar-chart comparing test accuracy across all experiments (saved as PNG).
3. A bar-chart comparing computational complexity (MACs) across all
   experiments, with a note that transfer_resize uses 224x224 input while
   all other experiments use 32x32 (so absolute MACs are not directly
   comparable across the two groups).

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HW2_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW2_DIR.parents[1]
_RESULTS_ROOT = _REPO_ROOT / "results"
_CKPT_ROOT = _RESULTS_ROOT / "checkpoints" / "HW2"
_PLOT_ROOT = _RESULTS_ROOT / "plots" / "HW2" / "comparison"

# Canonical experiment order for display
EXPERIMENT_ORDER = [
    "transfer_resize",
    "transfer_modify",
    "resnet_scratch",
    "resnet_scratch_ls",
    "simplecnn_kd",
    "mobilenet_kd_ls",
]

# Short display labels (two-line for bar-chart x-axis)
LABELS = {
    "transfer_resize":   "ResNet-18\nTransfer (resize)",
    "transfer_modify":   "ResNet-18\nTransfer (modify)",
    "resnet_scratch":    "ResNet-18\nScratch",
    "resnet_scratch_ls": "ResNet-18\nScratch + LS",
    "simplecnn_kd":      "SimpleCNN\n(KD)",
    "mobilenet_kd_ls":   "MobileNetV2\n(KD-LS)",
}

# Colour palette — one colour per experiment
COLORS = [
    "#4C72B0",  # transfer_resize  (blue)
    "#55A868",  # transfer_modify  (green)
    "#C44E52",  # resnet_scratch   (red)
    "#DD8452",  # resnet_scratch_ls(orange)
    "#8172B2",  # simplecnn_kd     (purple)
    "#937860",  # mobilenet_kd_ls  (brown)
]

# Experiments that use 224x224 input (MACs not directly comparable to 32x32)
LARGE_INPUT_EXPERIMENTS = {"transfer_resize"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_test_results(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load the test-results JSON for one experiment.

    Args:
        experiment_name: Name of the experiment directory under checkpoints/HW2/.

    Returns:
        Parsed JSON dict, or ``None`` if the file does not exist.
    """
    results_path = _CKPT_ROOT / experiment_name / f"{experiment_name}_test_results.json"
    if not results_path.exists():
        return None
    with results_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _parse_macs(macs_str: str) -> float:
    """Convert a ptflops MACs string to a float in MMac units.

    Handles suffixes: GMac, MMac, KMac (case-insensitive).

    Args:
        macs_str: String such as ``"557.78 MMac"`` or ``"1.2 GMac"``.

    Returns:
        Float value in MMac, or 0.0 if the string is ``"N/A"`` or unparseable.
    """
    if not macs_str or macs_str == "N/A":
        return 0.0
    try:
        val_str, unit = macs_str.split()
        val = float(val_str)
        unit_upper = unit.upper()
        if "G" in unit_upper:
            return val * 1_000.0
        if "M" in unit_upper:
            return val
        if "K" in unit_upper:
            return val / 1_000.0
        return val
    except (ValueError, AttributeError):
        return 0.0


def _parse_params(params_str: str) -> float:
    """Convert a ptflops parameter-count string to a float in millions.

    Args:
        params_str: String such as ``"11.17 M"`` or ``"3.4 K"``.

    Returns:
        Float value in millions, or 0.0 if unparseable.
    """
    if not params_str or params_str == "N/A":
        return 0.0
    try:
        val_str, unit = params_str.split()
        val = float(val_str)
        unit_upper = unit.upper()
        if "G" in unit_upper:
            return val * 1_000.0
        if "M" in unit_upper:
            return val
        if "K" in unit_upper:
            return val / 1_000.0
        return val
    except (ValueError, AttributeError):
        return 0.0


# ── Table ─────────────────────────────────────────────────────────────────────

def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        rows: List of result dicts, each with keys: experiment, accuracy,
              flops, params, input_size.
    """
    header = f"{'Experiment':<22} {'Accuracy':>10} {'MACs':>14} {'Params (M)':>12} {'Input':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for row in rows:
        flops_note = row["flops"] + (" *" if row["experiment"] in LARGE_INPUT_EXPERIMENTS else "")
        print(
            f"{row['experiment']:<22}"
            f" {row['accuracy']:>9.2%}"
            f" {flops_note:>14}"
            f" {row['params_m']:>12.2f}"
            f" {row['input_size']:>10}"
        )
    print(sep)
    print("  * MACs measured on 224x224 input; all others use 32x32.")
    print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Save a bar chart comparing test accuracy across experiments.

    Args:
        rows: List of result dicts with ``experiment`` and ``accuracy`` keys.
        save_path: Output PNG file path.
    """
    names = [LABELS.get(r["experiment"], r["experiment"]) for r in rows]
    accs = [r["accuracy"] * 100 for r in rows]
    colors = [COLORS[EXPERIMENT_ORDER.index(r["experiment"])]
              if r["experiment"] in EXPERIMENT_ORDER else "#999999"
              for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("HW2 — Test accuracy comparison")
    ax.set_ylim(0, min(100, max(accs) + 8))
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Accuracy comparison saved -> {save_path}")


def plot_complexity_comparison(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Save a bar chart comparing MACs (complexity) across experiments.

    Experiments that use 224x224 input are marked with an asterisk because
    their MACs are not directly comparable to 32x32-input experiments.

    Args:
        rows: List of result dicts with ``experiment``, ``macs_m``, and
              ``params_m`` keys.
        save_path: Output PNG file path.
    """
    names = [
        LABELS.get(r["experiment"], r["experiment"]) +
        ("\n*" if r["experiment"] in LARGE_INPUT_EXPERIMENTS else "")
        for r in rows
    ]
    macs = [r["macs_m"] for r in rows]
    colors = [COLORS[EXPERIMENT_ORDER.index(r["experiment"])]
              if r["experiment"] in EXPERIMENT_ORDER else "#999999"
              for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, macs, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, macs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(macs) * 0.01,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("MACs (MMac)")
    ax.set_title("HW2 — Computational complexity (MACs)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)

    fig.text(
        0.01, 0.01,
        "* Transfer-resize uses 224x224 input; all others use 32x32 — MACs not directly comparable.",
        fontsize=7, color="gray",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Complexity comparison saved -> {save_path}")


def plot_accuracy_vs_complexity(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Save a scatter plot of accuracy vs. MACs for 32x32-input experiments.

    Transfer-resize is excluded because its MACs are measured on a different
    input resolution.

    Args:
        rows: List of result dicts.
        save_path: Output PNG file path.
    """
    filtered = [r for r in rows if r["experiment"] not in LARGE_INPUT_EXPERIMENTS and r["macs_m"] > 0]
    if len(filtered) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in filtered:
        color = COLORS[EXPERIMENT_ORDER.index(r["experiment"])]
        ax.scatter(r["macs_m"], r["accuracy"] * 100, color=color, s=120, zorder=3)
        ax.annotate(
            LABELS.get(r["experiment"], r["experiment"]).replace("\n", " "),
            (r["macs_m"], r["accuracy"] * 100),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    ax.set_xlabel("MACs (MMac) — 32x32 input")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("HW2 — Accuracy vs. Complexity (32x32 experiments)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Accuracy vs. complexity saved -> {save_path}")


# ── CLI & entry point ─────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW2 -- compare test results across all experiments."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=str(_PLOT_ROOT),
        help="Directory where comparison plots are saved.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Print the table only; skip generating plot files.",
    )
    return parser


def main() -> None:
    """Load all available test results, print the comparison table, and save plots."""
    args = get_parser().parse_args()
    plot_dir = Path(args.plot_dir)

    rows: List[Dict[str, Any]] = []

    # Discover available experiments (in canonical order first, then any extras)
    found_dirs = {p.name for p in _CKPT_ROOT.iterdir() if p.is_dir()} if _CKPT_ROOT.exists() else set()
    ordered = [e for e in EXPERIMENT_ORDER if e in found_dirs]
    extras = sorted(found_dirs - set(EXPERIMENT_ORDER))

    for exp_name in ordered + extras:
        results = _load_test_results(exp_name)
        if results is None:
            continue
        rows.append({
            "experiment": exp_name,
            "accuracy": results.get("test_accuracy", 0.0),
            "flops": results.get("flops", "N/A"),
            "params": results.get("params", "N/A"),
            "macs_m": _parse_macs(results.get("flops", "N/A")),
            "params_m": _parse_params(results.get("params", "N/A")),
            "input_size": "224x224" if exp_name in LARGE_INPUT_EXPERIMENTS else "32x32",
        })

    if not rows:
        print(
            "No test results found under:\n"
            f"  {_CKPT_ROOT}\n"
            "Run 'python run_experiments.py' first."
        )
        return

    print_comparison_table(rows)

    if not args.no_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plot_accuracy_comparison(rows, str(plot_dir / "accuracy_comparison.png"))
        plot_complexity_comparison(rows, str(plot_dir / "complexity_comparison.png"))
        plot_accuracy_vs_complexity(rows, str(plot_dir / "accuracy_vs_complexity.png"))


if __name__ == "__main__":
    main()
