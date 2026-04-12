"""Compare and summarise all HW3 evaluation results.

Loads the JSON result files produced by each experiment and outputs:

1. A console table with clean accuracy, mCA (CIFAR-10-C), and adversarial
   accuracy (Linf / L2) for every experiment.
2. A grouped bar chart comparing robustness metrics across all experiments
   (saved as PNG).
3. A corruption accuracy bar chart for each model (saved as PNG).

Usage::

    cd corecode/HW3
    python compare_results.py
    python compare_results.py --plot_dir /custom/plot/dir
    python compare_results.py --no_plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HW3_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW3_DIR.parents[1]
_RESULTS_ROOT = _REPO_ROOT / "results"
_HW2_CKPT_ROOT = _RESULTS_ROOT / "checkpoints" / "HW2"
_HW3_CKPT_ROOT = _RESULTS_ROOT / "checkpoints" / "HW3"
_PLOT_ROOT = _RESULTS_ROOT / "plots" / "HW3" / "comparison"

# Canonical experiment display order
EXPERIMENT_ORDER = [
    "hw2_baseline",
    "augmix",
    "simplecnn_kd_augmix",
    "mobilenet_kd_ls_augmix",
]

LABELS = {
    "hw2_baseline":          "ResNet-18\n(HW2 baseline)",
    "augmix":                "ResNet-18\n(AugMix)",
    "simplecnn_kd_augmix":   "SimpleCNN\n(KD-AugMix)",
    "mobilenet_kd_ls_augmix":"MobileNetV2\n(KD-LS-AugMix)",
}

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` if the file does not exist.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict, or ``None``.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_checkpoint_dir(experiment_name: str) -> Optional[Path]:
    """Locate the checkpoint directory for a named experiment.

    Searches HW3 first, then HW2 (for the baseline).

    Args:
        experiment_name: Experiment directory name.

    Returns:
        Resolved checkpoint directory path, or ``None``.
    """
    hw3_path = _HW3_CKPT_ROOT / experiment_name
    if hw3_path.exists():
        return hw3_path

    # HW2 baseline: try resnet_scratch and resnet_scratch_ls
    if experiment_name == "hw2_baseline":
        for candidate in ("resnet_scratch", "resnet_scratch_ls"):
            p = _HW2_CKPT_ROOT / candidate
            if p.exists():
                return p

    return None


def _load_experiment_results(experiment_name: str) -> Dict[str, Any]:
    """Load all available result JSON files for one experiment.

    Args:
        experiment_name: Experiment directory name.

    Returns:
        Dict with keys: ``clean``, ``corrupted``, ``adversarial_linf``,
        ``adversarial_l2``, ``transferability`` — each mapping to the
        parsed JSON dict (or ``None`` if not found).
    """
    ckpt_dir = _find_checkpoint_dir(experiment_name)
    if ckpt_dir is None:
        return {}

    # Determine the stem name used in file names (may differ from dir name)
    stem = experiment_name
    if experiment_name == "hw2_baseline":
        for candidate in ("resnet_scratch", "resnet_scratch_ls"):
            p = _HW2_CKPT_ROOT / candidate
            if p.exists():
                ckpt_dir = p
                stem = candidate
                break

    results: Dict[str, Any] = {}

    # Try both the experiment_name-based and stem-based file names
    for sfx, key in [
        ("", "clean"),
        ("_corrupted", "corrupted"),
        ("_adversarial_linf", "adversarial_linf"),
        ("_adversarial_l2", "adversarial_l2"),
        ("_transferability", "transferability"),
    ]:
        for name in [stem, experiment_name]:
            p = ckpt_dir / f"{name}_test_results{sfx}.json"
            data = _load_json(p)
            if data is not None:
                results[key] = data
                break

    return results


# ── Console table ─────────────────────────────────────────────────────────────

def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        rows: List of dicts with keys: experiment, clean_acc, mca,
              adv_linf_acc, adv_l2_acc, transfer_acc.
    """
    header = (
        f"{'Experiment':<26} "
        f"{'Clean':>8} "
        f"{'mCA':>8} "
        f"{'Adv-Linf':>10} "
        f"{'Adv-L2':>8} "
        f"{'Transfer':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    def _fmt(val: Optional[float]) -> str:
        return f"{val:.2%}" if val is not None else "  N/A  "

    for row in rows:
        print(
            f"{row['experiment']:<26}"
            f" {_fmt(row.get('clean_acc'))}"
            f" {_fmt(row.get('mca'))}"
            f" {_fmt(row.get('adv_linf_acc'))}"
            f" {_fmt(row.get('adv_l2_acc'))}"
            f" {_fmt(row.get('transfer_acc'))}"
        )

    print(sep)
    print("  mCA = Mean Corruption Accuracy (CIFAR-10-C)")
    print("  Adv-Linf = PGD20-Linf accuracy  |  Adv-L2 = PGD20-L2 accuracy")
    print("  Transfer = student accuracy under teacher-generated adversarial examples")
    print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_robustness_bars(rows: List[Dict[str, Any]], save_path: str) -> None:
    """Save a grouped bar chart comparing robustness metrics across experiments.

    Args:
        rows: Row dicts from :func:`print_comparison_table`.
        save_path: Output PNG path.
    """
    metrics = ["clean_acc", "mca", "adv_linf_acc", "adv_l2_acc"]
    metric_labels = ["Clean", "mCA (corrupt.)", "Adv Linf", "Adv L2"]
    bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    exp_names = [r["experiment"] for r in rows]
    n_models = len(exp_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(9, n_models * 2.5), 5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, bar_colors)):
        vals = [
            (r.get(metric) or 0.0) * 100
            for r in rows
        ]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

    display_names = [LABELS.get(n, n) for n in exp_names]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("HW3 — Robustness comparison")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Robustness comparison saved -> {save_path}")


def plot_corruption_comparison(
    all_results: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    """Compare mean per-corruption accuracy across all experiments.

    Args:
        all_results: ``{experiment_name: loaded_results_dict}``.
        save_path: Output PNG path.
    """
    from test import CORRUPTIONS

    # Build a bar chart: one group per corruption, one bar per experiment
    available_exps = [e for e in EXPERIMENT_ORDER if e in all_results
                      and all_results[e].get("corrupted")]
    if not available_exps:
        print("No CIFAR-10-C results found for comparison.")
        return

    n_exps = len(available_exps)
    n_corruptions = len(CORRUPTIONS)
    x = np.arange(n_corruptions)
    width = 0.8 / n_exps

    fig, ax = plt.subplots(figsize=(max(14, n_corruptions * 0.8), 5))

    for i, exp_name in enumerate(available_exps):
        corr_data = all_results[exp_name]["corrupted"].get("per_corruption", {})
        means = []
        for corruption in CORRUPTIONS:
            sev_accs = corr_data.get(corruption, {})
            means.append(np.mean(list(sev_accs.values())) if sev_accs else 0.0)

        color = COLORS[EXPERIMENT_ORDER.index(exp_name) % len(COLORS)]
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, [m * 100 for m in means], width,
               label=LABELS.get(exp_name, exp_name),
               color=color, edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CORRUPTIONS], fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("HW3 — CIFAR-10-C per-corruption accuracy comparison")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Corruption comparison saved -> {save_path}")


# ── CLI & entry point ─────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW3 — compare results across all experiments."
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
    """Load all available results, print the comparison table, and save plots."""
    args = get_parser().parse_args()
    plot_dir = Path(args.plot_dir)

    # Discover experiments
    available_dirs: set = set()
    for root in (_HW3_CKPT_ROOT, _HW2_CKPT_ROOT):
        if root.exists():
            available_dirs.update(p.name for p in root.iterdir() if p.is_dir())

    # Map HW2 scratch dirs to the canonical baseline name
    if "resnet_scratch" in available_dirs or "resnet_scratch_ls" in available_dirs:
        available_dirs.add("hw2_baseline")

    ordered = [e for e in EXPERIMENT_ORDER if e in available_dirs]
    extras = sorted(available_dirs - set(EXPERIMENT_ORDER) -
                    {"resnet_scratch", "resnet_scratch_ls",
                     "transfer_resize", "transfer_modify", "simplecnn_kd",
                     "mobilenet_kd_ls"})

    all_results: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for exp_name in ordered + extras:
        exp_results = _load_experiment_results(exp_name)
        if not exp_results:
            continue
        all_results[exp_name] = exp_results

        clean_acc = (exp_results.get("clean") or {}).get("test_accuracy")
        mca = (exp_results.get("corrupted") or {}).get("mean_corruption_accuracy")
        adv_linf_acc = (exp_results.get("adversarial_linf") or {}).get("adversarial_accuracy")
        adv_l2_acc = (exp_results.get("adversarial_l2") or {}).get("adversarial_accuracy")
        transfer_acc = (exp_results.get("transferability") or {}).get("student_adversarial_accuracy")

        rows.append({
            "experiment": exp_name,
            "clean_acc": clean_acc,
            "mca": mca,
            "adv_linf_acc": adv_linf_acc,
            "adv_l2_acc": adv_l2_acc,
            "transfer_acc": transfer_acc,
        })

    if not rows:
        print(
            "No HW3 results found.\n"
            f"  HW3 checkpoints: {_HW3_CKPT_ROOT}\n"
            f"  HW2 checkpoints: {_HW2_CKPT_ROOT}\n"
            "Run 'python run_experiments.py' first."
        )
        return

    print_comparison_table(rows)

    if not args.no_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plot_robustness_bars(rows, str(plot_dir / "robustness_comparison.png"))
        plot_corruption_comparison(all_results, str(plot_dir / "corruption_comparison.png"))


if __name__ == "__main__":
    main()
