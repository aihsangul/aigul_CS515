from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Resolve the repository root: corecode/HW1/compare_experiments.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON dictionary.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_experiment_record(
    project_name: str,
    experiment_name: str,
    checkpoint_root: Path,
) -> Dict[str, Any]:
    """
    Build a single experiment record from saved history and test results.

    Args:
        project_name: Project/homework name.
        experiment_name: Experiment name.
        checkpoint_root: Root checkpoint directory.

    Returns:
        Dictionary containing merged metrics and config for the experiment.
    """
    exp_dir = checkpoint_root / project_name / experiment_name

    history_path = exp_dir / f"{experiment_name}_history.json"
    test_results_path = exp_dir / f"{experiment_name}_test_results.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not test_results_path.exists():
        raise FileNotFoundError(f"Missing test results file: {test_results_path}")

    history = load_json(history_path)
    test_results = load_json(test_results_path)

    cfg = history.get("config", {})
    hidden_sizes = cfg.get("hidden_sizes", [])
    activation = cfg.get("activation", "unknown")
    dropout = cfg.get("dropout", "unknown")
    use_batch_norm = cfg.get("use_batch_norm", False)
    optimizer = cfg.get("optimizer", "unknown")
    scheduler = cfg.get("scheduler", "unknown")
    regularizer = cfg.get("regularizer", "unknown")
    reg_lambda = cfg.get("reg_lambda", "unknown")
    weight_decay = cfg.get("weight_decay", "unknown")
    learning_rate = cfg.get("learning_rate", "unknown")
    batch_size = cfg.get("batch_size", "unknown")

    label = (
        f"{experiment_name}\n"
        f"{hidden_sizes} | {activation} | d={dropout} | "
        f"{'BN' if use_batch_norm else 'NoBN'}"
    )

    return {
        "project_name": project_name,
        "experiment_name": experiment_name,
        "label": label,
        "hidden_sizes": str(hidden_sizes),
        "activation": activation,
        "dropout": dropout,
        "use_batch_norm": use_batch_norm,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "regularizer": regularizer,
        "reg_lambda": reg_lambda,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "best_val_acc": history.get("best_val_acc"),
        "test_loss": test_results.get("test_loss"),
        "test_accuracy": test_results.get("test_accuracy"),
    }


def save_summary_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save aggregated experiment records to CSV.

    Args:
        records: List of experiment records.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No experiment records to save.")

    fieldnames = list(records[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_metric_bar(
    records: List[Dict[str, Any]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Plot a bar chart for a given metric across experiments.

    Args:
        records: Experiment records.
        metric_key: Metric field to plot.
        output_path: Output image path.
        title: Plot title.
        ylabel: Y-axis label.
    """
    labels = [record["label"] for record in records]
    values = [record[metric_key] for record in records]

    plt.figure(figsize=(max(10, len(labels) * 1.5), 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_bar_with_annotations(
    records: List[Dict[str, Any]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    decimals: int = 4,
) -> None:
    """
    Plot a bar chart and annotate bars with numeric values.

    Args:
        records: Experiment records.
        metric_key: Metric field to plot.
        output_path: Output image path.
        title: Plot title.
        ylabel: Y-axis label.
        decimals: Number of decimals in annotations.
    """
    labels = [record["label"] for record in records]
    values = [record[metric_key] for record in records]

    plt.figure(figsize=(max(10, len(labels) * 1.5), 6))
    bars = plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple saved experiments and generate ablation summary outputs."
    )

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument(
        "--experiment_names",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment names to compare.",
    )

    parser.add_argument("--checkpoint_root", type=str, default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--plot_root", type=str, default=str(_RESULTS_ROOT / "plots"))
    parser.add_argument("--summary_root", type=str, default=str(_RESULTS_ROOT / "summaries"))

    parser.add_argument(
        "--comparison_name",
        type=str,
        default="comparison",
        help="Name of the comparison output group.",
    )

    parser.add_argument(
        "--sort_by",
        choices=["test_accuracy", "best_val_acc", "best_val_loss", "experiment_name"],
        default="test_accuracy",
    )

    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    output_plot_dir = Path(args.plot_root) / args.project_name / args.comparison_name
    output_summary_dir = Path(args.summary_root) / args.project_name / args.comparison_name

    records = [
        build_experiment_record(
            project_name=args.project_name,
            experiment_name=experiment_name,
            checkpoint_root=checkpoint_root,
        )
        for experiment_name in args.experiment_names
    ]

    reverse = not args.ascending
    if args.sort_by == "best_val_loss":
        reverse = args.ascending is False and False  # lower is better by default
        if args.ascending:
            reverse = False
        else:
            reverse = False
    else:
        reverse = not args.ascending

    if args.sort_by == "best_val_loss":
        records = sorted(records, key=lambda x: x[args.sort_by])
    else:
        records = sorted(records, key=lambda x: x[args.sort_by], reverse=reverse)

    summary_csv_path = output_summary_dir / f"{args.comparison_name}_summary.csv"
    save_summary_csv(records, summary_csv_path)

    plot_metric_bar_with_annotations(
        records=records,
        metric_key="test_accuracy",
        output_path=output_plot_dir / f"{args.comparison_name}_test_accuracy.png",
        title=f"{args.project_name} - Test Accuracy Comparison",
        ylabel="Test Accuracy",
    )

    plot_metric_bar_with_annotations(
        records=records,
        metric_key="best_val_acc",
        output_path=output_plot_dir / f"{args.comparison_name}_best_val_acc.png",
        title=f"{args.project_name} - Best Validation Accuracy Comparison",
        ylabel="Best Validation Accuracy",
    )

    plot_metric_bar_with_annotations(
        records=records,
        metric_key="best_val_loss",
        output_path=output_plot_dir / f"{args.comparison_name}_best_val_loss.png",
        title=f"{args.project_name} - Best Validation Loss Comparison",
        ylabel="Best Validation Loss",
    )

    print("Summary CSV saved to:", summary_csv_path)
    print("Comparison plots saved to:", output_plot_dir)


if __name__ == "__main__":
    main()