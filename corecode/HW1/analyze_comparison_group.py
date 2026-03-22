from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Resolve the repository root: corecode/HW1/analyze_comparison_group.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_experiment_paths(
    checkpoint_root: Path,
    project_name: str,
    experiment_name: str,
) -> Dict[str, Path]:
    """
    Resolve the important file paths for a single experiment.

    Args:
        checkpoint_root: Root checkpoint directory.
        project_name: Project/homework name.
        experiment_name: Experiment name.

    Returns:
        Dictionary of resolved paths.
    """
    exp_dir = checkpoint_root / project_name / experiment_name
    return {
        "exp_dir": exp_dir,
        "history": exp_dir / f"{experiment_name}_history.json",
        "test_results": exp_dir / f"{experiment_name}_test_results.json",
        "checkpoint": exp_dir / f"{experiment_name}_best_model.pt",
    }


def summarize_experiment(
    checkpoint_root: Path,
    project_name: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Load one experiment and summarize its key metrics/configuration.

    Args:
        checkpoint_root: Root checkpoint directory.
        project_name: Project/homework name.
        experiment_name: Experiment name.

    Returns:
        Summary dictionary for the experiment.
    """
    paths = get_experiment_paths(checkpoint_root, project_name, experiment_name)

    if not paths["history"].exists():
        raise FileNotFoundError(f"Missing history file: {paths['history']}")
    if not paths["test_results"].exists():
        raise FileNotFoundError(f"Missing test results file: {paths['test_results']}")

    history = load_json(paths["history"])
    test_results = load_json(paths["test_results"])
    cfg = history.get("config", {})

    summary = {
        "project_name": project_name,
        "experiment_name": experiment_name,
        "hidden_sizes": str(cfg.get("hidden_sizes", "")),
        "activation": cfg.get("activation", ""),
        "dropout": cfg.get("dropout", ""),
        "use_batch_norm": cfg.get("use_batch_norm", ""),
        "optimizer": cfg.get("optimizer", ""),
        "scheduler": cfg.get("scheduler", ""),
        "regularizer": cfg.get("regularizer", ""),
        "reg_lambda": cfg.get("reg_lambda", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "batch_size": cfg.get("batch_size", ""),
        "best_epoch": history.get("best_epoch", ""),
        "best_val_loss": history.get("best_val_loss", ""),
        "best_val_acc": history.get("best_val_acc", ""),
        "test_loss": test_results.get("test_loss", ""),
        "test_accuracy": test_results.get("test_accuracy", ""),
    }

    return summary


def save_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save summary records as CSV.

    Args:
        records: List of experiment summaries.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_markdown_table(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save summary records as a Markdown table.

    Args:
        records: List of experiment summaries.
        output_path: Output Markdown path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "experiment_name",
        "hidden_sizes",
        "activation",
        "dropout",
        "use_batch_norm",
        "learning_rate",
        "scheduler",
        "regularizer",
        "best_val_loss",
        "best_val_acc",
        "test_accuracy",
    ]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    for record in records:
        row = [str(record[col]) for col in columns]
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_latex_table(records: List[Dict[str, Any]], output_path: Path, caption: str, label: str) -> None:
    """
    Save summary records as a LaTeX table.

    Args:
        records: List of experiment summaries.
        output_path: Output .tex path.
        caption: Table caption.
        label: Table label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "experiment_name",
        "hidden_sizes",
        "activation",
        "dropout",
        "use_batch_norm",
        "best_val_loss",
        "best_val_acc",
        "test_accuracy",
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \begin{tabular}{llllllll}")
    lines.append(r"        \toprule")
    lines.append(
        r"        Experiment & Hidden Sizes & Activation & Dropout & BN & Best Val Loss & Best Val Acc & Test Acc \\"
    )
    lines.append(r"        \midrule")

    for record in records:
        bn = "Yes" if record["use_batch_norm"] else "No"
        line = (
            f"        {record['experiment_name']} & "
            f"{record['hidden_sizes']} & "
            f"{record['activation']} & "
            f"{record['dropout']} & "
            f"{bn} & "
            f"{record['best_val_loss']:.4f} & "
            f"{record['best_val_acc']:.4f} & "
            f"{record['test_accuracy']:.4f} \\\\"
        )
        lines.append(line)

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_bar(
    records: List[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot a bar chart for a metric across experiments.

    Args:
        records: Experiment summary records.
        metric_key: Key of the metric to plot.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: Output image path.
    """
    labels = [record["experiment_name"] for record in records]
    values = [record[metric_key] for record in records]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    bars = plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_group_curves(
    checkpoint_root: Path,
    project_name: str,
    experiment_names: List[str],
    metric_key: str,
    split_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot epoch-wise curves for a given metric across multiple experiments.

    Args:
        checkpoint_root: Root checkpoint directory.
        project_name: Project/homework name.
        experiment_names: Experiments to compare.
        metric_key: Metric key in history JSON (e.g. train_loss, val_acc).
        split_name: Logical name for legend/title clarity.
        title: Plot title.
        ylabel: Y-axis label.
        output_path: Output image path.
    """
    plt.figure(figsize=(8, 5))

    for experiment_name in experiment_names:
        history_path = (
            checkpoint_root / project_name / experiment_name / f"{experiment_name}_history.json"
        )
        history = load_json(history_path)
        values = history[metric_key]
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker="o", label=experiment_name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a comparison group of completed experiments."
    )

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--comparison_name", type=str, required=True)
    parser.add_argument("--experiment_names", type=str, nargs="+", required=True)

    parser.add_argument("--checkpoint_root", type=str, default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--summary_root", type=str, default=str(_RESULTS_ROOT / "summaries"))
    parser.add_argument("--plot_root", type=str, default=str(_RESULTS_ROOT / "plots"))

    parser.add_argument(
        "--sort_by",
        choices=["experiment_name", "best_val_loss", "best_val_acc", "test_accuracy"],
        default="best_val_loss",
    )
    parser.add_argument("--ascending", action="store_true")

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    summary_dir = Path(args.summary_root) / args.project_name / args.comparison_name
    plot_dir = Path(args.plot_root) / args.project_name / args.comparison_name

    records = [
        summarize_experiment(checkpoint_root, args.project_name, exp_name)
        for exp_name in args.experiment_names
    ]

    records = sorted(
        records,
        key=lambda x: x[args.sort_by],
        reverse=not args.ascending if args.sort_by != "best_val_loss" else False,
    )

    ordered_experiment_names = [record["experiment_name"] for record in records]

    # Save tables
    save_csv(records, summary_dir / f"{args.comparison_name}_summary.csv")
    save_markdown_table(records, summary_dir / f"{args.comparison_name}_summary.md")
    save_latex_table(
        records,
        summary_dir / f"{args.comparison_name}_summary.tex",
        caption=f"{args.comparison_name.replace('_', ' ').title()} Results",
        label=f"tab:{args.comparison_name}",
    )

    # Bar charts
    plot_metric_bar(
        records,
        metric_key="best_val_loss",
        ylabel="Best Validation Loss",
        title=f"{args.project_name} - {args.comparison_name} - Best Validation Loss",
        output_path=plot_dir / f"{args.comparison_name}_best_val_loss.png",
    )

    plot_metric_bar(
        records,
        metric_key="best_val_acc",
        ylabel="Best Validation Accuracy",
        title=f"{args.project_name} - {args.comparison_name} - Best Validation Accuracy",
        output_path=plot_dir / f"{args.comparison_name}_best_val_acc.png",
    )

    plot_metric_bar(
        records,
        metric_key="test_accuracy",
        ylabel="Test Accuracy",
        title=f"{args.project_name} - {args.comparison_name} - Test Accuracy",
        output_path=plot_dir / f"{args.comparison_name}_test_accuracy.png",
    )

    # Curves
    plot_group_curves(
        checkpoint_root=checkpoint_root,
        project_name=args.project_name,
        experiment_names=ordered_experiment_names,
        metric_key="train_loss",
        split_name="train",
        title=f"{args.project_name} - {args.comparison_name} - Train Loss Curves",
        ylabel="Train Loss",
        output_path=plot_dir / f"{args.comparison_name}_train_loss_curves.png",
    )

    plot_group_curves(
        checkpoint_root=checkpoint_root,
        project_name=args.project_name,
        experiment_names=ordered_experiment_names,
        metric_key="val_loss",
        split_name="validation",
        title=f"{args.project_name} - {args.comparison_name} - Validation Loss Curves",
        ylabel="Validation Loss",
        output_path=plot_dir / f"{args.comparison_name}_val_loss_curves.png",
    )

    plot_group_curves(
        checkpoint_root=checkpoint_root,
        project_name=args.project_name,
        experiment_names=ordered_experiment_names,
        metric_key="train_acc",
        split_name="train",
        title=f"{args.project_name} - {args.comparison_name} - Train Accuracy Curves",
        ylabel="Train Accuracy",
        output_path=plot_dir / f"{args.comparison_name}_train_acc_curves.png",
    )

    plot_group_curves(
        checkpoint_root=checkpoint_root,
        project_name=args.project_name,
        experiment_names=ordered_experiment_names,
        metric_key="val_acc",
        split_name="validation",
        title=f"{args.project_name} - {args.comparison_name} - Validation Accuracy Curves",
        ylabel="Validation Accuracy",
        output_path=plot_dir / f"{args.comparison_name}_val_acc_curves.png",
    )

    print("Summary files saved to:", summary_dir)
    print("Plots saved to:", plot_dir)


if __name__ == "__main__":
    main()