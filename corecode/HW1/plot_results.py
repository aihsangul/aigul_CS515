from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

# Resolve the repository root: corecode/HW1/plot_results.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"

from plotting import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_learning_rate_curve,
    plot_loss_curve,
)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dictionary.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_title(project_name: str, experiment_name: str, history: Dict[str, Any]) -> str:
    """
    Build a descriptive title prefix using saved config metadata.

    Args:
        project_name: Project/homework name.
        experiment_name: Experiment name.
        history: Training history dictionary.

    Returns:
        Title prefix string.
    """
    cfg = history.get("config", {})
    hidden_sizes = cfg.get("hidden_sizes", "unknown")
    activation = cfg.get("activation", "unknown")
    dropout = cfg.get("dropout", "unknown")
    use_batch_norm = cfg.get("use_batch_norm", False)

    bn_text = "BN" if use_batch_norm else "NoBN"

    return (
        f"{project_name}/{experiment_name} | "
        f"MLP {hidden_sizes} | {activation.upper()} | "
        f"dropout={dropout} | {bn_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from saved training/test results."
    )

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)

    parser.add_argument("--checkpoint_root", type=str, default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--plot_root", type=str, default=str(_RESULTS_ROOT / "plots"))

    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--test_results_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--normalize_cm", action="store_true",
                        help="Normalize confusion matrix rows")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_root) / args.project_name / args.experiment_name
    plot_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.plot_root) / args.project_name / args.experiment_name
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    history_path = (
        Path(args.history_path)
        if args.history_path is not None
        else checkpoint_dir / f"{args.experiment_name}_history.json"
    )

    test_results_path = (
        Path(args.test_results_path)
        if args.test_results_path is not None
        else checkpoint_dir / f"{args.experiment_name}_test_results.json"
    )

    history = load_json(history_path)
    test_results = load_json(test_results_path)

    title_prefix = build_title(
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        history=history,
    )

    plot_loss_curve(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        output_path=str(plot_dir / f"{args.experiment_name}_loss.png"),
        title=f"{title_prefix} - Loss Curve",
    )

    plot_accuracy_curve(
        train_acc=history["train_acc"],
        val_acc=history["val_acc"],
        output_path=str(plot_dir / f"{args.experiment_name}_accuracy.png"),
        title=f"{title_prefix} - Accuracy Curve",
    )

    plot_learning_rate_curve(
        learning_rates=history["learning_rate"],
        output_path=str(plot_dir / f"{args.experiment_name}_lr.png"),
        title=f"{title_prefix} - Learning Rate Curve",
    )

    num_classes = len(test_results["confusion_matrix"])
    class_names = [str(i) for i in range(num_classes)]

    plot_confusion_matrix(
        confusion_matrix=test_results["confusion_matrix"],
        output_path=str(plot_dir / f"{args.experiment_name}_confusion_matrix.png"),
        class_names=class_names,
        normalize=args.normalize_cm,
        title=f"{title_prefix} - Confusion Matrix",
    )

    print("History file used:", history_path)
    print("Test results file used:", test_results_path)
    print("Plots saved to:", plot_dir)


if __name__ == "__main__":
    main()