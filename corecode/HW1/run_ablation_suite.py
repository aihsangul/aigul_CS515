from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Resolve the repository root: corecode/HW1/run_ablation_suite.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"


def build_experiments() -> Dict[str, List[dict]]:
    """
    Build the ablation experiment suite.

    Smoke test and baseline are intentionally excluded because they were already
    run manually.

    Returns:
        Dictionary mapping group names to experiment configurations.
    """
    common = {
        "mode": "both",
        "batch_size": 64,
        "epochs": 15,
        "lr": 1e-3,
        "hidden_sizes": [256, 128],
        "activation": "relu",
        "dropout": 0.2,
    }

    groups = {
        "architecture": [
            {
                "experiment_name": "arch_shallow_128",
                "hidden_sizes": [128],
            },
            {
                "experiment_name": "arch_medium_256_128",
                "hidden_sizes": [256, 128],
            },
            {
                "experiment_name": "arch_deep_512_256_128",
                "hidden_sizes": [512, 256, 128],
            },
        ],
        "activation": [
            {
                "experiment_name": "act_gelu",
                "activation": "gelu",
            },
            {
                "experiment_name": "act_gelu_deep",
                "hidden_sizes": [512, 256, 128],
                "activation": "gelu",
            },
        ],
        "dropout": [
            {
                "experiment_name": "dropout_00",
                "dropout": 0.0,
            },
            {
                "experiment_name": "dropout_02",
                "dropout": 0.2,
            },
            {
                "experiment_name": "dropout_05",
                "dropout": 0.5,
            },
        ],
        "training": [
            {
                "experiment_name": "train_lr_1e4",
                "lr": 1e-4,
            },
            {
                "experiment_name": "train_lr_3e3",
                "lr": 3e-3,
            },
            {
                "experiment_name": "train_plateau",
                "epochs": 20,
                "scheduler": "plateau",
                "scheduler_patience": 2,
                "scheduler_factor": 0.5,
            },
            {
                "experiment_name": "train_cosine",
                "epochs": 20,
                "scheduler": "cosine",
            },
        ],
        "batchnorm": [
            {
                "experiment_name": "bn_on",
                "use_batch_norm": True,
            },
        ],
        "regularization": [
            {
                "experiment_name": "reg_l1_1e5",
                "regularizer": "l1",
                "reg_lambda": 1e-5,
            },
            {
                "experiment_name": "reg_l2_1e5",
                "regularizer": "l2",
                "reg_lambda": 1e-5,
            },
            {
                "experiment_name": "reg_l2_1e4",
                "regularizer": "l2",
                "reg_lambda": 1e-4,
            },
        ],
    }

    # Merge each experiment with the common defaults
    merged_groups: Dict[str, List[dict]] = {}
    for group_name, experiments in groups.items():
        merged_groups[group_name] = []
        for exp in experiments:
            merged = dict(common)
            merged.update(exp)
            merged_groups[group_name].append(merged)

    return merged_groups


def experiment_checkpoint_path(project_name: str, experiment_name: str) -> Path:
    """
    Build the expected checkpoint path for an experiment.

    Args:
        project_name: Project/homework name.
        experiment_name: Experiment name.

    Returns:
        Path to the expected best-model checkpoint.
    """
    return (
        _RESULTS_ROOT
        / "checkpoints"
        / project_name
        / experiment_name
        / f"{experiment_name}_best_model.pt"
    )


def build_command(
    python_executable: str,
    project_name: str,
    device: str,
    experiment: dict,
) -> List[str]:
    """
    Convert an experiment dictionary into a CLI command.

    Args:
        python_executable: Python executable path.
        project_name: Project/homework name.
        device: Device string.
        experiment: Experiment dictionary.

    Returns:
        Command list for subprocess.run.
    """
    cmd = [
        python_executable,
        "main.py",
        "--project_name",
        project_name,
        "--experiment_name",
        experiment["experiment_name"],
        "--mode",
        experiment["mode"],
        "--epochs",
        str(experiment["epochs"]),
        "--batch_size",
        str(experiment["batch_size"]),
        "--lr",
        str(experiment["lr"]),
        "--activation",
        str(experiment["activation"]),
        "--dropout",
        str(experiment["dropout"]),
        "--device",
        device,
        "--hidden_sizes",
        *[str(x) for x in experiment["hidden_sizes"]],
    ]

    if experiment.get("use_batch_norm", False):
        cmd.append("--use_batch_norm")

    if "scheduler" in experiment:
        cmd.extend(["--scheduler", str(experiment["scheduler"])])

    if "scheduler_patience" in experiment:
        cmd.extend(["--scheduler_patience", str(experiment["scheduler_patience"])])

    if "scheduler_factor" in experiment:
        cmd.extend(["--scheduler_factor", str(experiment["scheduler_factor"])])

    if "regularizer" in experiment:
        cmd.extend(["--regularizer", str(experiment["regularizer"])])

    if "reg_lambda" in experiment:
        cmd.extend(["--reg_lambda", str(experiment["reg_lambda"])])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the remaining CS515 HW1 ablation experiments."
    )
    parser.add_argument("--project_name", type=str, default="HW1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=["architecture", "activation", "dropout", "training", "batchnorm", "regularization"],
        help="Subset of experiment groups to run.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python_executable", type=str, default=sys.executable)

    args = parser.parse_args()

    groups = build_experiments()

    selected_groups = []
    for group_name in args.groups:
        if group_name not in groups:
            raise ValueError(
                f"Unknown group '{group_name}'. Available groups: {list(groups.keys())}"
            )
        selected_groups.append(group_name)

    total_runs = sum(len(groups[group]) for group in selected_groups)
    current_run = 0

    print(f"Project: {args.project_name}")
    print(f"Device: {args.device}")
    print(f"Selected groups: {selected_groups}")
    print(f"Total experiments to process: {total_runs}")
    print("-" * 80)

    for group_name in selected_groups:
        print(f"\n=== Group: {group_name} ===")
        for experiment in groups[group_name]:
            current_run += 1
            experiment_name = experiment["experiment_name"]
            checkpoint_path = experiment_checkpoint_path(args.project_name, experiment_name)

            print(f"\n[{current_run}/{total_runs}] {experiment_name}")

            if args.skip_existing and checkpoint_path.exists():
                print(f"Skipping existing experiment: {experiment_name}")
                print(f"Found checkpoint at: {checkpoint_path}")
                continue

            cmd = build_command(
                python_executable=args.python_executable,
                project_name=args.project_name,
                device=args.device,
                experiment=experiment,
            )

            print("Command:")
            print(" ".join(cmd))

            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Experiment '{experiment_name}' failed with exit code {result.returncode}."
                )

    print("\nAll requested experiments have been processed.")


if __name__ == "__main__":
    main()