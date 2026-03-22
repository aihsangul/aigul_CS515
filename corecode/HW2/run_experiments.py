"""HW2 experiment runner.

Runs all six training modes in dependency order:

Phase 1 (independent):
    1. transfer_resize   -- ResNet-18, frozen backbone, images resized to 224
    2. transfer_modify   -- ResNet-18, modified stem, full fine-tune
    3. resnet_scratch    -- ResNet-18 from scratch (standard CE)
    4. resnet_scratch_ls -- ResNet-18 from scratch (label-smoothed CE)

Phase 2 (require teacher checkpoint from Phase 1):
    5. simplecnn_kd    -- SimpleCNN student, Hinton KD loss
    6. mobilenet_kd_ls -- MobileNetV2 student, modified KD loss

After Phase 1 the script automatically compares the test accuracy of
``resnet_scratch`` and ``resnet_scratch_ls`` and uses the **better
performing** model as the teacher for Phase 2 experiments.

Usage::

    cd corecode/HW2
    python run_experiments.py                 # run all experiments
    python run_experiments.py --phase 1       # phase 1 only
    python run_experiments.py --phase 2       # phase 2 only (teacher must exist)
    python run_experiments.py --device cpu    # override device for all runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
_HW2_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW2_DIR.parents[1]
_RESULTS_ROOT = _REPO_ROOT / "results"
_CKPT_ROOT = _RESULTS_ROOT / "checkpoints" / "HW2"

PYTHON = sys.executable


# ── Experiment definitions ────────────────────────────────────────────────────

PHASE_1_EXPERIMENTS: List[Tuple[str, List[str]]] = [
    (
        "transfer_resize",
        [
            "--training_mode", "transfer_resize",
            "--epochs", "30",
            "--lr", "1e-3",
            "--optimizer", "adam",
            "--scheduler", "cosine",
            "--batch_size", "128",
            "--early_stopping_patience", "7",
        ],
    ),
    (
        "transfer_modify",
        [
            "--training_mode", "transfer_modify",
            "--epochs", "50",
            "--lr", "1e-3",
            "--optimizer", "adam",
            "--scheduler", "cosine",
            "--batch_size", "128",
            "--early_stopping_patience", "10",
        ],
    ),
    (
        "resnet_scratch",
        [
            "--training_mode", "scratch",
            "--epochs", "100",
            "--lr", "1e-3",
            "--optimizer", "adam",
            "--scheduler", "cosine",
            "--batch_size", "128",
            "--early_stopping_patience", "15",
        ],
    ),
    (
        "resnet_scratch_ls",
        [
            "--training_mode", "scratch_ls",
            "--epochs", "100",
            "--lr", "1e-3",
            "--optimizer", "adam",
            "--scheduler", "cosine",
            "--batch_size", "128",
            "--label_smoothing", "0.1",
            "--early_stopping_patience", "15",
        ],
    ),
]


def _build_phase_2(teacher_ckpt: Path) -> List[Tuple[str, List[str]]]:
    """Build phase-2 experiment list with the resolved teacher checkpoint.

    Args:
        teacher_ckpt: Path to the frozen teacher model checkpoint.

    Returns:
        List of (experiment_name, extra_cli_args) pairs.
    """
    return [
        (
            "simplecnn_kd",
            [
                "--training_mode", "kd",
                "--epochs", "80",
                "--lr", "1e-3",
                "--optimizer", "adam",
                "--scheduler", "cosine",
                "--batch_size", "128",
                "--kd_temperature", "4.0",
                "--kd_alpha", "0.5",
                "--early_stopping_patience", "15",
                "--teacher_checkpoint", str(teacher_ckpt),
            ],
        ),
        (
            "mobilenet_kd_ls",
            [
                "--training_mode", "kd_ls",
                "--epochs", "80",
                "--lr", "1e-3",
                "--optimizer", "adam",
                "--scheduler", "cosine",
                "--batch_size", "128",
                "--kd_temperature", "4.0",
                "--early_stopping_patience", "15",
                "--teacher_checkpoint", str(teacher_ckpt),
            ],
        ),
    ]


# ── Teacher selection ─────────────────────────────────────────────────────────

def _load_test_accuracy(experiment_name: str) -> float:
    """Load the test accuracy from a saved test-results JSON.

    Args:
        experiment_name: Name of the experiment whose results to load.

    Returns:
        Test accuracy as a float, or -1.0 if the file does not exist.
    """
    results_path = _CKPT_ROOT / experiment_name / f"{experiment_name}_test_results.json"
    if not results_path.exists():
        return -1.0
    with results_path.open(encoding="utf-8") as fh:
        return float(json.load(fh).get("test_accuracy", -1.0))


def _pick_teacher_checkpoint() -> Path:
    """Select the better teacher between resnet_scratch and resnet_scratch_ls.

    Compares test accuracy from the saved test-results JSON files produced
    during Phase 1. Falls back to ``resnet_scratch`` if the label-smoothed
    results are unavailable or do not outperform the baseline.

    Returns:
        Path to the best checkpoint file.

    Raises:
        SystemExit: If neither checkpoint file exists on disk.
    """
    scratch_acc = _load_test_accuracy("resnet_scratch")
    ls_acc = _load_test_accuracy("resnet_scratch_ls")

    print("\n" + "-" * 70)
    print("Teacher selection")
    print(f"  resnet_scratch    test_acc = {scratch_acc:.4f}" if scratch_acc >= 0
          else "  resnet_scratch    results not found")
    print(f"  resnet_scratch_ls test_acc = {ls_acc:.4f}" if ls_acc >= 0
          else "  resnet_scratch_ls results not found")

    if ls_acc > scratch_acc:
        winner = "resnet_scratch_ls"
        winner_acc = ls_acc
    else:
        winner = "resnet_scratch"
        winner_acc = scratch_acc

    ckpt = _CKPT_ROOT / winner / f"{winner}_best_model.pt"

    if not ckpt.exists():
        print(f"\nERROR: Teacher checkpoint not found at:\n  {ckpt}")
        print("Run Phase 1 first (or omit --phase to run both phases together).")
        sys.exit(1)

    print(f"  Selected teacher  : '{winner}' (acc={winner_acc:.4f})")
    print("-" * 70)
    return ckpt


# ── Runner ────────────────────────────────────────────────────────────────────

def run_experiment(
    experiment_name: str,
    extra_args: List[str],
    device: str,
) -> int:
    """Invoke main.py as a subprocess for one experiment.

    Args:
        experiment_name: Name used for checkpoints / logs / plots.
        extra_args: Additional CLI arguments forwarded to main.py.
        device: Device string passed via ``--device``.

    Returns:
        Process return code (0 = success).
    """
    cmd = [
        PYTHON,
        str(_HW2_DIR / "main.py"),
        "--experiment_name", experiment_name,
        "--mode", "both",
        "--device", device,
        *extra_args,
    ]

    separator = "=" * 70
    print(f"\n{separator}")
    print(f"EXPERIMENT : {experiment_name}")
    print(separator)

    result = subprocess.run(cmd, cwd=str(_HW2_DIR))
    return result.returncode


def run_phase(
    experiments: List[Tuple[str, List[str]]],
    device: str,
    phase_label: str,
) -> None:
    """Run a list of experiments sequentially, aborting on first failure.

    Args:
        experiments: List of (name, extra_args) pairs.
        device: Device override.
        phase_label: Human-readable label for logging.

    Raises:
        SystemExit: If any experiment returns a non-zero exit code.
    """
    print(f"\n{'#' * 70}")
    print(f"# {phase_label}")
    print(f"{'#' * 70}")

    for name, args in experiments:
        rc = run_experiment(name, args, device)
        if rc != 0:
            print(f"\nERROR: experiment '{name}' failed (exit code {rc}). Aborting.")
            sys.exit(rc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the experiment runner.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW2 -- run all experiments in dependency order."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only phase 1 or phase 2. Omit to run both phases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for all experiments: auto | cpu | cuda | mps (default: auto).",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default="",
        help=(
            "Explicit path to the teacher checkpoint for phase 2. "
            "When omitted, the script auto-selects the better of "
            "resnet_scratch / resnet_scratch_ls based on test accuracy."
        ),
    )
    return parser


def main() -> None:
    """Entry point: parse args and dispatch experiment phases."""
    args = get_parser().parse_args()
    device: str = args.device
    phase: Optional[int] = args.phase

    if phase is None or phase == 1:
        run_phase(PHASE_1_EXPERIMENTS, device, "PHASE 1 -- Transfer Learning & Scratch Training")

    # Resolve teacher checkpoint
    if phase is None or phase == 2:
        if args.teacher_checkpoint:
            teacher_ckpt = Path(args.teacher_checkpoint)
            if not teacher_ckpt.exists():
                print(f"\nERROR: Provided teacher checkpoint not found:\n  {teacher_ckpt}")
                sys.exit(1)
            print(f"\nUsing provided teacher checkpoint: {teacher_ckpt}")
        else:
            teacher_ckpt = _pick_teacher_checkpoint()

        phase_2_experiments = _build_phase_2(teacher_ckpt)
        run_phase(phase_2_experiments, device, "PHASE 2 -- Knowledge Distillation")

    print("\n" + "=" * 70)
    print("All experiments completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
