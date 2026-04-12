"""HW3 experiment runner.

Runs all experiments in dependency order across five phases:

Phase 1 — AugMix teacher training
    Train ResNet-18 with AugMix + JSD consistency loss.

Phase 2 — Corruption robustness evaluation
    Evaluate both the HW2 scratch baseline and the AugMix model on the
    CIFAR-10 clean test set and CIFAR-10-C corruptions.

Phase 3 — Adversarial attack evaluation
    Run PGD20-Linf and PGD20-L2 on both models, generate Grad-CAM
    visualisations and adversarial T-SNE plots.

Phase 4 — Knowledge distillation with AugMix teacher
    Train SimpleCNN (kd_augmix) and MobileNetV2 (kd_ls_augmix) students
    using the AugMix-trained ResNet-18 as the teacher.

Phase 5 — Adversarial transferability
    Generate adversarial examples from the AugMix teacher and evaluate them
    on each KD student (black-box transfer attack).

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
_HW3_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW3_DIR.parents[1]
_RESULTS_ROOT = _REPO_ROOT / "results"
_CKPT_ROOT = _RESULTS_ROOT / "checkpoints"
_HW2_CKPT_ROOT = _CKPT_ROOT / "HW2"
_HW3_CKPT_ROOT = _CKPT_ROOT / "HW3"

PYTHON = sys.executable


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_main(
    experiment_name: str,
    extra_args: List[str],
    device: str,
    mode: str = "both",
) -> int:
    """Invoke main.py as a subprocess for one experiment.

    Args:
        experiment_name: Name used for checkpoints / logs / plots.
        extra_args: Additional CLI arguments forwarded to main.py.
        device: Device string.
        mode: ``"train"``, ``"test"``, or ``"both"``.

    Returns:
        Process return code (0 = success).
    """
    cmd = [
        PYTHON,
        str(_HW3_DIR / "main.py"),
        "--experiment_name", experiment_name,
        "--mode", mode,
        "--device", device,
        *extra_args,
    ]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT : {experiment_name}  (mode={mode})")
    print(sep)

    result = subprocess.run(cmd, cwd=str(_HW3_DIR))
    return result.returncode


def _abort_on_failure(rc: int, name: str) -> None:
    """Exit if a subprocess returned a non-zero code.

    Args:
        rc: Return code from the subprocess.
        name: Experiment name for error messaging.

    Raises:
        SystemExit: If ``rc != 0``.
    """
    if rc != 0:
        print(f"\nERROR: experiment '{name}' failed (exit code {rc}). Aborting.")
        sys.exit(rc)


def _augmix_checkpoint(experiment_name: str = "augmix") -> Path:
    """Return the path to the AugMix teacher checkpoint.

    Args:
        experiment_name: Experiment name used during Phase 1 training.

    Returns:
        Path to the best-model checkpoint file.

    Raises:
        SystemExit: If the checkpoint does not exist.
    """
    ckpt = _HW3_CKPT_ROOT / experiment_name / f"{experiment_name}_best_model.pt"
    if not ckpt.exists():
        print(f"\nERROR: AugMix checkpoint not found:\n  {ckpt}")
        print("Run Phase 1 first:  python run_experiments.py --phase 1")
        sys.exit(1)
    return ckpt


def _hw2_scratch_checkpoint(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Locate the HW2 baseline (scratch) checkpoint.

    Checks, in order:
    1. The explicit path provided via ``--hw2_scratch_checkpoint``.
    2. ``results/checkpoints/HW2/resnet_scratch/resnet_scratch_best_model.pt``
    3. ``results/checkpoints/HW2/resnet_scratch_ls/resnet_scratch_ls_best_model.pt``

    Returns ``None`` (with a warning) if no checkpoint is found.

    Args:
        explicit_path: Optional override path.

    Returns:
        Path to the checkpoint, or ``None`` if not found.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        print(f"WARNING: Provided HW2 checkpoint not found: {p}")
        return None

    for name in ("resnet_scratch", "resnet_scratch_ls"):
        p = _HW2_CKPT_ROOT / name / f"{name}_best_model.pt"
        if p.exists():
            print(f"Found HW2 scratch checkpoint: {p}")
            return p

    print(
        "WARNING: No HW2 scratch checkpoint found under:\n"
        f"  {_HW2_CKPT_ROOT}\n"
        "Phase 2 / 3 will skip evaluation of the HW2 baseline model.\n"
        "Train HW2 first or pass --hw2_scratch_checkpoint /path/to/ckpt.pt"
    )
    return None


def _hw2_baseline_experiment_name(hw2_ckpt: Path) -> str:
    """Infer the HW2 experiment name from the resolved checkpoint path."""
    return hw2_ckpt.parent.name


# ── Phase definitions ─────────────────────────────────────────────────────────

def run_phase1(device: str) -> None:
    """Phase 1: Train ResNet-18 with AugMix + JSD consistency loss.

    Args:
        device: Device string.
    """
    print("\n" + "#" * 70)
    print("# PHASE 1 — AugMix teacher training")
    print("#" * 70)

    rc = _run_main(
        experiment_name="augmix",
        extra_args=[
            "--training_mode", "augmix",
            "--epochs", "100",
            "--lr", "1e-3",
            "--optimizer", "adam",
            "--scheduler", "cosine",
            "--batch_size", "128",
            "--augmix_severity", "3",
            "--augmix_width", "3",
            "--augmix_depth", "-1",
            "--augmix_jsd_lambda", "12.0",
            "--early_stopping_patience", "15",
            "--eval_modes", "clean",
        ],
        device=device,
        mode="both",
    )
    _abort_on_failure(rc, "augmix")


def run_phase2(device: str, hw2_ckpt: Optional[Path]) -> None:
    """Phase 2: Corruption robustness evaluation (clean + CIFAR-10-C).

    Evaluates both the AugMix model and (if available) the HW2 baseline.

    Args:
        device: Device string.
        hw2_ckpt: Path to the HW2 scratch checkpoint, or ``None``.
    """
    print("\n" + "#" * 70)
    print("# PHASE 2 — Corruption robustness evaluation")
    print("#" * 70)

    # AugMix model: test-only eval with corruptions
    rc = _run_main(
        experiment_name="augmix",
        extra_args=[
            "--training_mode", "augmix",
            "--eval_modes", "clean", "corrupted",
        ],
        device=device,
        mode="test",
    )
    _abort_on_failure(rc, "augmix (phase2)")

    # HW2 baseline: evaluate under the HW3 harness (test-only)
    if hw2_ckpt is not None:
        hw2_exp = _hw2_baseline_experiment_name(hw2_ckpt)
        rc = _run_main(
            experiment_name=hw2_exp,
            extra_args=[
                "--project_name", "HW2",
                "--training_mode", "augmix",   # same model architecture
                "--eval_modes", "clean", "corrupted",
            ],
            device=device,
            mode="test",
        )
        # Non-fatal: baseline eval is informational
        if rc != 0:
            print(f"WARNING: HW2 baseline phase2 eval returned code {rc}. Continuing.")


def run_phase3(device: str, hw2_ckpt: Optional[Path]) -> None:
    """Phase 3: Adversarial attack evaluation (PGD, Grad-CAM, T-SNE).

    Args:
        device: Device string.
        hw2_ckpt: Path to the HW2 scratch checkpoint, or ``None``.
    """
    print("\n" + "#" * 70)
    print("# PHASE 3 — Adversarial robustness evaluation")
    print("#" * 70)

    adv_eval_modes = ["clean", "adversarial_linf", "adversarial_l2", "gradcam", "tsne_adv"]

    # AugMix model adversarial eval
    rc = _run_main(
        experiment_name="augmix",
        extra_args=[
            "--training_mode", "augmix",
            "--eval_modes", *adv_eval_modes,
        ],
        device=device,
        mode="test",
    )
    _abort_on_failure(rc, "augmix (phase3)")

    # HW2 baseline adversarial eval (if checkpoint available)
    if hw2_ckpt is not None:
        hw2_exp = _hw2_baseline_experiment_name(hw2_ckpt)
        rc = _run_main(
            experiment_name=hw2_exp,
            extra_args=[
                "--project_name", "HW2",
                "--training_mode", "augmix",
                "--eval_modes", *adv_eval_modes,
            ],
            device=device,
            mode="test",
        )
        if rc != 0:
            print(f"WARNING: HW2 baseline phase3 eval returned code {rc}. Continuing.")


def run_phase4(device: str) -> None:
    """Phase 4: Knowledge distillation with the AugMix teacher.

    Trains SimpleCNN (kd_augmix) and MobileNetV2 (kd_ls_augmix) students.

    Args:
        device: Device string.
    """
    print("\n" + "#" * 70)
    print("# PHASE 4 — Knowledge distillation with AugMix teacher")
    print("#" * 70)

    teacher_ckpt = str(_augmix_checkpoint())

    experiments: List[Tuple[str, str, List[str]]] = [
        (
            "simplecnn_kd_augmix",
            "kd_augmix",
            [
                "--kd_temperature", "4.0",
                "--kd_alpha", "0.5",
                "--epochs", "80",
                "--early_stopping_patience", "15",
                "--eval_modes", "clean",
            ],
        ),
        (
            "mobilenet_kd_ls_augmix",
            "kd_ls_augmix",
            [
                "--kd_temperature", "4.0",
                "--epochs", "80",
                "--early_stopping_patience", "15",
                "--eval_modes", "clean",
            ],
        ),
    ]

    for exp_name, mode, extra in experiments:
        rc = _run_main(
            experiment_name=exp_name,
            extra_args=[
                "--training_mode", mode,
                "--teacher_checkpoint", teacher_ckpt,
                "--lr", "1e-3",
                "--optimizer", "adam",
                "--scheduler", "cosine",
                "--batch_size", "128",
                *extra,
            ],
            device=device,
            mode="both",
        )
        _abort_on_failure(rc, exp_name)


def run_phase5(device: str) -> None:
    """Phase 5: Adversarial transferability from teacher to students.

    Generates PGD20-Linf adversarial examples from the AugMix teacher and
    evaluates them on both KD students.

    Args:
        device: Device string.
    """
    print("\n" + "#" * 70)
    print("# PHASE 5 — Adversarial transferability")
    print("#" * 70)

    teacher_ckpt = str(_augmix_checkpoint())

    for exp_name, mode in [
        ("simplecnn_kd_augmix", "kd_augmix"),
        ("mobilenet_kd_ls_augmix", "kd_ls_augmix"),
    ]:
        student_ckpt = _HW3_CKPT_ROOT / exp_name / f"{exp_name}_best_model.pt"
        if not student_ckpt.exists():
            print(f"WARNING: Student checkpoint not found: {student_ckpt}. "
                  "Run Phase 4 first.")
            continue

        rc = _run_main(
            experiment_name=exp_name,
            extra_args=[
                "--training_mode", mode,
                "--teacher_checkpoint", teacher_ckpt,
                "--eval_modes", "transferability",
            ],
            device=device,
            mode="test",
        )
        if rc != 0:
            print(f"WARNING: Transferability eval for '{exp_name}' returned code {rc}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the experiment runner.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW3 -- run all experiments in dependency order."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only a specific phase (1–5). Omit to run all phases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for all experiments: auto | cpu | cuda | mps (default: auto).",
    )
    parser.add_argument(
        "--hw2_scratch_checkpoint",
        type=str,
        default="",
        help=(
            "Path to the HW2 scratch baseline checkpoint. If omitted the "
            "runner searches the default HW2 checkpoint directory."
        ),
    )
    return parser


def main() -> None:
    """Entry point: parse args and dispatch experiment phases."""
    args = get_parser().parse_args()
    device: str = args.device
    phase: Optional[int] = args.phase
    hw2_ckpt = _hw2_scratch_checkpoint(args.hw2_scratch_checkpoint or None)

    if phase is None or phase == 1:
        run_phase1(device)

    if phase is None or phase == 2:
        run_phase2(device, hw2_ckpt)

    if phase is None or phase == 3:
        run_phase3(device, hw2_ckpt)

    if phase is None or phase == 4:
        run_phase4(device)

    if phase is None or phase == 5:
        run_phase5(device)

    print("\n" + "=" * 70)
    print("All requested experiments completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
