"""CS515 HW3 — main entry point.

Training modes
--------------
augmix
    ResNet-18 trained from scratch with AugMix + JSD consistency loss.

kd_augmix
    SimpleCNN student, Hinton KD loss, AugMix-trained ResNet-18 teacher.

kd_ls_augmix
    MobileNetV2 student, modified KD loss, AugMix-trained ResNet-18 teacher.

Evaluation modes (set via --eval_modes, multiple allowed)
----------------------------------------------------------
clean               Standard CIFAR-10 test set.
corrupted           CIFAR-10-C (19 corruptions × 5 severities).
adversarial_linf    PGD20-Linf (ε = 4/255).
adversarial_l2      PGD20-L2 (ε = 0.25).
gradcam             Grad-CAM on clean vs adversarial mis-classified samples.
tsne_adv            T-SNE with clean and adversarial embeddings together.
transferability     Transfer adversarial examples from teacher to student.

Usage::

    cd corecode/HW3

    # Train AugMix model
    python main.py --training_mode augmix --mode train --experiment_name augmix

    # Train + evaluate on all eval modes
    python main.py --training_mode augmix --mode both \
        --experiment_name augmix \
        --eval_modes clean corrupted adversarial_linf adversarial_l2 gradcam tsne_adv

    # Evaluate a pretrained model
    python main.py --training_mode augmix --mode test \
        --experiment_name augmix \
        --eval_modes clean adversarial_linf
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import ssl
import sys
from pathlib import Path
from typing import List, Optional

_HW3_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW3_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HW3_DIR))

import torch
import torch.nn as nn
import numpy as np

from corecode.models.resnet_cifar import build_resnet18_scratch
from corecode.models.simple_cnn import SimpleCNN
from corecode.models.mobilenet_cifar import build_mobilenetv2_cifar
from gradcam import visualize_gradcam_pairs
from logging_utils import log_environment_info, save_config_snapshot, setup_experiment_logger
from parameters import Config, get_config
from plotting import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_corruption_accuracy,
    plot_corruption_heatmap,
    plot_loss_curve,
    plot_lr_curve,
    plot_robustness_comparison,
    plot_tsne,
    plot_tsne_adversarial,
)
from test import get_test_loader, run_test, run_test_adversarial, run_test_corrupted, run_test_transferability
from train import CIFAR10_CLASSES, run_training

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger("cs515")


def _normalized_input_bounds(config: Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid per-channel bounds for normalized CIFAR-10 tensors."""
    mean = torch.tensor(config.data.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.data.std, device=device).view(1, 3, 1, 1)
    return (0.0 - mean) / std, (1.0 - mean) / std


def _linf_attack_budget_in_normalized_space(
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw-pixel L-inf epsilon and step-size into normalized tensor space."""
    std = torch.tensor(config.data.std, device=device).view(1, 3, 1, 1)
    eps = torch.full((1, 3, 1, 1), config.attack.pgd_eps_linf, device=device) / std
    step_size = torch.full((1, 3, 1, 1), config.attack.pgd_step_size_linf, device=device) / std
    return eps, step_size


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_paths(config: Config) -> None:
    """Create checkpoint, log, and plot directories if absent.

    Args:
        config: Full experiment configuration.
    """
    os.makedirs(config.run.checkpoint_dir, exist_ok=True)
    os.makedirs(config.run.log_dir, exist_ok=True)
    os.makedirs(config.run.plot_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: Config) -> nn.Module:
    """Instantiate the student / solo model for the requested training mode.

    Mode → model mapping:

    +---------------+------------------------------------------------------+
    | training_mode | Model                                                |
    +===============+======================================================+
    | augmix        | ResNet-18 from random weights (modified stem)        |
    | kd_augmix     | SimpleCNN (lightweight student)                      |
    | kd_ls_augmix  | MobileNetV2 (efficient student)                      |
    +---------------+------------------------------------------------------+

    Args:
        config: Full experiment configuration.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If ``training_mode`` is unrecognised.
    """
    mode = config.train.training_mode
    num_classes = config.data.num_classes

    if mode == "augmix":
        return build_resnet18_scratch(num_classes=num_classes)
    if mode == "kd_augmix":
        return SimpleCNN(num_classes=num_classes, dropout=config.model.dropout)
    if mode == "kd_ls_augmix":
        return build_mobilenetv2_cifar(num_classes=num_classes)

    raise ValueError(f"Unsupported training_mode: {mode!r}")


def build_teacher(config: Config, device: torch.device) -> nn.Module:
    """Load and freeze the AugMix-trained ResNet-18 teacher for KD experiments.

    Args:
        config: Full experiment configuration.
        device: Device to map the teacher weights to.

    Returns:
        Frozen teacher model in evaluation mode.

    Raises:
        FileNotFoundError: If the teacher checkpoint does not exist.
    """
    checkpoint_path = Path(config.train.teacher_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}. "
            "Train an AugMix ResNet-18 first with --training_mode augmix, "
            "then pass its checkpoint via --teacher_checkpoint."
        )

    teacher = build_resnet18_scratch(num_classes=config.data.num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    logger.info("Teacher loaded from %s (all parameters frozen).", checkpoint_path)
    return teacher


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_config_summary(
    exp_logger: logging.Logger,
    config: Config,
    model: nn.Module,
    device: torch.device,
) -> None:
    """Log a concise experiment summary before training begins.

    Args:
        exp_logger: Configured logger.
        config: Full experiment configuration.
        model: Instantiated model.
        device: Target device.
    """
    exp_logger.info("=" * 70)
    exp_logger.info("CS515 HW3 - AugMix & Adversarial Robustness")
    exp_logger.info("=" * 70)
    exp_logger.info("Training mode    : %s", config.train.training_mode)
    exp_logger.info("Run mode         : %s", config.run.mode)
    exp_logger.info("Eval modes       : %s", ", ".join(config.run.eval_modes))
    exp_logger.info("Experiment       : %s", config.run.experiment_name)
    exp_logger.info("Seed             : %d", config.run.seed)
    exp_logger.info("Device           : %s", device)
    exp_logger.info("Epochs           : %d", config.train.epochs)
    exp_logger.info("Batch size       : %d", config.train.batch_size)
    exp_logger.info("Learning rate    : %g", config.train.learning_rate)
    exp_logger.info("Optimizer        : %s", config.train.optimizer)
    exp_logger.info("Scheduler        : %s", config.train.scheduler)
    exp_logger.info("Weight decay     : %g", config.train.weight_decay)
    exp_logger.info("AugMix JSD       : %s  (λ=%g)",
                    config.augmix.jsd_loss, config.augmix.jsd_lambda)
    exp_logger.info("AugMix severity  : %d  width=%d  depth=%d",
                    config.augmix.severity, config.augmix.width, config.augmix.depth)
    exp_logger.info("KD temperature   : %g", config.train.kd_temperature)
    exp_logger.info("KD alpha         : %g", config.train.kd_alpha)
    exp_logger.info("Teacher ckpt     : %s", config.train.teacher_checkpoint or "N/A")
    exp_logger.info("PGD Linf ε       : %.4f  step=%.4f  steps=%d",
                    config.attack.pgd_eps_linf,
                    config.attack.pgd_step_size_linf,
                    config.attack.pgd_steps)
    exp_logger.info("PGD L2 ε         : %.4f  step=%.4f",
                    config.attack.pgd_eps_l2, config.attack.pgd_step_size_l2)
    exp_logger.info("Checkpoint dir   : %s", config.run.checkpoint_dir)
    exp_logger.info("Log dir          : %s", config.run.log_dir)
    exp_logger.info("Trainable params : %s", f"{count_trainable_parameters(model):,}")
    exp_logger.info("=" * 70)
    exp_logger.info("Model:\n%s", model)
    exp_logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluations(
    model: nn.Module,
    config: Config,
    device: torch.device,
    plot_dir: str,
    exp: str,
    teacher: Optional[nn.Module] = None,
) -> None:
    """Dispatch all requested evaluation modes.

    Args:
        model: Trained (and loaded from checkpoint) model.
        config: Full experiment configuration.
        device: Target device.
        plot_dir: Directory where plots are saved.
        exp: Experiment name (used in file names).
        teacher: Loaded teacher model (required for ``transferability`` mode).
    """
    eval_modes = config.run.eval_modes
    test_loader = get_test_loader(config)

    # ── Clean accuracy ────────────────────────────────────────────────────
    if "clean" in eval_modes:
        clean_results = run_test(model=model, config=config, device=device)
        plot_confusion_matrix(
            clean_results["confusion_matrix"],
            os.path.join(plot_dir, f"{exp}_confusion_matrix.png"),
            class_names=list(CIFAR10_CLASSES),
        )
        plot_tsne(model, test_loader, device,
                  os.path.join(plot_dir, f"{exp}_tsne.png"))

    # ── CIFAR-10-C ────────────────────────────────────────────────────────
    if "corrupted" in eval_modes:
        corr_results = run_test_corrupted(model=model, config=config, device=device)
        per_corr = corr_results.get("per_corruption", {})
        if per_corr:
            plot_corruption_accuracy(
                per_corr,
                os.path.join(plot_dir, f"{exp}_corruption_accuracy.png"),
                title=f"CIFAR-10-C accuracy — {exp}",
            )
            plot_corruption_heatmap(
                per_corr,
                os.path.join(plot_dir, f"{exp}_corruption_heatmap.png"),
                title=f"CIFAR-10-C heatmap — {exp}",
            )

    # ── Adversarial Linf ──────────────────────────────────────────────────
    adv_linf_results = None
    if "adversarial_linf" in eval_modes:
        adv_linf_results = run_test_adversarial(
            model=model, config=config, device=device, norm="linf"
        )

    # ── Adversarial L2 ────────────────────────────────────────────────────
    adv_l2_results = None
    if "adversarial_l2" in eval_modes:
        adv_l2_results = run_test_adversarial(
            model=model, config=config, device=device, norm="l2"
        )

    # ── Grad-CAM (uses L-inf adversarial examples) ────────────────────────
    if "gradcam" in eval_modes:
        # Use Linf adversarial examples (generate if not already done)
        if adv_linf_results is None:
            adv_linf_results = run_test_adversarial(
                model=model, config=config, device=device, norm="linf"
            )

        mis_clean = adv_linf_results.get("_misclassified_clean", [])
        mis_adv = adv_linf_results.get("_misclassified_adv", [])
        mis_info = adv_linf_results.get("misclassified_indices", {})

        if mis_clean and mis_adv:
            logger.info("Generating Grad-CAM for %d misclassified sample(s)...",
                        len(mis_clean))
            try:
                visualize_gradcam_pairs(
                    model=model,
                    clean_images=torch.stack(mis_clean),
                    adv_images=torch.stack(mis_adv),
                    true_labels=mis_info.get("true_labels", []),
                    clean_preds=mis_info.get("clean_preds", []),
                    adv_preds=mis_info.get("adv_preds", []),
                    save_path=os.path.join(plot_dir, f"{exp}_gradcam.png"),
                )
                logger.info("Grad-CAM saved.")
            except Exception as exc:
                logger.warning("Grad-CAM generation failed: %s", exc)
        else:
            logger.warning("No misclassified adversarial samples found for Grad-CAM.")

    # ── Adversarial T-SNE ─────────────────────────────────────────────────
    if "tsne_adv" in eval_modes:
        if adv_linf_results is None:
            adv_linf_results = run_test_adversarial(
                model=model, config=config, device=device, norm="linf"
            )

        # Collect a batch of clean + adversarial images for T-SNE
        logger.info("Collecting images for adversarial T-SNE...")
        clean_batch: List[torch.Tensor] = []
        adv_batch: List[torch.Tensor] = []
        label_batch: List[torch.Tensor] = []

        from attacks import pgd_attack
        model.eval()
        n_collected = 0
        max_tsne = config.attack.tsne_max_samples
        attack_eps, attack_step_size = _linf_attack_budget_in_normalized_space(config, device)
        clamp_min, clamp_max = _normalized_input_bounds(config, device)

        for imgs, lbls in test_loader:
            if n_collected >= max_tsne:
                break
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            adv_imgs = pgd_attack(
                model=model, images=imgs, labels=lbls, norm="linf",
                eps=attack_eps,
                step_size=attack_step_size,
                steps=config.attack.pgd_steps,
                random_start=config.attack.pgd_random_start,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            clean_batch.append(imgs.cpu())
            adv_batch.append(adv_imgs.cpu())
            label_batch.append(lbls.cpu())
            n_collected += imgs.size(0)

        if clean_batch:
            clean_tensor = torch.cat(clean_batch, dim=0)[:max_tsne]
            adv_tensor = torch.cat(adv_batch, dim=0)[:max_tsne]
            label_tensor = torch.cat(label_batch, dim=0)[:max_tsne]

            plot_tsne_adversarial(
                model=model,
                clean_images=clean_tensor,
                adv_images=adv_tensor,
                labels=label_tensor,
                device=device,
                save_path=os.path.join(plot_dir, f"{exp}_tsne_adversarial.png"),
                max_samples=max_tsne,
            )

    # ── Adversarial transferability ───────────────────────────────────────
    if "transferability" in eval_modes:
        if teacher is None:
            logger.warning(
                "Skipping transferability evaluation — no teacher model provided. "
                "Pass the teacher via --teacher_checkpoint."
            )
        else:
            run_test_transferability(
                teacher_model=teacher,
                student_model=model,
                config=config,
                device=device,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point: parse config, build models, train and/or evaluate."""
    config = get_config()

    set_seed(config.run.seed)
    ensure_output_paths(config)

    exp_logger, log_path = setup_experiment_logger(config)
    config_snapshot_path = save_config_snapshot(config, log_path)

    try:
        device = torch.device(config.run.device)
        model = build_model(config).to(device)

        exp_logger.info("Log file         : %s", log_path)
        exp_logger.info("Config snapshot  : %s", config_snapshot_path)
        log_environment_info(exp_logger)
        log_config_summary(exp_logger, config, model, device)

        # Load teacher for KD modes
        teacher = None
        if config.train.training_mode in ("kd_augmix", "kd_ls_augmix"):
            teacher = build_teacher(config, device)

        # Also load teacher if transferability evaluation is requested
        if "transferability" in config.run.eval_modes and teacher is None:
            if config.train.teacher_checkpoint:
                teacher = build_teacher(config, device)
            else:
                exp_logger.warning(
                    "transferability eval requested but --teacher_checkpoint not set."
                )

        plot_dir = config.run.plot_dir
        exp = config.run.experiment_name

        # ── Train ─────────────────────────────────────────────────────────
        if config.run.mode in ("train", "both"):
            history = run_training(
                model=model, config=config, device=device, teacher=teacher
            )
            os.makedirs(plot_dir, exist_ok=True)
            plot_loss_curve(history, os.path.join(plot_dir, f"{exp}_loss.png"))
            plot_accuracy_curve(history, os.path.join(plot_dir, f"{exp}_accuracy.png"))
            plot_lr_curve(history, os.path.join(plot_dir, f"{exp}_lr.png"))

        # ── Test / evaluate ───────────────────────────────────────────────
        if config.run.mode in ("test", "both"):
            checkpoint_path = Path(config.run.save_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at: {checkpoint_path}. "
                    "Train the model first (--mode train or --mode both)."
                )

            state_dict = torch.load(config.run.save_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            os.makedirs(plot_dir, exist_ok=True)
            run_evaluations(
                model=model,
                config=config,
                device=device,
                plot_dir=plot_dir,
                exp=exp,
                teacher=teacher,
            )

        exp_logger.info("Run completed successfully.")

    except Exception as exc:
        exp_logger.exception("Run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
