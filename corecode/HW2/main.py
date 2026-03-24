from __future__ import annotations

import logging
import os

# Must be set before any OpenMP-linked library (numpy, torch) is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import ssl
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so ``corecode`` is importable,
# and that the HW2 directory is on sys.path for local module imports.
_HW2_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HW2_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HW2_DIR))

# Import torch before numpy to control DLL loading order on Windows.
import torch
import torch.nn as nn
import numpy as np

from corecode.models.resnet_cifar import (
    build_resnet18_scratch,
    build_resnet18_transfer_modify,
    build_resnet18_transfer_resize,
)
from corecode.models.simple_cnn import SimpleCNN
from corecode.models.mobilenet_cifar import build_mobilenetv2_cifar
from logging_utils import log_environment_info, save_config_snapshot, setup_experiment_logger
from parameters import Config, get_config
from plotting import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_lr_curve,
    plot_tsne,
)
from test import get_test_loader, run_test
from train import CIFAR10_CLASSES, run_training

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger("cs515")


# Reproducibility

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

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


# Output paths

def ensure_output_paths(config: Config) -> None:
    """Create checkpoint, log, and plot directories if they do not yet exist.

    Args:
        config: Full experiment configuration.
    """
    os.makedirs(config.run.checkpoint_dir, exist_ok=True)
    os.makedirs(config.run.log_dir, exist_ok=True)
    os.makedirs(config.run.plot_dir, exist_ok=True)


# Model factory

def build_model(config: Config) -> nn.Module:
    """Instantiate the student (or solo) model for the requested training mode.

    Mode -> student model mapping:

    +-----------------+-----------------------------------------------------+
    | training_mode   | Model                                               |
    +=================+=====================================================+
    | transfer_resize | ResNet-18 (pre-trained, frozen backbone)            |
    | transfer_modify | ResNet-18 (pre-trained, modified stem, all layers)  |
    | scratch         | ResNet-18 (random weights, adapted stem)            |
    | scratch_ls      | ResNet-18 (random weights, adapted stem)            |
    | kd              | SimpleCNN (student)                                 |
    | kd_ls           | MobileNetV2 (student)                               |
    +-----------------+-----------------------------------------------------+

    Args:
        config: Full experiment configuration.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If "training_mode" is unrecognized.
    """
    mode = config.train.training_mode
    num_classes = config.data.num_classes

    if mode == "transfer_resize":
        return build_resnet18_transfer_resize(num_classes=num_classes, freeze_backbone=True)
    if mode == "transfer_modify":
        return build_resnet18_transfer_modify(num_classes=num_classes)
    if mode in ("scratch", "scratch_ls"):
        return build_resnet18_scratch(num_classes=num_classes)
    if mode == "kd":
        return SimpleCNN(num_classes=num_classes, dropout=config.model.dropout)
    if mode == "kd_ls":
        return build_mobilenetv2_cifar(num_classes=num_classes)

    raise ValueError(f"Unsupported training_mode: {mode!r}")


def build_teacher(config: Config, device: torch.device) -> nn.Module:
    """Load and freeze the teacher ResNet-18 for KD / KD-LS experiments.

    The teacher is always a ResNet-18 trained from scratch on CIFAR-10
    (adapted stem), loaded from the path specified by
    ``config.train.teacher_checkpoint``.

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
            "Train a ResNet-18 with --training_mode scratch or scratch_ls first, "
            "then pass its best-model path via --teacher_checkpoint."
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
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Logging summary
# ─────────────────────────────────────────────────────────────────────────────

def log_config_summary(
    logger: logging.Logger,
    config: Config,
    model: nn.Module,
    device: torch.device,
) -> None:
    """Log a concise experiment summary before training begins.

    Args:
        logger: Configured logger.
        config: Full experiment configuration.
        model: Instantiated student / solo model.
        device: Target device.
    """
    logger.info("=" * 70)
    logger.info("CS515 HW2 - Transfer Learning & Knowledge Distillation")
    logger.info("=" * 70)
    logger.info("Training mode    : %s", config.train.training_mode)
    logger.info("Run mode         : %s", config.run.mode)
    logger.info("Experiment       : %s", config.run.experiment_name)
    logger.info("Seed             : %d", config.run.seed)
    logger.info("Device           : %s", device)
    logger.info("Dataset          : %s", config.data.dataset)
    logger.info("Num classes      : %d", config.data.num_classes)
    logger.info("Epochs           : %d", config.train.epochs)
    logger.info("Batch size       : %d", config.train.batch_size)
    logger.info("Learning rate    : %g", config.train.learning_rate)
    logger.info("Optimizer        : %s", config.train.optimizer)
    logger.info("Scheduler        : %s", config.train.scheduler)
    logger.info("Weight decay     : %g", config.train.weight_decay)
    logger.info("Label smoothing  : %g", config.train.label_smoothing)
    logger.info("KD temperature   : %g", config.train.kd_temperature)
    logger.info("KD alpha         : %g", config.train.kd_alpha)
    logger.info("Teacher ckpt     : %s", config.train.teacher_checkpoint or "N/A")
    logger.info("Checkpoint dir   : %s", config.run.checkpoint_dir)
    logger.info("Log dir          : %s", config.run.log_dir)
    logger.info("Trainable params : %s", f"{count_trainable_parameters(model):,}")
    logger.info("=" * 70)
    logger.info("Model:\n%s", model)
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point: parse config, build models, train and/or test."""
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
        if config.train.training_mode in ("kd", "kd_ls"):
            teacher = build_teacher(config, device)

        plot_dir = config.run.plot_dir
        exp = config.run.experiment_name

        if config.run.mode in ("train", "both"):
            history = run_training(model=model, config=config, device=device, teacher=teacher)
            os.makedirs(plot_dir, exist_ok=True)
            plot_loss_curve(history, os.path.join(plot_dir, f"{exp}_loss.png"))
            plot_accuracy_curve(history, os.path.join(plot_dir, f"{exp}_accuracy.png"))
            plot_lr_curve(history, os.path.join(plot_dir, f"{exp}_lr.png"))

        if config.run.mode in ("test", "both"):
            if (
                config.run.mode == "test"
                and not Path(config.run.save_path).exists()
            ):
                raise FileNotFoundError(
                    f"Checkpoint not found at: {config.run.save_path}. "
                    "Train the model first or provide a valid checkpoint path."
                )
            results = run_test(model=model, config=config, device=device)
            os.makedirs(plot_dir, exist_ok=True)
            plot_confusion_matrix(
                results["confusion_matrix"],
                os.path.join(plot_dir, f"{exp}_confusion_matrix.png"),
                class_names=list(CIFAR10_CLASSES),
            )
            test_loader = get_test_loader(config)
            plot_tsne(model, test_loader, device, os.path.join(plot_dir, f"{exp}_tsne.png"))

        exp_logger.info("Run completed successfully.")

    except Exception as exc:
        exp_logger.exception("Run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
