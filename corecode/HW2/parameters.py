from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

# Resolve paths relative to the repository root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DATA_ROOT = _REPO_ROOT / "data"

# All supported training modes for HW2
TRAINING_MODES = [
    "transfer_resize",  # Option A-1: resize to 224, freeze backbone, train FC only
    "transfer_modify",  # Option A-2: modify conv1/maxpool, fine-tune all layers
    "scratch",          # Part B: ResNet-18 from scratch, standard CE loss
    "scratch_ls",       # Part B: ResNet-18 from scratch, CE with label smoothing
    "kd",               # Part B: SimpleCNN student + ResNet teacher (KD loss)
    "kd_ls",            # Part B: MobileNetV2 student + ResNet teacher (modified KD)
]


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    dataset: str = "cifar10"
    data_dir: str = str(_DATA_ROOT)
    val_split: float = 0.1
    num_workers: int = 2
    pin_memory: bool = False
    num_classes: int = 10

    # CIFAR-10 channel statistics
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2470, 0.2435, 0.2616)

    # ImageNet channel statistics (used when images are resized to 224×224)
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Configuration for the student model architecture."""

    num_classes: int = 10
    # Dropout probability used in the SimpleCNN classifier head
    dropout: float = 0.5


@dataclass
class TrainConfig:
    """Configuration for training, validation, scheduler, and loss functions."""

    training_mode: str = "scratch"
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3

    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 1e-4

    scheduler: str = "cosine"
    step_size: int = 20
    gamma: float = 0.1
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    early_stopping_patience: int = 10
    min_delta: float = 0.0

    # Label smoothing — active only in ``scratch_ls`` mode
    label_smoothing: float = 0.1

    # Knowledge distillation — active in ``kd`` and ``kd_ls`` modes
    teacher_checkpoint: str = ""
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5  # weight for the hard CE term; (1-alpha) for soft term


@dataclass
class RunConfig:
    """Configuration for execution, logging, checkpointing, and output paths."""

    mode: str = "both"  # train | test | both
    seed: int = 42
    device: str = "auto"

    project_name: str = "HW2"
    experiment_name: str = "resnet_cifar10"

    checkpoint_root: str = str(_RESULTS_ROOT / "checkpoints")
    log_root: str = str(_RESULTS_ROOT / "logs")
    plot_root: str = str(_RESULTS_ROOT / "plots")

    checkpoint_dir: str = str(_RESULTS_ROOT / "checkpoints" / "HW2" / "resnet_cifar10")
    log_dir: str = str(_RESULTS_ROOT / "logs" / "HW2" / "resnet_cifar10")
    plot_dir: str = str(_RESULTS_ROOT / "plots" / "HW2" / "resnet_cifar10")

    save_path: str = str(
        _RESULTS_ROOT / "checkpoints" / "HW2" / "resnet_cifar10" / "resnet_cifar10_best_model.pt"
    )
    log_interval: int = 50


@dataclass
class Config:
    """Top-level experiment configuration composed of individual dataclasses."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig


# Helpers
def _resolve_device(device: str) -> str:
    """Resolve the device string, auto-detecting CUDA or MPS when available.

    Args:
        device: One of "auto", "cpu", "cuda", or "mps".

    Returns:
        Resolved device string.

    Raises:
        ValueError: If an unsupported device string is supplied.
    """
    device = device.lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device in {"cpu", "cuda", "mps"}:
        return device

    raise ValueError("device must be one of: auto, cpu, cuda, mps")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments and raise on invalid combinations.

    Args:
        args: Parsed argument namespace.

    Raises:
        ValueError: If any argument value is invalid.
    """
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("val_split must be in (0, 1).")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if args.lr <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    if args.kd_temperature <= 0.0:
        raise ValueError("kd_temperature must be positive.")
    if not 0.0 <= args.kd_alpha <= 1.0:
        raise ValueError("kd_alpha must be in [0, 1].")
    if args.training_mode in ("kd", "kd_ls") and not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint must be set for kd and kd_ls modes.")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be a positive integer.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.step_size <= 0:
        raise ValueError("step_size must be a positive integer.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not 0.0 < args.scheduler_factor <= 1.0:
        raise ValueError("scheduler_factor must be in (0, 1].")
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("momentum must be in [0, 1).")


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the HW2 training script.

    Returns:
        Configured :class:'argparse.ArgumentParser'.
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW2 - Transfer Learning and Knowledge Distillation on CIFAR-10"
    )

    # Run / execution
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="Run training, testing, or both.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--project_name", type=str, default="HW2")
    parser.add_argument("--experiment_name", type=str, default="resnet_cifar10")
    parser.add_argument("--checkpoint_root", type=str,
                        default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--log_root", type=str,
                        default=str(_RESULTS_ROOT / "logs"))
    parser.add_argument("--plot_root", type=str,
                        default=str(_RESULTS_ROOT / "plots"))
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Print training progress every N batches.")

    # Data
    parser.add_argument("--data_dir", type=str, default=str(_DATA_ROOT))
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")

    # Model
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability for the SimpleCNN classifier head.")

    # Training
    parser.add_argument(
        "--training_mode",
        choices=TRAINING_MODES,
        default="scratch",
        help=(
            "Experiment type: "
            "transfer_resize (Option A-1), transfer_modify (Option A-2), "
            "scratch, scratch_ls, kd, kd_ls."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--scheduler",
                        choices=["none", "step", "plateau", "cosine"],
                        default="cosine")
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon (used in scratch_ls mode).")

    # Knowledge distillation
    parser.add_argument("--teacher_checkpoint", type=str, default="",
                        help="Path to the teacher model checkpoint (required for kd/kd_ls).")
    parser.add_argument("--kd_temperature", type=float, default=4.0,
                        help="Temperature T for softened distributions in KD.")
    parser.add_argument("--kd_alpha", type=float, default=0.5,
                        help="Weight for the hard CE term in KD (soft term = 1 - alpha).")

    return parser


def get_config() -> Config:
    """Parse CLI arguments and build the full :class:`Config` object.

    Returns:
        Populated :class:`Config` dataclass.
    """
    parser = get_parser()
    args = parser.parse_args()
    _validate_args(args)

    resolved_device = _resolve_device(args.device)

    checkpoint_dir = Path(args.checkpoint_root) / args.project_name / args.experiment_name
    log_dir = Path(args.log_root) / args.project_name / args.experiment_name
    plot_dir = Path(args.plot_root) / args.project_name / args.experiment_name
    save_path = checkpoint_dir / f"{args.experiment_name}_best_model.pt"

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        val_split=args.val_split,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_cfg = ModelConfig(
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        training_mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        label_smoothing=args.label_smoothing,
        teacher_checkpoint=args.teacher_checkpoint,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
    )

    run_cfg = RunConfig(
        mode=args.mode,
        seed=args.seed,
        device=resolved_device,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
        plot_root=args.plot_root,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        plot_dir=str(plot_dir),
        save_path=str(save_path),
        log_interval=args.log_interval,
    )

    return Config(data=data_cfg, model=model_cfg, train=train_cfg, run=run_cfg)
