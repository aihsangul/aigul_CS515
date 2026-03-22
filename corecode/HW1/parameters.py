from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch

# Resolve the repository root: corecode/HW1/parameters.py -> repo root (two levels up)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DATA_ROOT = _REPO_ROOT / "data"


@dataclass
class DataConfig:
    """Configuration related to dataset loading and preprocessing."""
    dataset: str = "mnist"
    data_dir: str = str(_DATA_ROOT)
    val_split: float = 0.1
    num_workers: int = 2
    pin_memory: bool = False

    input_size: int = 784
    num_classes: int = 10
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)


@dataclass
class ModelConfig:
    """Configuration related to the MLP architecture."""
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10

    activation: str = "relu"
    dropout: float = 0.3
    use_batch_norm: bool = False


@dataclass
class TrainConfig:
    """Configuration related to training, validation, scheduler, and regularization."""
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3

    optimizer: str = "adam"
    momentum: float = 0.9

    scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    early_stopping_patience: int = 5
    min_delta: float = 0.0

    regularizer: str = "none"
    reg_lambda: float = 0.0
    weight_decay: float = 0.0


@dataclass
class RunConfig:
    """Configuration related to execution, logging, saving, and project structure."""
    mode: str = "both"
    seed: int = 42
    device: str = "auto"

    project_name: str = "HW1"
    experiment_name: str = "mlp_mnist"

    checkpoint_root: str = str(_RESULTS_ROOT / "checkpoints")
    log_root: str = str(_RESULTS_ROOT / "logs")
    plot_root: str = str(_RESULTS_ROOT / "plots")

    checkpoint_dir: str = str(_RESULTS_ROOT / "checkpoints" / "HW1" / "mlp_mnist")
    log_dir: str = str(_RESULTS_ROOT / "logs" / "HW1" / "mlp_mnist")
    plot_dir: str = str(_RESULTS_ROOT / "plots" / "HW1" / "mlp_mnist")

    save_path: str = str(_RESULTS_ROOT / "checkpoints" / "HW1" / "mlp_mnist" / "mlp_mnist_best_model.pt")
    log_interval: int = 100


@dataclass
class Config:
    """Top-level experiment configuration composed of individual dataclasses."""
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig


def _resolve_device(device: str) -> str:
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
    if args.val_split <= 0.0 or args.val_split >= 1.0:
        raise ValueError("val_split must be in the range (0, 1).")

    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("dropout must be in the range [0, 1).")

    if len(args.hidden_sizes) == 0:
        raise ValueError("hidden_sizes must contain at least one hidden layer size.")

    if any(size <= 0 for size in args.hidden_sizes):
        raise ValueError("All hidden layer sizes must be positive integers.")

    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if args.lr <= 0.0:
        raise ValueError("learning rate must be positive.")

    if args.reg_lambda < 0.0:
        raise ValueError("reg_lambda must be non-negative.")

    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")

    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative.")

    if args.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be non-negative.")

    if args.log_interval <= 0:
        raise ValueError("log_interval must be a positive integer.")

    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    if args.step_size <= 0:
        raise ValueError("step_size must be a positive integer.")

    if args.min_lr < 0.0:
        raise ValueError("min_lr must be non-negative.")

    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("gamma must be in the range (0, 1].")

    if not (0.0 < args.scheduler_factor <= 1.0):
        raise ValueError("scheduler_factor must be in the range (0, 1].")

    if args.momentum < 0.0 or args.momentum >= 1.0:
        raise ValueError("momentum must be in the range [0, 1).")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CS515 HW1a - MNIST classification with configurable MLP"
    )

    # Run / execution arguments
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, cuda, or mps")

    parser.add_argument("--project_name", type=str, default="HW1")
    parser.add_argument("--experiment_name", type=str, default="mlp_mnist")

    parser.add_argument("--checkpoint_root", type=str, default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--log_root", type=str, default=str(_RESULTS_ROOT / "logs"))
    parser.add_argument("--plot_root", type=str, default=str(_RESULTS_ROOT / "plots"))

    parser.add_argument("--log_interval", type=int, default=100)

    # Data arguments
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--data_dir", type=str, default=str(_DATA_ROOT))
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable pin_memory in DataLoader")

    # Model arguments
    parser.add_argument("--hidden_sizes", type=int, nargs="+",
                        default=[512, 256, 128],
                        help="Hidden layer sizes, e.g. --hidden_sizes 512 256 128")
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_batch_norm", action="store_true",
                        help="Use BatchNorm1d before activation")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--scheduler", choices=["none", "step", "plateau", "cosine"],
                        default="none")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument("--regularizer", choices=["none", "l1", "l2"], default="none")
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Usually used as L2 regularization in optimizer")

    return parser


def get_config() -> Config:
    parser = get_parser()
    args = parser.parse_args()
    _validate_args(args)

    input_size = 784
    num_classes = 10
    mean = (0.1307,)
    std = (0.3081,)

    resolved_device = _resolve_device(args.device)

    weight_decay = args.weight_decay
    if args.regularizer == "l2" and weight_decay == 0.0:
        weight_decay = args.reg_lambda

    checkpoint_dir = Path(args.checkpoint_root) / args.project_name / args.experiment_name
    log_dir = Path(args.log_root) / args.project_name / args.experiment_name
    plot_dir = Path(args.plot_root) / args.project_name / args.experiment_name
    save_path = checkpoint_dir / f"{args.experiment_name}_best_model.pt"

    data_cfg = DataConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        val_split=args.val_split,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        input_size=input_size,
        num_classes=num_classes,
        mean=mean,
        std=std,
    )

    model_cfg = ModelConfig(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=num_classes,
        activation=args.activation,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        regularizer=args.regularizer,
        reg_lambda=args.reg_lambda,
        weight_decay=weight_decay,
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

    return Config(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        run=run_cfg,
    )