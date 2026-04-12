from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

# Resolve paths relative to the repository root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DATA_ROOT = _REPO_ROOT / "data"

# ── Training modes ────────────────────────────────────────────────────────────
# New modes specific to HW3 (AugMix training + KD with AugMix teacher)
TRAINING_MODES = [
    "augmix",       # ResNet-18 from scratch trained with AugMix + JSD consistency loss
    "kd_augmix",    # SimpleCNN student, Hinton KD loss, AugMix-trained ResNet-18 teacher
    "kd_ls_augmix", # MobileNetV2 student, modified KD loss, AugMix-trained ResNet-18 teacher
]

# Evaluation modes (can be combined via --eval_modes)
EVAL_MODES = [
    "clean",            # Standard CIFAR-10 test set
    "corrupted",        # CIFAR-10-C (all 19 corruptions, 5 severities)
    "adversarial_linf", # PGD20 with L-infinity norm (ε = 4/255)
    "adversarial_l2",   # PGD20 with L2 norm (ε = 0.25)
    "gradcam",          # Grad-CAM on adversarial vs clean misclassified samples
    "tsne_adv",         # T-SNE with adversarial samples visualised
    "transferability",  # Adversarial transferability: teacher → student
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

    # CIFAR-10-C data directory (contains .npy files for each corruption type)
    cifar10c_dir: str = str(_DATA_ROOT / "CIFAR-10-C")


@dataclass
class ModelConfig:
    """Configuration for the student model architecture."""

    num_classes: int = 10
    # Dropout probability for the SimpleCNN classifier head (kd_augmix mode)
    dropout: float = 0.5


@dataclass
class AugMixConfig:
    """Configuration for the AugMix augmentation framework.

    Reference: Hendrycks et al., "AugMix: A Simple Method to Improve Robustness
    and Uncertainty under Data Shift", ICLR 2020.
    """

    severity: int = 3
    """Augmentation severity level (1–10). Higher values apply stronger ops."""

    width: int = 3
    """Number of independent augmentation chains to mix."""

    depth: int = -1
    """Number of ops per chain. -1 means sample uniformly from {1, 2, 3}."""

    alpha: float = 1.0
    """Concentration parameter for the Dirichlet / Beta mixing coefficients."""

    jsd_loss: bool = True
    """If True, add JSD consistency loss across three augmented views."""

    jsd_lambda: float = 12.0
    """Weight λ for the JSD consistency term in the total loss."""


@dataclass
class AttackConfig:
    """Configuration for PGD adversarial attacks.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to
    Adversarial Attacks", ICLR 2018.
    """

    pgd_steps: int = 20
    """Number of PGD iterations."""

    # L-infinity attack
    pgd_eps_linf: float = 4.0 / 255.0
    """L-infinity perturbation budget ε."""

    pgd_step_size_linf: float = 1.0 / 255.0
    """L-infinity per-step size α (2/ε recommended; default 1/255)."""

    # L2 attack
    pgd_eps_l2: float = 0.25
    """L2 perturbation budget ε."""

    pgd_step_size_l2: float = 0.05
    """L2 per-step size α."""

    pgd_random_start: bool = True
    """Initialise PGD from a random perturbation inside the ε-ball."""

    # Samples for Grad-CAM + adversarial T-SNE visualisation
    num_gradcam_samples: int = 2
    """Number of misclassified adversarial samples to visualise with Grad-CAM."""

    tsne_max_samples: int = 1000
    """Maximum total samples (clean + adversarial) for adversarial T-SNE."""


@dataclass
class TrainConfig:
    """Configuration for training, validation, scheduler, and loss functions."""

    training_mode: str = "augmix"
    epochs: int = 100
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

    early_stopping_patience: int = 15
    min_delta: float = 0.0

    # Label smoothing (not used in HW3 modes but kept for compatibility)
    label_smoothing: float = 0.0

    # Knowledge distillation
    teacher_checkpoint: str = ""
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5


@dataclass
class RunConfig:
    """Configuration for execution, logging, checkpointing, and output paths."""

    mode: str = "both"       # train | test | both
    eval_modes: List[str] = None  # type: ignore[assignment]
    seed: int = 42
    device: str = "auto"

    project_name: str = "HW3"
    experiment_name: str = "augmix"

    checkpoint_root: str = str(_RESULTS_ROOT / "checkpoints")
    log_root: str = str(_RESULTS_ROOT / "logs")
    plot_root: str = str(_RESULTS_ROOT / "plots")

    checkpoint_dir: str = str(_RESULTS_ROOT / "checkpoints" / "HW3" / "augmix")
    log_dir: str = str(_RESULTS_ROOT / "logs" / "HW3" / "augmix")
    plot_dir: str = str(_RESULTS_ROOT / "plots" / "HW3" / "augmix")

    save_path: str = str(
        _RESULTS_ROOT / "checkpoints" / "HW3" / "augmix" / "augmix_best_model.pt"
    )
    log_interval: int = 50

    def __post_init__(self) -> None:
        """Set default eval_modes list after dataclass initialisation."""
        if self.eval_modes is None:
            self.eval_modes = ["clean"]


@dataclass
class Config:
    """Top-level experiment configuration composed of individual dataclasses."""

    data: DataConfig
    model: ModelConfig
    augmix: AugMixConfig
    attack: AttackConfig
    train: TrainConfig
    run: RunConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_device(device: str) -> str:
    """Resolve the device string, auto-detecting CUDA or MPS when available.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

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
    if args.kd_temperature <= 0.0:
        raise ValueError("kd_temperature must be positive.")
    if not 0.0 <= args.kd_alpha <= 1.0:
        raise ValueError("kd_alpha must be in [0, 1].")
    if args.training_mode in ("kd_augmix", "kd_ls_augmix") and not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required for kd_augmix and kd_ls_augmix modes.")
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
    if not 1 <= args.augmix_severity <= 10:
        raise ValueError("augmix_severity must be in [1, 10].")
    if args.augmix_width < 1:
        raise ValueError("augmix_width must be >= 1.")
    if args.pgd_steps < 1:
        raise ValueError("pgd_steps must be >= 1.")
    if args.pgd_eps_linf <= 0.0:
        raise ValueError("pgd_eps_linf must be positive.")
    if args.pgd_eps_l2 <= 0.0:
        raise ValueError("pgd_eps_l2 must be positive.")
    for em in args.eval_modes:
        if em not in EVAL_MODES:
            raise ValueError(f"Unknown eval_mode: {em!r}. Choose from {EVAL_MODES}.")


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the HW3 training/evaluation script.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "CS515 HW3 - Data Augmentation (AugMix) and Adversarial Robustness on CIFAR-10"
        )
    )

    # ── Run / execution ───────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="Run training, testing, or both.")
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=["clean"],
        help=(
            "One or more evaluation modes: "
            "clean | corrupted | adversarial_linf | adversarial_l2 | "
            "gradcam | tsne_adv | transferability"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--project_name", type=str, default="HW3")
    parser.add_argument("--experiment_name", type=str, default="augmix")
    parser.add_argument("--checkpoint_root", type=str,
                        default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--log_root", type=str,
                        default=str(_RESULTS_ROOT / "logs"))
    parser.add_argument("--plot_root", type=str,
                        default=str(_RESULTS_ROOT / "plots"))
    parser.add_argument("--log_interval", type=int, default=50)

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default=str(_DATA_ROOT))
    parser.add_argument("--cifar10c_dir", type=str,
                        default=str(_DATA_ROOT / "CIFAR-10-C"),
                        help="Directory containing CIFAR-10-C .npy files.")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability for the SimpleCNN classifier head.")

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--training_mode",
        choices=TRAINING_MODES,
        default="augmix",
        help=(
            "Experiment type: augmix (ResNet-18 with AugMix), "
            "kd_augmix (SimpleCNN KD), kd_ls_augmix (MobileNetV2 KD-LS)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
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

    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # ── AugMix ────────────────────────────────────────────────────────────
    parser.add_argument("--augmix_severity", type=int, default=3,
                        help="Augmentation severity level (1–10).")
    parser.add_argument("--augmix_width", type=int, default=3,
                        help="Number of augmentation chains.")
    parser.add_argument("--augmix_depth", type=int, default=-1,
                        help="Ops per chain; -1 = random from {1,2,3}.")
    parser.add_argument("--augmix_alpha", type=float, default=1.0,
                        help="Dirichlet/Beta concentration parameter.")
    parser.add_argument("--no_augmix_jsd", action="store_true",
                        help="Disable the JSD consistency loss term.")
    parser.add_argument("--augmix_jsd_lambda", type=float, default=12.0,
                        help="Weight λ for the JSD consistency term.")

    # ── Knowledge distillation ────────────────────────────────────────────
    parser.add_argument("--teacher_checkpoint", type=str, default="",
                        help="Path to the AugMix-trained teacher checkpoint.")
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)

    # ── PGD attack ────────────────────────────────────────────────────────
    parser.add_argument("--pgd_steps", type=int, default=20,
                        help="Number of PGD iterations.")
    parser.add_argument("--pgd_eps_linf", type=float, default=4.0 / 255.0,
                        help="L-inf PGD epsilon (default 4/255).")
    parser.add_argument("--pgd_step_size_linf", type=float, default=1.0 / 255.0,
                        help="L-inf PGD step size (default 1/255).")
    parser.add_argument("--pgd_eps_l2", type=float, default=0.25,
                        help="L2 PGD epsilon.")
    parser.add_argument("--pgd_step_size_l2", type=float, default=0.05,
                        help="L2 PGD step size.")
    parser.add_argument("--no_pgd_random_start", action="store_true",
                        help="Disable random initialisation inside the ε-ball.")
    parser.add_argument("--num_gradcam_samples", type=int, default=2,
                        help="Number of misclassified adversarial samples for Grad-CAM.")
    parser.add_argument("--tsne_max_samples", type=int, default=1000,
                        help="Max samples for adversarial T-SNE plot.")

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
        cifar10c_dir=args.cifar10c_dir,
    )

    model_cfg = ModelConfig(dropout=args.dropout)

    augmix_cfg = AugMixConfig(
        severity=args.augmix_severity,
        width=args.augmix_width,
        depth=args.augmix_depth,
        alpha=args.augmix_alpha,
        jsd_loss=not args.no_augmix_jsd,
        jsd_lambda=args.augmix_jsd_lambda,
    )

    attack_cfg = AttackConfig(
        pgd_steps=args.pgd_steps,
        pgd_eps_linf=args.pgd_eps_linf,
        pgd_step_size_linf=args.pgd_step_size_linf,
        pgd_eps_l2=args.pgd_eps_l2,
        pgd_step_size_l2=args.pgd_step_size_l2,
        pgd_random_start=not args.no_pgd_random_start,
        num_gradcam_samples=args.num_gradcam_samples,
        tsne_max_samples=args.tsne_max_samples,
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
        teacher_checkpoint=args.teacher_checkpoint,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
    )

    run_cfg = RunConfig(
        mode=args.mode,
        eval_modes=args.eval_modes,
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
        augmix=augmix_cfg,
        attack=attack_cfg,
        train=train_cfg,
        run=run_cfg,
    )
