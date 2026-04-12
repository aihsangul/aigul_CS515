"""Training routines for HW3 (AugMix and knowledge distillation with AugMix teacher).

Supported training modes
------------------------
augmix
    ResNet-18 from random initialisation trained with the AugMix augmentation
    framework.  When ``config.augmix.jsd_loss`` is ``True``, each training
    step processes three views of every batch (clean, aug1, aug2) and adds a
    Jensen-Shannon divergence consistency term to the cross-entropy loss.

kd_augmix
    SimpleCNN student trained with Hinton-style knowledge distillation, using
    an AugMix-trained ResNet-18 as the (frozen) teacher.

kd_ls_augmix
    MobileNetV2 student trained with the modified KD loss (teacher probability
    for the true class only), using the same AugMix-trained teacher.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from augmix import AugMixDataset, compute_jsd_loss
from parameters import Config

logger = logging.getLogger("cs515")

# CIFAR-10 class names
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data transforms and loaders
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_transform(config: Config) -> transforms.Normalize:
    """Return the CIFAR-10 normalisation transform.

    Args:
        config: Full experiment configuration.

    Returns:
        Configured :class:`~torchvision.transforms.Normalize` transform.
    """
    return transforms.Normalize(config.data.mean, config.data.std)


def get_val_transforms(config: Config) -> transforms.Compose:
    """Build validation / test-time transforms (no augmentation).

    Args:
        config: Full experiment configuration.

    Returns:
        Composed evaluation transform.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        _normalize_transform(config),
    ])


def get_train_val_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    For the ``augmix`` training mode, the training set is wrapped in
    :class:`~augmix.AugMixDataset` which returns three views per sample.
    All other modes use standard CIFAR-10 augmentation.

    Args:
        config: Full experiment configuration.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    val_transform = get_val_transforms(config)
    normalize = _normalize_transform(config)

    # Standard train augmentation (for non-AugMix modes and validation)
    standard_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if config.train.training_mode == "augmix":
        # Load raw PIL images for AugMix (no ToTensor at this stage)
        raw_transform = transforms.Compose([])  # identity — keep PIL
        full_train_raw = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            download=True,
            transform=raw_transform,
        )
        full_val = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            download=False,
            transform=val_transform,
        )

        total_size = len(full_train_raw)
        val_size = int(total_size * config.data.val_split)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(config.run.seed)
        train_indices, val_indices = random_split(
            range(total_size), [train_size, val_size], generator=generator
        )

        train_raw_subset = torch.utils.data.Subset(full_train_raw, train_indices.indices)
        val_dataset = torch.utils.data.Subset(full_val, val_indices.indices)

        # Preprocess applied after AugMix (ToTensor + normalise only)
        preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = AugMixDataset(
            dataset=train_raw_subset,
            preprocess=preprocess,
            severity=config.augmix.severity,
            width=config.augmix.width,
            depth=config.augmix.depth,
            alpha=config.augmix.alpha,
            jsd_loss=config.augmix.jsd_loss,
        )

    else:
        # Standard training for KD modes
        full_train = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            download=True,
            transform=standard_train_transform,
        )
        full_val = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            download=False,
            transform=val_transform,
        )

        total_size = len(full_train)
        val_size = int(total_size * config.data.val_split)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(config.run.seed)
        train_indices, val_indices = random_split(
            range(total_size), [train_size, val_size], generator=generator
        )

        train_dataset = torch.utils.data.Subset(full_train, train_indices.indices)
        val_dataset = torch.utils.data.Subset(full_val, val_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser and scheduler builders
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(params: Any, config: Config) -> optim.Optimizer:
    """Build an optimiser over the given parameter iterable.

    Args:
        params: Iterable of parameters or parameter groups.
        config: Full experiment configuration.

    Returns:
        Configured optimiser.

    Raises:
        ValueError: If the requested optimiser is unsupported.
    """
    if config.train.optimizer == "adam":
        return optim.Adam(
            params,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    if config.train.optimizer == "sgd":
        return optim.SGD(
            params,
            lr=config.train.learning_rate,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.train.optimizer!r}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Config,
) -> Optional[Any]:
    """Build a learning-rate scheduler.

    Args:
        optimizer: Optimiser instance.
        config: Full experiment configuration.

    Returns:
        Scheduler instance, or ``None`` when ``scheduler == "none"``.

    Raises:
        ValueError: If the requested scheduler is unsupported.
    """
    if config.train.scheduler == "none":
        return None
    if config.train.scheduler == "step":
        return StepLR(optimizer, step_size=config.train.step_size, gamma=config.train.gamma)
    if config.train.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience,
            min_lr=config.train.min_lr,
        )
    if config.train.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {config.train.scheduler!r}")


def get_current_lr(optimizer: optim.Optimizer) -> float:
    """Return the current learning rate from the first parameter group.

    Args:
        optimizer: Optimiser instance.

    Returns:
        Current learning rate as a float.
    """
    return optimizer.param_groups[0]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions (reused from HW2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Compute the Hinton-style knowledge distillation loss.

    Total loss = alpha * CE(student, hard_labels)
               + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))

    Args:
        student_logits: Raw logits from the student model, shape (B, C).
        teacher_logits: Raw logits from the teacher model (detached), shape (B, C).
        labels: Ground-truth class indices, shape (B,).
        temperature: Softening temperature T > 1.
        alpha: Weight for the hard CE term; (1 - alpha) weights the soft term.

    Returns:
        Scalar loss tensor.
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = (
        F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    )

    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def compute_kd_ls_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    num_classes: int,
) -> torch.Tensor:
    """Compute the modified KD loss combining label smoothing with teacher guidance.

    The teacher probability for the true class replaces the hard label; the
    remaining mass is distributed uniformly over the other classes.

    Args:
        student_logits: Raw logits from the student, shape (B, C).
        teacher_logits: Raw logits from the teacher (detached), shape (B, C).
        labels: Ground-truth class indices, shape (B,).
        temperature: Temperature for the teacher soft distribution.
        num_classes: Total number of classes C.

    Returns:
        Scalar loss tensor.
    """
    batch_size = labels.size(0)
    device = student_logits.device

    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    true_class_prob = teacher_probs[torch.arange(batch_size, device=device), labels]

    soft_labels = torch.full(
        (batch_size, num_classes), fill_value=0.0, dtype=torch.float32, device=device,
    )
    remaining = (1.0 - true_class_prob) / (num_classes - 1)
    soft_labels += remaining.unsqueeze(1)
    soft_labels[torch.arange(batch_size, device=device), labels] = true_class_prob

    log_probs = F.log_softmax(student_logits, dim=1)
    loss = -(soft_labels * log_probs).sum(dim=1).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Single-epoch training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    teacher: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Handles three distinct loss regimes:

    - ``augmix`` with JSD: batch contains (x_clean, x_aug1, x_aug2, y);
      total loss = CE(clean) + λ * JSD(clean, aug1, aug2).
    - ``kd_augmix`` / ``kd_ls_augmix``: standard KD losses using a frozen
      AugMix-trained teacher.
    - Fallback: standard cross-entropy (e.g., ``augmix`` without JSD).

    Args:
        model: Model to train.
        loader: Training DataLoader.
        optimizer: Optimiser instance.
        criterion: CE loss (used in non-KD modes).
        device: Target device.
        config: Full experiment configuration.
        teacher: Frozen teacher model (required for KD modes).

    Returns:
        Tuple of (epoch_train_loss, epoch_train_accuracy).
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    mode = config.train.training_mode
    use_jsd = (mode == "augmix") and config.augmix.jsd_loss

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        if use_jsd:
            # Batch: (x_clean, x_aug1, x_aug2, labels)
            x_clean, x_aug1, x_aug2, labels = batch
            x_clean = x_clean.to(device, non_blocking=True)
            x_aug1 = x_aug1.to(device, non_blocking=True)
            x_aug2 = x_aug2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits_clean = model(x_clean)
            logits_aug1 = model(x_aug1)
            logits_aug2 = model(x_aug2)

            ce_loss = criterion(logits_clean, labels)
            jsd_loss = compute_jsd_loss(logits_clean, logits_aug1, logits_aug2)
            loss = ce_loss + config.augmix.jsd_lambda * jsd_loss

            student_logits = logits_clean  # for accuracy tracking

        else:
            # Standard batch: (images, labels)
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            student_logits = model(images)

            if mode == "kd_augmix":
                with torch.no_grad():
                    teacher_logits = teacher(images)  # type: ignore[misc]
                loss = compute_kd_loss(
                    student_logits, teacher_logits, labels,
                    config.train.kd_temperature, config.train.kd_alpha,
                )
            elif mode == "kd_ls_augmix":
                with torch.no_grad():
                    teacher_logits = teacher(images)  # type: ignore[misc]
                loss = compute_kd_ls_loss(
                    student_logits, teacher_logits, labels,
                    config.train.kd_temperature, config.data.num_classes,
                )
            else:
                loss = criterion(student_logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.detach().item() * batch_size
        running_correct += (torch.argmax(student_logits, dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % config.run.log_interval == 0:
            logger.info(
                "  [Batch %4d/%d] train_loss=%.4f | train_acc=%.4f",
                batch_idx + 1, len(loader),
                running_loss / total_samples,
                running_correct / total_samples,
            )

    return running_loss / total_samples, running_correct / total_samples


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set.

    Always uses plain cross-entropy so validation loss is interpretable.

    Args:
        model: Model to evaluate.
        loader: Validation DataLoader.
        criterion: Plain CE loss (no smoothing).
        device: Target device.

    Returns:
        Tuple of (average_val_loss, average_val_accuracy).
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

    return running_loss / total_samples, running_correct / total_samples


# ─────────────────────────────────────────────────────────────────────────────
# History persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_history(history: Dict[str, Any], config: Config) -> None:
    """Persist training history as a JSON file next to the checkpoint.

    Args:
        history: Training history dictionary.
        config: Full experiment configuration.
    """
    save_path = Path(config.run.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = save_path.parent / f"{config.run.experiment_name}_history.json"

    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    config: Config,
    device: torch.device,
    teacher: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Execute the full training loop.

    Supports all three HW3 training modes. The teacher model (if provided) is
    expected to be already loaded, moved to ``device``, and frozen.

    Args:
        model: Student (or solo) model to train.
        config: Full experiment configuration.
        device: Target device.
        teacher: Frozen teacher model for KD modes; ``None`` otherwise.

    Returns:
        Dictionary containing per-epoch metrics and best-epoch statistics.
    """
    train_loader, val_loader = get_train_val_loaders(config)

    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, config)
    scheduler = build_scheduler(optimizer, config)

    os.makedirs(Path(config.run.save_path).parent, exist_ok=True)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_acc": None,
        "config": {
            "training_mode": config.train.training_mode,
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "optimizer": config.train.optimizer,
            "scheduler": config.train.scheduler,
            "weight_decay": config.train.weight_decay,
            "augmix_jsd": config.augmix.jsd_loss,
            "augmix_jsd_lambda": config.augmix.jsd_lambda,
            "kd_temperature": config.train.kd_temperature,
            "kd_alpha": config.train.kd_alpha,
        },
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    logger.info("Starting training | mode=%s", config.train.training_mode)
    logger.info("Train size: %d | Val size: %d",
                len(train_loader.dataset), len(val_loader.dataset))

    for epoch in range(1, config.train.epochs + 1):
        logger.info("Epoch %d/%d", epoch, config.train.epochs)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            teacher=teacher,
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            device=device,
        )

        if scheduler is not None:
            if config.train.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = get_current_lr(optimizer)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        logger.info(
            "  Train | loss=%.4f acc=%.4f | Val | loss=%.4f acc=%.4f | lr=%.6f",
            train_loss, train_acc, val_loss, val_acc, current_lr,
        )

        improved = val_loss < (best_val_loss - config.train.min_delta)

        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.run.save_path)
            logger.info("  Saved best model → %s (val_loss=%.4f)",
                        config.run.save_path, best_val_loss)
        else:
            patience_counter += 1
            logger.info("  No improvement. Patience: %d/%d",
                        patience_counter, config.train.early_stopping_patience)

        if (
            config.train.early_stopping_patience > 0
            and patience_counter >= config.train.early_stopping_patience
        ):
            logger.info("  Early stopping triggered at epoch %d.", epoch)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc

    save_history(history, config)

    logger.info("Training complete.")
    logger.info("Best epoch    : %d", best_epoch)
    logger.info("Best val loss : %.4f", best_val_loss)
    logger.info("Best val acc  : %.4f", best_val_acc)
    logger.info("Checkpoint    : %s", config.run.save_path)

    return history
