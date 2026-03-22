import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from parameters import Config

logger = logging.getLogger("cs515")


def get_transforms(config: Config) -> transforms.Compose:
    """
    Build dataset transforms.

    For HW1a, the dataset is MNIST, so we only apply:
    - ToTensor
    - Normalize

    Args:
        config: Full experiment configuration.

    Returns:
        A torchvision Compose transform.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def get_train_val_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders using a split from the MNIST training set.

    The homework explicitly asks for validation-loss-based tuning, so we should
    not use the test set as validation. Instead, we split the official training
    set into train and validation subsets.

    Args:
        config: Full experiment configuration.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    transform = get_transforms(config)

    full_train_dataset = datasets.MNIST(
        root=config.data.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * config.data.val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config.run.seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

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


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 penalty across all trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Scalar tensor representing the L1 norm sum.
    """
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for parameter in model.parameters():
        l1_penalty = l1_penalty + parameter.abs().sum()
    return l1_penalty


def build_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """
    Build optimizer according to configuration.

    Args:
        model: PyTorch model.
        config: Full experiment configuration.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer choice is unsupported.
    """
    if config.train.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

    if config.train.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.train.optimizer}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Config,
) -> Optional[Any]:
    """
    Build learning-rate scheduler according to configuration.

    Supported schedulers:
    - none
    - step
    - plateau
    - cosine

    Args:
        optimizer: Optimizer instance.
        config: Full experiment configuration.

    Returns:
        Scheduler instance or None.

    Raises:
        ValueError: If scheduler choice is unsupported.
    """
    if config.train.scheduler == "none":
        return None

    if config.train.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=config.train.step_size,
            gamma=config.train.gamma,
        )

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

    raise ValueError(f"Unsupported scheduler: {config.train.scheduler}")


def get_current_learning_rate(optimizer: optim.Optimizer) -> float:
    """
    Read the current learning rate from the optimizer.

    Args:
        optimizer: Optimizer instance.

    Returns:
        Current learning rate.
    """
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    If L1 regularization is selected, the penalty is added manually to the loss.
    L2 regularization is expected to be handled through optimizer weight_decay.

    Args:
        model: PyTorch model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device.
        config: Full experiment configuration.

    Returns:
        Tuple of (average_train_loss, average_train_accuracy).
    """
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if config.train.regularizer == "l1" and config.train.reg_lambda > 0.0:
            l1_penalty = compute_l1_penalty(model)
            loss = loss + config.train.reg_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.detach().item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % config.run.log_interval == 0:
            avg_loss = running_loss / total_samples
            avg_acc = running_correct / total_samples
            logger.info(
                "  [Batch %4d/%d] train_loss=%.4f | train_acc=%.4f",
                batch_idx + 1,
                len(loader),
                avg_loss,
                avg_acc,
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Validation loss is computed without manual L1 penalty because it should
    reflect predictive performance on held-out data rather than the training
    optimization objective.

    Args:
        model: PyTorch model.
        loader: Validation DataLoader.
        criterion: Loss function.
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

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def save_history(history: Dict[str, Any], config: Config) -> None:
    """
    Save training history as JSON for later plotting/reporting.

    Args:
        history: Training history dictionary.
        config: Full experiment configuration.
    """
    save_path = Path(config.run.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = save_path.parent / f"{config.run.experiment_name}_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def run_training(
    model: nn.Module,
    config: Config,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full training routine.

    Main behaviors:
    - Creates train/validation loaders
    - Uses CrossEntropyLoss
    - Supports Adam/SGD
    - Supports StepLR / ReduceLROnPlateau / CosineAnnealingLR
    - Saves best model checkpoint using validation loss
    - Applies early stopping based on validation loss
    - Stores training history for plots/reporting

    Args:
        model: PyTorch model.
        config: Full experiment configuration.
        device: Target device.

    Returns:
        Dictionary containing training history and best validation statistics.
    """
    train_loader, val_loader = get_train_val_loaders(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
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
            "hidden_sizes": config.model.hidden_sizes,
            "activation": config.model.activation,
            "dropout": config.model.dropout,
            "use_batch_norm": config.model.use_batch_norm,
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "optimizer": config.train.optimizer,
            "scheduler": config.train.scheduler,
            "regularizer": config.train.regularizer,
            "reg_lambda": config.train.reg_lambda,
            "weight_decay": config.train.weight_decay,
        },
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    logger.info("Starting training...")
    logger.info(
        "Train size: %s | Val size: %s",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    for epoch in range(1, config.train.epochs + 1):
        logger.info("Epoch %s/%s", epoch, config.train.epochs)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            if config.train.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = get_current_learning_rate(optimizer)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        logger.info(
            "  Train | loss: %.4f | acc: %.4f | Val | loss: %.4f | acc: %.4f | LR: %.6f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        improved = val_loss < (best_val_loss - config.train.min_delta)

        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.run.save_path)

            logger.info(
                "  Saved best model to %s (best_val_loss=%.4f)",
                config.run.save_path,
                best_val_loss,
            )
        else:
            patience_counter += 1
            logger.info(
                "  No improvement in validation loss. Early-stop counter: %d/%d",
                patience_counter,
                config.train.early_stopping_patience,
            )

        if (
            config.train.early_stopping_patience > 0
            and patience_counter >= config.train.early_stopping_patience
        ):
            logger.info("  Early stopping triggered.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc

    save_history(history, config)

    logger.info("Training complete.")
    logger.info("Best epoch     : %s", best_epoch)
    logger.info("Best val loss  : %.4f", best_val_loss)
    logger.info("Best val acc   : %.4f", best_val_acc)
    logger.info("Checkpoint     : %s", config.run.save_path)

    return history