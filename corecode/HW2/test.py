from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import Config
from train import CIFAR10_CLASSES, get_val_transforms

logger = logging.getLogger("cs515")


# Test data loader

def get_test_loader(config: Config) -> DataLoader:
    """Create the CIFAR-10 test DataLoader.

    Uses the official CIFAR-10 test split (10 000 images). Transform is chosen
    based on "config.train.training_mode" (resize to 224 for transfer_resize,
    otherwise standard CIFAR-10 normalisation).

    Args:
        config: Full experiment configuration.

    Returns:
        DataLoader for the CIFAR-10 test set.
    """
    transform = get_val_transforms(config)

    test_dataset = datasets.CIFAR10(
        root=config.data.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def initialize_confusion_matrix(num_classes: int) -> List[List[int]]:
    """Initialise a square confusion matrix filled with zeros.

    Args:
        num_classes: Number of classification classes.

    Returns:
        Nested list of shape (num_classes, num_classes).
    """
    return [[0] * num_classes for _ in range(num_classes)]


def update_confusion_matrix(
    confusion_matrix: List[List[int]],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Update the confusion matrix in-place for one batch.

    Rows represent true labels; columns represent predicted labels.

    Args:
        confusion_matrix: Existing confusion matrix (modified in-place).
        targets: Ground-truth labels tensor.
        predictions: Predicted labels tensor.
    """
    for true_label, pred_label in zip(targets.tolist(), predictions.tolist()):
        confusion_matrix[true_label][pred_label] += 1


def compute_per_class_accuracy(confusion_matrix: List[List[int]]) -> List[float]:
    """Compute per-class accuracy from the confusion matrix.

    Args:
        confusion_matrix: Square confusion matrix.

    Returns:
        List of per-class accuracies (one per class).
    """
    per_class: List[float] = []
    for class_idx, row in enumerate(confusion_matrix):
        correct = row[class_idx]
        total = sum(row)
        per_class.append(correct / total if total > 0 else 0.0)
    return per_class


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs counting
# ─────────────────────────────────────────────────────────────────────────────

def count_flops(
    model: nn.Module,
    input_res: Tuple[int, int, int],
) -> Optional[Tuple[str, str]]:
    """Count MACs and parameters using ``ptflops``.

    Silently skips if ``ptflops`` is not installed.

    Args:
        model: PyTorch model to profile.
        input_res: Input resolution as (C, H, W).

    Returns:
        Tuple of (macs_string, params_string) if ``ptflops`` is available,
        otherwise ``None``.
    """
    try:
        from ptflops import get_model_complexity_info  # type: ignore[import]
    except ImportError:
        logger.warning(
            "ptflops not installed — skipping FLOPs count. "
            "Install with: pip install ptflops"
        )
        return None

    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_test_results(results: Dict[str, Any], config: Config) -> None:
    """Save test results as JSON next to the checkpoint file.

    Args:
        results: Dictionary containing evaluation outputs.
        config: Full experiment configuration.
    """
    save_path = Path(config.run.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = save_path.parent / f"{config.run.experiment_name}_test_results.json"

    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main test routine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_test(
    model: nn.Module,
    config: Config,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate the saved best model on the official CIFAR-10 test set.

    Outputs:
    - Average test loss (standard CE)
    - Overall test accuracy
    - Per-class accuracy for all 10 CIFAR-10 classes
    - Confusion matrix
    - FLOPs / parameter count (via ``ptflops`` if installed)

    Args:
        model: Model instance (weights loaded inside this function).
        config: Full experiment configuration.
        device: Target device.

    Returns:
        Dictionary containing all test metrics.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    checkpoint_path = Path(config.run.save_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Train the model first or provide a valid --save_path."
        )

    logger.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_loader = get_test_loader(config)
    criterion = nn.CrossEntropyLoss()
    num_classes = config.data.num_classes
    confusion_matrix = initialize_confusion_matrix(num_classes)

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    logger.info("Starting test evaluation on %d samples...", len(test_loader.dataset))

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        update_confusion_matrix(confusion_matrix, labels.cpu(), predictions.cpu())

    test_loss = running_loss / total_samples
    test_accuracy = total_correct / total_samples
    per_class_accuracy = compute_per_class_accuracy(confusion_matrix)

    # Input resolution depends on training mode
    flops_input = (3, 224, 224) if config.train.training_mode == "transfer_resize" else (3, 32, 32)
    flops_result = count_flops(model, flops_input)

    results: Dict[str, Any] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "num_samples": total_samples,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
        "flops": flops_result[0] if flops_result else "N/A",
        "params": flops_result[1] if flops_result else "N/A",
        "config": {
            "training_mode": config.train.training_mode,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "optimizer": config.train.optimizer,
            "scheduler": config.train.scheduler,
            "label_smoothing": config.train.label_smoothing,
            "kd_temperature": config.train.kd_temperature,
            "kd_alpha": config.train.kd_alpha,
        },
    }

    save_test_results(results, config)

    logger.info("=" * 70)
    logger.info("Test Results  [%s]", config.run.experiment_name)
    logger.info("=" * 70)
    logger.info("Test loss        : %.4f", test_loss)
    logger.info(
        "Overall accuracy : %.4f  (%d / %d)",
        test_accuracy, total_correct, total_samples,
    )
    if flops_result:
        logger.info("MACs             : %s", flops_result[0])
        logger.info("Parameters       : %s", flops_result[1])
    logger.info("-" * 70)
    logger.info("Per-class accuracy:")
    for class_idx, acc in enumerate(per_class_accuracy):
        class_correct = confusion_matrix[class_idx][class_idx]
        class_total = sum(confusion_matrix[class_idx])
        logger.info(
            "  %-12s : %.4f  (%d / %d)",
            CIFAR10_CLASSES[class_idx], acc, class_correct, class_total,
        )
    logger.info("=" * 70)

    return results
