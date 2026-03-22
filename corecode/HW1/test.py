import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import Config
from train import get_transforms

logger = logging.getLogger("cs515")


def get_test_loader(config: Config) -> DataLoader:
    """
    Create the MNIST test DataLoader.

    This uses the official MNIST test split only. The validation split is handled
    separately in train.py from the official training set.

    Args:
        config: Full experiment configuration.

    Returns:
        DataLoader for the MNIST test set.
    """
    transform = get_transforms(config)

    test_dataset = datasets.MNIST(
        root=config.data.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    return test_loader


def initialize_confusion_matrix(num_classes: int) -> List[List[int]]:
    """
    Initialize a square confusion matrix filled with zeros.

    Args:
        num_classes: Number of classification classes.

    Returns:
        Confusion matrix as a nested list.
    """
    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]


def update_confusion_matrix(
    confusion_matrix: List[List[int]],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """
    Update the confusion matrix using a batch of targets and predictions.

    Rows correspond to true labels, columns correspond to predicted labels.

    Args:
        confusion_matrix: Existing confusion matrix.
        targets: Ground-truth labels.
        predictions: Predicted labels.
    """
    for true_label, pred_label in zip(targets.tolist(), predictions.tolist()):
        confusion_matrix[true_label][pred_label] += 1


def compute_per_class_accuracy(confusion_matrix: List[List[int]]) -> List[float]:
    """
    Compute per-class accuracy from the confusion matrix.

    Args:
        confusion_matrix: Square confusion matrix.

    Returns:
        List of per-class accuracies.
    """
    per_class_accuracy: List[float] = []

    for class_idx, row in enumerate(confusion_matrix):
        correct = row[class_idx]
        total = sum(row)
        accuracy = correct / total if total > 0 else 0.0
        per_class_accuracy.append(accuracy)

    return per_class_accuracy


def save_test_results(results: Dict[str, Any], config: Config) -> None:
    """
    Save test results as JSON for later use in the report.

    Args:
        results: Dictionary containing evaluation outputs.
        config: Full experiment configuration.
    """
    save_path = Path(config.run.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = save_path.parent / f"{config.run.experiment_name}_test_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


@torch.no_grad()
def run_test(
    model: nn.Module,
    config: Config,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate the best saved model on the official MNIST test set.

    Main outputs:
    - Average test loss
    - Overall test accuracy
    - Per-class accuracy
    - Confusion matrix

    Args:
        model: PyTorch model instance.
        config: Full experiment configuration.
        device: Target device.

    Returns:
        Dictionary containing test metrics and confusion matrix.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = Path(config.run.save_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Train the model first or provide the correct --save_path."
        )

    test_loader = get_test_loader(config)
    criterion = nn.CrossEntropyLoss()

    logger.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_classes = config.data.num_classes
    confusion_matrix = initialize_confusion_matrix(num_classes)

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    logger.info("Starting test evaluation...")
    logger.info("Test size: %s", len(test_loader.dataset))

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

    results: Dict[str, Any] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "num_samples": total_samples,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
        "config": {
            "hidden_sizes": config.model.hidden_sizes,
            "activation": config.model.activation,
            "dropout": config.model.dropout,
            "use_batch_norm": config.model.use_batch_norm,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "optimizer": config.train.optimizer,
            "scheduler": config.train.scheduler,
            "regularizer": config.train.regularizer,
            "reg_lambda": config.train.reg_lambda,
            "weight_decay": config.train.weight_decay,
        },
    }

    save_test_results(results, config)

    logger.info("=" * 70)
    logger.info("Test Results")
    logger.info("=" * 70)
    logger.info("Average test loss : %.4f", test_loss)
    logger.info(
        "Overall accuracy  : %.4f (%d/%d)",
        test_accuracy,
        total_correct,
        total_samples,
    )
    logger.info("-" * 70)
    logger.info("Per-class accuracy:")
    for class_idx, accuracy in enumerate(per_class_accuracy):
        class_total = sum(confusion_matrix[class_idx])
        class_correct = confusion_matrix[class_idx][class_idx]
        logger.info(
            "  Class %d: %.4f (%d/%d)",
            class_idx,
            accuracy,
            class_correct,
            class_total,
        )
    logger.info("=" * 70)

    return results