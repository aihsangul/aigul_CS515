from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from utils import ensure_dir


def plot_loss_curve(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    output_path: str,
    title: str = "Training and Validation Loss",
) -> None:
    """
    Plot training and validation loss over epochs.

    Args:
        train_loss: Training loss history.
        val_loss: Validation loss history.
        output_path: Output image path.
        title: Figure title.
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_accuracy_curve(
    train_acc: Sequence[float],
    val_acc: Sequence[float],
    output_path: str,
    title: str = "Training and Validation Accuracy",
) -> None:
    """
    Plot training and validation accuracy over epochs.

    Args:
        train_acc: Training accuracy history.
        val_acc: Validation accuracy history.
        output_path: Output image path.
        title: Figure title.
    """
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_rate_curve(
    learning_rates: Sequence[float],
    output_path: str,
    title: str = "Learning Rate Over Epochs",
) -> None:
    """
    Plot learning rate over epochs.

    Args:
        learning_rates: Learning-rate history.
        output_path: Output image path.
        title: Figure title.
    """
    epochs = range(1, len(learning_rates) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, learning_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    output_path: str,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix using matplotlib.

    Args:
        confusion_matrix: Square confusion matrix.
        output_path: Output image path.
        class_names: Class labels for axes.
        normalize: Whether to normalize each row.
        title: Figure title.
    """
    cm = np.array(confusion_matrix, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
        fmt = ".2f"
    else:
        fmt = ".0f"

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(np.arange(num_classes), class_names)
    plt.yticks(np.arange(num_classes), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = format(cm[i, j], fmt)
            text_color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, value, ha="center", va="center", color=text_color)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect hidden-layer embeddings and labels from a model.

    This assumes the model implements `forward_features()` and returns the
    representation before the final classifier.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader providing images and labels.
        device: Execution device.
        max_samples: Maximum number of samples to collect.

    Returns:
        Tuple of:
            - embeddings as numpy array of shape (N, D)
            - labels as numpy array of shape (N,)
    """
    model.eval()

    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    collected = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        features = model.forward_features(images).cpu()
        features_list.append(features)
        labels_list.append(labels.cpu())

        collected += images.size(0)
        if collected >= max_samples:
            break

    embeddings = torch.cat(features_list, dim=0)[:max_samples].numpy()
    label_array = torch.cat(labels_list, dim=0)[:max_samples].numpy()

    return embeddings, label_array


def plot_tsne_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: str,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    random_state: int = 42,
    title: str = "t-SNE of Learned Features",
) -> None:
    """
    Compute t-SNE on model embeddings and save the resulting scatter plot.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader.
        device: Execution device.
        output_path: Output image path.
        max_samples: Number of samples to visualize.
        perplexity: t-SNE perplexity.
        random_state: Random seed for t-SNE.
        title: Figure title.
    """
    embeddings, labels = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        max_samples=max_samples,
    )

    if len(embeddings) < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")

    adjusted_perplexity = min(perplexity, len(embeddings) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=10)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.colorbar(scatter, ticks=np.arange(10))
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()