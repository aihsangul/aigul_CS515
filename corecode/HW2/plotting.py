from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # non-interactive backend for saving figures

from train import CIFAR10_CLASSES

logger = logging.getLogger("cs515")


# ─────────────────────────────────────────────────────────────────────────────
# Curve plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history: Dict[str, Any], save_path: str) -> None:
    """Plot training and validation loss over epochs.

    Args:
        history: Training history dictionary with ``train_loss`` and ``val_loss``.
        save_path: Absolute path where the figure is saved (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    if history.get("best_epoch"):
        ax.axvline(x=history["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved -> %s", save_path)


def plot_accuracy_curve(history: Dict[str, Any], save_path: str) -> None:
    """Plot training and validation accuracy over epochs.

    Args:
        history: Training history dictionary with ``train_acc`` and ``val_acc``.
        save_path: Absolute path where the figure is saved (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], label="Train acc")
    ax.plot(epochs, history["val_acc"], label="Val acc")
    if history.get("best_epoch"):
        ax.axvline(x=history["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Accuracy curve saved -> %s", save_path)


def plot_lr_curve(history: Dict[str, Any], save_path: str) -> None:
    """Plot the learning-rate schedule over epochs.

    Args:
        history: Training history dictionary with ``learning_rate``.
        save_path: Absolute path where the figure is saved (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["learning_rate"]) + 1)
    ax.plot(epochs, history["learning_rate"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("LR curve saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    save_path: str,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        confusion_matrix: Square nested list of counts.
        save_path: Absolute path where the figure is saved (PNG).
        class_names: Optional list of class name strings.
        normalize: If True, normalise each row to [0, 1].
    """
    cm = np.array(confusion_matrix, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    num_classes = cm.shape[0]
    names = class_names or [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE embedding visualisation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings from the model.

    For ResNet-18 and MobileNetV2 the feature vector before the final FC layer
    is obtained via ``forward_features`` if defined, otherwise by hooking the
    global average pooling output. For SimpleCNN the flatten output before the
    final linear is used.

    Args:
        model: Model to extract features from (must be in eval mode).
        loader: DataLoader to iterate over.
        device: Target device.
        max_samples: Maximum number of samples to collect.

    Returns:
        Tuple of (features array (N, D), labels array (N,)).
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total = 0

    # Hook to capture the output of the last global pooling / flatten operation
    activations: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        activations.append(out.detach().cpu())

    # Register hook on the appropriate layer
    hook_handle = None
    if hasattr(model, "avgpool"):  # ResNet-18
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # SimpleCNN / MobileNet: hook the Flatten or first Linear input
        for layer in model.classifier:
            if isinstance(layer, nn.Flatten):
                hook_handle = layer.register_forward_hook(hook_fn)
                break
        if hook_handle is None:
            hook_handle = model.classifier[0].register_forward_hook(hook_fn)

    for images, labels in loader:
        if total >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        activations.clear()
        _ = model(images)

        if activations:
            feat = activations[0]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            all_features.append(feat.numpy())
            all_labels.append(labels.numpy())
            total += feat.size(0)

    if hook_handle is not None:
        hook_handle.remove()

    features_arr = np.concatenate(all_features, axis=0)[:max_samples]
    labels_arr = np.concatenate(all_labels, axis=0)[:max_samples]
    return features_arr, labels_arr


def plot_tsne(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Compute t-SNE on model embeddings and save a scatter plot.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to sample from.
        device: Target device.
        save_path: Absolute path where the figure is saved (PNG).
        max_samples: Maximum number of samples to embed.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn not installed — skipping t-SNE plot.")
        return

    logger.info("Collecting embeddings for t-SNE (%d samples max)...", max_samples)
    features, labels = collect_embeddings(model, loader, device, max_samples=max_samples)

    logger.info("Running t-SNE on %d samples (dim=%d)...", len(features), features.shape[1])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))  # type: ignore[attr-defined]
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=10,
            color=palette[class_idx],
        )

    ax.set_title("t-SNE of learned feature embeddings")
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("t-SNE plot saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all plots from saved JSON files
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(experiment_dir: str) -> None:
    """Load training history and test results from JSON and generate all plots.

    Searches ``experiment_dir`` for ``*_history.json`` and
    ``*_test_results.json`` files and produces the standard set of plots.

    Args:
        experiment_dir: Directory containing the experiment JSON outputs.
    """
    exp_dir = Path(experiment_dir)

    history_files = list(exp_dir.glob("*_history.json"))
    test_files = list(exp_dir.glob("*_test_results.json"))

    for history_path in history_files:
        stem = history_path.stem.replace("_history", "")
        with history_path.open() as fh:
            history = json.load(fh)

        plot_loss_curve(history, str(exp_dir / f"{stem}_loss.png"))
        plot_accuracy_curve(history, str(exp_dir / f"{stem}_accuracy.png"))
        if "learning_rate" in history:
            plot_lr_curve(history, str(exp_dir / f"{stem}_lr.png"))

    for test_path in test_files:
        stem = test_path.stem.replace("_test_results", "")
        with test_path.open() as fh:
            results = json.load(fh)

        if "confusion_matrix" in results:
            plot_confusion_matrix(
                results["confusion_matrix"],
                str(exp_dir / f"{stem}_confusion_matrix.png"),
                class_names=list(CIFAR10_CLASSES),
            )
