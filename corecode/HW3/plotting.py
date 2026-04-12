"""Plotting utilities for HW3.

Includes all HW2 curve / confusion-matrix / T-SNE plots plus new plots for:
- CIFAR-10-C corruption accuracy (bar chart + heatmap)
- Adversarial T-SNE: clean samples vs adversarial samples in feature space
- Robustness comparison bar chart across models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # non-interactive backend

from test import CORRUPTIONS
from train import CIFAR10_CLASSES

logger = logging.getLogger("cs515")


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plots (identical to HW2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history: Dict[str, Any], save_path: str) -> None:
    """Plot training and validation loss over epochs.

    Args:
        history: Training history with ``train_loss`` and ``val_loss``.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    if history.get("best_epoch"):
        ax.axvline(x=history["best_epoch"], color="gray", linestyle="--",
                   label="Best epoch")
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
        history: Training history with ``train_acc`` and ``val_acc``.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], label="Train acc")
    ax.plot(epochs, history["val_acc"], label="Val acc")
    if history.get("best_epoch"):
        ax.axvline(x=history["best_epoch"], color="gray", linestyle="--",
                   label="Best epoch")
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
        history: Training history with ``learning_rate``.
        save_path: Output PNG path.
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
        save_path: Output PNG path.
        class_names: Optional class name list.
        normalize: If True, normalise each row to [0, 1].
    """
    cm = np.array(confusion_matrix, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    num_classes = cm.shape[0]
    names = class_names or [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10-C corruption accuracy plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_corruption_accuracy(
    corruption_results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "CIFAR-10-C accuracy per corruption",
) -> None:
    """Plot mean per-corruption accuracy as a bar chart.

    Each bar represents the mean accuracy across the five severity levels for
    one corruption type.

    Args:
        corruption_results: Dict mapping corruption name → {``"severity_k"``: acc}.
        save_path: Output PNG path.
        title: Figure title.
    """
    labels = []
    mean_accs = []
    for corruption in CORRUPTIONS:
        if corruption not in corruption_results:
            continue
        sev_accs = list(corruption_results[corruption].values())
        if sev_accs:
            labels.append(corruption.replace("_", "\n"))
            mean_accs.append(np.mean(sev_accs))

    if not labels:
        logger.warning("No CIFAR-10-C results to plot.")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 5))
    bars = ax.bar(x, [a * 100 for a in mean_accs], color="#4C72B0", edgecolor="white")

    for bar, val in zip(bars, mean_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=np.mean(mean_accs) * 100, color="red", linestyle="--",
               label=f"mCA = {np.mean(mean_accs)*100:.1f}%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Corruption accuracy bar chart saved -> %s", save_path)


def plot_corruption_heatmap(
    corruption_results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "CIFAR-10-C accuracy heatmap (corruption × severity)",
) -> None:
    """Plot a heatmap of accuracy across corruption types and severity levels.

    Args:
        corruption_results: Dict mapping corruption name → {``"severity_k"``: acc}.
        save_path: Output PNG path.
        title: Figure title.
    """
    available = [c for c in CORRUPTIONS if c in corruption_results]
    if not available:
        logger.warning("No CIFAR-10-C results to plot heatmap.")
        return

    num_corruptions = len(available)
    heatmap = np.zeros((num_corruptions, 5))

    for row, corruption in enumerate(available):
        for sev in range(1, 6):
            key = f"severity_{sev}"
            heatmap[row, sev - 1] = corruption_results[corruption].get(key, np.nan)

    fig, ax = plt.subplots(figsize=(8, max(6, num_corruptions * 0.35)))
    im = ax.imshow(heatmap * 100, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_yticks(range(num_corruptions))
    ax.set_yticklabels(available, fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Sev {s}" for s in range(1, 6)], fontsize=9)
    ax.set_title(title)

    # Annotate cells
    for row in range(num_corruptions):
        for col in range(5):
            val = heatmap[row, col]
            if not np.isnan(val):
                ax.text(col, row, f"{val*100:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if val < 0.5 else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Corruption heatmap saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial T-SNE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_embeddings_with_adv(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings for clean and adversarial images.

    Args:
        model: Model with a ``avgpool`` layer (ResNet) or ``classifier``
               containing a Flatten layer.
        clean_images: Clean images tensor (N, C, H, W).
        adv_images: Adversarial images tensor (N, C, H, W).
        labels: Ground-truth labels tensor (N,).
        device: Target device.

    Returns:
        Tuple of (features array (2N, D), labels array (2N,),
                  is_adv array (2N,) bool).
    """
    model.eval()
    activations: list = []

    def hook_fn(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        activations.append(out.detach().cpu())

    # Register hook
    hook_handle = None
    if hasattr(model, "avgpool"):
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for layer in model.classifier:
            if isinstance(layer, nn.Flatten):
                hook_handle = layer.register_forward_hook(hook_fn)
                break
        if hook_handle is None:
            hook_handle = model.classifier[0].register_forward_hook(hook_fn)

    all_feats = []
    all_labels = []
    all_is_adv = []

    for imgs, is_adv_flag in [(clean_images, False), (adv_images, True)]:
        activations.clear()
        _ = model(imgs.to(device))
        if activations:
            feat = activations[0]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            all_feats.append(feat.numpy())
            all_labels.append(labels.numpy())
            all_is_adv.append(
                np.full(len(imgs), fill_value=is_adv_flag, dtype=bool)
            )

    if hook_handle is not None:
        hook_handle.remove()

    if not all_feats:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_is_adv, axis=0),
    )


def plot_tsne_adversarial(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    save_path: str,
    max_samples: int = 1000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """T-SNE scatter plot showing clean and adversarial samples in feature space.

    Clean samples are shown with filled markers; adversarial samples with
    hollow ('x') markers.  Both share the same class colour palette.

    Args:
        model: Trained model used for feature extraction.
        clean_images: Clean test images (N, C, H, W).
        adv_images: Adversarial images generated from the same set (N, C, H, W).
        labels: Ground-truth labels (N,).
        device: Target device.
        save_path: Output PNG path.
        max_samples: Maximum number of per-group samples (clips N).
        perplexity: T-SNE perplexity parameter.
        random_state: Random seed.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn not installed — skipping adversarial T-SNE.")
        return

    n = min(max_samples, len(clean_images))
    clean_images = clean_images[:n]
    adv_images = adv_images[:n]
    labels = labels[:n]

    logger.info("Collecting embeddings for adversarial T-SNE (%d samples each)...", n)
    features, all_labels, is_adv = _collect_embeddings_with_adv(
        model, clean_images, adv_images, labels, device
    )

    if len(features) == 0:
        logger.warning("No embeddings collected — skipping adversarial T-SNE.")
        return

    logger.info("Running T-SNE on %d points (dim=%d)...", len(features), features.shape[1])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        clean_mask = (all_labels == class_idx) & (~is_adv)
        adv_mask = (all_labels == class_idx) & is_adv

        if clean_mask.any():
            ax.scatter(emb[clean_mask, 0], emb[clean_mask, 1],
                       color=palette[class_idx], alpha=0.6, s=12,
                       label=class_name, marker="o")
        if adv_mask.any():
            ax.scatter(emb[adv_mask, 0], emb[adv_mask, 1],
                       color=palette[class_idx], alpha=0.8, s=20,
                       marker="x", linewidths=1.0)

    # Legend: class colours (circles) + style legend
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, label="Clean"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None",
               markersize=6, markeredgewidth=1.5, label="Adversarial"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    # Keep one handle per class (the first filled scatter)
    class_handles = handles[:len(CIFAR10_CLASSES)]
    ax.legend(handles=class_handles + style_legend,
              markerscale=2, fontsize=8, loc="best", ncol=2)

    ax.set_title("T-SNE: Clean (●) vs Adversarial (×) feature embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Adversarial T-SNE saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Standard model T-SNE (clean only — same as HW2)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings from the model.

    Args:
        model: Model to extract features from.
        loader: DataLoader to iterate.
        device: Target device.
        max_samples: Maximum number of samples.

    Returns:
        Tuple of (features array (N, D), labels array (N,)).
    """
    model.eval()
    all_features: list = []
    all_labels: list = []
    total = 0
    activations: list = []

    def hook_fn(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        activations.append(out.detach().cpu())

    hook_handle = None
    if hasattr(model, "avgpool"):
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for layer in model.classifier:
            if isinstance(layer, nn.Flatten):
                hook_handle = layer.register_forward_hook(hook_fn)
                break
        if hook_handle is None:
            hook_handle = model.classifier[0].register_forward_hook(hook_fn)

    for images, lbl in loader:
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
            all_labels.append(lbl.numpy())
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
    """Compute T-SNE on clean embeddings and save a scatter plot.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to sample from.
        device: Target device.
        save_path: Output PNG path.
        max_samples: Maximum number of samples.
        perplexity: T-SNE perplexity.
        random_state: Random seed.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn not installed — skipping T-SNE.")
        return

    logger.info("Collecting clean embeddings for T-SNE (%d max)...", max_samples)
    features, labels = collect_embeddings(model, loader, device, max_samples)

    logger.info("Running T-SNE on %d samples (dim=%d)...", len(features), features.shape[1])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_idx
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=class_name, alpha=0.6, s=10, color=palette[class_idx])

    ax.set_title("T-SNE of learned feature embeddings")
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("T-SNE saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Robustness comparison",
) -> None:
    """Bar chart comparing clean, corruption, and adversarial accuracy across models.

    Args:
        results: Nested dict ``{model_name: {metric_name: float}}``.
                 Expected metric keys: ``clean``, ``corrupted``,
                 ``adversarial_linf``, ``adversarial_l2``.
        save_path: Output PNG path.
        title: Figure title.
    """
    model_names = list(results.keys())
    metrics = ["clean", "corrupted", "adversarial_linf", "adversarial_l2"]
    metric_labels = ["Clean", "mCA (corrupt.)", "Adv Linf", "Adv L2"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results[m].get(metric, 0.0) * 100 for m in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Robustness comparison saved -> %s", save_path)
