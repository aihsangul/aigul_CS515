"""Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-18.

Reference:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
    Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. ICCV 2017.

Usage::

    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(image_tensor, class_idx)          # (H, W) numpy array
    overlay  = cam.overlay(image_tensor, heatmap)   # (H, W, 3) uint8

The :func:`visualize_gradcam_pairs` function generates a side-by-side figure
comparing Grad-CAM overlays on clean versus adversarial samples where the
adversarial perturbation induces a mis-classification.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

matplotlib.use("Agg")


class GradCAM:
    """Grad-CAM hook attached to a target convolutional layer.

    Args:
        model: The neural network to inspect.
        target_layer: The convolutional layer whose feature maps and
                      gradients are used to compute the CAM.  For
                      ResNet-18 this is typically ``model.layer4[-1]``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Internal hooks ────────────────────────────────────────────────────

    def _save_activations(
        self,
        _module: nn.Module,
        _inp: Tuple,
        output: Tensor,
    ) -> None:
        """Store forward activations from the target layer."""
        self._activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_in: Tuple,
        grad_out: Tuple,
    ) -> None:
        """Store gradients flowing back through the target layer."""
        self._gradients = grad_out[0].detach()

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(
        self,
        image: Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            image: Preprocessed image tensor of shape (1, C, H, W) or
                   (C, H, W).  Must be on the same device as the model.
            class_idx: Target class index.  If ``None``, the predicted
                       class (argmax of logits) is used.

        Returns:
            Normalised heatmap of shape (H_feat, W_feat) with values in
            ``[0, 1]``.  Use :meth:`overlay` to upsample and blend onto
            the original image.
        """
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.requires_grad_(False)

        # Forward pass
        logits = self.model(image)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass for the target class
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        # Global average pooling of gradients → channel weights
        grads = self._gradients  # (1, C, H', W')
        acts = self._activations  # (1, C, H', W')

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not capture activations/gradients. "
                               "Ensure the target_layer is in the forward path.")

        weights = grads[0].mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * acts[0]).sum(dim=0)  # (H', W')
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay(
        self,
        image: Tensor,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Upsample heatmap and blend onto the original image.

        Args:
            image: Preprocessed image tensor (C, H, W) or (1, C, H, W).
                   Values assumed to be normalised; will be unnormalised
                   for display using CIFAR-10 statistics.
            heatmap: Grad-CAM heatmap as returned by :meth:`__call__`,
                     shape (H_feat, W_feat), values in [0, 1].
            alpha: Blending weight for the heatmap overlay.

        Returns:
            RGB overlay as a uint8 numpy array of shape (H, W, 3).
        """
        if image.dim() == 4:
            image = image[0]

        # Denormalise using CIFAR-10 statistics
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img_np = (image.cpu() * std + mean).clamp(0, 1).numpy()
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)

        h, w = img_np.shape[:2]

        # Resize heatmap to match image spatial dimensions
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = np.array(heatmap_pil.resize((w, h), PILImage.BILINEAR))

        # Apply colour map
        colormap = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3]  # (H, W, 3)
        colormap = (colormap * 255).astype(np.uint8)

        overlay = (alpha * colormap + (1.0 - alpha) * img_np).astype(np.uint8)
        return overlay

    def remove_hooks(self) -> None:
        """Remove the forward and backward hooks from the target layer."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def visualize_gradcam_pairs(
    model: nn.Module,
    clean_images: Tensor,
    adv_images: Tensor,
    true_labels: List[int],
    clean_preds: List[int],
    adv_preds: List[int],
    save_path: str,
    target_layer: Optional[nn.Module] = None,
) -> None:
    """Save a side-by-side Grad-CAM figure for clean and adversarial samples.

    Each row shows one sample with four columns:
    - Original clean image
    - Grad-CAM overlay on clean image (correct prediction)
    - Adversarial image (with invisible perturbation)
    - Grad-CAM overlay on adversarial image (mis-classification)

    Args:
        model: Trained ResNet-18 (or compatible) model in eval mode.
        clean_images: Clean image tensors, shape (N, C, H, W).
        adv_images: Adversarial image tensors, shape (N, C, H, W).
        true_labels: Ground-truth class indices (length N).
        clean_preds: Model predictions on clean images (length N).
        adv_preds: Model predictions on adversarial images (length N).
        save_path: Output PNG file path.
        target_layer: Layer to attach the Grad-CAM hook to.  Defaults to
                      ``model.layer4[-1]`` (last ResNet block).
    """
    if target_layer is None:
        # Default to the last residual block of ResNet-18
        if hasattr(model, "layer4"):
            target_layer = model.layer4[-1]
        else:
            raise ValueError(
                "Cannot infer target_layer automatically. "
                "Pass target_layer explicitly."
            )

    cam = GradCAM(model, target_layer)
    n = len(clean_images)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    col_titles = ["Clean image", "Grad-CAM (clean)", "Adversarial image", "Grad-CAM (adv)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    # Denormalise helper (CIFAR-10 stats)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def to_display(t: Tensor) -> np.ndarray:
        img = (t.cpu() * std + mean).clamp(0, 1)
        return (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    for i in range(n):
        img_c = clean_images[i]
        img_a = adv_images[i]
        true = true_labels[i]
        pred_c = clean_preds[i]
        pred_a = adv_preds[i]

        true_name = CIFAR10_CLASSES[true]
        pred_c_name = CIFAR10_CLASSES[pred_c]
        pred_a_name = CIFAR10_CLASSES[pred_a]

        # Grad-CAM heatmaps
        heatmap_clean = cam(img_c.unsqueeze(0).to(next(model.parameters()).device),
                            class_idx=pred_c)
        heatmap_adv = cam(img_a.unsqueeze(0).to(next(model.parameters()).device),
                          class_idx=pred_a)

        # Column 0: clean image
        axes[i, 0].imshow(to_display(img_c))
        axes[i, 0].set_xlabel(f"True: {true_name}\nPred: {pred_c_name}", fontsize=9)

        # Column 1: Grad-CAM on clean
        axes[i, 1].imshow(cam.overlay(img_c, heatmap_clean))
        axes[i, 1].set_xlabel(f"Pred: {pred_c_name}", fontsize=9)

        # Column 2: adversarial image (barely perceptible difference)
        axes[i, 2].imshow(to_display(img_a))
        axes[i, 2].set_xlabel(f"True: {true_name}\nPred: {pred_a_name}", fontsize=9)

        # Column 3: Grad-CAM on adversarial
        axes[i, 3].imshow(cam.overlay(img_a, heatmap_adv))
        axes[i, 3].set_xlabel(f"Pred: {pred_a_name}", fontsize=9)

        for col in range(4):
            axes[i, col].axis("off")

    fig.suptitle("Grad-CAM: Clean vs Adversarial (rows = misclassified samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cam.remove_hooks()
