"""PGD adversarial attack implementations.

Reference:
    Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
    Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.

Supported threat models:
    - L-infinity (PGD-Linf): each pixel may move at most ε in [0,∞) norm.
    - L2 (PGD-L2): the total perturbation Euclidean norm is bounded by ε.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _as_broadcast_tensor(value: float | Tensor, reference: Tensor) -> Tensor:
    """Convert a scalar or tensor value to a tensor broadcastable to ``reference``."""
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.full_like(reference, float(value))


# ─────────────────────────────────────────────────────────────────────────────
# L-infinity PGD
# ─────────────────────────────────────────────────────────────────────────────

def pgd_linf(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    eps: float | Tensor,
    step_size: float | Tensor,
    steps: int,
    random_start: bool = True,
    clamp_min: float | Tensor = 0.0,
    clamp_max: float | Tensor = 1.0,
) -> Tensor:
    """Projected Gradient Descent attack under L-infinity norm.

    Generates adversarial examples by iteratively applying sign gradient
    updates and projecting back onto the ε-ball in L-infinity norm.

    Args:
        model: Target model (set to eval mode before calling).
        images: Clean input images, shape (B, C, H, W).  Values in
                ``[clamp_min, clamp_max]`` (typically ``[0, 1]`` for
                unnormalised images, but the perturbation is added to
                the *normalised* tensor — see note below).
        labels: Ground-truth labels, shape (B,).
        eps: L-infinity perturbation budget ε.
        step_size: Per-step perturbation size α.
        steps: Number of PGD iterations.
        random_start: If True, initialise δ uniformly in [-ε, ε].
        clamp_min: Minimum pixel value after clamping.
        clamp_max: Maximum pixel value after clamping.

    Returns:
        Adversarial images, same shape as ``images``.

    Note:
        The inputs are assumed to already be normalised (e.g., with CIFAR-10
        channel statistics). Perturbations are applied *in normalised space*
        so the ε values should be chosen accordingly.  The typical workflow
        in this codebase is to pass normalised tensors from the DataLoader
        and interpret ε as acting on the [0, 1] scale before normalisation
        (the default values 4/255 ≈ 0.016 are already in this regime).
    """
    model.eval()
    images = images.detach().clone()

    eps_tensor = _as_broadcast_tensor(eps, images)
    step_tensor = _as_broadcast_tensor(step_size, images)
    clamp_min_tensor = _as_broadcast_tensor(clamp_min, images)
    clamp_max_tensor = _as_broadcast_tensor(clamp_max, images)

    if random_start:
        delta = (torch.rand_like(images) * 2.0 - 1.0) * eps_tensor
    else:
        delta = torch.zeros_like(images)

    delta = delta.requires_grad_(True)

    for _ in range(steps):
        adv = images + delta
        adv = adv.clamp(clamp_min, clamp_max)

        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta.data = delta.data + step_tensor * grad_sign
            delta.data = torch.clamp(delta.data, min=-eps_tensor, max=eps_tensor)
            delta.data = torch.clamp(
                images + delta.data,
                min=clamp_min_tensor,
                max=clamp_max_tensor,
            ) - images

        delta.grad.zero_()

    adversarial = torch.clamp(
        images + delta.detach(),
        min=clamp_min_tensor,
        max=clamp_max_tensor,
    )
    return adversarial


# ─────────────────────────────────────────────────────────────────────────────
# L2 PGD
# ─────────────────────────────────────────────────────────────────────────────

def pgd_l2(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
    clamp_min: float | Tensor = 0.0,
    clamp_max: float | Tensor = 1.0,
) -> Tensor:
    """Projected Gradient Descent attack under L2 norm.

    Applies gradient updates and projects the perturbation onto the
    ε-ball in L2 norm after each step.

    Args:
        model: Target model (set to eval mode before calling).
        images: Clean input images, shape (B, C, H, W).
        labels: Ground-truth labels, shape (B,).
        eps: L2 perturbation budget ε.
        step_size: Per-step perturbation size α.
        steps: Number of PGD iterations.
        random_start: If True, initialise δ as a random L2-norm-bounded vector.
        clamp_min: Minimum pixel value after clamping.
        clamp_max: Maximum pixel value after clamping.

    Returns:
        Adversarial images, same shape as ``images``.
    """
    model.eval()
    images = images.detach().clone()
    batch_size = images.size(0)
    clamp_min_tensor = _as_broadcast_tensor(clamp_min, images)
    clamp_max_tensor = _as_broadcast_tensor(clamp_max, images)

    if random_start:
        delta = torch.randn_like(images)
        # Normalise to the ε-ball boundary and scale by a random factor
        norms = delta.view(batch_size, -1).norm(dim=1, keepdim=True)
        norms = norms.view(batch_size, 1, 1, 1)
        delta = delta / (norms + 1e-12) * eps
        delta = delta * torch.empty(batch_size, 1, 1, 1,
                                    device=images.device).uniform_(0.0, 1.0)
    else:
        delta = torch.zeros_like(images)

    delta = delta.requires_grad_(True)

    for _ in range(steps):
        adv = torch.clamp(images + delta, min=clamp_min_tensor, max=clamp_max_tensor)

        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad = delta.grad
            # Normalise gradient to unit L2 norm per sample
            grad_norms = grad.view(batch_size, -1).norm(dim=1, keepdim=True)
            grad_norms = grad_norms.view(batch_size, 1, 1, 1)
            grad_unit = grad / (grad_norms + 1e-12)

            delta.data = delta.data + step_size * grad_unit

            # Project onto L2 ε-ball
            norms = delta.data.view(batch_size, -1).norm(dim=1, keepdim=True)
            norms = norms.view(batch_size, 1, 1, 1)
            factor = torch.clamp(eps / (norms + 1e-12), max=1.0)
            delta.data = delta.data * factor

            # Enforce image constraints
            delta.data = torch.clamp(
                images + delta.data,
                min=clamp_min_tensor,
                max=clamp_max_tensor,
            ) - images

        delta.grad.zero_()

    adversarial = torch.clamp(
        images + delta.detach(),
        min=clamp_min_tensor,
        max=clamp_max_tensor,
    )
    return adversarial


# ─────────────────────────────────────────────────────────────────────────────
# Unified interface
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    norm: str,
    eps: float | Tensor,
    step_size: float | Tensor,
    steps: int,
    random_start: bool = True,
    clamp_min: float | Tensor = 0.0,
    clamp_max: float | Tensor = 1.0,
) -> Tensor:
    """Dispatch to the appropriate PGD variant based on the threat model norm.

    Args:
        model: Target model.
        images: Clean input images, shape (B, C, H, W).
        labels: Ground-truth labels, shape (B,).
        norm: Threat model — ``"linf"`` or ``"l2"``.
        eps: Perturbation budget.
        step_size: Per-step size.
        steps: Number of PGD iterations.
        random_start: Random initialisation inside the ε-ball.
        clamp_min: Minimum pixel value.
        clamp_max: Maximum pixel value.

    Returns:
        Adversarial images, same shape as ``images``.

    Raises:
        ValueError: If ``norm`` is not ``"linf"`` or ``"l2"``.
    """
    if norm == "linf":
        return pgd_linf(
            model, images, labels,
            eps=eps, step_size=step_size, steps=steps,
            random_start=random_start,
            clamp_min=clamp_min, clamp_max=clamp_max,
        )
    if norm == "l2":
        return pgd_l2(
            model, images, labels,
            eps=eps, step_size=step_size, steps=steps,
            random_start=random_start,
            clamp_min=clamp_min, clamp_max=clamp_max,
        )
    raise ValueError(f"Unsupported norm: {norm!r}. Choose 'linf' or 'l2'.")


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level adversarial evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
) -> float:
    """Compute classification accuracy for a batch (no gradients needed).

    Args:
        model: Model to evaluate.
        images: Input images, shape (B, C, H, W).
        labels: Ground-truth labels, shape (B,).

    Returns:
        Fraction of correctly classified samples.
    """
    model.eval()
    logits = model(images)
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
