"""AugMix data augmentation and JSD consistency loss.

Reference:
    Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., &
    Lakshminarayanan, B. (2020). AugMix: A Simple Method to Improve Robustness
    and Uncertainty under Data Shift. ICLR 2020.
    https://openreview.net/forum?id=S1gmrxHFvB
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Elementary PIL augmentation operations
# ─────────────────────────────────────────────────────────────────────────────

def _autocontrast(img: Image.Image, _severity: float) -> Image.Image:
    """Apply automatic contrast normalisation."""
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _severity: float) -> Image.Image:
    """Equalise the image histogram."""
    return ImageOps.equalize(img)


def _rotate(img: Image.Image, severity: float) -> Image.Image:
    """Rotate the image by up to ``severity * 30`` degrees."""
    degrees = severity * 30.0
    sign = random.choice([-1.0, 1.0])
    return img.rotate(sign * degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _solarize(img: Image.Image, severity: float) -> Image.Image:
    """Invert pixels above a severity-dependent threshold."""
    threshold = int(256 - severity * 256 * 0.9)  # avoid full inversion
    return ImageOps.solarize(img, threshold)


def _shear_x(img: Image.Image, severity: float) -> Image.Image:
    """Apply a horizontal shear of up to ``severity * 0.3``."""
    level = severity * 0.3
    sign = random.choice([-1.0, 1.0])
    return img.transform(
        img.size, Image.AFFINE, (1, sign * level, 0, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _shear_y(img: Image.Image, severity: float) -> Image.Image:
    """Apply a vertical shear of up to ``severity * 0.3``."""
    level = severity * 0.3
    sign = random.choice([-1.0, 1.0])
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, sign * level, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _translate_x(img: Image.Image, severity: float) -> Image.Image:
    """Translate the image horizontally by up to ``severity * 33%`` of width."""
    pixels = severity * img.size[0] * 0.33
    sign = random.choice([-1.0, 1.0])
    return img.transform(
        img.size, Image.AFFINE, (1, 0, sign * pixels, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _translate_y(img: Image.Image, severity: float) -> Image.Image:
    """Translate the image vertically by up to ``severity * 33%`` of height."""
    pixels = severity * img.size[1] * 0.33
    sign = random.choice([-1.0, 1.0])
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, sign * pixels),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _color(img: Image.Image, severity: float) -> Image.Image:
    """Adjust colour saturation."""
    level = 1.0 + severity * 1.8 * random.choice([-1.0, 1.0])
    level = max(0.1, level)
    return ImageEnhance.Color(img).enhance(level)


def _contrast(img: Image.Image, severity: float) -> Image.Image:
    """Adjust image contrast."""
    level = 1.0 + severity * 1.8 * random.choice([-1.0, 1.0])
    level = max(0.1, level)
    return ImageEnhance.Contrast(img).enhance(level)


def _brightness(img: Image.Image, severity: float) -> Image.Image:
    """Adjust image brightness."""
    level = 1.0 + severity * 1.8 * random.choice([-1.0, 1.0])
    level = max(0.1, level)
    return ImageEnhance.Brightness(img).enhance(level)


def _sharpness(img: Image.Image, severity: float) -> Image.Image:
    """Adjust image sharpness."""
    level = 1.0 + severity * 1.8 * random.choice([-1.0, 1.0])
    level = max(0.1, level)
    return ImageEnhance.Sharpness(img).enhance(level)


def _posterize(img: Image.Image, severity: float) -> Image.Image:
    """Reduce the number of bits per colour channel."""
    bits = max(1, int(8 - severity * 4))
    return ImageOps.posterize(img, bits)


# All available augmentation operations (excluding colour-jitter-style ops
# that interact badly with normalisation at test time).
_AUG_OPS: List[Callable[[Image.Image, float], Image.Image]] = [
    _autocontrast,
    _equalize,
    _rotate,
    _solarize,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
    _color,
    _contrast,
    _brightness,
    _sharpness,
    _posterize,
]


# ─────────────────────────────────────────────────────────────────────────────
# AugMix core
# ─────────────────────────────────────────────────────────────────────────────

def _apply_chain(
    img: Image.Image,
    severity: float,
    depth: int,
) -> Image.Image:
    """Apply a single randomly-sampled augmentation chain.

    Each chain consists of ``depth`` operations chosen uniformly at random
    from :data:`_AUG_OPS`, applied sequentially.

    Args:
        img: PIL image to augment.
        severity: Fractional severity in [0, 1].
        depth: Number of operations to apply.

    Returns:
        Augmented PIL image.
    """
    ops = random.choices(_AUG_OPS, k=depth)
    for op in ops:
        img = op(img, severity)
    return img


def augment_and_mix(
    img: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """Apply AugMix to a single PIL image.

    Generates ``width`` augmented chains, mixes them with Dirichlet weights,
    then blends the mixture with the original image using a Beta coefficient.

    Args:
        img: Input PIL image.
        severity: Integer severity level (1–10).
        width: Number of augmentation chains.
        depth: Ops per chain; -1 samples uniformly from {1, 2, 3}.
        alpha: Concentration parameter for Dirichlet / Beta distributions.

    Returns:
        AugMix-augmented PIL image.
    """
    sev_float = severity / 10.0  # normalise to [0, 1]

    # Mixing coefficients: w_i ~ Dirichlet(alpha), m ~ Beta(alpha, alpha)
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    img_array = np.array(img, dtype=np.float32)
    mix = np.zeros_like(img_array, dtype=np.float32)

    for w in ws:
        d = random.randint(1, 3) if depth == -1 else depth
        aug = _apply_chain(img, sev_float, d)
        mix += w * np.array(aug, dtype=np.float32)

    # Blend original and mixture
    result = (1.0 - m) * img_array + m * mix
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper that returns three views per sample
# ─────────────────────────────────────────────────────────────────────────────

class AugMixDataset(Dataset):
    """Wraps a dataset and returns (x_clean, x_aug1, x_aug2, y) per sample.

    When used with the JSD consistency loss, the model receives three views of
    each image:

    - ``x_clean``: standard training augmentation (random crop + flip)
    - ``x_aug1``, ``x_aug2``: two independent AugMix views

    The base :class:`~torch.utils.data.Dataset` must return ``(PIL_image, label)``
    pairs (e.g., a raw ``torchvision.datasets.CIFAR10`` without ToTensor).

    Args:
        dataset: Source dataset yielding (PIL image, int label).
        preprocess: Transform applied to *all three* views after AugMix
                    (e.g., ``ToTensor()`` + normalisation).
        severity: AugMix severity level (1–10).
        width: Number of augmentation chains.
        depth: Ops per chain; -1 = random from {1, 2, 3}.
        alpha: Dirichlet / Beta concentration parameter.
    """

    def __init__(
        self,
        dataset: Dataset,
        preprocess: transforms.Compose,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        jsd_loss: bool = True,
    ) -> None:
        self.dataset = dataset
        self.preprocess = preprocess
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.jsd_loss = jsd_loss

        # Light augmentation applied to the clean view
        self._base_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Return three views of the sample at ``idx``.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x_clean, x_aug1, x_aug2, label).
        """
        img, label = self.dataset[idx]  # PIL image, int

        if self.jsd_loss:
            x_clean = self.preprocess(self._base_aug(img))
            x_aug1 = self.preprocess(augment_and_mix(
                img, severity=self.severity, width=self.width,
                depth=self.depth, alpha=self.alpha,
            ))
            x_aug2 = self.preprocess(augment_and_mix(
                img, severity=self.severity, width=self.width,
                depth=self.depth, alpha=self.alpha,
            ))
            return x_clean, x_aug1, x_aug2, label

        x_aug = self.preprocess(augment_and_mix(
            img, severity=self.severity, width=self.width,
            depth=self.depth, alpha=self.alpha,
        ))
        return x_aug, label


# ─────────────────────────────────────────────────────────────────────────────
# JSD consistency loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_jsd_loss(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence consistency loss across three views.

    The JSD loss encourages the model to produce similar predictions for an
    image regardless of which augmentation was applied, thereby improving
    robustness to distribution shift.

    Definition::

        p_mix = (p_clean + p_aug1 + p_aug2) / 3
        JSD   = (KL(p_clean || p_mix) + KL(p_aug1 || p_mix) + KL(p_aug2 || p_mix)) / 3

    Args:
        logits_clean: Raw logits for the clean view, shape (B, C).
        logits_aug1: Raw logits for the first augmented view, shape (B, C).
        logits_aug2: Raw logits for the second augmented view, shape (B, C).

    Returns:
        Scalar JSD loss tensor.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    # Mixture distribution
    p_mix = (p_clean + p_aug1 + p_aug2) / 3.0
    log_p_mix = torch.log(p_mix + 1e-8)

    # KL(p_i || p_mix) = sum( p_i * (log(p_i) - log(p_mix)) )
    def _kl(p: torch.Tensor) -> torch.Tensor:
        return (p * (torch.log(p + 1e-8) - log_p_mix)).sum(dim=1).mean()

    jsd = (_kl(p_clean) + _kl(p_aug1) + _kl(p_aug2)) / 3.0
    return jsd
