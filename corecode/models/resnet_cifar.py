from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18_transfer_resize(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    -> Load pre-trained ResNet-18 for CIFAR-10 using the image-resize strategy.

    CIFAR-10 images are expected to be resized to 224x224 externally (via data
    transforms) so they match the original ImageNet input dimensions. The
    backbone weights are optionally frozen and only the final FC layer is
    updated during training.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all parameters except the FC layer.

    Returns:
        ResNet-18 with ImageNet weights and an adapted FC head.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer (trainable by default after the loop above)
    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet18_transfer_modify(num_classes: int = 10) -> nn.Module:
    """Load pre-trained ResNet-18 adapted for 32x32 CIFAR-10 input.

    The original ResNet-18 stem (conv1: 7x7, stride 2 + MaxPool) aggressively
    downsamples the spatial resolution. For 32x32 inputs this collapses the
    feature maps too early. The following modifications preserve resolution:

    Modifications:
        - conv1: 7x7 stride-2 -> 3x3 stride-1, padding 1 (no bias, matching BN)
        - maxpool: replaced with "nn.Identity()"
        - fc: replaced for "num_classes"

    All layers remain trainable (full fine-tuning from ImageNet weights).

    Args:
        num_classes: Number of output classes.

    Returns:
        Modified ResNet-18 fine-tunable on 32x32 inputs.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore[assignment]

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet18_scratch(num_classes: int = 10) -> nn.Module:
    """Build ResNet-18 from random weights for CIFAR-10.

    Same stem modifications as :func:'build_resnet18_transfer_modify' but
    initialised randomly ("weights=None"). Used for the from-scratch
    baseline and as the teacher in knowledge distillation experiments.

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNet-18 with random weights and an adapted stem for 32x32 inputs.
    """
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore[assignment]

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
