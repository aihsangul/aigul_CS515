from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenetv2_cifar(num_classes: int = 10) -> nn.Module:
    """Build MobileNetV2 for CIFAR-10 classification from random initialisation.

    The standard MobileNetV2 stem uses a stride-2 convolution followed by
    stride-2 inverted residuals, quickly reducing a 32×32 input to 1×1 before
    the classifier. To prevent this, the first convolution's stride is changed
    from 2 to 1 so the spatial resolution is 32×32 throughout the early layers.

    Modifications:
        - ``features[0][0]``: stride 2 → 1 (keeps 32×32 spatial size)
        - ``classifier[1]``: replaced for ``num_classes``

    Args:
        num_classes: Number of output classes.

    Returns:
        MobileNetV2 adapted for 32×32 CIFAR-10 inputs, randomly initialised.
    """
    model = models.mobilenet_v2(weights=None)

    # Adapt the first conv for 32×32 inputs
    first_conv: nn.Conv2d = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        first_conv.in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=1,                       # was 2
        padding=first_conv.padding,
        bias=False,
    )

    # Adapt classifier head
    last_channel: int = model.last_channel
    model.classifier[1] = nn.Linear(last_channel, num_classes)

    return model
