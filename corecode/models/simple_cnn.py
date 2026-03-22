from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple convolutional neural network for CIFAR-10 classification.

    Architecture:
        Block 1: Conv(3→32) → BN → ReLU → Conv(32→64) → BN → ReLU → MaxPool(2)
        Block 2: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2)
        Block 3: Conv(128→256) → BN → ReLU → MaxPool(2)
        Classifier: Flatten → Linear(4096→512) → ReLU → Dropout → Linear(512→C)

    Designed for 32×32 RGB input (CIFAR-10). After three MaxPool(2) layers the
    spatial size is 4×4, giving 256·4·4 = 4096 features before the FC head.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        """Initialise SimpleCNN.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability applied in the classifier head.
        """
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ────────────────────────────────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 → 16
            # ── Block 2 ────────────────────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 → 8
            # ── Block 3 ────────────────────────────────────────────────────
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)
