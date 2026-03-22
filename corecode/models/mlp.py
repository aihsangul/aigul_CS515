from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable multi-layer perceptron for image classification.

    This model is designed for homework experiments on MNIST and supports:
    - Variable number of hidden layers
    - Variable hidden widths
    - ReLU / GELU activation
    - Optional BatchNorm1d
    - Optional Dropout

    The input is flattened with nn.Flatten before being passed through the
    fully connected layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_size: Flattened input dimension. For MNIST, this is 784.
            hidden_sizes: Sizes of hidden layers, e.g. [512, 256, 128].
            num_classes: Number of output classes.
            activation: Activation function name, either "relu" or "gelu".
            dropout: Dropout probability. Set to 0.0 to disable dropout.
            use_batch_norm: Whether to use BatchNorm1d after linear layers.

        Raises:
            ValueError: If activation is unsupported or dropout is invalid.
        """
        super().__init__()

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes
        self.activation_name = activation.lower()
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        self.flatten = nn.Flatten()

        self.hidden_blocks = nn.ModuleList()
        in_features = input_size

        for hidden_dim in self.hidden_sizes:
            block_layers = [nn.Linear(in_features, hidden_dim)]

            if self.use_batch_norm:
                block_layers.append(nn.BatchNorm1d(hidden_dim))

            block_layers.append(self._make_activation())

            if self.dropout_rate > 0.0:
                block_layers.append(nn.Dropout(self.dropout_rate))

            self.hidden_blocks.append(nn.Sequential(*block_layers))
            in_features = hidden_dim

        self.classifier = nn.Linear(in_features, num_classes)

    def _make_activation(self) -> nn.Module:
        """
        Build the requested activation module.

        Returns:
            A PyTorch activation module.

        Raises:
            ValueError: If activation type is unsupported.
        """
        if self.activation_name == "relu":
            return nn.ReLU()
        if self.activation_name == "gelu":
            return nn.GELU()

        raise ValueError(
            f"Unsupported activation '{self.activation_name}'. "
            "Choose either 'relu' or 'gelu'."
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hidden representation before the final classifier.

        Args:
            x: Input tensor of shape (B, C, H, W) or already flattened.

        Returns:
            Hidden feature tensor after the last hidden layer.
        """
        x = self.flatten(x)

        for block in self.hidden_blocks:
            x = block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, input_size).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits