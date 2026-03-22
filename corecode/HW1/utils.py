from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn


PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Ensure the parent directory of a file path exists.

    Args:
        path: File path.

    Returns:
        Parent directory path.
    """
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def save_json(data: Dict[str, Any], path: PathLike) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data: Dictionary to save.
        path: Output file path.
    """
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: PathLike) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Args:
        path: Input file path.

    Returns:
        Loaded dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters of a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve a concrete torch device.

    Args:
        device: Requested device string.

    Returns:
        torch.device object.
    """
    device = device.lower()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)