from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve the repository root: corecode/HW1/tsne_analysis.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DATA_ROOT = _REPO_ROOT / "data"

import sys
sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from corecode.models import MLP
from plotting import plot_tsne_embeddings


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dictionary.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Build MNIST test DataLoader with the same normalization used in training.

    Args:
        data_dir: Dataset root directory.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        Test DataLoader.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def resolve_device(device: str) -> torch.device:
    """
    Resolve the torch device.

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


def resolve_model_config(
    history_config: Dict[str, Any],
    hidden_sizes: Optional[list[int]],
    activation: Optional[str],
    dropout: Optional[float],
    use_batch_norm: Optional[bool],
) -> Dict[str, Any]:
    """
    Resolve model configuration, preferring CLI overrides over saved history config.

    Args:
        history_config: Config block loaded from history JSON.
        hidden_sizes: Optional CLI override.
        activation: Optional CLI override.
        dropout: Optional CLI override.
        use_batch_norm: Optional CLI override.

    Returns:
        Dictionary with resolved model configuration.

    Raises:
        ValueError: If a required value cannot be resolved.
    """
    resolved_hidden_sizes = (
        hidden_sizes if hidden_sizes is not None else history_config.get("hidden_sizes")
    )
    resolved_activation = (
        activation if activation is not None else history_config.get("activation")
    )
    resolved_dropout = (
        dropout if dropout is not None else history_config.get("dropout")
    )
    resolved_use_batch_norm = (
        use_batch_norm
        if use_batch_norm is not None
        else history_config.get("use_batch_norm")
    )

    if resolved_hidden_sizes is None:
        raise ValueError("Could not resolve hidden_sizes from CLI or history config.")
    if resolved_activation is None:
        raise ValueError("Could not resolve activation from CLI or history config.")
    if resolved_dropout is None:
        raise ValueError("Could not resolve dropout from CLI or history config.")
    if resolved_use_batch_norm is None:
        raise ValueError("Could not resolve use_batch_norm from CLI or history config.")

    return {
        "hidden_sizes": resolved_hidden_sizes,
        "activation": resolved_activation,
        "dropout": resolved_dropout,
        "use_batch_norm": resolved_use_batch_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="t-SNE analysis for trained MLP features on MNIST."
    )

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)

    parser.add_argument("--checkpoint_root", type=str, default=str(_RESULTS_ROOT / "checkpoints"))
    parser.add_argument("--plot_root", type=str, default=str(_RESULTS_ROOT / "plots"))

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default=str(_DATA_ROOT))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    # Optional manual overrides; if omitted, values are read from history.json
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--activation", choices=["relu", "gelu"], default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--use_batch_norm", dest="use_batch_norm", action="store_true")
    parser.add_argument("--no_batch_norm", dest="use_batch_norm", action="store_false")
    parser.set_defaults(use_batch_norm=None)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_root) / args.project_name / args.experiment_name
    plot_dir = Path(args.plot_root) / args.project_name / args.experiment_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else checkpoint_dir / f"{args.experiment_name}_best_model.pt"
    )

    history_path = (
        Path(args.history_path)
        if args.history_path is not None
        else checkpoint_dir / f"{args.experiment_name}_history.json"
    )

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else plot_dir / f"{args.experiment_name}_tsne.png"
    )

    history = load_json(history_path)
    history_config = history.get("config", {})

    resolved_cfg = resolve_model_config(
        history_config=history_config,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
    )

    device = resolve_device(args.device)

    model = MLP(
        input_size=784,
        hidden_sizes=resolved_cfg["hidden_sizes"],
        num_classes=10,
        activation=resolved_cfg["activation"],
        dropout=resolved_cfg["dropout"],
        use_batch_norm=resolved_cfg["use_batch_norm"],
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    plot_tsne_embeddings(
        model=model,
        loader=test_loader,
        device=device,
        output_path=str(output_path),
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        random_state=args.random_state,
        title=f"t-SNE of MLP Hidden Features ({args.project_name}/{args.experiment_name})",
    )

    print("Resolved model config:", resolved_cfg)
    print("t-SNE plot saved to:", output_path)


if __name__ == "__main__":
    main()