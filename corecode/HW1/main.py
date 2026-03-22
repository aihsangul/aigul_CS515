import logging
import os
import random
import ssl
import sys
from pathlib import Path

# Add repo root to sys.path so corecode package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from logging_utils import (
    log_environment_info,
    save_config_snapshot,
    setup_experiment_logger,
)
from corecode.models import MLP
from parameters import Config, get_config
from test import run_test
from train import run_training


ssl._create_default_https_context = ssl._create_unverified_context


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


def ensure_output_paths(config: Config) -> None:
    """
    Create output directories if they do not already exist.

    Args:
        config: Full experiment configuration.
    """
    os.makedirs(config.run.checkpoint_dir, exist_ok=True)
    os.makedirs(config.run.log_dir, exist_ok=True)
    os.makedirs(config.run.plot_dir, exist_ok=True)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(config: Config) -> nn.Module:
    """
    Build the model specified by the configuration.

    Args:
        config: Full experiment configuration.

    Returns:
        Instantiated PyTorch model.
    """
    if config.data.dataset != "mnist":
        raise ValueError(
            f"Unsupported dataset '{config.data.dataset}'. "
            "HW1a is scoped to MNIST classification with MLP."
        )

    model = MLP(
        input_size=config.model.input_size,
        hidden_sizes=config.model.hidden_sizes,
        num_classes=config.model.num_classes,
        activation=config.model.activation,
        dropout=config.model.dropout,
        use_batch_norm=config.model.use_batch_norm,
    )
    return model


def log_config_summary(
    logger: logging.Logger,
    config: Config,
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    Log a concise experiment summary.

    Args:
        logger: Configured logger.
        config: Full experiment configuration.
        model: Instantiated model.
        device: Torch device used for execution.
    """
    logger.info("=" * 70)
    logger.info("CS515 HW1a - MNIST Classification with MLP")
    logger.info("=" * 70)
    logger.info("Mode             : %s", config.run.mode)
    logger.info("Project name     : %s", config.run.project_name)
    logger.info("Experiment name  : %s", config.run.experiment_name)
    logger.info("Seed             : %s", config.run.seed)
    logger.info("Device           : %s", device)
    logger.info("Dataset          : %s", config.data.dataset)
    logger.info("Input size       : %s", config.data.input_size)
    logger.info("Num classes      : %s", config.data.num_classes)
    logger.info("Hidden sizes     : %s", config.model.hidden_sizes)
    logger.info("Activation       : %s", config.model.activation)
    logger.info("Dropout          : %s", config.model.dropout)
    logger.info("BatchNorm        : %s", config.model.use_batch_norm)
    logger.info("Epochs           : %s", config.train.epochs)
    logger.info("Batch size       : %s", config.train.batch_size)
    logger.info("Learning rate    : %s", config.train.learning_rate)
    logger.info("Optimizer        : %s", config.train.optimizer)
    logger.info("Scheduler        : %s", config.train.scheduler)
    logger.info("Regularizer      : %s", config.train.regularizer)
    logger.info("reg_lambda       : %s", config.train.reg_lambda)
    logger.info("Weight decay     : %s", config.train.weight_decay)
    logger.info("Checkpoint dir   : %s", config.run.checkpoint_dir)
    logger.info("Checkpoint path  : %s", config.run.save_path)
    logger.info("Log directory    : %s", config.run.log_dir)
    logger.info("Plot directory   : %s", config.run.plot_dir)
    logger.info("Trainable params : %s", f"{count_trainable_parameters(model):,}")
    logger.info("=" * 70)
    logger.info("Model architecture:\n%s", model)
    logger.info("=" * 70)


def main() -> None:
    """
    Main entry point for running training and/or testing.
    """
    config = get_config()

    set_seed(config.run.seed)
    ensure_output_paths(config)

    logger, log_path = setup_experiment_logger(config)
    config_snapshot_path = save_config_snapshot(config, log_path)

    try:
        device = torch.device(config.run.device)
        model = build_model(config).to(device)

        logger.info("Log file path: %s", log_path)
        logger.info("Config snapshot path: %s", config_snapshot_path)
        log_environment_info(logger)
        log_config_summary(logger, config, model, device)

        if config.run.mode in ("train", "both"):
            run_training(model=model, config=config, device=device)

        if config.run.mode in ("test", "both"):
            if not os.path.exists(config.run.save_path) and config.run.mode == "test":
                raise FileNotFoundError(
                    f"Checkpoint not found at: {config.run.save_path}. "
                    "Train the model first or provide a valid checkpoint path."
                )
            run_test(model=model, config=config, device=device)

        logger.info("Run completed successfully.")

    except Exception as exc:
        logger.exception("Run failed with an exception: %s", exc)
        raise


if __name__ == "__main__":
    main()