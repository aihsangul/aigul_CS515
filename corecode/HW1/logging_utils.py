from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from parameters import Config


def setup_experiment_logger(config: Config) -> Tuple[logging.Logger, str]:
    """
    Create a logger that writes to both terminal and a timestamped log file.

    Args:
        config: Full experiment configuration.

    Returns:
        Tuple of:
            - configured logger
            - log file path as string
    """
    log_dir = Path(config.run.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{config.run.experiment_name}_{timestamp}.log"

    logger = logging.getLogger("cs515")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear old handlers to avoid duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, str(log_path)


def save_config_snapshot(config: Config, log_path: str) -> str:
    """
    Save the full config as a JSON snapshot next to the log file.

    Args:
        config: Full experiment configuration.
        log_path: Path to the log file.

    Returns:
        Path to the saved JSON config snapshot.
    """
    log_file = Path(log_path)
    config_path = log_file.with_suffix(".config.json")

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    return str(config_path)


def log_environment_info(logger: logging.Logger) -> None:
    """
    Log basic environment information.

    Args:
        logger: Configured logger.
    """
    logger.info("Environment information")
    logger.info("Python platform: %s", platform.platform())
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    if hasattr(torch.backends, "mps"):
        logger.info("MPS available: %s", torch.backends.mps.is_available())