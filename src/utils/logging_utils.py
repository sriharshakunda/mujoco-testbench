"""Logging utilities for training and evaluation.

Provides centralized logging configuration for both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_file: str = "training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logging with both console and file handlers.

    Args:
        log_dir: Directory to save log files (optional)
        log_file: Name of log file
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance by name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
