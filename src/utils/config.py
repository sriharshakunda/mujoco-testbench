"""Configuration loading and management utilities.

This module provides utilities for loading and managing configuration from
YAML files, with support for command-line overrides and validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If configuration file does not exist
        yaml.YAMLError: If YAML is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path where to save configuration

    Raises:
        IOError: If configuration cannot be saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    except IOError as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override configuration into base configuration.

    Recursively merges override config into base config, with override values
    taking precedence over base values.

    Args:
        base: Base configuration dictionary
        override: Configuration dictionary to merge in

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """Get configuration value using dot notation.

    Supports nested access like 'model.hidden_dims' or 'training.learning_rate'.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to configuration value
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that configuration contains required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports dot notation)

    Returns:
        True if all required keys present

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []

    for key in required_keys:
        if get_config_value(config, key) is None:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return True
