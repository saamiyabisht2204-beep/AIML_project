"""
Logging and configuration utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict
import yaml


def setup_logger(name: str = "AIML_Project", log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a formatted console and file logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration file safely.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
