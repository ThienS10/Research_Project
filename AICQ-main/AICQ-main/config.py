import json
import os
from typing import Any, Dict, Optional


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


"""
Configuration loader

This module now treats config.json as the single source of truth. No Python-side
defaults are provided here; missing keys in config.json will raise errors when
accessed. Edit config.json to change runtime behavior.
"""


 


def _abs_path(path_value: str) -> str:
    """Resolve a possibly relative path against the project directory.

    Args:
        path_value: Absolute or relative path.

    Returns:
        str: Absolute path.
    """
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_DIR, path_value)


def get_config() -> Dict[str, Any]:
    """Load configuration strictly from config.json and resolve paths.

    Returns:
        Dict[str, Any]: The loaded and resolved configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = os.path.join(PROJECT_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    # Resolve file paths relative to project dir
    data_cfg = config.get("data", {})
    if "csv_path" in data_cfg:
        data_cfg["csv_path"] = _abs_path(data_cfg["csv_path"])
    if "csv_option1" in data_cfg:
        data_cfg["csv_option1"] = _abs_path(data_cfg["csv_option1"])
    if "csv_option2" in data_cfg:
        data_cfg["csv_option2"] = _abs_path(data_cfg["csv_option2"])
    if "json_option1" in data_cfg:
        data_cfg["json_option1"] = _abs_path(data_cfg["json_option1"])
    if "json_option2" in data_cfg:
        data_cfg["json_option2"] = _abs_path(data_cfg["json_option2"])
    config["data"] = data_cfg

    if "model" in config and "path" in config["model"]:
        config["model"]["path"] = _abs_path(config["model"]["path"])

    if "logging" in config and "file" in config["logging"]:
        config["logging"]["file"] = _abs_path(config["logging"]["file"])

    _CONFIG_CACHE = config
    return config


