"""
src/config.py
=============
Loads and exposes the project-wide configuration defined in config.yaml.
All other modules should import their settings from here — never hardcode paths.
"""

from pathlib import Path
import yaml

# Project root is two levels above this file: src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml and resolve all relative paths to absolute paths.

    Parameters
    ----------
    config_path : Path, optional
        Explicit path to the YAML file. Defaults to PROJECT_ROOT/config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary with 'paths' values as absolute Path objects.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in the 'paths' section to absolute Path objects
    for key, rel_path in cfg.get("paths", {}).items():
        abs_path = PROJECT_ROOT / rel_path
        cfg["paths"][key] = abs_path
        abs_path.mkdir(parents=True, exist_ok=True)  # ensure directories exist

    return cfg


# Singleton — loaded once at import time for convenience
CFG = load_config()
