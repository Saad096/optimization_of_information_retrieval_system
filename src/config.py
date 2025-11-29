from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# Project root = folder that contains `src/`, `configs/`, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_config_path(path: Optional[str] = None) -> Path:
    """
    Resolve the path to the YAML config file.

    If `path` is absolute, use it directly.
    If `path` is relative, it is interpreted relative to PROJECT_ROOT.
    If `path` is None, default to `configs/config.yaml` under PROJECT_ROOT.
    """
    if path is not None:
        p = Path(path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    return PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the YAML config file and return it as a Python dict.
    """
    cfg_path = get_config_path(path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg
