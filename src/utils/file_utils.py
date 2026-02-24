"""File utility functions."""
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    """Save a dict to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_prompt(path: str | Path) -> str:
    """Load a prompt template from a markdown file."""
    with open(path, "r") as f:
        return f.read()


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
