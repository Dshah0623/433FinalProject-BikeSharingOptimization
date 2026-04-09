from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else project_root() / "configs" / "project_config.yml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project_root"] = str(project_root())
    return config
