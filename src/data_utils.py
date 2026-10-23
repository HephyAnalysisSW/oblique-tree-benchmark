from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
# Allow command-line entry points under src/ to import project common utilities.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(handle)
        else:
            config = json.load(handle)
    config["_config_path"] = str(path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_metadata(path: str | Path = "data/metadata.json") -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    arrays = np.load(resolve(data_cfg["dataset_path"]))
    metadata = load_metadata(data_cfg.get("metadata_path", "data/metadata.json"))
    return {
        "X_train": arrays["X_train"],
        "y_train": arrays["y_train"],
        "X_val": arrays["X_val"],
        "y_val": arrays["y_val"],
        "X_test": arrays["X_test"],
        "y_test": arrays["y_test"],
        "scaler_mean": arrays["scaler_mean"] if "scaler_mean" in arrays else None,
        "scaler_scale": arrays["scaler_scale"] if "scaler_scale" in arrays else None,
        "metadata": metadata,
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
