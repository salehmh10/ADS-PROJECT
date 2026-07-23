"""Production JSON normalization and atomic writing for Stage 5A2 recovery 2."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_json(value: Any) -> Any:
    """Convert supported Python, NumPy, pandas, and pathlib values to strict JSON types."""
    if value is None:
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, np.ndarray):
        return [normalize_json(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [normalize_json(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [normalize_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(normalize_json(key)): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=lambda item: str(item))
        return [normalize_json(item) for item in items]
    if hasattr(value, "item"):
        return normalize_json(value.item())
    raise TypeError(f"Unsupported JSON proof value: {type(value).__module__}.{type(value).__name__}")


def atomic_json(payload: Any, path: Path) -> None:
    """Normalize, write strict JSON to a sibling temporary file, then atomically replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        normalized = normalize_json(payload)
        encoded = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
