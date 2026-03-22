"""Convert nested structures to JSON-serializable Python types."""

from __future__ import annotations

from typing import Any


def to_json_serializable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [to_json_serializable(x) for x in obj]
    return obj
