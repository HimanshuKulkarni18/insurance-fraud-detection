"""Artifact filenames and loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from insurance_fraud.jsonutil import to_json_serializable

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


def artifacts_paths(artifacts_dir: Path) -> tuple[Path, Path]:
    return artifacts_dir / MODEL_FILENAME, artifacts_dir / METADATA_FILENAME


def load_model_and_metadata(artifacts_dir: Path) -> tuple[Any, dict[str, Any]]:
    model_path, meta_path = artifacts_paths(artifacts_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    model = joblib.load(model_path)
    with meta_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def save_json(path: Path, data: dict[str, Any], *, safe: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json_serializable(data) if safe else data
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
