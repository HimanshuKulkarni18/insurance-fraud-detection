"""Unit tests for JSON helpers and scoring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from insurance_fraud.artifacts import save_json
from insurance_fraud.jsonutil import to_json_serializable
from insurance_fraud.pipeline import build_fitted_pipeline
from insurance_fraud.schema import FEATURE_COLUMNS, TARGET_COL
from insurance_fraud.scoring import score_dataframe
from insurance_fraud.synthetic import generate_claims


def test_to_json_serializable_numpy() -> None:
    raw = {"x": np.float64(1.5), "y": np.int64(3), "z": np.array([1, 2])}
    out = to_json_serializable(raw)
    json.dumps(out)
    assert out["x"] == 1.5
    assert out["y"] == 3.0
    assert out["z"] == [1, 2]


def test_save_json_nested_numpy(tmp_path) -> None:
    path = tmp_path / "x.json"
    save_json(path, {"m": {"a": np.float32(0.25)}}, safe=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["m"]["a"] == 0.25


def test_score_dataframe_raises_on_missing_columns() -> None:
    df = generate_claims(5, 0.2, 0)[FEATURE_COLUMNS + [TARGET_COL]]
    model = build_fitted_pipeline(0, calibrate=False)
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COL])
    bad = df[FEATURE_COLUMNS].drop(columns=["claim_amount"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        score_dataframe(model, bad, 0.5)


def test_score_dataframe_adds_columns() -> None:
    df = generate_claims(20, 0.15, 1)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    model = build_fitted_pipeline(1, calibrate=False)
    model.fit(X, y)
    out = score_dataframe(model, X, 0.4)
    assert "fraud_probability" in out.columns
    assert "predicted_fraud" in out.columns
    assert len(out) == len(df)
