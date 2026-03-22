"""Shared inference helpers for CLI, API, and UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insurance_fraud.schema import DTYPES, FEATURE_COLUMNS, ID_COL, TARGET_COL


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def align_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in DTYPES.items():
        if col not in out.columns or col == TARGET_COL:
            continue
        if col == ID_COL:
            continue
        if dtype == "category":
            out[col] = out[col].astype("category")
        else:
            out[col] = out[col].astype(dtype)
    return out


def score_dataframe(model: Any, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    validate_feature_columns(df)
    aligned = align_feature_dtypes(df)
    X = aligned[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba
    out["predicted_fraud"] = (proba >= threshold).astype(np.int64)
    return out
