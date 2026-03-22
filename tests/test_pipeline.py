"""Tests for preprocessing + classifier pipeline."""

from __future__ import annotations

import numpy as np

from insurance_fraud.pipeline import build_fitted_pipeline
from insurance_fraud.schema import FEATURE_COLUMNS, TARGET_COL
from insurance_fraud.synthetic import generate_claims


def test_pipeline_fit_and_predict_proba() -> None:
    df = generate_claims(200, fraud_rate=0.1, seed=0)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL].astype(np.int64)
    model = build_fitted_pipeline(0, calibrate=False)
    model.fit(X, y)
    prob = model.predict_proba(X)
    assert prob.shape == (len(df), 2)
    assert np.all((prob >= 0) & (prob <= 1))
