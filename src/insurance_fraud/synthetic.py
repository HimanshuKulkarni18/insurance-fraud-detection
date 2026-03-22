"""Synthetic claim data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_fraud.schema import (
    ALL_COLUMNS,
    DTYPES,
    ID_COL,
    INJURY_TYPES,
    POLICE,
    REGIONS,
    SHOP_NETWORK,
    TARGET_COL,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_claims(
    n_samples: int,
    fraud_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    injury = rng.choice(np.array(INJURY_TYPES), size=n_samples)
    region = rng.choice(np.array(REGIONS), size=n_samples)
    police = rng.choice(np.array(POLICE), size=n_samples, p=[0.55, 0.45])
    shop = rng.choice(np.array(SHOP_NETWORK), size=n_samples, p=[0.5, 0.35, 0.15])

    claim_amount = rng.lognormal(mean=8.0, sigma=1.1, size=n_samples).clip(200, 500_000)
    policy_age_months = rng.integers(1, 240, size=n_samples)
    num_prior_claims = rng.poisson(0.8, size=n_samples).clip(0, 15)
    days_to_report = rng.integers(0, 120, size=n_samples)

    severe = (injury == "severe").astype(float)
    no_police = (police == "no").astype(float)
    out_net = (shop == "out_of_network").astype(float)
    high_prior = (num_prior_claims >= 3).astype(float)
    fast_report = (days_to_report <= 1).astype(float)
    amt_z = (np.log1p(claim_amount) - 8.5) / 1.0

    logit = (
        -3.2
        + 1.4 * amt_z
        + 0.9 * severe
        + 0.7 * no_police
        + 0.8 * out_net
        + 0.5 * high_prior
        + 0.4 * fast_report
        + 0.15 * (policy_age_months < 6).astype(float)
        + rng.normal(0, 0.35, size=n_samples)
    )
    p_fraud = _sigmoid(logit)

    target_rate = np.clip(fraud_rate, 0.01, 0.4)
    threshold = float(np.quantile(p_fraud, 1.0 - target_rate))
    is_fraud = (p_fraud >= threshold).astype(np.int64)

    noise_flip = rng.random(n_samples) < 0.02
    is_fraud = np.where(noise_flip, 1 - is_fraud, is_fraud)

    claim_ids = np.array([f"C{i:08d}" for i in range(n_samples)])

    df = pd.DataFrame(
        {
            ID_COL: claim_ids,
            "claim_amount": claim_amount.astype(np.float64),
            "policy_age_months": policy_age_months.astype(np.int64),
            "num_prior_claims": num_prior_claims.astype(np.int64),
            "days_to_report": days_to_report.astype(np.int64),
            "injury_type": injury,
            "region": region,
            "has_police_report": police,
            "repair_shop_network": shop,
            TARGET_COL: is_fraud,
        }
    )

    for col, dtype in DTYPES.items():
        if dtype == "category":
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(dtype)

    return df[ALL_COLUMNS]
