"""Sklearn preprocessing + classifier pipeline."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance_fraud.schema import FEATURE_COLUMNS

NUMERIC_FEATURES = [
    "claim_amount",
    "policy_age_months",
    "num_prior_claims",
    "days_to_report",
]
CATEGORICAL_FEATURES = [
    "injury_type",
    "region",
    "has_police_report",
    "repair_shop_network",
]


def build_preprocess_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_base_classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=random_state,
    )


def build_fitted_pipeline(
    random_state: int,
    *,
    calibrate: bool = True,
    cv: int = 3,
) -> Pipeline | CalibratedClassifierCV:
    preprocess = build_preprocess_transformer()
    clf = build_base_classifier(random_state)
    inner = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", clf),
        ]
    )
    if not calibrate:
        return inner
    return CalibratedClassifierCV(inner, method="isotonic", cv=cv)


def feature_matrix_columns() -> list[str]:
    """Columns used as model input (order matches training)."""
    return list(FEATURE_COLUMNS)
