"""Column names and dtypes for claim records."""

from __future__ import annotations

TARGET_COL = "is_fraud"
ID_COL = "claim_id"

FEATURE_COLUMNS: list[str] = [
    "claim_amount",
    "policy_age_months",
    "num_prior_claims",
    "days_to_report",
    "injury_type",
    "region",
    "has_police_report",
    "repair_shop_network",
]

ALL_COLUMNS: list[str] = [ID_COL, *FEATURE_COLUMNS, TARGET_COL]

DTYPES: dict[str, str] = {
    ID_COL: "string",
    "claim_amount": "float64",
    "policy_age_months": "int64",
    "num_prior_claims": "int64",
    "days_to_report": "int64",
    "injury_type": "category",
    "region": "category",
    "has_police_report": "category",
    "repair_shop_network": "category",
    TARGET_COL: "int64",
}

INJURY_TYPES = ("minor", "moderate", "severe")
REGIONS = ("NE", "SE", "MW", "SW", "W")
POLICE = ("yes", "no")
SHOP_NETWORK = ("in_network", "out_of_network", "unknown")
