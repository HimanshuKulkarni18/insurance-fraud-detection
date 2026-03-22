"""FastAPI app to score a single claim as JSON."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from insurance_fraud.artifacts import load_model_and_metadata
from insurance_fraud.schema import FEATURE_COLUMNS
from insurance_fraud.scoring import score_dataframe


class ClaimPayload(BaseModel):
    claim_amount: float = Field(..., gt=0)
    policy_age_months: int = Field(..., ge=1)
    num_prior_claims: int = Field(..., ge=0)
    days_to_report: int = Field(..., ge=0)
    injury_type: str
    region: str
    has_police_report: str
    repair_shop_network: str


class ScoreResponse(BaseModel):
    fraud_probability: float
    predicted_fraud: bool
    threshold_used: float


def _payload_to_frame(payload: ClaimPayload) -> Any:
    import pandas as pd

    row = {name: getattr(payload, name) for name in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def create_app(artifacts_dir: Path) -> FastAPI:
    model: Any = None
    metadata: dict[str, Any] = {}
    threshold: float = 0.5

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal model, metadata, threshold
        model, metadata = load_model_and_metadata(artifacts_dir)
        threshold = float(metadata.get("threshold", 0.5))
        yield

    app = FastAPI(title="Insurance fraud scoring", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score_claim(body: ClaimPayload) -> ScoreResponse:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        X = _payload_to_frame(body)
        scored = score_dataframe(model, X, threshold)
        prob = float(scored["fraud_probability"].iloc[0])
        return ScoreResponse(
            fraud_probability=prob,
            predicted_fraud=bool(scored["predicted_fraud"].iloc[0]),
            threshold_used=threshold,
        )

    return app


def _default_artifacts_dir() -> Path:
    import os

    return Path(os.environ.get("INSURANCE_FRAUD_ARTIFACTS_DIR", "artifacts"))


app = create_app(_default_artifacts_dir())
