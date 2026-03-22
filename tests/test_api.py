"""FastAPI scoring endpoint smoke test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from insurance_fraud.api import create_app
from insurance_fraud.synthetic import generate_claims


def test_score_endpoint(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "claims.csv"
    generate_claims(800, fraud_rate=0.1, seed=11).to_csv(csv_path, index=False)
    art = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/train.py"),
            "--data",
            str(csv_path),
            "--artifacts-dir",
            str(art),
            "--no-calibrate",
            "--seed",
            "5",
        ],
        cwd=project,
        check=True,
    )

    app = create_app(art)
    payload = {
        "claim_amount": 12000.0,
        "policy_age_months": 24,
        "num_prior_claims": 1,
        "days_to_report": 3,
        "injury_type": "minor",
        "region": "NE",
        "has_police_report": "yes",
        "repair_shop_network": "in_network",
    }
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        r2 = client.post("/score", json=payload)
        assert r2.status_code == 200
        data = r2.json()
    assert "fraud_probability" in data
    assert "predicted_fraud" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
