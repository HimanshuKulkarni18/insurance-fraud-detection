"""Smoke tests for training and CLI predict."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from insurance_fraud.artifacts import load_model_and_metadata
from insurance_fraud.synthetic import generate_claims


def test_train_predict_scripts(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "claims.csv"
    generate_claims(2500, fraud_rate=0.08, seed=7).to_csv(csv_path, index=False)
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
            "3",
        ],
        cwd=project,
        check=True,
    )
    model, meta = load_model_and_metadata(art)
    assert model is not None
    assert "threshold" in meta
    assert (art / "metrics.json").is_file()
    out_csv = tmp_path / "scored.csv"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/predict.py"),
            "--artifacts-dir",
            str(art),
            "--input",
            str(csv_path),
            "--output",
            str(out_csv),
        ],
        cwd=project,
        check=True,
    )
    text = out_csv.read_text(encoding="utf-8")
    assert "fraud_probability" in text
    assert "predicted_fraud" in text
