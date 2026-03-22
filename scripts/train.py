#!/usr/bin/env python3
"""Train fraud model, evaluate, save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    from insurance_fraud.artifacts import (
        METADATA_FILENAME,
        METRICS_FILENAME,
        MODEL_FILENAME,
        save_json,
    )
    from insurance_fraud.pipeline import build_fitted_pipeline, feature_matrix_columns
    from insurance_fraud.schema import DTYPES, FEATURE_COLUMNS, TARGET_COL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Training CSV path")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for model.joblib and metadata.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip probability calibration (faster, fewer samples needed)",
    )
    parser.add_argument("--calibration-cv", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    for col, dtype in DTYPES.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if dtype == "category":
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(dtype)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL].astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=args.seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=args.seed,
    )

    model = build_fitted_pipeline(
        args.seed,
        calibrate=not args.no_calibrate,
        cv=max(2, args.calibration_cv),
    )
    model.fit(X_train, y_train)

    def scores(X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        return proba[:, 1]

    def eval_split(name: str, Xs: pd.DataFrame, ys: np.ndarray) -> dict[str, Any]:
        prob = scores(Xs)
        roc = float(roc_auc_score(ys, prob))
        ap = float(average_precision_score(ys, prob))
        prec, rec, thresh = precision_recall_curve(ys, prob)
        if len(thresh) == 0:
            thr_f1 = 0.5
        else:
            f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
            idx = int(np.argmax(f1s))
            thr_f1 = float(thresh[idx])
        pred_default = (prob >= 0.5).astype(np.int64)
        pred_f1 = (prob >= thr_f1).astype(np.int64)
        top5_thr = float(np.quantile(prob, 0.95))
        pred_top5 = (prob >= top5_thr).astype(np.int64)

        cm_default = confusion_matrix(ys, pred_default).tolist()
        report_default = classification_report(
            ys, pred_default, output_dict=True, zero_division=0
        )

        return {
            "roc_auc": roc,
            "average_precision": ap,
            "threshold_f1_optimal": thr_f1,
            "f1_at_0.5": float(f1_score(ys, pred_default, zero_division=0)),
            "f1_at_f1_optimal_threshold": float(f1_score(ys, pred_f1, zero_division=0)),
            "threshold_top_5pct_score": top5_thr,
            "fraction_flagged_top_5pct": float(np.mean(pred_top5)),
            "confusion_matrix_at_0.5": cm_default,
            "classification_report_at_0.5": report_default,
        }

    val_metrics = eval_split("validation", X_val, y_val.values)
    test_metrics = eval_split("test", X_test, y_test.values)

    chosen_threshold = val_metrics["threshold_f1_optimal"]

    metadata: dict[str, Any] = {
        "feature_columns": feature_matrix_columns(),
        "target_column": TARGET_COL,
        "threshold": chosen_threshold,
        "threshold_note": "F1-optimal on validation set probabilities",
        "random_seed": args.seed,
        "calibrated": not args.no_calibrate,
        "splits": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifacts_dir / MODEL_FILENAME)
    save_json(args.artifacts_dir / METADATA_FILENAME, metadata)
    save_json(args.artifacts_dir / METRICS_FILENAME, metadata["metrics"])

    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved model and metadata to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
