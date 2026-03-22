#!/usr/bin/env python3
"""Generate synthetic labeled insurance claims for fraud detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from insurance_fraud.schema import TARGET_COL
    from insurance_fraud.synthetic import generate_claims

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/claims.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.06,
        help="Approximate fraction of fraudulent claims",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_claims(args.n_samples, args.fraud_rate, args.seed)
    df.to_csv(args.output, index=False)
    fraud_n = int(df[TARGET_COL].sum())
    print(f"Wrote {len(df)} rows ({fraud_n} fraud) to {args.output}")


if __name__ == "__main__":
    main()
