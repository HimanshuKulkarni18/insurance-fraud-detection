#!/usr/bin/env python3
"""Score claims with a trained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import pandas as pd

    from insurance_fraud.artifacts import load_model_and_metadata
    from insurance_fraud.scoring import score_dataframe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with claim features")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    model, metadata = load_model_and_metadata(args.artifacts_dir)
    threshold = float(metadata.get("threshold", 0.5))

    df = pd.read_csv(args.input)
    out = score_dataframe(model, df, threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
