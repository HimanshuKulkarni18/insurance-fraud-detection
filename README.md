# Insurance claim fraud detection






Python prototype for **binary fraud risk** on insurance claims: synthetic data, **scikit-learn** pipeline (gradient boosting + optional calibration), **CLI** batch scoring, **FastAPI** JSON API, and a **Streamlit** web UI.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## Highlights

- **Imbalanced fraud**: training reports **ROC-AUC**, **PR-AUC (average precision)**, F1 at 0.5 and at an F1-tuned threshold, plus “top 5% score” stats — not raw accuracy alone.
- **Artifacts**: `artifacts/model.joblib`, `metadata.json` (threshold + full metadata), `metrics.json` (validation/test metrics for dashboards or CI).
- **One scoring path** for CLI, API, and UI via `insurance_fraud.scoring`.

## Quick start

```bash
git clone <your-repo-url>
cd insurance-fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_data.py --output data/raw/claims.csv --n-samples 20000 --fraud-rate 0.06 --seed 42
python scripts/train.py --data data/raw/claims.csv --artifacts-dir artifacts --seed 42
streamlit run ui/streamlit_app.py
```

## Documentation

| Doc | Content |
|-----|--------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Full workflow, schema, metrics, API/UI usage |
| [docs/GITHUB.md](docs/GITHUB.md) | **Secure** GitHub setup — **do not share passwords or tokens** |

## Commands (summary)

| Task | Command |
|------|--------|
| Generate CSV | `python scripts/generate_data.py --output data/raw/claims.csv` |
| Train | `python scripts/train.py --data data/raw/claims.csv --artifacts-dir artifacts` |
| Fast train (no calibration) | add `--no-calibrate` |
| CLI scores | `python scripts/predict.py --artifacts-dir artifacts --input claims.csv --output scores.csv` |
| Web UI | `streamlit run ui/streamlit_app.py` |
| API | `uvicorn insurance_fraud.api:app --app-dir src --reload` |
| Tests | `pytest tests/ -v` |

## Project layout

```
insurance-fraud-detection/
├── data/raw/              # generated CSVs (gitignored)
├── scripts/               # generate_data, train, predict
├── src/insurance_fraud/   # schema, pipeline, scoring, API
├── ui/streamlit_app.py    # browser UI
├── tests/                 # pytest
└── docs/                  # user guide + GitHub how-to
```

## Security note

This is a **prototype / educational** model on **synthetic** data. Do not use production claim decisions without domain review, fairness checks, and a proper MLOps process.

## License

MIT — see [LICENSE](LICENSE).
