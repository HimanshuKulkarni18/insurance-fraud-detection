# User guide — insurance claim fraud detection

This project scores **insurance claims** for **fraud risk** using a machine learning model trained on tabular features (amounts, timing, injury type, region, police report, repair network, etc.). It ships with a **synthetic data generator** so you can run everything locally without external datasets.

## What you get

| Piece | Purpose |
|--------|--------|
| `scripts/generate_data.py` | Build a labeled CSV (`is_fraud`) |
| `scripts/train.py` | Train model, write `artifacts/model.joblib`, `metadata.json`, `metrics.json` |
| `scripts/predict.py` | Batch-score a CSV from the command line |
| `insurance_fraud.api` | JSON API (`POST /score`) for integrations |
| `ui/streamlit_app.py` | Browser UI for single claims and batch CSV |

## Install

```bash
cd insurance-fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## End-to-end workflow

1. **Generate data**

   ```bash
   python scripts/generate_data.py --output data/raw/claims.csv --n-samples 20000 --fraud-rate 0.06 --seed 42
   ```

2. **Train**

   ```bash
   python scripts/train.py --data data/raw/claims.csv --artifacts-dir artifacts --seed 42
   ```

   Training prints **validation** and **test** metrics: ROC-AUC, **average precision (PR-AUC)** — important when fraud is rare — F1 at 0.5 and at an F1-tuned threshold, plus a “top 5% score” threshold summary.

   For faster iteration or very small CSVs, use `--no-calibrate` (skips probability calibration).

3. **Score with CLI**

   ```bash
   python scripts/predict.py --artifacts-dir artifacts --input data/raw/claims.csv --output scores.csv
   ```

   Output adds `fraud_probability` and `predicted_fraud` (using the threshold saved in `metadata.json`).

4. **Web UI (Streamlit)**

   ```bash
   streamlit run ui/streamlit_app.py
   ```

   Open the URL shown in the terminal. Point **Artifacts directory** at the folder that contains `model.joblib` and `metadata.json` (default: `artifacts/`). Use **Single claim** or upload a **Batch CSV**.

5. **REST API**

   ```bash
   uvicorn insurance_fraud.api:app --app-dir src --reload
   ```

   - `GET /health` — liveness  
   - `POST /score` — JSON body with the eight feature fields (see `ClaimPayload` in `src/insurance_fraud/api.py`)

   Optional: `INSURANCE_FRAUD_ARTIFACTS_DIR=/path/to/artifacts` to change where the app loads models from.

## Feature schema

Required columns for scoring (training CSV must include these plus `claim_id` and `is_fraud`):

- `claim_amount` (float)  
- `policy_age_months` (int)  
- `num_prior_claims` (int)  
- `days_to_report` (int)  
- `injury_type`: `minor`, `moderate`, `severe`  
- `region`: `NE`, `SE`, `MW`, `SW`, `W`  
- `has_police_report`: `yes`, `no`  
- `repair_shop_network`: `in_network`, `out_of_network`, `unknown`

## Metrics and thresholds

- The **saved threshold** is chosen on the **validation** set to maximize **F1** on predicted probabilities.  
- **Do not rely on accuracy alone** when fraud is rare; use PR-AUC and business rules for how many cases you can review.  
- `artifacts/metrics.json` mirrors the metrics block embedded in `metadata.json` for dashboards or CI.

## Tests

```bash
pytest tests/ -v
```

## Replacing synthetic data

Use your own CSV if column names and categorical levels match (or extend `src/insurance_fraud/schema.py` and retrain). Unknown categorical values at inference are handled via `OneHotEncoder(handle_unknown="ignore")`.
