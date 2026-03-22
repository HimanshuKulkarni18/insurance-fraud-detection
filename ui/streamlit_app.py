"""
Insurance claim fraud risk — Streamlit UI.

Run from project root:
  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from insurance_fraud.artifacts import load_model_and_metadata
from insurance_fraud.schema import (
    INJURY_TYPES,
    POLICE,
    REGIONS,
    SHOP_NETWORK,
)
from insurance_fraud.scoring import score_dataframe

st.set_page_config(
    page_title="Claim fraud risk",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_scorer(artifacts_dir: str):
    path = Path(artifacts_dir)
    return load_model_and_metadata(path)


def main() -> None:
    st.markdown(
        """
        <style>
        .hero { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;
                letter-spacing: -0.02em; color: #0f172a; }
        .sub { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="hero">Insurance claim fraud risk</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub">Score a single claim or upload a CSV batch. Probabilities come from your '
        "trained model (calibrated if you trained with calibration enabled).</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model")
        default_art = ROOT / "artifacts"
        art_dir = st.text_input(
            "Artifacts directory",
            value=str(default_art),
            help="Folder containing model.joblib and metadata.json",
        )
        if st.button("Reload model", type="secondary"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.markdown("**Quick start**")
        st.code(
            "python scripts/generate_data.py\n"
            "python scripts/train.py --data data/raw/claims.csv --artifacts-dir artifacts",
            language="bash",
        )

    try:
        model, meta = load_scorer(art_dir)
        threshold = float(meta.get("threshold", 0.5))
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train a model first (see sidebar), then refresh this page.")
        return

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Decision threshold", f"{threshold:.3f}")
    with mcol2:
        val_ap = meta.get("metrics", {}).get("validation", {}).get("average_precision")
        if val_ap is not None:
            st.metric("Val PR-AUC (last train)", f"{float(val_ap):.3f}")
    with mcol3:
        st.metric("Calibrated", "yes" if meta.get("calibrated") else "no")

    tab1, tab2 = st.tabs(["Single claim", "Batch CSV"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            claim_amount = st.number_input("Claim amount", min_value=1.0, value=12500.0, step=100.0)
            policy_age_months = st.number_input("Policy age (months)", min_value=1, value=36, step=1)
            num_prior_claims = st.number_input("Prior claims", min_value=0, value=0, step=1)
            days_to_report = st.number_input("Days to report", min_value=0, value=4, step=1)
        with c2:
            injury_type = st.selectbox("Injury type", INJURY_TYPES)
            region = st.selectbox("Region", REGIONS)
            has_police_report = st.selectbox("Police report", POLICE)
            repair_shop_network = st.selectbox("Repair shop network", SHOP_NETWORK)

        if st.button("Score claim", type="primary"):
            row = pd.DataFrame(
                [
                    {
                        "claim_amount": claim_amount,
                        "policy_age_months": policy_age_months,
                        "num_prior_claims": num_prior_claims,
                        "days_to_report": days_to_report,
                        "injury_type": injury_type,
                        "region": region,
                        "has_police_report": has_police_report,
                        "repair_shop_network": repair_shop_network,
                    }
                ]
            )
            scored = score_dataframe(model, row, threshold)
            prob = float(scored["fraud_probability"].iloc[0])
            flagged = bool(scored["predicted_fraud"].iloc[0])

            st.divider()
            ac1, ac2 = st.columns([2, 1])
            with ac1:
                st.progress(min(max(prob, 0.0), 1.0), text=f"Fraud probability: {prob:.1%}")
                if flagged:
                    st.error("**Predicted fraud** at the saved threshold (investigate / escalate).")
                else:
                    st.success("**Below threshold** — auto-route for normal processing (still monitor).")
            with ac2:
                st.metric("Fraud probability", f"{prob:.1%}")
                st.metric("Above threshold?", "Yes" if flagged else "No")

    with tab2:
        up = st.file_uploader("Claims CSV (must include all feature columns)", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            try:
                batch = score_dataframe(model, raw, threshold)
            except ValueError as e:
                st.warning(str(e))
                st.dataframe(raw.head(), use_container_width=True)
                return
            st.dataframe(batch, use_container_width=True, height=360)
            csv_bytes = batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scores CSV",
                data=csv_bytes,
                file_name="claim_scores.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
