"""
Streamlit UI for Fraud Detection API
"""
import streamlit as st
import os
import requests
import json
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraud Detection System")
st.markdown("Powered by MLOps pipeline — PaySim dataset")

# ── Sidebar: API health ───────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

    st.divider()
    st.header("Model Info")
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.metric("Model",    info.get("best_model", "N/A"))
            st.metric("Features", info.get("feature_count", "N/A"))
            st.metric("Recall",   info.get("best_recall", "N/A"))
            st.metric("F1 Score", info.get("best_f1", "N/A"))
    except:
        st.warning("Could not fetch model info")

    st.divider()
    st.header("Live Metrics")
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            m = r.json()
            st.metric("Total Predictions", m.get("total_predictions", 0))
            st.metric("Fraud Detected",    m.get("fraud_detected", 0))
            st.metric("Live Fraud Rate",   m.get("fraud_rate_live", 0))
            st.metric("Avg Response (ms)", m.get("avg_response_ms", 0))
    except:
        st.warning("Could not fetch metrics")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Transaction", "Batch Prediction"])

with tab1:
    st.header("Single Transaction Check")
    col1, col2 = st.columns(2)

    with col1:
        tx_type = st.selectbox("Transaction Type",
                               ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount  = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
        step    = st.number_input("Step (hour)", min_value=1, value=1)

    with col2:
        oldbalanceOrg  = st.number_input("Sender Old Balance",  min_value=0.0, value=50000.0)
        newbalanceOrig = st.number_input("Sender New Balance",  min_value=0.0, value=40000.0)
        oldbalanceDest = st.number_input("Receiver Old Balance",min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Receiver New Balance",min_value=0.0, value=10000.0)

    if st.button("Check Transaction", type="primary"):
        payload = {
            "step": step, "type": tx_type, "amount": amount,
            "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                result = r.json()
                risk   = result["risk_level"]
                prob   = result["fraud_probability"]
                fraud  = result["is_fraud"]

                if fraud == 1:
                    st.error(f"🚨 FRAUD DETECTED — Risk: {risk} | Probability: {prob:.2%}")
                else:
                    st.success(f"✅ LEGITIMATE — Risk: {risk} | Probability: {prob:.2%}")

                st.progress(prob)
                with st.expander("Full Response"):
                    st.json(result)
            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")

with tab2:
    st.header("Batch Prediction via CSV")
    st.markdown("Upload a CSV with columns: `step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        required = ["step","type","amount","oldbalanceOrg",
                    "newbalanceOrig","oldbalanceDest","newbalanceDest"]
        missing  = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        elif st.button("Run Batch Prediction", type="primary"):
            transactions = df[required].head(100).to_dict(orient="records")
            payload = {"transactions": transactions}
            try:
                r = requests.post(f"{API_URL}/batch_predict", json=payload, timeout=30)
                if r.status_code == 200:
                    res = r.json()
                    preds_df = pd.DataFrame(res["predictions"])
                    result_df = pd.concat([df.head(100).reset_index(drop=True),
                                           preds_df], axis=1)
                    st.success(f"Processed {res['total']} transactions — "
                               f"{res['fraud_count']} fraud detected")
                    st.dataframe(result_df)
                    csv = result_df.to_csv(index=False)
                    st.download_button("Download Results", csv,
                                       "fraud_predictions.csv", "text/csv")
                else:
                    st.error(f"API error: {r.status_code} — {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
