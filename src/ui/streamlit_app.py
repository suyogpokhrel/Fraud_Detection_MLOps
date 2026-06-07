"""
Streamlit Frontend for Fraud Detection Dashboard
Phase 8B: Interactive UI for real-time predictions and model monitoring
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .fraud-high { color: #d62728; font-weight: bold; }
    .fraud-medium { color: #ff7f0e; font-weight: bold; }
    .fraud-low { color: #2ca02c; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# API endpoint
API_URL = "http://127.0.0.1:8000"

# Sidebar
st.sidebar.title("⚙️ Configuration")
page = st.sidebar.radio("Select Page", ["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Info"])

# Helper functions
def check_api_connection():
    """Verify API is running"""
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except:
        return False

def make_prediction(transaction_data):
    """Call FastAPI prediction endpoint"""
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Main pages
if page == "🔍 Single Prediction":
    st.title("🔍 Single Transaction Prediction")
    
    # Check API
    if not check_api_connection():
        st.error("⚠️ API not available. Start FastAPI: `python src/api/fastapi_app.py`")
    else:
        st.success("✓ API Connected")
    
    # Transaction input form
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            step = st.number_input("Step", min_value=1, max_value=1000, value=1)
            amount = st.number_input("Amount", min_value=0.0, value=9839.64)
            oldbalanceOrg = st.number_input("Sender Original Balance", min_value=0.0, value=170136.00)
        
        with col2:
            newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=160296.36)
            oldbalanceDest = st.number_input("Receiver Original Balance", min_value=0.0, value=0.00)
            newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.00)
        
        with col3:
            tx_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
            acct_type = st.selectbox("Account Type", ["Current", "Savings"])
            branch = st.number_input("Branch ID", min_value=0, max_value=134, value=50)
        
        col4, col5 = st.columns(2)
        with col4:
            unusuallogin = st.number_input("Unusual Login Score", min_value=0, max_value=100, value=5)
        with col5:
            isFlaggedFraud = st.number_input("System Fraud Flag", min_value=0, max_value=1, value=0)
        
        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)
    
    # Make prediction
    if submitted:
        with st.spinner("🔄 Analyzing transaction..."):
            transaction_data = {
                "step": int(step),
                "amount": float(amount),
                "oldbalanceOrg": float(oldbalanceOrg),
                "newbalanceOrig": float(newbalanceOrig),
                "oldbalanceDest": float(oldbalanceDest),
                "newbalanceDest": float(newbalanceDest),
                "type": tx_type,
                "acct_type": acct_type,
                "branch": int(branch),
                "unusuallogin": int(unusuallogin),
                "isFlaggedFraud": int(isFlaggedFraud)
            }
            
            result = make_prediction(transaction_data)
            
            if result:
                st.divider()
                st.subheader("📋 Prediction Result")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    prediction = "🚨 FRAUD" if result['is_fraud'] else "✅ LEGITIMATE"
                    st.metric("Classification", prediction)
                
                with col2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
                
                with col3:
                    st.metric("Confidence", f"{result['fraud_confidence']:.1f}%")
                
                with col4:
                    risk = result['risk_level']
                    risk_color = "🔴" if risk == "High" else "🟠" if risk == "Medium" else "🟢"
                    st.metric("Risk Level", f"{risk_color} {risk}")
                
                # Detailed analysis
                st.subheader("📊 Transaction Analysis")
                analysis_df = pd.DataFrame({
                    "Property": ["Amount", "Sender Balance Change", "Receiver Balance Change", "Account Type", "Transaction Type"],
                    "Value": [
                        f"${amount:,.2f}",
                        f"${newbalanceOrig - oldbalanceOrg:,.2f}",
                        f"${newbalanceDest - oldbalanceDest:,.2f}",
                        acct_type,
                        tx_type
                    ]
                })
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)

elif page == "📊 Batch Analysis":
    st.title("📊 Batch Transaction Analysis")
    
    st.info("Upload a CSV file with transaction data to analyze multiple transactions at once.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} transactions")
        
        if st.button("🚀 Analyze Batch", use_container_width=True):
            with st.spinner("Analyzing transactions..."):
                predictions = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    tx_data = {
                        "step": int(row.get("step", 1)),
                        "amount": float(row.get("amount", 0)),
                        "oldbalanceOrg": float(row.get("oldbalanceOrg", 0)),
                        "newbalanceOrig": float(row.get("newbalanceOrig", 0)),
                        "oldbalanceDest": float(row.get("oldbalanceDest", 0)),
                        "newbalanceDest": float(row.get("newbalanceDest", 0)),
                        "type": str(row.get("type", "PAYMENT")),
                        "acct_type": str(row.get("acct_type", "Current")),
                        "branch": int(row.get("branch", 0)),
                        "unusuallogin": int(row.get("unusuallogin", 0)),
                        "isFlaggedFraud": int(row.get("isFlaggedFraud", 0))
                    }
                    
                    result = make_prediction(tx_data)
                    if result:
                        predictions.append(result)
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                st.subheader("📈 Results Summary")
                
                fraud_count = sum(1 for p in predictions if p.get('is_fraud', False))
                fraud_rate = fraud_count / len(predictions) * 100 if predictions else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(predictions))
                with col2:
                    st.metric("Fraudulent", fraud_count)
                with col3:
                    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                with col4:
                    avg_confidence = np.mean([p.get('fraud_confidence', 0) for p in predictions])
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Detailed results table
                results_df = pd.DataFrame(predictions)
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

elif page == "📈 Model Info":
    st.title("📈 Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model-info")
        model_info = response.json()
        
        st.subheader("🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_info.get("model_type", "N/A"))
            st.metric("Recall", f"{model_info.get('recall', 0):.1%}")
            st.metric("Precision", f"{model_info.get('precision', 0):.1%}")
        
        with col2:
            st.metric("ROC-AUC", f"{model_info.get('roc_auc', 0):.4f}")
            st.metric("Fraud Threshold", model_info.get("fraud_threshold", 0.5))
            st.metric("Features", model_info.get("features", "N/A"))
        
        st.info(f"Model trained: {model_info.get('training_date', 'N/A')}")
        
        # Model explanation
        st.subheader("📚 How the Model Works")
        st.markdown("""
        This fraud detection model is a **Logistic Regression classifier** trained on 10,113 financial transactions.
        
        **Key Features:**
        - **Account Balance Changes**: Tracks money movement before/after transactions
        - **Transaction Amounts**: Analyzes relative transaction size vs. available balance
        - **Account Types**: Distinguishes between Current and Savings accounts
        - **Transaction Types**: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
        - **Risk Indicators**: Unusual login patterns and system fraud flags
        
        **Performance Metrics:**
        - **Recall**: 78.57% (catches most frauds)
        - **Precision**: 15.49% (higher false positive rate, acceptable for fraud detection)
        - **ROC-AUC**: 0.8738 (good discrimination between fraud/legitimate)
        
        **Risk Levels:**
        - 🟢 **Low** (< 30% fraud probability): Safe to proceed
        - 🟠 **Medium** (30-70%): Requires review
        - 🔴 **High** (> 70%): Likely fraudulent, block transaction
        """)
        
    except Exception as e:
        st.error(f"Failed to fetch model info: {str(e)}")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
    **Fraud Detection MLOps Project**
    
    Phase 8: API Deployment
    
    - FastAPI Backend: http://127.0.0.1:8000
    - Streamlit UI: http://127.0.0.1:8501
""")
