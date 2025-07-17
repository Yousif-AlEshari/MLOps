# streamlit_client.py
import streamlit as st
import requests

st.title("Transaction Fraud Detection (via FastAPI)")

# Input fields
high_transaction = st.number_input("High Transaction", min_value=0, max_value=1, step=1)
repeated_logins = st.number_input("Repeated Logins", min_value=0, max_value=1, step=1)
long_transaction = st.number_input("Long Transaction", min_value=0, max_value=1, step=1)

if st.button("Predict"):
    data = {
        "HighTransaction": high_transaction,
        "RepeatedLogins": repeated_logins,
        "LongTransaction": long_transaction
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        result = response.json().get("Prediction: ", ["Error"])
        st.success(f"Prediction: {result[0]}")
    except Exception as e:
        st.error(f"API request failed: {e}")
