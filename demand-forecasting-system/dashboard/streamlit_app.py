import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("📊 Retail Demand Forecasting")

# Upload CSV
st.subheader("📤 Upload Sales CSV")
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post(f"{API_URL}/api/upload-data", files=files)
    if response.status_code == 200:
        st.success("CSV uploaded successfully!")
    else:
        st.error("Upload failed")

# Run pipeline
if st.button("Run Forecast"):
    st.info("Running pipeline...")
    try:
        res = requests.post(f"{API_URL}/api/run-pipeline")
        if res.json().get("status") == "success":
            st.success("Pipeline executed successfully!")
        else:
            st.error(f"Pipeline failed: {res.json().get('message')}")
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")

st.subheader("📈 Forecast Output")
try:
    resp = requests.get(f"{API_URL}/api/forecast").json()
    if isinstance(resp, list) and len(resp) > 0:
        df_forecast = pd.DataFrame(resp)
        st.dataframe(df_forecast)
    elif isinstance(resp, dict) and resp.get("error"):
        st.error(f"Error fetching forecast: {resp['error']}")
    else:
        st.info("No forecast data yet.")
except Exception as e:
    st.error(f"Error fetching forecast: {str(e)}")

st.subheader("📦 Inventory Plan")
try:
    resp = requests.get(f"{API_URL}/api/inventory").json()
    if isinstance(resp, list) and len(resp) > 0:
        df_inventory = pd.DataFrame(resp)
        st.dataframe(df_inventory)
    elif isinstance(resp, dict) and resp.get("error"):
        st.error(f"Error fetching inventory: {resp['error']}")
    else:
        st.info("No inventory data yet.")
except Exception as e:
    st.error(f"Error fetching inventory: {str(e)}")


