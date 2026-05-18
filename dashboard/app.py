"""
Streamlit dashboard entry point.

Run:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Agri Price Forecast",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧅 Agri Price Forecast Dashboard")
st.markdown("""
AI/ML-powered forecasting for agricultural commodity prices.
Helps the Department of Consumer Affairs make data-driven market intervention decisions.

**Navigate using the sidebar pages →**
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Commodities Tracked", "1 (Onion)", delta="8 planned")
with col2:
    st.metric("Models", "5", delta="SARIMA · Prophet · LSTM · TFT · Ensemble")
with col3:
    st.metric("Data Source", "Agmarknet via data.gov.in")

st.markdown("---")
st.info("Use the **sidebar** to navigate to Forecast, Map, Model Comparison, or Explainability pages.")