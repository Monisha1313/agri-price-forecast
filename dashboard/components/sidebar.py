"""Shared sidebar component for all dashboard pages."""
import streamlit as st
from src.utils.config import COMMODITIES, ONION_PRIMARY_MARKETS


def render_sidebar() -> dict:
    """Render sidebar controls and return selected values."""
    st.sidebar.title("Agri Price Forecast")
    st.sidebar.markdown("---")

    commodity = st.sidebar.selectbox(
        "Commodity",
        options=list(COMMODITIES.keys()),
        format_func=lambda k: COMMODITIES[k]["display_name"],
    )

    market = st.sidebar.selectbox(
        "Market",
        options=["All Markets"] + ONION_PRIMARY_MARKETS,
    )
    market = None if market == "All Markets" else market

    days = st.sidebar.slider("History (days)", min_value=30, max_value=365, value=180, step=30)

    model = st.sidebar.selectbox(
        "Forecast Model",
        options=["ensemble", "lstm", "xgb", "prophet", "arima"],
        format_func=lambda m: m.upper(),
    )

    horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Agmarknet via data.gov.in")
    st.sidebar.caption("Models: LSTM · TFT · XGBoost · Prophet")

    return {
        "commodity": commodity,
        "market":    market,
        "days":      days,
        "model":     model,
        "horizon":   horizon,
    }