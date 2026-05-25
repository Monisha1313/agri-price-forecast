import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability")
st.caption("Understanding what drives onion price predictions")

# Load feature matrix
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
feat_path = os.path.join(BASE, 'data/processed/features_onion.csv')
if not os.path.exists(feat_path):
    st.warning("Feature matrix not found. Run: `python -m src.features.engineer`")
    st.stop()

df = pd.read_csv(feat_path, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# --- SHAP plots if available ---
shap_bar = os.path.join(BASE, 'docs/shap_bar.png')
shap_bee = os.path.join(BASE, 'docs/shap_beeswarm.png')
shap_dep = os.path.join(BASE, 'docs/shap_dependence.png')

if os.path.exists(shap_bar):
    st.subheader("SHAP Feature Importance (XGBoost)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(shap_bar, caption='Feature Importance (Mean |SHAP|)', use_column_width=True)
    with col2:
        if os.path.exists(shap_bee):
            st.image(shap_bee, caption='SHAP Beeswarm — direction of impact', use_column_width=True)
    if os.path.exists(shap_dep):
        st.image(shap_dep, caption='SHAP Dependence — top 3 features', use_column_width=True)
else:
    st.info("Run notebook `05_xgboost.ipynb` to generate SHAP plots.")

st.markdown("---")

# --- Feature correlation with price ---
st.subheader("Feature Correlations with Modal Price")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['is_real'] + [c for c in numeric_cols if c.startswith('target')]
feature_cols = [c for c in numeric_cols if c not in exclude and c != 'modal_price']

corr = df[feature_cols + ['modal_price']].corr()['modal_price'].drop('modal_price')
corr_sorted = corr.abs().sort_values(ascending=False).head(20)
corr_vals   = corr[corr_sorted.index]

colours = ['#16A34A' if v > 0 else '#DC2626' for v in corr_vals.values]
fig = go.Figure(go.Bar(
    x=corr_vals.values,
    y=corr_vals.index,
    orientation='h',
    marker_color=colours,
))
fig.update_layout(
    title='Top 20 Feature Correlations with Modal Price',
    xaxis_title='Pearson Correlation',
    template='plotly_white',
    height=500,
    yaxis=dict(autorange='reversed')
)
st.plotly_chart(fig, use_container_width=True)
st.caption("🟢 Green = positive correlation (feature increases → price increases) | 🔴 Red = negative")

st.markdown("---")

# --- Rolling statistics visualisation ---
st.subheader("Price vs Rolling Averages")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['date'], y=df['modal_price'],
    mode='lines', name='Modal Price', line=dict(color='#2563EB', width=1.5)))
if 'rolling_mean_7d' in df.columns:
    fig2.add_trace(go.Scatter(x=df['date'], y=df['rolling_mean_7d'],
        mode='lines', name='7-day MA', line=dict(color='#F59E0B', width=1.5, dash='dash')))
if 'rolling_mean_30d' in df.columns:
    fig2.add_trace(go.Scatter(x=df['date'], y=df['rolling_mean_30d'],
        mode='lines', name='30-day MA', line=dict(color='#DC2626', width=2, dash='dash')))
fig2.update_layout(
    title='Price vs Rolling Moving Averages',
    xaxis_title='Date', yaxis_title='Price (₹/quintal)',
    hovermode='x unified', template='plotly_white', height=380
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- Key insights ---
st.subheader("Key Research Findings")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Model", "SARIMA", delta="R²=0.613")
    st.caption("Classical time-series model outperforms deep learning on this dataset size")
with col2:
    st.metric("Dataset Size", "641 rows", delta="Real observations only")
    st.caption("After removing 37.9% forward-filled duplicate rows")
with col3:
    st.metric("Price Volatility", "CV=41.8%", delta="Highly volatile")
    st.caption("Coefficient of variation — onion prices are extremely unpredictable")

st.markdown("""
**Why SARIMA outperforms deep learning here:**
- Dataset too small (641 rows) for LSTM/GRU to learn complex patterns
- Onion prices are highly autoregressive — yesterday's price is the strongest predictor
- Deep learning requires 2,000+ sequences for reliable training
- **Research contribution:** Establishes minimum data requirements for deep learning in agricultural price forecasting
""")