import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Performance Comparison")
st.caption("All models trained on 2023–2024 data, evaluated on 2025 held-out test set")

METRICS_FILES = {
    'SARIMA (Baseline)': 'docs/metrics_sarima.json',
    'GRU':               'docs/metrics_gru.json',
    'LSTM':              'docs/metrics_lstm.json',
    'XGBoost':           'docs/metrics_xgboost.json',
    'LightGBM':          'docs/metrics_lightgbm.json',
    'Ensemble':          'docs/metrics_ensemble.json',
}

all_metrics = []
for label, path in METRICS_FILES.items():
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
            m['model'] = label
            all_metrics.append(m)

if not all_metrics:
    st.warning("No metrics found. Run the notebooks first: 02_baseline_arima, 03_lstm_experiments, 05_xgboost, 06_ensemble")
    st.stop()

df = pd.DataFrame(all_metrics)[['model','rmse','mae','mape','r2']].round(3)
df = df.sort_values('rmse').reset_index(drop=True)

# Highlight best row
best_rmse = df['rmse'].min()

st.subheader("Results Table")
st.dataframe(
    df.style.highlight_min(subset=['rmse','mae','mape'], color='#bbf7d0')
             .highlight_max(subset=['r2'], color='#bbf7d0'),
    use_container_width=True, hide_index=True
)

# Bar charts
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    colours = ['#16A34A' if v == best_rmse else '#2563EB' for v in df['rmse']]
    fig.add_trace(go.Bar(x=df['model'], y=df['rmse'], marker_color=colours, name='RMSE'))
    fig.update_layout(
        title='RMSE Comparison (lower = better)',
        xaxis_tickangle=30, template='plotly_white', height=380,
        yaxis_title='RMSE (₹/quintal)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure()
    colours2 = ['#16A34A' if v == df['mape'].min() else '#2563EB' for v in df['mape']]
    fig2.add_trace(go.Bar(x=df['model'], y=df['mape'], marker_color=colours2, name='MAPE'))
    fig2.update_layout(
        title='MAPE Comparison (lower = better)',
        xaxis_tickangle=30, template='plotly_white', height=380,
        yaxis_title='MAPE (%)'
    )
    st.plotly_chart(fig2, use_container_width=True)

# R² chart
fig3 = go.Figure()
colours3 = ['#16A34A' if v == df['r2'].max() else '#2563EB' for v in df['r2']]
fig3.add_trace(go.Bar(x=df['model'], y=df['r2'], marker_color=colours3))
fig3.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Zero baseline')
fig3.update_layout(
    title='R² Score (higher = better, >0 means better than predicting mean)',
    xaxis_tickangle=30, template='plotly_white', height=350,
    yaxis_title='R²'
)
st.plotly_chart(fig3, use_container_width=True)

st.info("🟢 Green bars = best performing model for that metric")