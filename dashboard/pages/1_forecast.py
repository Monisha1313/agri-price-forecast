import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")
st.title("📈 Onion Price Forecast")

# Load data
try:
    from src.data.database import init_db, get_raw_prices_df
    init_db()
    df = get_raw_prices_df(commodity='onion')
    if not df.empty:
     df['date'] = pd.to_datetime(df['date'])
     # Use last 365 days of available data
     latest_date = df['date'].max()
     cutoff = latest_date - pd.Timedelta(days=365)
     df = df[df['date'] >= cutoff]
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.warning("No data found. Run: `python -m src.data.loader_csv --commodity onion`")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
daily = df.groupby('date').agg(
    modal_price=('modal_price','mean'),
    min_price=('min_price','mean'),
    max_price=('max_price','mean'),
).reset_index()

# Metrics
latest  = daily['modal_price'].iloc[-1]
prev7   = daily['modal_price'].iloc[-8] if len(daily) > 8 else latest
prev30  = daily['modal_price'].iloc[-31] if len(daily) > 31 else latest

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Price (₹/q)",  f"₹{latest:,.0f}", delta=f"{(latest-prev7)/prev7*100:+.1f}% vs 7d ago")
col2.metric("7-day Change",        f"₹{latest-prev7:+,.0f}")
col3.metric("30-day High",         f"₹{daily['modal_price'].tail(30).max():,.0f}")
col4.metric("30-day Avg",          f"₹{daily['modal_price'].tail(30).mean():,.0f}")

# Price history chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily['date'], y=daily['modal_price'],
    mode='lines', name='Modal Price',
    line=dict(color='#2563EB', width=2)
))
fig.add_trace(go.Scatter(
    x=pd.concat([daily['date'], daily['date'][::-1]]),
    y=pd.concat([daily['max_price'], daily['min_price'][::-1]]),
    fill='toself', fillcolor='rgba(37,99,235,0.1)',
    line=dict(color='rgba(0,0,0,0)'), name='Min-Max Range'
))
rolling = daily['modal_price'].rolling(30).mean()
fig.add_trace(go.Scatter(
    x=daily['date'], y=rolling,
    mode='lines', name='30-day MA',
    line=dict(color='#DC2626', width=1.5, dash='dash')
))
fig.update_layout(
    title='Onion Price History (All India Average)',
    xaxis_title='Date', yaxis_title='Price (₹/quintal)',
    hovermode='x unified', template='plotly_white', height=420
)
st.plotly_chart(fig, use_container_width=True)

# Seasonality
st.subheader("Monthly Seasonality")
daily['month'] = daily['date'].dt.month
monthly = daily.groupby('month')['modal_price'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig2 = go.Figure(go.Bar(
    x=month_names, y=[monthly.get(i, 0) for i in range(1,13)],
    marker_color='#2563EB'
))
fig2.update_layout(
    title='Average Price by Month',
    xaxis_title='Month', yaxis_title='Avg Price (₹/q)',
    template='plotly_white', height=350
)
st.plotly_chart(fig2, use_container_width=True)

# Raw data table
with st.expander("View raw data"):
    st.dataframe(daily.sort_values('date', ascending=False).head(50), use_container_width=True)