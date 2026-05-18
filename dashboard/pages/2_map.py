import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium


st.set_page_config(page_title="Price Map", page_icon="🗺️", layout="wide")
st.title("🗺️ India Onion Price Map")
st.caption("Bubble size and colour = average modal price over last 30 days")

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
    st.warning("No data found. Run the loader first.")
    st.stop()

df['date'] = pd.to_datetime(df['date'])

STATE_COORDS = {
    "Maharashtra":      [19.75, 75.71],
    "Karnataka":        [15.31, 75.71],
    "Rajasthan":        [27.02, 74.21],
    "Gujarat":          [22.25, 71.19],
    "Andhra Pradesh":   [15.91, 79.73],
    "Madhya Pradesh":   [22.97, 78.65],
    "Delhi":            [28.61, 77.20],
    "Uttar Pradesh":    [26.84, 80.94],
    "Tamil Nadu":       [11.12, 78.65],
    "West Bengal":      [22.98, 87.85],
    "Punjab":           [31.14, 75.34],
    "Haryana":          [29.05, 76.09],
    "Kerala":           [10.85, 76.27],
    "Telangana":        [18.11, 79.01],
    "Odisha":           [20.94, 85.09],
}

state_avg = df.groupby('state')['modal_price'].mean().reset_index()
state_avg.columns = ['state', 'avg_price']

m = folium.Map(location=[20.59, 78.96], zoom_start=5, tiles='CartoDB positron')

min_p = state_avg['avg_price'].min()
max_p = state_avg['avg_price'].max()

for _, row in state_avg.iterrows():
    coords = STATE_COORDS.get(row['state'])
    if not coords:
        continue
    norm  = (row['avg_price'] - min_p) / (max_p - min_p + 1)
    r     = int(255 * norm)
    b     = int(255 * (1 - norm))
    color = f'#{r:02x}33{b:02x}'
    radius = 8 + norm * 25

    folium.CircleMarker(
        location=coords,
        radius=radius,
        popup=folium.Popup(f"<b>{row['state']}</b><br>₹{row['avg_price']:,.0f}/quintal", max_width=200),
        tooltip=f"{row['state']}: ₹{row['avg_price']:,.0f}/q",
        color=color, fill=True, fill_color=color, fill_opacity=0.7,
    ).add_to(m)

st_folium(m, width=900, height=520)

st.subheader("State-wise Average Prices (Last 30 Days)")
state_avg = state_avg.sort_values('avg_price', ascending=False)
state_avg['avg_price'] = state_avg['avg_price'].round(0).astype(int)
state_avg.columns = ['State', 'Avg Price (₹/quintal)']
st.dataframe(state_avg, use_container_width=True, hide_index=True)