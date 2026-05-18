"""Reusable Plotly chart helpers for the dashboard."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


def price_history_chart(df: pd.DataFrame, title: str = "Onion Price History") -> go.Figure:
    """Line chart of historical modal prices."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["modal_price"],
        mode="lines", name="Modal Price",
        line=dict(color="#2563EB", width=2),
    ))
    if "min_price" in df.columns and "max_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["max_price"], df["min_price"][::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Min-Max Range",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    return fig


def forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = "Price Forecast",
) -> go.Figure:
    """Combined history + forecast chart with confidence interval."""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=history_df["date"], y=history_df["modal_price"],
        mode="lines", name="Historical",
        line=dict(color="#2563EB", width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_price"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#DC2626", width=2, dash="dash"),
    ))

    # Confidence interval
    if "lower_bound" in forecast_df.columns and forecast_df["lower_bound"].notna().any():
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
            fill="toself", fillcolor="rgba(220,38,38,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Interval",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    return fig


def model_comparison_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    metrics = ["rmse", "mae", "mape"]
    fig = go.Figure()
    colours = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=metrics_df["model"],
                y=metrics_df[metric],
                marker_color=colours[i % len(colours)],
            ))
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Error",
        template="plotly_white",
        height=400,
    )
    return fig


def seasonality_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly average price to show seasonality."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    monthly = df.groupby("month")["modal_price"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])
    fig = px.bar(
        monthly, x="month_name", y="modal_price",
        title="Average Price by Month (Seasonality)",
        labels={"modal_price": "Avg Price (₹/q)", "month_name": "Month"},
        color="modal_price", color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white", height=350)
    return fig