"""
Seasonal decomposition helpers.

Used in EDA notebooks and as features for models.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from src.utils.logger import get_logger

logger = get_logger(__name__)


def stl_decompose(series: pd.Series, period: int = 7) -> dict:
    """
    STL decomposition: trend + seasonal + residual.
    Returns dict of Series.
    """
    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()
    return {
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }


def add_stl_features(df: pd.DataFrame, col: str = "modal_price", period: int = 7) -> pd.DataFrame:
    """Add STL trend/seasonal/residual as columns."""
    try:
        series = df.set_index("date")[col]
        decomp = stl_decompose(series, period=period)
        df = df.set_index("date")
        df["stl_trend"]    = decomp["trend"]
        df["stl_seasonal"] = decomp["seasonal"]
        df["stl_residual"] = decomp["residual"]
        df = df.reset_index()
        logger.info("STL decomposition added (period=%d)", period)
    except Exception as exc:
        logger.warning("STL decomposition failed: %s", exc)
    return df


def detect_seasonality_periods(series: pd.Series) -> list[int]:
    """
    Use autocorrelation to detect dominant seasonality periods.
    Returns list of lag values with high autocorrelation.
    """
    from statsmodels.tsa.stattools import acf
    lags = min(365, len(series) // 2)
    acf_vals = acf(series.dropna(), nlags=lags, fft=True)
    threshold = 0.3
    peaks = [i for i in range(1, len(acf_vals)) if acf_vals[i] > threshold]
    return peaks[:5]