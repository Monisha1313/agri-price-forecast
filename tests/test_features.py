"""Tests for the feature engineering pipeline."""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.features.engineer import FeatureEngineer


def make_dummy_price_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 2000 + np.random.randn(n).cumsum() * 50
    return pd.DataFrame({"date": dates, "modal_price": prices,
                         "min_price": prices * 0.9, "max_price": prices * 1.1,
                         "market": "TestMarket", "state": "TestState"})


def test_lag_features_created():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    df = make_dummy_price_df()
    df = fe._add_lag_features(df)
    assert "lag_7d" in df.columns
    assert "lag_30d" in df.columns
    assert df["lag_1d"].iloc[1] == df["modal_price"].iloc[0]


def test_calendar_features():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    df = make_dummy_price_df()
    df = fe._add_calendar_features(df)
    assert "month" in df.columns
    assert "month_sin" in df.columns
    assert df["month_sin"].between(-1, 1).all()


def test_rolling_features():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    df = make_dummy_price_df(100)
    df = fe._add_rolling_features(df)
    assert "rolling_mean_30d" in df.columns
    assert "rolling_std_7d" in df.columns