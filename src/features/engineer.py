"""
Feature engineering pipeline for agri price forecasting.

Takes raw price data from the DB, merges with weather,
adds lag features, rolling statistics, and calendar features.
Outputs a clean DataFrame ready for model training.

Usage:
    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.build_features(commodity="onion", market="Lasalgaon")
    df.to_csv("data/processed/features_onion.csv", index=False)
"""

from __future__ import annotations
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd

from src.data.database import get_raw_prices_df
from src.data.scraper_weather import WeatherScraper
from src.utils.config import (
    FORECAST_HORIZON, LAG_DAYS, LOOKBACK_WINDOW,
    PROCESSED_DIR, ROLLING_WINDOWS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Builds the full feature matrix for model training."""

    def __init__(self):
        self.weather_scraper = WeatherScraper()

    def build_features(
        self,
        commodity: str = "onion",
        market: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        logger.info("Building features for %s (market=%s)", commodity, market or "all")

        price_df = self._load_prices(commodity, market, start_date, end_date)
        if price_df.empty:
            logger.error("No price data found. Run the scraper first.")
            return pd.DataFrame()

        logger.info("Loaded %d raw price rows", len(price_df))
        daily_df = self._aggregate_daily(price_df)
        daily_df = self._fill_missing_dates(daily_df)
        daily_df = self._add_lag_features(daily_df)
        daily_df = self._add_rolling_features(daily_df)
        daily_df = self._add_calendar_features(daily_df)
        daily_df = self._add_momentum_features(daily_df)
        daily_df = self._merge_weather(daily_df)
        daily_df = self._add_future_targets(daily_df)

        min_lag = max(LAG_DAYS)
        daily_df = daily_df.iloc[min_lag:].copy()
        # Only keep real observations for training
        daily_df = daily_df[daily_df["is_real"] == 1].copy()
        daily_df = daily_df.dropna(subset=["modal_price", "lag_7d"])

        logger.info("Feature matrix ready: %d rows x %d cols", len(daily_df), len(daily_df.columns))
        if save:
            self._save(daily_df, commodity, market)
        return daily_df

    def _load_prices(self, commodity, market, start_date, end_date):
        df = get_raw_prices_df(commodity=commodity, market=market, start_date=start_date, end_date=end_date)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        rolling_mean = df["modal_price"].rolling(30, min_periods=5).mean()
        rolling_std  = df["modal_price"].rolling(30, min_periods=5).std()
        outliers = (df["modal_price"] > rolling_mean + 3 * rolling_std) | \
                   (df["modal_price"] < rolling_mean - 3 * rolling_std)
        if outliers.sum() > 0:
            logger.warning("Removing %d outlier rows", outliers.sum())
            df = df[~outliers]
        return df

    def _aggregate_daily(self, df):
        daily = df.groupby("date").agg(
            modal_price=("modal_price", "mean"),
            min_price=("min_price", "mean"),
            max_price=("max_price", "mean"),
            n_markets=("market", "nunique"),
        ).reset_index()
        return daily.sort_values("date").reset_index(drop=True)

    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
    
      df = df.set_index("date")
      full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
      df = df.reindex(full_idx)
      df.index.name = "date"

      # Track which rows are real vs filled
      df["is_real"] = (~df["modal_price"].isna()).astype(int)

      # Forward fill price for feature continuity
      df["modal_price"] = df["modal_price"].ffill()
      df["min_price"]   = df["min_price"].ffill()
      df["max_price"]   = df["max_price"].ffill()
      df["n_markets"]   = df["n_markets"].fillna(0)

      return df.reset_index()

    def _add_lag_features(self, df):
        for lag in LAG_DAYS:
            df[f"lag_{lag}d"] = df["modal_price"].shift(lag)
        return df

    def _add_rolling_features(self, df):
        price = df["modal_price"]
        for w in ROLLING_WINDOWS:
            df[f"rolling_mean_{w}d"] = price.rolling(w, min_periods=w // 2).mean()
            df[f"rolling_std_{w}d"]  = price.rolling(w, min_periods=w // 2).std()
            df[f"rolling_min_{w}d"]  = price.rolling(w, min_periods=w // 2).min()
            df[f"rolling_max_{w}d"]  = price.rolling(w, min_periods=w // 2).max()
        df["price_vs_30d_mean"] = df["modal_price"] / (df["rolling_mean_30d"] + 1e-8)
        df["price_vs_90d_mean"] = df["modal_price"] / (df["rolling_mean_90d"] + 1e-8)
        return df

    def _add_calendar_features(self, df):
        df["day_of_week"]  = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"]        = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"]      = df["date"].dt.quarter
        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["year"]         = df["date"].dt.year
        df["season"]       = df["month"].map({
            1:"rabi", 2:"rabi", 3:"rabi",
            4:"zaid", 5:"zaid", 6:"zaid",
            7:"kharif", 8:"kharif", 9:"kharif",
            10:"kharif", 11:"rabi", 12:"rabi",
        })
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
        season_dummies  = pd.get_dummies(df["season"], prefix="season")
        df = pd.concat([df, season_dummies], axis=1)
        return df

    def _add_momentum_features(self, df):
        df["pct_change_1d"]  = df["modal_price"].pct_change(1)
        df["pct_change_7d"]  = df["modal_price"].pct_change(7)
        df["pct_change_30d"] = df["modal_price"].pct_change(30)
        roll_mean = df["modal_price"].rolling(30, min_periods=10).mean()
        roll_std  = df["modal_price"].rolling(30, min_periods=10).std()
        df["zscore_30d"]   = (df["modal_price"] - roll_mean) / (roll_std + 1e-8)
        df["price_spread"] = df["max_price"] - df["min_price"]
        return df

    def _merge_weather(self, df):
        try:
            weather_df = self.weather_scraper.get_weather_df()
            if weather_df.empty:
                logger.warning("No weather data — run scraper_weather.py first. Skipping weather features.")
                return df
            weather_df["date"] = pd.to_datetime(weather_df["date"])
            df = df.merge(weather_df, on="date", how="left")
            weather_cols = ["temp_max", "temp_min", "precip", "et0"]
            df[weather_cols] = df[weather_cols].ffill().bfill()
            logger.info("Weather features merged")
        except Exception as exc:
            logger.warning("Weather merge failed: %s", exc)
        return df

    def _add_future_targets(self, df: pd.DataFrame) -> pd.DataFrame:
     """
     Add future price targets only for real observations.
     Filled rows get NaN targets so they're excluded from training.
     """
     for h in range(1, FORECAST_HORIZON + 1):
        shifted = df["modal_price"].shift(-h)
        # Only set target if the future row is a real observation
        future_is_real = df["is_real"].shift(-h).fillna(0)
        df[f"target_{h}d"] = shifted.where(future_is_real == 1, other=np.nan)
     return df

    def _save(self, df, commodity, market):
        suffix = f"_{market.lower().replace(' ', '_')}" if market else ""
        filename = f"features_{commodity}{suffix}.csv"
        path = PROCESSED_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved: %s (%d rows)", path, len(df))


def build_onion_features(market=None, save=True):
    return FeatureEngineer().build_features(commodity="onion", market=market, save=save)


if __name__ == "__main__":
    from src.data.database import init_db
    init_db()
    df = build_onion_features()
    print(df.tail())
    print(f"\nShape: {df.shape}")