"""
Weather data fetcher using the Open-Meteo API.

Open-Meteo is a free, open-source weather API with no API key required.
We fetch historical and forecast weather for key onion-growing districts.

Variables fetched:
  - max/min temperature (crop stress)
  - precipitation (affects supply)
  - evapotranspiration (crop water stress proxy)
  - soil moisture (germination / growth)

Run directly:
    python -m src.data.scraper_weather --days 30

Or import:
    from src.data.scraper_weather import WeatherScraper
    scraper = WeatherScraper()
    scraper.fetch_and_store(days_back=90)
"""

from __future__ import annotations
from src.data.database import engine, WeatherObservation

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from src.data.database import WeatherObservation, get_session
from src.utils.config import (
    OPENMETEO_BASE_URL,
    OPENMETEO_TIMEOUT,
    RAW_WEATHER_DIR,
    WEATHER_LOCATIONS,
    WEATHER_VARIABLES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherScraper:
    """
    Fetches historical weather data from Open-Meteo for key
    onion-growing districts in India.

    Open-Meteo archive API:
        https://archive-api.open-meteo.com/v1/archive
        No API key needed. Free for non-commercial use.
    """

    _HEADERS = {
        "User-Agent": "agri-price-forecast/1.0 (research project)",
        "Accept":     "application/json",
    }

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(self._HEADERS)

    def fetch_and_store(self, days_back: int = 30) -> int:
        """
        Fetch weather for all configured locations for the last `days_back` days.
        Returns total rows inserted.
        """
        end_date   = date.today() - timedelta(days=7)  # Archive lag 
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            "Fetching weather for %d locations: %s to %s",
            len(WEATHER_LOCATIONS), start_date, end_date,
        )

        total_inserted = 0
        for location_name, coords in WEATHER_LOCATIONS.items():
            try:
                inserted = self._fetch_location(
                    location_name=location_name,
                    lat=coords["lat"],
                    lon=coords["lon"],
                    start_date=start_date,
                    end_date=end_date,
                )
                total_inserted += inserted
                logger.info(
                    "%s: %d new rows inserted", location_name, inserted
                )
            except Exception as exc:
                logger.error(
                    "Failed to fetch weather for %s: %s",
                    location_name, exc, exc_info=True,
                )
            time.sleep(0.5)  # Polite delay between locations

        logger.info("Weather fetch complete. Total inserted: %d", total_inserted)
        return total_inserted

    def _fetch_location(
        self,
        location_name: str,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> int:
        """Fetch and store weather for one location."""
        params = {
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      start_date.isoformat(),
            "end_date":        end_date.isoformat(),
            "daily":           ",".join(WEATHER_VARIABLES),
            "timezone":        "Asia/Kolkata",
        }

        resp = self.session.get(
            OPENMETEO_BASE_URL, params=params, timeout=OPENMETEO_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            logger.warning("No weather data returned for %s", location_name)
            return 0

        # Save raw JSON backup
        self._save_raw(data, location_name, start_date, end_date)

        # Build records
        records = []
        for i, date_str in enumerate(dates):
            obs_date = date.fromisoformat(date_str)
            records.append(WeatherObservation(
                location_name=location_name,
                obs_date=obs_date,
                temp_max=self._safe_get(daily, "temperature_2m_max", i),
                temp_min=self._safe_get(daily, "temperature_2m_min", i),
                precipitation=self._safe_get(daily, "precipitation_sum", i),
                et0_evapotranspiration=self._safe_get(daily, "et0_fao_evapotranspiration", i),
                soil_moisture=self._safe_get(daily, "soil_moisture_0_to_7cm", i),
            ))

        # Upsert
        inserted = 0
        with get_session() as session:
            for rec in records:
                exists = session.query(WeatherObservation).filter_by(
                    location_name=rec.location_name,
                    obs_date=rec.obs_date,
                ).first()
                if exists is None:
                    session.add(rec)
                    inserted += 1

        return inserted

    @staticmethod
    def _safe_get(daily: dict, key: str, idx: int) -> Optional[float]:
        vals = daily.get(key, [])
        if idx < len(vals) and vals[idx] is not None:
            return float(vals[idx])
        return None

    def _save_raw(self, data: dict, location: str, start: date, end: date) -> None:
        import json
        filename = f"weather_{location}_{start}_{end}.json"
        path = RAW_WEATHER_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f)
        logger.debug("Raw weather saved: %s", filename)

    def get_weather_df(self, start_date=None, end_date=None):
     from sqlalchemy import select
     with engine.connect() as conn:
        stmt = select(
            WeatherObservation.obs_date,
            WeatherObservation.location_name,
            WeatherObservation.temp_max,
            WeatherObservation.temp_min,
            WeatherObservation.precipitation,
            WeatherObservation.et0_evapotranspiration,
        )
        if start_date:
            stmt = stmt.where(WeatherObservation.obs_date >= start_date)
        if end_date:
            stmt = stmt.where(WeatherObservation.obs_date <= end_date)
        rows = conn.execute(stmt).fetchall()

     if not rows:
        return pd.DataFrame()

     df = pd.DataFrame(rows, columns=["date","location","temp_max","temp_min","precip","et0"])
     df["date"] = pd.to_datetime(df["date"])
     return df.groupby("date").agg({"temp_max":"mean","temp_min":"mean","precip":"mean","et0":"mean"}).reset_index()


if __name__ == "__main__":
    import argparse
    from src.data.database import init_db

    parser = argparse.ArgumentParser(description="Fetch weather data")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    init_db()
    scraper = WeatherScraper()
    scraper.fetch_and_store(days_back=args.days)