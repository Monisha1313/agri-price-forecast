"""
Scheduler for automated daily data pipeline.

Runs two jobs every day:
  1. Fetch latest onion prices from data.gov.in
  2. Fetch latest weather from Open-Meteo

Run directly (blocking — keeps running until Ctrl+C):
    python -m src.data.scheduler

Typically you'd run this in a separate terminal or as a background service.
"""

from __future__ import annotations

import os
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.data.database import init_db
from src.data.scraper_agmarknet import DataGovScraper
from src.data.scraper_weather import WeatherScraper
from src.utils.logger import get_logger

logger = get_logger(__name__)

scheduler = BlockingScheduler(timezone="Asia/Kolkata")


def job_fetch_prices() -> None:
    """Daily job: fetch latest onion prices."""
    logger.info("=== Scheduled job: fetch prices ===")
    try:
        api_key = os.getenv("DATAGOVIND_API_KEY", "")
        scraper = DataGovScraper(api_key=api_key)
        inserted = scraper.fetch_and_store(commodity="onion", days_back=3)
        logger.info("Price job done. %d new records.", inserted)
    except Exception as exc:
        logger.error("Price job failed: %s", exc, exc_info=True)


def job_fetch_weather() -> None:
    """Daily job: fetch latest weather."""
    logger.info("=== Scheduled job: fetch weather ===")
    try:
        scraper = WeatherScraper()
        inserted = scraper.fetch_and_store(days_back=3)
        logger.info("Weather job done. %d new records.", inserted)
    except Exception as exc:
        logger.error("Weather job failed: %s", exc, exc_info=True)


def start_scheduler() -> None:
    init_db()

    # Run prices at 8 AM IST daily (Agmarknet updates overnight)
    scheduler.add_job(
        job_fetch_prices,
        trigger=CronTrigger(hour=8, minute=0),
        id="fetch_prices",
        name="Fetch onion prices",
        replace_existing=True,
    )

    # Run weather at 8:30 AM IST daily
    scheduler.add_job(
        job_fetch_weather,
        trigger=CronTrigger(hour=8, minute=30),
        id="fetch_weather",
        name="Fetch weather data",
        replace_existing=True,
    )

    logger.info("Scheduler started. Jobs: fetch_prices @ 08:00, fetch_weather @ 08:30 IST")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()