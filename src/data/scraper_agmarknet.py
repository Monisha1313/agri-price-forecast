"""
Agmarknet price data fetcher via data.gov.in Open Government API.

data.gov.in exposes the same Agmarknet mandi price data as a clean
REST/JSON API — no HTML scraping, no ViewState tokens, no blocking.

Resource used:
  Current daily prices of all commodities from all markets
  Resource ID: 9ef84268-d588-465a-a308-a864a43d0070

API key: free from https://data.gov.in — register, then find key on APIs page.

Run directly:
    python -m src.data.scraper_agmarknet --commodity onion --days 30
    python -m src.data.scraper_agmarknet --commodity onion --history

Or import:
    from src.data.scraper_agmarknet import DataGovScraper
    scraper = DataGovScraper(api_key="your_key")
    scraper.fetch_and_store(commodity="onion", days_back=90)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from src.data.database import init_db, upsert_raw_prices
from src.utils.config import (
    AGMARKNET_DELAY,
    AGMARKNET_HISTORY_YEARS,
    AGMARKNET_MAX_RETRIES,
    AGMARKNET_BACKOFF_FACTOR,
    AGMARKNET_TIMEOUT,
    COMMODITIES,
    RAW_AGMARKNET_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATAGOV_BASE_URL = "https://api.data.gov.in/resource"
RESOURCE_ID      = "9ef84268-d588-465a-a308-a864a43d0070"
DATAGOV_MAX_LIMIT = 5000

DATAGOV_COMMODITY_NAMES = {
    "onion": "Onion",
}


@dataclass
class PriceRecord:
    commodity:   str
    market:      str
    state:       str
    price_date:  date
    modal_price: float
    district:    Optional[str] = None
    min_price:   Optional[float] = None
    max_price:   Optional[float] = None
    variety:     Optional[str] = None
    grade:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "commodity":   self.commodity,
            "market":      self.market,
            "state":       self.state,
            "district":    self.district,
            "price_date":  self.price_date,
            "min_price":   self.min_price,
            "max_price":   self.max_price,
            "modal_price": self.modal_price,
            "variety":     self.variety,
            "grade":       self.grade,
        }


class DataGovScraper:
    """Fetches Agmarknet data from the data.gov.in REST API."""

    _HEADERS = {
        "User-Agent": "agri-price-forecast/1.0 (research project)",
        "Accept":     "application/json",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("DATAGOVIND_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "data.gov.in API key required.\n"
                "1. Go to https://data.gov.in and register (free)\n"
                "2. Click 'APIs' in top nav after login — key is shown there\n"
                "3. Add to .env: DATAGOVIND_API_KEY=your_key_here"
            )
        self.session = requests.Session()
        self.session.headers.update(self._HEADERS)

    def fetch_and_store(self, commodity: str = "onion", days_back: int = 30) -> int:
        if commodity not in COMMODITIES:
            raise ValueError(f"Unknown commodity: {commodity}")

        commodity_display = DATAGOV_COMMODITY_NAMES.get(commodity, commodity.title())
        end_date   = date.today()
        start_date = end_date - timedelta(days=days_back)

        logger.info("Fetching %s | %s to %s", commodity_display, start_date, end_date)

        records = self._fetch_all_pages(commodity_display, start_date, end_date)

        if not records:
            logger.warning("No records returned. Check API key and commodity name.")
            return 0

        dicts = [r.to_dict() for r in records]
        for d in dicts:
            d["commodity"] = commodity

        inserted = upsert_raw_prices(dicts)
        logger.info("%d new records inserted (%d fetched total)", inserted, len(dicts))
        self._save_raw_csv(dicts, commodity, start_date, end_date)
        return inserted

    def fetch_history(self, commodity: str = "onion") -> int:
        total_days = AGMARKNET_HISTORY_YEARS * 365
        chunk_days = 90
        total_inserted = 0
        end_date   = date.today()
        start_date = end_date - timedelta(days=total_days)
        current_end = end_date

        logger.info("Starting %d-year backfill in %d-day chunks", AGMARKNET_HISTORY_YEARS, chunk_days)

        while current_end > start_date:
            current_start = max(current_end - timedelta(days=chunk_days), start_date)
            days = (current_end - current_start).days
            try:
                inserted = self.fetch_and_store(commodity=commodity, days_back=days)
                total_inserted += inserted
            except Exception as exc:
                logger.error("Chunk %s–%s failed: %s", current_start, current_end, exc)
            current_end = current_start - timedelta(days=1)
            time.sleep(AGMARKNET_DELAY)

        logger.info("Backfill complete. Total inserted: %d", total_inserted)
        return total_inserted

    def _fetch_all_pages(self, commodity_name: str, start_date: date, end_date: date) -> list[PriceRecord]:
        all_records: list[PriceRecord] = []
        offset = 0

        while True:
            batch = self._fetch_page_with_retry(commodity_name, start_date, end_date, offset, DATAGOV_MAX_LIMIT)
            if not batch:
                break
            all_records.extend(batch)
            logger.debug("offset=%d got %d records (total: %d)", offset, len(batch), len(all_records))
            if len(batch) < DATAGOV_MAX_LIMIT:
                break
            offset += DATAGOV_MAX_LIMIT
            time.sleep(AGMARKNET_DELAY)

        return all_records

    def _fetch_page_with_retry(self, commodity_name, start_date, end_date, offset, limit) -> list[PriceRecord]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, AGMARKNET_MAX_RETRIES + 1):
            try:
                return self._fetch_page(commodity_name, start_date, end_date, offset, limit)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (401, 403):
                    raise RuntimeError("API key rejected. Check DATAGOVIND_API_KEY in .env") from exc
                logger.warning("HTTP error attempt %d: %s", attempt, exc)
                last_exc = exc
            except Exception as exc:
                logger.error("Error attempt %d: %s", attempt, exc)
                last_exc = exc
            if attempt < AGMARKNET_MAX_RETRIES:
                wait = AGMARKNET_DELAY * (AGMARKNET_BACKOFF_FACTOR ** (attempt - 1))
                time.sleep(wait)
        raise RuntimeError(f"All {AGMARKNET_MAX_RETRIES} attempts failed") from last_exc

    def _fetch_page(self, commodity_name, start_date, end_date, offset, limit) -> list[PriceRecord]:
        url = f"{DATAGOV_BASE_URL}/{RESOURCE_ID}"
        params = {
            "api-key":            self.api_key,
            "format":             "json",
            "limit":              limit,
            "offset":             offset,
            "filters[Commodity]": commodity_name,
        }

        resp = self.session.get(url, params=params, timeout=AGMARKNET_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        raw_records = data.get("records", [])
        if not raw_records:
            return []

        parsed = []
        for rec in raw_records:
            pr = self._parse_record(rec, commodity_name)
            if pr is None:
                continue
            if not (start_date <= pr.price_date <= end_date):
                continue
            parsed.append(pr)
        return parsed

    def _parse_record(self, rec: dict, commodity_name: str) -> Optional[PriceRecord]:
        def safe_float(val) -> Optional[float]:
            if val is None or str(val).strip() in ("", "N/A", "-"):
                return None
            try:
                return float(str(val).replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        def parse_date(val) -> Optional[date]:
            if not val:
                return None
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
                try:
                    return datetime.strptime(str(val).strip(), fmt).date()
                except ValueError:
                    continue
            return None

        arrival_date = rec.get("Arrival_Date") or rec.get("arrival_date") or rec.get("date")
        modal_raw    = rec.get("Modal_x0020_Price") or rec.get("Modal Price") or rec.get("modal_price")
        min_raw      = rec.get("Min_x0020_Price")   or rec.get("Min Price")   or rec.get("min_price")
        max_raw      = rec.get("Max_x0020_Price")   or rec.get("Max Price")   or rec.get("max_price")

        price_date  = parse_date(arrival_date)
        modal_price = safe_float(modal_raw)

        if price_date is None or modal_price is None:
            return None
        if not (100 <= modal_price <= 25000):
            return None

        return PriceRecord(
            commodity=commodity_name,
            market=str(rec.get("Market", "Unknown")).strip(),
            state=str(rec.get("State", "Unknown")).strip(),
            district=str(rec.get("District", "")).strip() or None,
            price_date=price_date,
            min_price=safe_float(min_raw),
            max_price=safe_float(max_raw),
            modal_price=modal_price,
            variety=str(rec.get("Variety", "")).strip() or None,
            grade=str(rec.get("Grade", "")).strip() or None,
        )

    def _save_raw_csv(self, records, commodity, start_date, end_date):
        if not records:
            return
        df = pd.DataFrame(records)
        filename = f"{commodity}_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(RAW_AGMARKNET_DIR / filename, index=False)
        logger.debug("Raw CSV saved: %s", filename)


# Alias so the rest of the codebase doesn't need to change
AgmarknetScraper = DataGovScraper


def _parse_args():
    parser = argparse.ArgumentParser(description="Fetch agri prices from data.gov.in")
    parser.add_argument("--commodity", default="onion", choices=list(COMMODITIES.keys()))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--history", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    api_key = os.getenv("DATAGOVIND_API_KEY", "")
    init_db()
    try:
        scraper = DataGovScraper(api_key=api_key)
    except ValueError as e:
        print(f"\nERROR: {e}\n")
        raise SystemExit(1)

    if args.history:
        scraper.fetch_history(commodity=args.commodity)
    else:
        scraper.fetch_and_store(commodity=args.commodity, days_back=args.days)