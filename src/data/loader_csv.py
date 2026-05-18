"""
CSV loader for Kaggle / Agmarknet CSV datasets.

Reads CSV files from data/raw/agmarknet/, filters for a commodity,
and loads into the database.

Run:
    python -m src.data.loader_csv --commodity onion
    python -m src.data.loader_csv --commodity onion --file data/raw/agmarknet/Agriculture_price_dataset.csv
"""

from __future__ import annotations

import argparse
from gettext import find
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database import init_db, upsert_raw_prices
from src.utils.config import COMMODITIES, RAW_AGMARKNET_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map commodity key → names as they appear in the CSV Commodity column
COMMODITY_CSV_NAMES = {
    "onion":   ["Onion", "onion", "ONION"],
    "potato":  ["Potato", "potato", "POTATO"],
    "tomato":  ["Tomato", "tomato", "TOMATO"],
}


def load_csv(
    commodity: str = "onion",
    filepath: Optional[str] = None,
) -> int:
    """
    Load a commodity from CSV into the database.

    Args:
        commodity: commodity key (e.g. "onion")
        filepath:  path to CSV file. If None, auto-detects from raw dir.

    Returns:
        Number of new rows inserted.
    """
    if commodity not in COMMODITIES:
        raise ValueError(f"Unknown commodity: {commodity}")

    # Find CSV file
    if filepath:
        csv_path = Path(filepath)
    else:
        files = sorted(glob.glob(str(RAW_AGMARKNET_DIR / "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {RAW_AGMARKNET_DIR}. "
                "Download from Kaggle and place in data/raw/agmarknet/"
            )
        csv_path = Path(files[0])
        logger.info("Auto-detected CSV: %s", csv_path)

    logger.info("Loading %s from %s", commodity, csv_path)

    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Raw CSV: %d rows, %d cols", len(df), len(df.columns))

    # Normalise column names — strip whitespace, consistent naming
    df.columns = [c.strip() for c in df.columns]

    # Detect column name variants
    col_map = _detect_columns(df.columns.tolist())
    if col_map is None:
        raise ValueError(
            f"Unrecognised CSV structure. Columns found: {df.columns.tolist()}\n"
            "Expected columns like: STATE, District Name, Market Name, "
            "Commodity, Modal_Price, Price Date"
        )

    logger.info("Column mapping: %s", col_map)

    # Filter for target commodity
    
    commodity_names = COMMODITY_CSV_NAMES.get(commodity, [commodity.title()])

    if col_map["commodity"] and col_map["commodity"] in df.columns:
     mask = df[col_map["commodity"]].isin(commodity_names)
     df_filtered = df[mask].copy()
     if df_filtered.empty:
        # Try case-insensitive
        df_filtered = df[
            df[col_map["commodity"]].str.lower() == commodity.lower()
        ].copy()
     if df_filtered.empty:
        # Commodity is likely in the filename, not the column — use all rows
        logger.info(
            "Commodity column '%s' has no '%s' values — "
            "assuming entire file is for this commodity.",
            col_map["commodity"], commodity,
        )
        df_filtered = df.copy()
    else:
     df_filtered = df.copy()

    logger.info("Filtered to %d rows for commodity '%s'", len(df_filtered), commodity)

    # Parse and clean
    records = _build_records(df_filtered, col_map, commodity)
    if not records:
        logger.warning("No valid records after parsing.")
        return 0

    # Insert into DB
    inserted = upsert_raw_prices(records)
    logger.info(
        "Done. %d new records inserted (%d parsed, %d in file).",
        inserted, len(records), len(df_filtered),
    )
    return inserted


def _detect_columns(columns: list[str]) -> Optional[dict]:
    """
    Map standard field names to actual CSV column names.
    Handles variations across different CSV exports.
    """
    lower = {c.lower().strip(): c for c in columns}

    def find(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    state     = find(["state name", "state", "State Name", "STATE"])
    district  = find(["district name", "district", "District Name"])
    market    = find(["market name", "market", "Market Name"])
    commodity = find(["commodity", "Commodity", "group", "Group"])
    min_price = find(["min price (rs./quintal)", "min_price", "min price", "Min Price (Rs./Quintal)"])
    max_price = find(["max price (rs./quintal)", "max_price", "max price", "Max Price (Rs./Quintal)"])
    modal_price = find(["modal price (rs./quintal)", "modal_price", "modal price", "Modal Price (Rs./Quintal)"])
    price_date  = find(["reported date", "price date", "date", "Reported Date", "Price Date"])

    # Modal price and date are mandatory
    if modal_price is None or price_date is None:
        return None
    variety = find(["variety", "Variety", "VARIETY"])
    grade   = find(["grade", "Grade", "GRADE"])
    return {
        "state":       state,
        "district":    district,
        "market":      market,
        "commodity":   commodity,
        "variety":     variety,
        "grade":       grade,
        "min_price":   min_price,
        "max_price":   max_price,
        "modal_price": modal_price,
        "price_date":  price_date,
    }


def _build_records(df: pd.DataFrame, col_map: dict, commodity: str) -> list[dict]:
    """Convert DataFrame rows to list of dicts ready for DB insertion."""

    def safe_float(val) -> Optional[float]:
        if pd.isna(val):
            return None
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    def parse_date(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
                    "%d-%m-%Y", "%d-%b-%Y", "%Y/%m/%d",
                    "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                continue
        return None

    records = []
    skipped = 0

    for _, row in df.iterrows():
        price_date  = parse_date(row.get(col_map["price_date"]))
        modal_price = safe_float(row.get(col_map["modal_price"]))

        if price_date is None or modal_price is None:
            skipped += 1
            continue

        # Sanity check
        if not (100 <= modal_price <= 25000):
            skipped += 1
            continue

        records.append({
            "commodity":   commodity,
            "state":       str(row.get(col_map["state"] or "", "Unknown")).strip(),
            "arrivals":    safe_float(row.get("Arrivals (Tonnes)") or row.get("Arrivals")),
            "district":    str(row.get(col_map["district"] or "", "")).strip() or None,
            "market":      str(row.get(col_map["market"] or "", "Unknown")).strip(),
            "variety":     str(row.get(col_map["variety"] or "", "")).strip() or None,
            "grade":       str(row.get(col_map["grade"] or "", "")).strip() or None,
            "price_date":  price_date,
            "min_price":   safe_float(row.get(col_map["min_price"])) if col_map["min_price"] else None,
            "max_price":   safe_float(row.get(col_map["max_price"])) if col_map["max_price"] else None,
            "modal_price": modal_price,
        })

    if skipped:
        logger.warning("Skipped %d rows (bad date/price or outlier)", skipped)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CSV data into database")
    parser.add_argument("--commodity", default="onion", choices=list(COMMODITIES.keys()))
    parser.add_argument("--file", default=None, help="Path to CSV file (auto-detects if not set)")
    args = parser.parse_args()

    init_db()
    inserted = load_csv(commodity=args.commodity, filepath=args.file)
    print(f"\nInserted {inserted} new records for '{args.commodity}'")