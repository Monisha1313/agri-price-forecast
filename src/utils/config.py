"""
Central configuration for the agri-price-forecast project.
All constants, paths, API endpoints, and model hyperparameters live here.
Import this module everywhere — never hardcode paths or magic numbers elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root — works regardless of where Python is invoked from
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]   # agri-price-forecast/

load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR          = ROOT_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
EXTERNAL_DIR      = DATA_DIR / "external"
MODELS_DIR        = ROOT_DIR / "models"
LOGS_DIR          = ROOT_DIR / "logs"
MLRUNS_DIR        = ROOT_DIR / "mlruns"
NOTEBOOKS_DIR     = ROOT_DIR / "notebooks"

# Sub-directories for raw data
RAW_AGMARKNET_DIR = RAW_DIR / "agmarknet"
RAW_WEATHER_DIR   = RAW_DIR / "weather"
RAW_NDVI_DIR      = RAW_DIR / "ndvi"

# Create all directories on import (safe — won't overwrite existing)
for _dir in [
    RAW_AGMARKNET_DIR, RAW_WEATHER_DIR, RAW_NDVI_DIR,
    PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, LOGS_DIR, MLRUNS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "agri_prices.db"          # SQLite for development
DB_URL  = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Commodities
# Adding more here later requires zero changes elsewhere — just add to this dict
# ---------------------------------------------------------------------------
COMMODITIES = {
    "onion": {
        "display_name": "Onion",
        "agmarknet_code": "Onion",          # Exact string used in Agmarknet
        "unit": "quintal",                   # Price per quintal (100 kg)
        "typical_range_inr": (500, 8000),   # Sanity-check bounds for validation
    },
    # Future: potato, tomato, tur, gram, urad, moong, masur
}

DEFAULT_COMMODITY = "onion"

# Key states for onion (major producing + consuming markets)
ONION_KEY_STATES = [
    "Maharashtra", "Karnataka", "Madhya Pradesh",
    "Rajasthan", "Gujarat", "Andhra Pradesh",
]

# Primary markets to track (Agmarknet market names)
ONION_PRIMARY_MARKETS = [
    "Lasalgaon",    # Largest onion market in Asia, Maharashtra
    "Pimpalgaon",   # Major market, Maharashtra
    "Solapur",      # Maharashtra
    "Hubli",        # Karnataka
    "Bangalore",    # Karnataka
    "Delhi",        # North India consumption hub
]

# ---------------------------------------------------------------------------
# Agmarknet scraper settings
# ---------------------------------------------------------------------------
AGMARKNET_BASE_URL = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
AGMARKNET_TIMEOUT  = 30          # seconds per request
AGMARKNET_DELAY    = 2.0         # polite delay between requests (seconds)
AGMARKNET_MAX_RETRIES = 3
AGMARKNET_BACKOFF_FACTOR = 2.0   # exponential backoff multiplier

# How far back to fetch on a fresh initialisation
AGMARKNET_HISTORY_YEARS = 3

# ---------------------------------------------------------------------------
# Open-Meteo weather settings
# ---------------------------------------------------------------------------
OPENMETEO_BASE_URL    = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_TIMEOUT     = 20

# Weather variables to fetch (Open-Meteo variable names)
WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]

# Lat/lon for key onion-growing districts
WEATHER_LOCATIONS = {
    "Nashik":    {"lat": 20.0059, "lon": 73.7897},   # Lasalgaon mkt
    "Solapur":   {"lat": 17.6868, "lon": 75.9064},
    "Hubli":     {"lat": 15.3647, "lon": 75.1240},
    "Kota":      {"lat": 25.2138, "lon": 75.8648},   # Rajasthan
}

# ---------------------------------------------------------------------------
# Feature engineering settings
# ---------------------------------------------------------------------------
LAG_DAYS         = [1, 2, 3, 7, 14, 21, 30]    # Lag feature windows
ROLLING_WINDOWS  = [7, 14, 30, 60, 90]          # Rolling stat windows
FORECAST_HORIZON = 7                             # Days ahead to predict
LOOKBACK_WINDOW  = 60                            # Input sequence length for LSTM/TFT

# ---------------------------------------------------------------------------
# Model hyperparameters — these are starting defaults, MLflow tracks the rest
# ---------------------------------------------------------------------------
ARIMA_ORDER        = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (1, 1, 1, 7)   # Weekly seasonality

LSTM_HIDDEN_SIZE   = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.2
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE    = 32
LSTM_MAX_EPOCHS    = 100
LSTM_PATIENCE      = 10             # Early stopping patience

XGB_N_ESTIMATORS   = 500
XGB_LEARNING_RATE  = 0.05
XGB_MAX_DEPTH      = 6
XGB_SUBSAMPLE      = 0.8

# Train / validation / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# Date-based split (stronger for time-series papers)
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))
API_RELOAD  = os.getenv("API_RELOAD", "true").lower() == "true"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", str(MLRUNS_DIR))
MLFLOW_EXPERIMENT_NAME  = "agri-price-forecast"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE   = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"