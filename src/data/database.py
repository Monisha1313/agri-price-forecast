"""
Database layer for agri-price-forecast.

Handles:
  - SQLAlchemy engine + session factory
  - Table schema definitions (ORM models)
  - CRUD helpers used by scrapers, feature pipeline, and API

Supports SQLite (development, zero setup) and PostgreSQL (production).
Switch by setting DATABASE_URL in .env.
"""

from __future__ import annotations

import contextlib
from datetime import date, datetime
from typing import Generator, Optional

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import DB_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------
# connect_args only required for SQLite (disables same-thread check so it
# works inside background schedulers and FastAPI's thread pool)
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(
    DB_URL,
    connect_args=_connect_args,
    echo=False,          # Set True to log all SQL (noisy but useful for debugging)
    pool_pre_ping=True,  # Reconnect automatically on dropped connections
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# ORM base class
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table: raw_prices
# One row per (commodity, market, date) observation from Agmarknet.
# ---------------------------------------------------------------------------
class RawPrice(Base):
    __tablename__ = "raw_prices"

    id            = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    commodity     = Column(String(64),  nullable=False, index=True)
    market        = Column(String(128), nullable=False, index=True)
    state         = Column(String(64),  nullable=False, index=True)
    arrivals      = Column(Float, nullable=True)   # Tonnes arriving at mandi
    district      = Column(String(64),  nullable=True)
    price_date    = Column(Date,        nullable=False, index=True)
    min_price     = Column(Float,       nullable=True)   # INR per quintal
    max_price     = Column(Float,       nullable=True)
    modal_price   = Column(Float,       nullable=False)  # Most traded price — use this
    variety       = Column(String(64),  nullable=True)
    grade         = Column(String(32),  nullable=True)
    scraped_at    = Column(DateTime,    default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("commodity", "market", "price_date", "variety",
                         name="uq_raw_price"),
        Index("ix_raw_prices_commodity_date", "commodity", "price_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<RawPrice {self.commodity} | {self.market} | "
            f"{self.price_date} | ₹{self.modal_price}>"
        )


# ---------------------------------------------------------------------------
# Table: weather_observations
# One row per (location, date) with key weather metrics.
# ---------------------------------------------------------------------------
class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id                     = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    location_name          = Column(String(64), nullable=False, index=True)
    obs_date               = Column(Date,       nullable=False, index=True)
    temp_max               = Column(Float,      nullable=True)   # °C
    temp_min               = Column(Float,      nullable=True)
    precipitation          = Column(Float,      nullable=True)   # mm
    et0_evapotranspiration = Column(Float,      nullable=True)   # mm (crop stress)
    soil_moisture          = Column(Float,      nullable=True)   # m³/m³
    fetched_at             = Column(DateTime,   default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("location_name", "obs_date", name="uq_weather_obs"),
        Index("ix_weather_location_date", "location_name", "obs_date"),
    )


# ---------------------------------------------------------------------------
# Table: processed_features
# Final feature-engineered rows consumed by model training.
# ---------------------------------------------------------------------------
class ProcessedFeature(Base):
    __tablename__ = "processed_features"

    id              = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    commodity       = Column(String(64), nullable=False, index=True)
    market          = Column(String(128), nullable=False, index=True)
    feature_date    = Column(Date,       nullable=False, index=True)
    modal_price     = Column(Float,      nullable=False)  # Target variable

    # Lag features
    lag_1d          = Column(Float, nullable=True)
    lag_2d          = Column(Float, nullable=True)
    lag_3d          = Column(Float, nullable=True)
    lag_7d          = Column(Float, nullable=True)
    lag_14d         = Column(Float, nullable=True)
    lag_21d         = Column(Float, nullable=True)
    lag_30d         = Column(Float, nullable=True)

    # Rolling statistics
    rolling_mean_7d  = Column(Float, nullable=True)
    rolling_mean_14d = Column(Float, nullable=True)
    rolling_mean_30d = Column(Float, nullable=True)
    rolling_std_7d   = Column(Float, nullable=True)
    rolling_std_30d  = Column(Float, nullable=True)

    # Calendar features
    day_of_week     = Column(Integer, nullable=True)   # 0=Mon, 6=Sun
    day_of_month    = Column(Integer, nullable=True)
    month           = Column(Integer, nullable=True)
    week_of_year    = Column(Integer, nullable=True)
    is_weekend      = Column(Integer, nullable=True)   # 0/1

    # Weather features (averaged over key growing locations)
    avg_temp_max    = Column(Float, nullable=True)
    avg_temp_min    = Column(Float, nullable=True)
    avg_precip      = Column(Float, nullable=True)
    avg_et0         = Column(Float, nullable=True)

    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("commodity", "market", "feature_date", name="uq_feature_row"),
        Index("ix_features_commodity_date", "commodity", "feature_date"),
    )


# ---------------------------------------------------------------------------
# Table: model_predictions
# Stores every model's predictions for audit trail + dashboard display.
# ---------------------------------------------------------------------------
class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id              = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    commodity       = Column(String(64),  nullable=False, index=True)
    market          = Column(String(128), nullable=False, index=True)
    model_name      = Column(String(64),  nullable=False, index=True)   # "lstm", "tft", "xgb", "ensemble"
    model_version   = Column(String(32),  nullable=True)
    prediction_date = Column(Date,        nullable=False, index=True)   # Date this prediction was made
    target_date     = Column(Date,        nullable=False, index=True)   # Date being predicted
    predicted_price = Column(Float,       nullable=False)
    actual_price    = Column(Float,       nullable=True)   # Filled in after the date passes
    lower_bound     = Column(Float,       nullable=True)   # Prediction interval
    upper_bound     = Column(Float,       nullable=True)
    created_at      = Column(DateTime,    default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("commodity", "market", "model_name", "prediction_date",
                         "target_date", name="uq_prediction"),
    )


# ---------------------------------------------------------------------------
# Initialise schema
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialised at %s", DB_URL)


# ---------------------------------------------------------------------------
# Session context manager — use this everywhere
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with get_session() as session:
            session.add(some_object)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helper: bulk upsert raw prices
# ---------------------------------------------------------------------------
def upsert_raw_prices(records: list[dict]) -> int:
    if not records:
        return 0

    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    inserted = 0
    batch_size = 1000

    with engine.connect() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            stmt = sqlite_insert(RawPrice).values(batch)
            stmt = stmt.on_conflict_do_nothing()
            result = conn.execute(stmt)
            inserted += result.rowcount
            conn.commit()

    logger.debug("Upserted raw prices: %d new / %d total", inserted, len(records))
    return inserted


# ---------------------------------------------------------------------------
# Helper: fetch raw prices as a DataFrame
# ---------------------------------------------------------------------------
def get_raw_prices_df(
    commodity: str,
    market: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Return raw prices for a commodity as a pandas DataFrame.
    Optionally filter by market and date range.
    """
    from sqlalchemy import select
    with engine.connect() as conn:
     stmt = select(
        RawPrice.price_date, RawPrice.commodity, RawPrice.market,
        RawPrice.state, RawPrice.modal_price, RawPrice.min_price,
        RawPrice.max_price, RawPrice.variety,
     ).where(RawPrice.commodity == commodity)
     if market:
        stmt = stmt.where(RawPrice.market == market)
     if start_date:
        stmt = stmt.where(RawPrice.price_date >= start_date)
     if end_date:
        stmt = stmt.where(RawPrice.price_date <= end_date)
     stmt = stmt.order_by(RawPrice.price_date)
     rows = conn.execute(stmt).fetchall()

  
    df = pd.DataFrame([{
    "date":        r[0],
    "commodity":   r[1],
    "market":      r[2],
    "state":       r[3],
    "modal_price": r[4],
    "min_price":   r[5],
    "max_price":   r[6],
    "variety":     r[7],
    } for r in rows])

    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Helper: check DB health (used by API /health endpoint)
# ---------------------------------------------------------------------------
def check_db_health() -> bool:
    """Return True if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False