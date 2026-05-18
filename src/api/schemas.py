"""Pydantic schemas for API request/response validation."""
from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    commodity: str = Field(default="onion", description="Commodity key")
    market:    Optional[str] = Field(default=None, description="Market/mandi name")
    model:     str = Field(default="ensemble", description="Model: arima|lstm|xgb|prophet|ensemble")
    horizon:   int = Field(default=7, ge=1, le=30, description="Days ahead to forecast")


class PredictionPoint(BaseModel):
    date:            date
    predicted_price: float
    lower_bound:     Optional[float] = None
    upper_bound:     Optional[float] = None


class PredictResponse(BaseModel):
    commodity:   str
    market:      Optional[str]
    model:       str
    predictions: list[PredictionPoint]
    generated_at: str


class HistoryPoint(BaseModel):
    date:        date
    modal_price: float
    min_price:   Optional[float]
    max_price:   Optional[float]
    market:      str


class HistoryResponse(BaseModel):
    commodity: str
    market:    Optional[str]
    from_date: date
    to_date:   date
    data:      list[HistoryPoint]
    count:     int


class ModelMetrics(BaseModel):
    model:  str
    rmse:   float
    mae:    float
    mape:   float
    smape:  float
    r2:     float


class CompareResponse(BaseModel):
    commodity: str
    results:   list[ModelMetrics]


class HealthResponse(BaseModel):
    status: str
    db:     str
    models: list[str]