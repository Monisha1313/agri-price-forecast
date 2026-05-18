"""
FastAPI application entry point.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import commodities, compare, history, predict
from src.api.schemas import HealthResponse
from src.data.database import check_db_health, init_db
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Agri Price Forecast API",
    description=(
        "AI/ML-powered price forecasting for agricultural commodities. "
        "Predicts onion prices using LSTM, TFT, XGBoost, and ensemble models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router,     prefix="/api/v1", tags=["Predictions"])
app.include_router(history.router,     prefix="/api/v1", tags=["History"])
app.include_router(commodities.router, prefix="/api/v1", tags=["Commodities"])
app.include_router(compare.router,     prefix="/api/v1", tags=["Model Comparison"])


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("API started. DB initialised.")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Agri Price Forecast API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    db_ok = check_db_health()
    saved_models = [p.stem for p in MODELS_DIR.glob("*.pkl")] + \
                   [p.stem for p in MODELS_DIR.glob("*.pt")]
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db="connected" if db_ok else "error",
        models=saved_models or ["none trained yet"],
    )