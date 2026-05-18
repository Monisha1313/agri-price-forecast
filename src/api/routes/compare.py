"""GET /compare — return stored model metrics comparison."""
from fastapi import APIRouter, HTTPException
from src.api.schemas import CompareResponse, ModelMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/compare/{commodity}", response_model=CompareResponse)
async def compare_models(commodity: str):
    """
    Return model metrics comparison.
    Populated after model training. Returns placeholder until then.
    """
    # TODO: Load from MLflow or a saved metrics JSON after training
    placeholder = [
        ModelMetrics(model="SARIMA",   rmse=0, mae=0, mape=0, smape=0, r2=0),
        ModelMetrics(model="Prophet",  rmse=0, mae=0, mape=0, smape=0, r2=0),
        ModelMetrics(model="LSTM",     rmse=0, mae=0, mape=0, smape=0, r2=0),
        ModelMetrics(model="XGBoost",  rmse=0, mae=0, mape=0, smape=0, r2=0),
        ModelMetrics(model="Ensemble", rmse=0, mae=0, mape=0, smape=0, r2=0),
    ]
    return CompareResponse(commodity=commodity, results=placeholder)