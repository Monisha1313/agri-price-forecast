"""POST /predict — run a forecast for a commodity."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from fastapi import APIRouter, HTTPException
from src.api.schemas import PredictRequest, PredictResponse, PredictionPoint
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Generate a price forecast.
    Currently returns a placeholder — replace with loaded model inference
    once models are trained and saved.
    """
    try:
        # TODO: load model from MODELS_DIR and run inference
        # For now: return a simple placeholder so the API is live
        today = date.today()
        predictions = []
        for i in range(1, req.horizon + 1):
            predictions.append(PredictionPoint(
                date=today + timedelta(days=i),
                predicted_price=0.0,    # Replace with model output
                lower_bound=None,
                upper_bound=None,
            ))

        return PredictResponse(
            commodity=req.commodity,
            market=req.market,
            model=req.model,
            predictions=predictions,
            generated_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))