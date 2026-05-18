"""GET /history/{commodity} — return historical price data."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.api.schemas import HistoryPoint, HistoryResponse
from src.data.database import get_raw_prices_df
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/history/{commodity}", response_model=HistoryResponse)
async def get_history(
    commodity: str,
    market:    Optional[str] = Query(default=None),
    days:      int           = Query(default=90, ge=1, le=1095),
):
    try:
        end_date   = date.today()
        start_date = end_date - timedelta(days=days)
        df = get_raw_prices_df(
            commodity=commodity,
            market=market,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {commodity}. Run the scraper first."
            )

        data = [
            HistoryPoint(
                date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                modal_price=row["modal_price"],
                min_price=row.get("min_price"),
                max_price=row.get("max_price"),
                market=row["market"],
            )
            for _, row in df.iterrows()
        ]

        return HistoryResponse(
            commodity=commodity,
            market=market,
            from_date=start_date,
            to_date=end_date,
            data=data,
            count=len(data),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("History endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))