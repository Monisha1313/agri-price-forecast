"""GET /commodities — list supported commodities."""
from fastapi import APIRouter
from src.utils.config import COMMODITIES

router = APIRouter()

@router.get("/commodities")
async def list_commodities():
    return {
        "commodities": [
            {"key": k, "display_name": v["display_name"], "unit": v["unit"]}
            for k, v in COMMODITIES.items()
        ]
    }