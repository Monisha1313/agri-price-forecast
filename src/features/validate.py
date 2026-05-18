"""
Data validation checks for the feature pipeline.
Lightweight replacement for Great Expectations — no extra setup needed.
"""
from __future__ import annotations
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_feature_df(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Run basic quality checks on the feature DataFrame.
    Returns (passed: bool, issues: list of strings).
    """
    issues = []

    if df.empty:
        return False, ["DataFrame is empty"]

    required_cols = ["date", "modal_price", "lag_7d", "rolling_mean_30d", "month"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    null_pct = df["modal_price"].isna().mean()
    if null_pct > 0.05:
        issues.append(f"modal_price has {null_pct:.1%} nulls (threshold 5%)")

    if "modal_price" in df.columns:
        if (df["modal_price"] < 100).any():
            issues.append("Some modal_price values < 100 (possible data error)")
        if (df["modal_price"] > 20000).any():
            issues.append("Some modal_price values > 20000 (possible outlier)")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        if dates.duplicated().any():
            issues.append(f"{dates.duplicated().sum()} duplicate dates found")

    passed = len(issues) == 0
    if passed:
        logger.info("Data validation passed (%d rows, %d cols)", len(df), len(df.columns))
    else:
        for issue in issues:
            logger.warning("Validation issue: %s", issue)

    return passed, issues