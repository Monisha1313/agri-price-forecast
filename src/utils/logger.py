"""
Centralised logging setup for the agri-price-forecast project.

Usage (in any module):
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Scraping started")
    logger.warning("Rate limit hit, backing off")
    logger.error("Failed to connect", exc_info=True)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from src.utils.config import LOG_LEVEL, LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name, fully configured.
    Calling this multiple times with the same name returns the same logger
    (standard Python logging behaviour) — safe to call at module level.
    """
    logger = logging.getLogger(name)

    # Only add handlers once — prevents duplicate log lines on re-import
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # --- Console handler (stdout, coloured level name) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(_ColouredFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # --- Rotating file handler (10 MB max, keep 5 backups) ---
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log messages from bubbling up to the root logger
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Coloured formatter for console output — makes warnings and errors stand out
# ---------------------------------------------------------------------------
_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[41m",    # Red background
}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Convenience: a project-level root logger for one-liners in scripts
# ---------------------------------------------------------------------------
project_logger = get_logger("agri_forecast")