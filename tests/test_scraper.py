"""Tests for the data scraper."""
import pytest
from unittest.mock import patch, MagicMock
from src.data.scraper_agmarknet import DataGovScraper, PriceRecord
from datetime import date


def test_parse_record_valid():
    scraper = DataGovScraper.__new__(DataGovScraper)
    rec = {
        "State": "Maharashtra", "District": "Nashik", "Market": "Lasalgaon",
        "Commodity": "Onion", "Variety": "Other", "Grade": "FAQ",
        "Arrival_Date": "15/01/2024",
        "Min_x0020_Price": "1200", "Max_x0020_Price": "2500", "Modal_x0020_Price": "1800",
    }
    result = scraper._parse_record(rec, "Onion")
    assert result is not None
    assert result.modal_price == 1800.0
    assert result.state == "Maharashtra"
    assert result.price_date == date(2024, 1, 15)


def test_parse_record_outlier_rejected():
    scraper = DataGovScraper.__new__(DataGovScraper)
    rec = {
        "State": "Maharashtra", "Market": "Lasalgaon",
        "Arrival_Date": "15/01/2024", "Modal_x0020_Price": "99999",
    }
    result = scraper._parse_record(rec, "Onion")
    assert result is None


def test_parse_record_missing_date():
    scraper = DataGovScraper.__new__(DataGovScraper)
    rec = {"State": "Maharashtra", "Market": "Lasalgaon", "Modal_x0020_Price": "1800"}
    result = scraper._parse_record(rec, "Onion")
    assert result is None


def test_no_api_key_raises():
    with pytest.raises(ValueError, match="API key"):
        DataGovScraper(api_key="")