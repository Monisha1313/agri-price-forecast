"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_list_commodities():
    resp = client.get("/api/v1/commodities")
    assert resp.status_code == 200
    data = resp.json()
    assert "commodities" in data
    assert any(c["key"] == "onion" for c in data["commodities"])


def test_predict_returns_response():
    resp = client.post("/api/v1/predict", json={
        "commodity": "onion", "model": "ensemble", "horizon": 7
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 7