"""Tests for model wrappers."""
import pytest
import numpy as np
from src.models.evaluate import rmse, mae, mape, evaluate_all


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_mape_known():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    expected = (10/100 + 20/200) / 2 * 100
    assert mape(y_true, y_pred) == pytest.approx(expected, rel=1e-3)


def test_evaluate_all_returns_dict():
    y = np.random.rand(50) * 2000
    result = evaluate_all(y, y + np.random.randn(50) * 50, model_name="test")
    assert "rmse" in result
    assert "mape" in result
    assert "r2" in result