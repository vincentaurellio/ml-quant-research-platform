import numpy as np
import pandas as pd
import pytest

from src.models.baseline import calculate_errors, calculate_daily_ic


def test_perfect_predictions_have_zero_error():
    y_true = np.array([0.02, -0.02, 0.03, -0.03])

    metrics = calculate_errors(y_true, y_true)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_prediction_errors_are_calculated_correctly():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])

    metrics = calculate_errors(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["mae"] == pytest.approx(1.0)


def test_daily_ic_detects_reversed_rankings():
    dates = pd.to_datetime(["2025-01-02"] * 3)
    y_true = np.array([0.03, 0.00, -0.03])
    y_pred = np.array([-1.0, 0.0, 1.0])

    metrics = calculate_daily_ic(dates, y_true, y_pred)

    assert metrics["mean_ic"] == pytest.approx(-1.0)
    assert metrics["positive_ic_fraction"] == 0.0
    assert metrics["number_of_ic_dates"] == 1


def test_constant_predictions_have_no_defined_ic():
    dates = pd.to_datetime(["2025-01-02"] * 3)
    y_true = np.array([0.03, 0.00, -0.03])
    y_pred = np.zeros(3)

    metrics = calculate_daily_ic(dates, y_true, y_pred)

    assert metrics["number_of_ic_dates"] == 0