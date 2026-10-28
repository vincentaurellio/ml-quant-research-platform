import numpy as np
import pandas as pd
import pytest

from src.models.baseline import calculate_metrics, calculate_daily_ic


def test_perfect_predictions_have_zero_error():
    y_true = np.array([0.02, -0.02, 0.03, -0.03])

    metrics = calculate_metrics(y_true, y_true)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_prediction_errors_are_calculated_correctly():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["mae"] == pytest.approx(1.0)


def test_daily_ic_detects_reversed_rankings():
    dates = pd.to_datetime(["2025-01-02"] * 3)
    y_true = np.array([0.03, 0.00, -0.03])
    y_pred = np.array([-1.0, 0.0, 1.0])

    daily_ic = calculate_daily_ic(dates, y_true, y_pred)

    assert len(daily_ic) == 1
    assert daily_ic.iloc[0] == pytest.approx(-1.0)


def test_daily_ic_skips_constant_predictions_and_sorts_dates():
    dates = pd.to_datetime([
        "2025-01-03", "2025-01-03", "2025-01-03",
        "2025-01-02", "2025-01-02", "2025-01-02",
    ])
    y_true = np.array([0.03, 0.00, -0.03, 0.03, 0.00, -0.03])
    y_pred = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])

    daily_ic = calculate_daily_ic(dates, y_true, y_pred)

    assert len(daily_ic) == 1
    assert daily_ic.index[0] == pd.Timestamp("2025-01-03")
    assert daily_ic.iloc[0] == pytest.approx(1.0)
