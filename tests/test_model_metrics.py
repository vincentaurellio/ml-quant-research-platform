import numpy as np
import pandas as pd
import pytest

from src.models.baseline import calculate_metrics, calculate_daily_ic, calculate_ic_diagnostics


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


def test_spearman_ic_perfectly_aligned_rankings():
    dates = pd.to_datetime(["2025-01-02"] * 4)
    y_true = np.array([0.01, 0.02, 0.03, 0.04])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])

    daily_ic = calculate_daily_ic(
        dates, y_true, y_pred, method="spearman"
    )

    assert daily_ic.iloc[0] == pytest.approx(1.0)


def test_spearman_ic_reversed_rankings():
    dates = pd.to_datetime(["2025-01-02"] * 4)
    y_true = np.array([0.01, 0.02, 0.03, 0.04])
    y_pred = np.array([40.0, 30.0, 20.0, 10.0])

    daily_ic = calculate_daily_ic(
        dates, y_true, y_pred, method="spearman"
    )

    assert daily_ic.iloc[0] == pytest.approx(-1.0)


def test_ic_rejects_unknown_correlation_method():
    with pytest.raises(ValueError, match="method"):
        calculate_daily_ic(
            pd.to_datetime(["2025-01-02"] * 3),
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
            method="kendall",
        )


def test_ic_diagnostics_identical_metrics_have_no_disagreement():
    daily_ic = pd.DataFrame({
        "pearson_ic": [0.2, -0.3, 0.5],
        "spearman_ic": [0.2, -0.3, 0.5],
    })

    daily_ic.index = pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ])

    diagnostics = calculate_ic_diagnostics(daily_ic)

    assert diagnostics["comparison"]["opposite_sign_fraction"] == 0
    assert diagnostics["comparison"]["mean_absolute_difference"] == 0
    assert diagnostics["comparison"]["number_of_dates"] == 3


def test_ic_diagnostics_detect_opposite_signs():
    daily_ic = pd.DataFrame({
        "pearson_ic": [0.2, -0.3, 0.5],
        "spearman_ic": [-0.2, -0.1, 0.4],
    })

    daily_ic.index = pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ])

    diagnostics = calculate_ic_diagnostics(daily_ic)

    assert diagnostics["comparison"]["opposite_sign_fraction"] == pytest.approx(1 / 3)


def test_ic_diagnostics_identifies_largest_disagreements():
    daily_ic = pd.DataFrame({
        "pearson_ic": [0.1, 0.8, -0.5],
        "spearman_ic": [0.1, 0.2, 0.5],
    })
    daily_ic.index = pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ])

    diagnostics = calculate_ic_diagnostics(daily_ic)

    largest = diagnostics["largest_disagreements"][0]

    assert largest["date"] == "2025-01-03"
    assert largest["absolute_difference"] == pytest.approx(1.0)
