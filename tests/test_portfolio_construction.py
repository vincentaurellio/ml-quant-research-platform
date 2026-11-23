import numpy as np
import pandas as pd
import pytest

from src.portfolio.construction import construct_portfolio


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-02"] * 20),
        "Ticker": [f"STOCK{i:02d}" for i in range(20)],
        "prediction": np.arange(20, dtype=float),
    })


def test_selects_correct_long_and_short_positions(sample_predictions):
    portfolio = construct_portfolio(sample_predictions)

    longs = portfolio.loc[portfolio["weight"] > 0, "Ticker"]
    shorts = portfolio.loc[portfolio["weight"] < 0, "Ticker"]

    assert set(longs) == {
        "STOCK15", "STOCK16", "STOCK17", "STOCK18", "STOCK19"
    }
    assert set(shorts) == {
        "STOCK00", "STOCK01", "STOCK02", "STOCK03", "STOCK04"
    }


def test_portfolio_has_neutral_exposure(sample_predictions):
    portfolio = construct_portfolio(sample_predictions)

    assert portfolio["weight"].sum() == pytest.approx(0.0)
    assert portfolio["weight"].abs().sum() == pytest.approx(1.0)


def test_each_position_has_equal_weight(sample_predictions):
    portfolio = construct_portfolio(sample_predictions)

    assert (portfolio.loc[
        portfolio["weight"] > 0, "weight"
    ] == 0.10).all()

    assert (portfolio.loc[
        portfolio["weight"] < 0, "weight"
    ] == -0.10).all()


def test_supports_multiple_dates(sample_predictions):
    second_date = sample_predictions.copy()
    second_date["Date"] = pd.Timestamp("2025-01-03")

    data = pd.concat([sample_predictions, second_date], ignore_index=True)
    portfolio = construct_portfolio(data)

    exposure = portfolio.groupby("Date")["weight"].sum()

    assert len(exposure) == 2
    assert np.allclose(exposure.values, 0.0)


def test_rejects_insufficient_stocks(sample_predictions):
    with pytest.raises(ValueError, match="Not enough stocks"):
        construct_portfolio(sample_predictions.head(8))


def test_rejects_duplicate_stocks(sample_predictions):
    data = pd.concat([
        sample_predictions,
        sample_predictions.iloc[[0]],
    ], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        construct_portfolio(data)


def test_rejects_non_finite_predictions(sample_predictions):
    sample_predictions.loc[0, "prediction"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        construct_portfolio(sample_predictions)
