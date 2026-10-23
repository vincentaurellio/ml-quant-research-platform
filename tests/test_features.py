
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2025-01-01", periods=80)
    rows = []

    for ticker, base, step in [
        ("AAA", 100.0, 1.0),
        ("BBB", 200.0, 2.0),
    ]:
        for i, date in enumerate(dates):
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "Open": base + step * i,
                "High": base + step * i + 1,
                "Low": base + step * i - 1,
                "Close": base + step * i,
                "Volume": 1000 + 10 * i,
            })

    return pd.DataFrame(rows)


def test_momentum(sample_data):
    features = build_features(sample_data)

    row = features[
        (features["Ticker"] == "AAA")
        & (features["Date"] == pd.bdate_range(
            "2025-01-01", periods=6
        )[-1])
    ].iloc[0]

    expected = 105.0 / 100.0 - 1

    assert row["momentum_5"] == pytest.approx(expected)


def test_volume_ratio(sample_data):
    features = build_features(sample_data)

    row = features[
        (features["Ticker"] == "AAA")
        & (features["Date"] == pd.bdate_range(
            "2025-01-01", periods=21
        )[-1])
    ].iloc[0]

    previous_volumes = np.array(
        [1000 + 10 * i for i in range(20)]
    )
    expected = 1200 / previous_volumes.mean()

    assert row["volume_ratio_20"] == pytest.approx(expected)


def test_stock_separation(sample_data):
    features = build_features(sample_data)

    row = features[
        (features["Ticker"] == "AAA")
        & (features["Date"] == pd.bdate_range(
            "2025-01-01", periods=6
        )[-1])
    ].iloc[0]

    # AAA must use AAA's own prices: 105 / 100 - 1.
    assert row["momentum_5"] == pytest.approx(0.05)


def test_duplicate_observations(sample_data):
    duplicated = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_features(duplicated)


def test_future_price_invariance(sample_data):
    original = build_features(sample_data)

    cutoff = pd.bdate_range("2025-01-01", periods=65)[-1]

    modified_data = sample_data.copy()
    future_mask = modified_data["Date"] > cutoff

    modified_data.loc[future_mask, "Close"] *= 1.5

    modified = build_features(modified_data)

    columns = [
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "volume_ratio_20",
        "momentum_rank_20",
    ]

    original_past = (
        original.loc[original["Date"] <= cutoff]
        .set_index(["Date", "Ticker"])[columns]
        .sort_index()
    )

    modified_past = (
        modified.loc[modified["Date"] <= cutoff]
        .set_index(["Date", "Ticker"])[columns]
        .sort_index()
    )

    pd.testing.assert_frame_equal(original_past, modified_past)
