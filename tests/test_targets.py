import pandas as pd
import pytest

from src.features.build_targets import build_targets


def make_sample_data():
    dates = pd.bdate_range("2025-01-01", periods=10)
    rows = []

    for ticker, base in [("AAA", 100.0), ("BBB", 200.0)]:
        for i, date in enumerate(dates):
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "Open": base + i,
                "Close": base + i,
                "Volume": 1000,
            })

    return pd.DataFrame(rows)


def test_forward_return():
    data = make_sample_data()
    result = build_targets(data)

    row = result[
        (result["Ticker"] == "AAA")
        & (result["Date"] == pd.bdate_range(
            "2025-01-01", periods=1
        )[0])
    ].iloc[0]

    # Entry: 101 at the next open.
    # Exit: 106 at the open six rows ahead.
    expected = 106.0 / 101.0 - 1

    assert row["future_return_5"] == pytest.approx(expected)


def test_relative_targets_are_centered_by_date():
    result = build_targets(make_sample_data())

    valid = result.dropna(subset=["target_relative_return_5"])

    means = valid.groupby("Date")["target_relative_return_5"].mean()

    assert (means.abs() < 1e-12).all()


def test_last_six_observations_have_no_forward_target():
    result = build_targets(make_sample_data())

    for ticker, group in result.groupby("Ticker"):
        ordered = group.sort_values("Date")

        assert ordered["future_return_5"].tail(6).isna().all()
        assert ordered["target_relative_return_5"].tail(6).isna().all()


def test_duplicate_observations():
    data = make_sample_data()
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_targets(duplicated)
