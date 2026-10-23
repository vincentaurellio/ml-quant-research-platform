
import pandas as pd
import pytest

from src.evaluation.splits import chronological_split


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2025-01-01", periods=60)
    rows = []

    for ticker, base in [("AAA", 100.0), ("BBB", 200.0)]:
        for i, date in enumerate(dates):
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "Close": base + i,
                "target_relative_return_5": 0.01 * (i + 1),
            })

    return pd.DataFrame(rows)


def test_splits_are_chronological_and_disjoint(sample_data):
    dates = pd.bdate_range("2025-01-01", periods=60)

    splits = chronological_split(
        sample_data,
        train_end=str(dates[29].date()),
        validation_end=str(dates[44].date()),
        gap=6,
    )

    train_max = splits["train"]["Date"].max()
    validation_min = splits["validation"]["Date"].min()
    validation_max = splits["validation"]["Date"].max()
    test_min = splits["test"]["Date"].min()

    assert train_max < validation_min
    assert validation_max < test_min


def test_forward_labels_do_not_cross_split_boundaries(sample_data):
    dates = pd.bdate_range("2025-01-01", periods=60)

    splits = chronological_split(
        sample_data,
        train_end=str(dates[29].date()),
        validation_end=str(dates[44].date()),
        gap=6,
    )

    # The target at date t uses information through date t+6.
    train_exit_date = (
        splits["train"]["Date"].max() + pd.offsets.BDay(6)
    )
    validation_exit_date = (
        splits["validation"]["Date"].max() + pd.offsets.BDay(6)
    )

    assert train_exit_date < splits["validation"]["Date"].min()
    assert validation_exit_date < splits["test"]["Date"].min()


def test_splits_contain_only_labeled_rows(sample_data):
    data = sample_data.copy()
    data.loc[data["Date"] == data["Date"].max(),
             "target_relative_return_5"] = None

    dates = pd.bdate_range("2025-01-01", periods=60)

    splits = chronological_split(
        data,
        train_end=str(dates[29].date()),
        validation_end=str(dates[44].date()),
        gap=6,
    )

    for frame in splits.values():
        assert frame["target_relative_return_5"].notna().all()


def test_invalid_boundaries(sample_data):
    with pytest.raises(ValueError, match="train_end"):
        chronological_split(
            sample_data,
            train_end="2025-02-01",
            validation_end="2025-01-01",
        )


def test_negative_gap(sample_data):
    with pytest.raises(ValueError, match="gap"):
        chronological_split(
            sample_data,
            train_end="2025-01-30",
            validation_end="2025-02-15",
            gap=-1,
        )
