
import pandas as pd


def chronological_split(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
    gap: int = 6,
    date_col: str = "Date",
    target_col: str = "target_relative_return_5",
) -> dict[str, pd.DataFrame]:
    """Split labeled observations chronologically with purged boundaries."""

    if gap < 0:
        raise ValueError("gap must be non-negative")

    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")

    if date_col not in data.columns or target_col not in data.columns:
        raise ValueError("Required date or target column is missing")

    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    if data[date_col].isna().any():
        raise ValueError("Dates must not contain missing values")

    if data.duplicated([date_col, "Ticker"]).any():
        raise ValueError("Duplicate date/ticker observations detected")

    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must precede validation_end")

    # Use all observed dates to define trading-date gaps.
    all_dates = pd.DatetimeIndex(data[date_col].unique()).sort_values()

    train_dates = all_dates[all_dates <= train_cutoff]
    validation_dates = all_dates[
        (all_dates > train_cutoff) & (all_dates <= validation_cutoff)
    ]
    test_dates = all_dates[all_dates > validation_cutoff]

    if min(len(train_dates), len(validation_dates), len(test_dates)) == 0:
        raise ValueError("Each split must contain at least one date")

    if len(train_dates) <= gap or len(validation_dates) <= gap:
        raise ValueError("Not enough dates for the requested purge gap")

    # Remove dates whose forward labels could cross the next boundary.
    if gap:
        train_dates = train_dates[:-gap]
        validation_dates = validation_dates[:-gap]

    labeled = data.dropna(subset=[target_col])

    train = labeled[labeled[date_col].isin(train_dates)].copy()
    validation = labeled[
        labeled[date_col].isin(validation_dates)
    ].copy()
    test = labeled[labeled[date_col].isin(test_dates)].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A split contains no labeled observations")

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
