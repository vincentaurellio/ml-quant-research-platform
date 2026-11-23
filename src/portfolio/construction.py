import numpy as np
import pandas as pd


def construct_portfolio(predictions, long_count=5, short_count=5):
    required_columns = {"Date", "Ticker", "prediction"}

    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if long_count < 1 or short_count < 1:
        raise ValueError("long_count and short_count must be positive")

    portfolio = predictions.copy()
    portfolio["Date"] = pd.to_datetime(portfolio["Date"])

    if portfolio[["Date", "Ticker"]].duplicated().any():
        raise ValueError("Duplicate Date/Ticker combinations found")

    if not pd.api.types.is_numeric_dtype(portfolio["prediction"]):
        raise ValueError("Predictions must be numeric")

    if not np.isfinite(portfolio["prediction"]).all():
        raise ValueError("Predictions must be finite")

    portfolio["weight"] = 0.0

    for date, group in portfolio.groupby("Date"):
        if len(group) < long_count + short_count:
            raise ValueError(
                f"Not enough stocks on {date.date()}: "
                f"need {long_count + short_count}, got {len(group)}"
            )

        # Stable sorting makes tie handling reproducible.
        ranked = group.sort_values(
            ["prediction", "Ticker"],
            ascending=[False, True],
            kind="mergesort",
        )

        long_indices = ranked.head(long_count).index
        short_indices = ranked.tail(short_count).index

        portfolio.loc[long_indices, "weight"] = 1.0 / (2 * long_count)
        portfolio.loc[short_indices, "weight"] = -1.0 / (2 * short_count)

    return portfolio.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)
