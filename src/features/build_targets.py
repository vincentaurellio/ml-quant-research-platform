
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.parquet"


def build_targets(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Ticker", "Open"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate date/ticker observations detected.")

    if (data["Open"].dropna() <= 0).any():
        raise ValueError("Opening prices must be positive.")

    grouped = data.groupby("Ticker", sort=False)["Open"]

    # Signal at close t; enter at open t+1; exit at open t+6.
    entry_open = grouped.shift(-1)
    exit_open = grouped.shift(-6)

    data["future_return_5"] = exit_open / entry_open - 1

    # Relative return versus the average stock return on the same date.
    cross_section_mean = data.groupby("Date")["future_return_5"].transform(
        "mean"
    )

    data["target_relative_return_5"] = (
        data["future_return_5"] - cross_section_mean
    )

    return data.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def main():
    data = pd.read_parquet(INPUT_PATH)
    dataset = build_targets(data)

    dataset.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved model dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(dataset):,}")
    print(f"Tickers: {dataset['Ticker'].nunique()}")
    print(
        "Rows with valid targets:",
        dataset["target_relative_return_5"].notna().sum(),
    )
    print(
        "Date range:",
        dataset["Date"].min(),
        "to",
        dataset["Date"].max(),
    )
    print("\nTarget summary:")
    print(
        dataset[
            ["future_return_5", "target_relative_return_5"]
        ].describe().round(4)
    )


if __name__ == "__main__":
    main()
