
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "market_data.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "features.parquet"


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Ticker", "Close", "Volume"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate date/ticker observations detected.")

    if (data["Close"].dropna() <= 0).any():
        raise ValueError("Close prices must be positive.")

    grouped = data.groupby("Ticker", sort=False)

    # Historical daily returns
    data["daily_return"] = grouped["Close"].pct_change(fill_method=None)

    # Price momentum over different lookback windows
    for window in [5, 20, 60]:
        data[f"momentum_{window}"] = grouped["Close"].transform(
            lambda s: s / s.shift(window) - 1
        )

    # Trailing volatility of daily returns
    data["volatility_20"] = data.groupby("Ticker")["daily_return"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )

    # Compare today's volume with the previous 20 sessions
    previous_volume_avg = data.groupby("Ticker")["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    data["volume_ratio_20"] = data["Volume"] / previous_volume_avg.where(previous_volume_avg > 0)

    # Relative momentum among the stocks observed on each date
    data["momentum_rank_20"] = data.groupby("Date")["momentum_20"].rank(method="average", pct=True)

    return data.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def main():
    data = pd.read_parquet(RAW_PATH)
    features = build_features(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    feature_columns = [
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "volume_ratio_20",
        "momentum_rank_20",
    ]

    print(f"Saved features to: {OUTPUT_PATH}")
    print(f"Rows: {len(features):,}")
    print(f"Tickers: {features['Ticker'].nunique()}")
    print(f"Date range: {features['Date'].min()} to {features['Date'].max()}")
    print("\nFeature missing-value counts:")
    print(features[feature_columns].isna().sum())
    print("\nSample:")
    print(features[["Date", "Ticker"] + feature_columns].tail(10))


if __name__ == "__main__":
    main()
