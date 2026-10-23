
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import yfinance as yf


TICKERS = [
    "AAPL", "MSFT", "NVDA", "ORCL",
    "GOOGL", "META", "NFLX",
    "AMZN", "WMT", "COST",
    "JPM", "GS",
    "JNJ", "ABBV",
    "CAT", "GE",
    "XOM", "CVX",
    "AMD", "AVGO",
]

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_market_data() -> pd.DataFrame:
    print(f"Downloading {len(TICKERS)} tickers...")

    downloaded = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if downloaded.empty:
        raise RuntimeError("No market data returned.")

    frames = []

    for ticker in TICKERS:
        try:
            if isinstance(downloaded.columns, pd.MultiIndex):
                if ticker in downloaded.columns.get_level_values(0):
                    stock = downloaded[ticker].copy()
                elif ticker in downloaded.columns.get_level_values(1):
                    stock = downloaded.xs(
                        ticker, axis=1, level=1
                    ).copy()
                else:
                    print(f"WARNING: No columns found for {ticker}")
                    continue
            else:
                if len(TICKERS) != 1:
                    raise ValueError(
                        "Unexpected column structure for multiple tickers."
                    )
                stock = downloaded.copy()

            required = ["Open", "High", "Low", "Close", "Volume"]

            if not all(col in stock.columns for col in required):
                print(f"WARNING: Missing required columns for {ticker}")
                continue

            stock = stock[required].dropna(
                subset=["Open", "High", "Low", "Close"]
            )

            if stock.empty:
                print(f"WARNING: No usable rows for {ticker}")
                continue

            stock = stock.reset_index()
            stock["Ticker"] = ticker
            frames.append(stock)

            print(f"{ticker}: {len(stock)} rows")

        except (KeyError, ValueError) as exc:
            print(f"WARNING: Could not process {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No usable ticker data was downloaded.")

    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate date/ticker observations detected.")

    return data


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    data = download_market_data()

    output_path = RAW_DIR / "market_data.parquet"
    metadata_path = RAW_DIR / "download_metadata.json"

    data.to_parquet(output_path, index=False)

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers_requested": TICKERS,
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "rows_downloaded": len(data),
        "tickers_downloaded": sorted(data["Ticker"].unique().tolist()),
        "columns": data.columns.tolist(),
        "auto_adjust": True,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDownload complete.")
    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total rows: {len(data):,}")
    print(f"Unique tickers: {data['Ticker'].nunique()}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")


if __name__ == "__main__":
    main()
