from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.utils import (
    FEATURES,
    TARGET,
    prepare_data,
)
from src.portfolio.construction import construct_portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting.joblib"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    # Load the untouched test split and trained model.
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    test = prepare_data(test)

    model = joblib.load(MODEL_PATH)

    # Generate predictions.
    predictions = model.predict(test[FEATURES])

    if not np.isfinite(predictions).all():
        raise ValueError("Model generated non-finite predictions")

    portfolio_data = test[
        ["Date", "Ticker", TARGET]
    ].copy()

    portfolio_data["prediction"] = predictions

    # Construct daily long-short portfolios.
    portfolio = construct_portfolio(
        portfolio_data.rename(columns={TARGET: "actual_return"}),
        long_count=5,
        short_count=5,
    )

    # Save the resulting dataset.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "portfolio.parquet"
    portfolio.to_parquet(output_path, index=False)

    print("Portfolio generation completed.")
    print(f"Rows: {len(portfolio)}")
    print(f"Dates: {portfolio['Date'].nunique()}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
