
from pathlib import Path

import pandas as pd

from src.evaluation.splits import chronological_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def main():
    data = pd.read_parquet(INPUT_PATH)

    splits = chronological_split(data, train_end="2021-12-31", validation_end="2023-12-31", gap=6)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, frame in splits.items():
        output_path = OUTPUT_DIR / f"{name}.parquet"
        frame.to_parquet(output_path, index=False)

        print(
            f"{name}: {len(frame):,} rows | "
            f"{frame['Date'].min().date()} to "
            f"{frame['Date'].max().date()} | "
            f"{frame['Ticker'].nunique()} tickers"
        )


if __name__ == "__main__":
    main()
