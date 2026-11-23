from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.utils import (
    FEATURES,
    TARGET,
    prepare_data,
    calculate_metrics,
    calculate_daily_ic,
    calculate_ic_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


MODEL_PARAMETERS = {
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def main():
    # Load the same chronological splits used by Ridge.
    train = prepare_data(pd.read_parquet(SPLIT_DIR / "train.parquet"))
    validation = prepare_data(pd.read_parquet(SPLIT_DIR / "validation.parquet"))

    X_train, y_train = train[FEATURES], train[TARGET]
    X_validation, y_validation = validation[FEATURES], validation[TARGET]

    # Build and train the model.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gradient_boosting", HistGradientBoostingRegressor(
            **MODEL_PARAMETERS
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    # Calculate prediction error metrics.
    model_metrics = calculate_metrics(y_validation, predictions)

    # Calculate daily Pearson and Spearman IC.
    daily_pearson_ic = calculate_daily_ic(validation["Date"], y_validation, predictions, method="pearson")
    daily_spearman_ic = calculate_daily_ic(validation["Date"], y_validation, predictions, method="spearman")

    model_metrics.update({
        "mean_ic": float(daily_pearson_ic.mean()),
        "median_ic": float(daily_pearson_ic.median()),
        "ic_std": float(daily_pearson_ic.std()),
        "positive_ic_fraction": float(
            (daily_pearson_ic > 0).mean()
        ),
        "number_of_ic_dates": int(len(daily_pearson_ic)),
        "mean_spearman_ic": float(daily_spearman_ic.mean()),
        "median_spearman_ic": float(daily_spearman_ic.median()),
        "spearman_positive_ic_fraction": float(
            (daily_spearman_ic > 0).mean()
        ),
    })

    # Combine daily IC results.
    daily_ic_report = pd.concat([
        daily_pearson_ic.rename("pearson_ic"),
        daily_spearman_ic.rename("spearman_ic"),
    ], axis=1)

    daily_ic_report.index.name = "Date"

    # Calculate annual IC results.
    annual_ic_report = (
        daily_ic_report.assign(year=daily_ic_report.index.year)
        .groupby("year")
        .agg(
            pearson_mean_ic=("pearson_ic", "mean"),
            pearson_median_ic=("pearson_ic", "median"),
            pearson_positive_fraction=(
                "pearson_ic",
                lambda values: (values > 0).mean(),
            ),
            spearman_mean_ic=("spearman_ic", "mean"),
            spearman_median_ic=("spearman_ic", "median"),
            spearman_positive_fraction=(
                "spearman_ic",
                lambda values: (values > 0).mean(),
            ),
            number_of_dates=("pearson_ic", "count"),
        )
    )

    diagnostics = calculate_ic_diagnostics(daily_ic_report)

    # Create output directories.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model and reports.
    joblib.dump(model, MODEL_DIR / "gradient_boosting.joblib")

    (REPORT_DIR / "gradient_boosting_metrics.json").write_text(
        json.dumps({
            "model": "HistGradientBoostingRegressor",
            "parameters": MODEL_PARAMETERS,
            "features": FEATURES,
            "target": TARGET,
            "validation_metrics": model_metrics,
            "training_rows": len(train),
            "validation_rows": len(validation),
        }, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    daily_ic_report.to_csv(REPORT_DIR / "gradient_boosting_daily_ic.csv")
    annual_ic_report.to_csv(REPORT_DIR / "gradient_boosting_annual_ic.csv")

    (REPORT_DIR / "gradient_boosting_ic_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Display results.
    print("HistGradientBoostingRegressor:")
    print(json.dumps(model_metrics, indent=2, allow_nan=False))

    print("\nAnnual IC:")
    print(annual_ic_report.to_string())

    print("\nIC diagnostics:")
    print(json.dumps(diagnostics, indent=2, allow_nan=False))

    print("\nSaved model and reports.")


if __name__ == "__main__":
    main()
