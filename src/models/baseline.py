from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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



def main():
    train = prepare_data(pd.read_parquet(SPLIT_DIR / "train.parquet"))
    validation = prepare_data(pd.read_parquet(SPLIT_DIR / "validation.parquet"))

    X_train, y_train = train[FEATURES], train[TARGET]
    X_validation, y_validation = validation[FEATURES], validation[TARGET]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    baseline_predictions = np.zeros(len(validation))

    model_metrics = calculate_metrics(y_validation, predictions)
    zero_metrics = calculate_metrics(y_validation, baseline_predictions)

    daily_ic = calculate_daily_ic(validation["Date"], y_validation, predictions)
    daily_spearman_ic = calculate_daily_ic(validation["Date"], y_validation, predictions, method="spearman")

    model_metrics.update({
        "mean_ic": float(daily_ic.mean()),
        "median_ic": float(daily_ic.median()),
        "ic_std": float(daily_ic.std()),
        "positive_ic_fraction": float((daily_ic > 0).mean()),
        "number_of_ic_dates": int(len(daily_ic)),
    })

    zero_ic = calculate_daily_ic(validation["Date"], y_validation, baseline_predictions)
    zero_metrics.update({
        "mean_ic": float(zero_ic.mean()) if len(zero_ic) else None,
        "median_ic": float(zero_ic.median()) if len(zero_ic) else None,
        "ic_std": float(zero_ic.std()) if len(zero_ic) else None,
        "positive_ic_fraction": (
            float((zero_ic > 0).mean()) if len(zero_ic) else None
        ),
        "number_of_ic_dates": int(len(zero_ic)),
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    daily_ic_report = pd.concat([daily_ic.rename("pearson_ic"), daily_spearman_ic.rename("spearman_ic")], axis=1)
    daily_ic_report.index.name = "Date"

    annual_ic_report = daily_ic_report.assign(year=daily_ic_report.index.year)\
    .groupby("year")\
    .agg(
        pearson_mean_ic=("pearson_ic", "mean"),
        pearson_median_ic=("pearson_ic", "median"),
        pearson_positive_fraction=(
            "pearson_ic", lambda values: (values > 0).mean()
        ),
        spearman_mean_ic=("spearman_ic", "mean"),
        spearman_median_ic=("spearman_ic", "median"),
        spearman_positive_fraction=(
            "spearman_ic", lambda values: (values > 0).mean()
        ),
        number_of_dates=("pearson_ic", "count"),
    )

    daily_ic_report.to_csv(REPORT_DIR / "daily_ic.csv")
    annual_ic_report.to_csv(REPORT_DIR / "annual_ic.csv")

    joblib.dump(model, MODEL_DIR / "ridge_baseline.joblib")

    report = {
        "model": "Ridge regression",
        "alpha": 1.0,
        "features": FEATURES,
        "target": TARGET,
        "validation_metrics": model_metrics,
        "zero_baseline_metrics": zero_metrics,
        "training_rows": len(train),
        "validation_rows": len(validation),
    }

    (REPORT_DIR / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    diagnostics = calculate_ic_diagnostics(daily_ic_report)
    (REPORT_DIR / "ic_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Zero-prediction baseline:")
    print(json.dumps(zero_metrics, indent=2, allow_nan=False))
    print("\nRidge regression:")
    print(json.dumps(model_metrics, indent=2, allow_nan=False))
    print("\nAnnual IC:")
    print(annual_ic_report.to_string())
    print("\nIC diagnostics:")
    print(json.dumps(diagnostics, indent=2, allow_nan=False))
    print("\nSaved model and reports.")


if __name__ == "__main__":
    main()
