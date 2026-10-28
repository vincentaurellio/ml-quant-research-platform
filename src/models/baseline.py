
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES = [
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "volume_ratio_20",
    "momentum_rank_20",
]

TARGET = "target_relative_return_5"


def prepare_data(data):
    data = data.copy()
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET])
    return data


def calculate_errors(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def calculate_daily_ic(dates, y_true, y_pred):
    evaluation = pd.DataFrame({
        "Date": dates,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })

    daily_ic = pd.Series({
        date: group["y_true"].corr(group["y_pred"])
        for date, group in evaluation.groupby("Date")
        if group["y_true"].nunique() > 1 and group["y_pred"].nunique() > 1
    }).dropna()

    return {
        "mean_ic": float(daily_ic.mean()),
        "median_ic": float(daily_ic.median()),
        "ic_std": float(daily_ic.std()),
        "positive_ic_fraction": float((daily_ic > 0).mean()),
        "number_of_ic_dates": int(len(daily_ic)),
    }


def calculate_metrics(dates, y_true, y_pred):
    errors = calculate_errors(y_true, y_pred)
    ic_metrics = calculate_daily_ic(dates, y_true, y_pred)
    return {**errors, **ic_metrics}


def main():
    train = prepare_data(
        pd.read_parquet(SPLIT_DIR / "train.parquet")
    )
    validation = prepare_data(
        pd.read_parquet(SPLIT_DIR / "validation.parquet")
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    baseline_predictions = np.zeros(len(validation))

    model_metrics = calculate_metrics(validation["Date"], y_validation, predictions)
    zero_metrics = calculate_metrics(validation["Date"], y_validation, baseline_predictions)

    print("Zero-prediction baseline:")
    print(json.dumps(zero_metrics, indent=2))

    print("\nRidge regression:")
    print(json.dumps(model_metrics, indent=2))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model to models/ridge_baseline.joblib")
    print("Saved metrics to reports/baseline_metrics.json")


if __name__ == "__main__":
    main()
