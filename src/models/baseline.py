
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
    return data.dropna(subset=[TARGET])


def calculate_metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def calculate_daily_ic(dates, y_true, y_pred):
    evaluation = pd.DataFrame({
        "Date": pd.to_datetime(dates).to_numpy(),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })

    daily_ic = pd.Series({
        date: group["y_true"].corr(group["y_pred"])
        for date, group in evaluation.groupby("Date")
        if group["y_true"].nunique() > 1
        and group["y_pred"].nunique() > 1
    }, dtype=float)

    daily_ic.index = pd.to_datetime(daily_ic.index)
    daily_ic.index.name = "Date"
    return daily_ic.sort_index()


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

    annual_ic = pd.DataFrame({
        "year": daily_ic.index.year,
        "daily_ic": daily_ic.to_numpy(),
    }).groupby("year")["daily_ic"].agg(
        mean_ic="mean",
        median_ic="median",
        positive_ic_fraction=lambda values: (values > 0).mean(),
        number_of_dates="count",
    )

    daily_ic.to_csv(REPORT_DIR / "daily_ic.csv")
    annual_ic.to_csv(REPORT_DIR / "annual_ic.csv")

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

    print("Zero-prediction baseline:")
    print(json.dumps(zero_metrics, indent=2, allow_nan=False))
    print("\nRidge regression:")
    print(json.dumps(model_metrics, indent=2, allow_nan=False))
    print("\nAnnual IC:")
    print(annual_ic.to_string())
    print("\nSaved model and reports.")


if __name__ == "__main__":
    main()
