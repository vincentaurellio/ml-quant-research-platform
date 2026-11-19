
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


def calculate_daily_ic(dates, y_true, y_pred, method="pearson"):
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
    evaluation = pd.DataFrame({
        "Date": pd.to_datetime(dates).to_numpy(),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })

    daily_ic = pd.Series({
        date: group["y_true"].corr(group["y_pred"], method=method)
        for date, group in evaluation.groupby("Date")
        if group["y_true"].nunique() > 1
        and group["y_pred"].nunique() > 1
    }, dtype=float)

    daily_ic.index = pd.to_datetime(daily_ic.index)
    daily_ic.index.name = "Date"
    return daily_ic.sort_index()


def calculate_ic_diagnostics(daily_ic_report):
    pearson = daily_ic_report["pearson_ic"].dropna()
    spearman = daily_ic_report["spearman_ic"].dropna()

    comparison = daily_ic_report.dropna(
        subset=["pearson_ic", "spearman_ic"]
    ).copy()

    differences = (
        comparison["pearson_ic"] - comparison["spearman_ic"]
    )

    return {
        "pearson": {
            "mean": float(pearson.mean()),
            "std": float(pearson.std()),
            "median": float(pearson.median()),
            "quantiles": {
                str(q): float(pearson.quantile(q))
                for q in [0.10, 0.25, 0.75, 0.90]
            },
        },
        "spearman": {
            "mean": float(spearman.mean()),
            "std": float(spearman.std()),
            "median": float(spearman.median()),
            "quantiles": {
                str(q): float(spearman.quantile(q))
                for q in [0.10, 0.25, 0.75, 0.90]
            },
        },
        "comparison": {
            "number_of_dates": int(len(comparison)),
            "opposite_sign_fraction": float(
                (
                    comparison["pearson_ic"]
                    * comparison["spearman_ic"] < 0
                ).mean()
            ),
            "mean_absolute_difference": float(differences.abs().mean()),
            "median_absolute_difference": float(differences.abs().median()),
        },
        "largest_disagreements": [
            {
                "date": date.strftime("%Y-%m-%d"),
                "pearson_ic": float(row["pearson_ic"]),
                "spearman_ic": float(row["spearman_ic"]),
                "absolute_difference": float(
                    abs(row["pearson_ic"] - row["spearman_ic"])
                ),
            }
            for date, row in comparison.assign(
                absolute_difference=differences.abs()
            ).nlargest(10, "absolute_difference").iterrows()
        ],
    }


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
