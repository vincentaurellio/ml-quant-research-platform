
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


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
