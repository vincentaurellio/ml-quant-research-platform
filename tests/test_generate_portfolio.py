import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.utils import FEATURES
from src.portfolio.construction import construct_portfolio


def test_generated_predictions_can_construct_portfolio():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(40, len(FEATURES))),
        columns=FEATURES,
    )

    y = (
        0.5 * X["momentum_5"]
        - 0.3 * X["volatility_20"]
        + rng.normal(scale=0.1, size=40)
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gradient_boosting", HistGradientBoostingRegressor(
            max_iter=10,
            random_state=42,
        )),
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    data = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-02"] * 20 + ["2025-01-03"] * 20),
        "Ticker": [f"STOCK{i:02d}" for i in range(20)] * 2,
        "actual_return": y,
        "prediction": predictions,
    })

    portfolio = construct_portfolio(data)

    assert len(portfolio) == 40
    assert portfolio["Date"].nunique() == 2

    exposure = portfolio.groupby("Date")["weight"].agg(
        net_exposure="sum",
        gross_exposure=lambda weights: weights.abs().sum(),
    )

    assert np.allclose(exposure["net_exposure"], 0.0)
    assert np.allclose(exposure["gross_exposure"], 1.0)


def test_saved_model_predictions_are_reproducible(tmp_path):
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(40, len(FEATURES))),
        columns=FEATURES,
    )
    y = rng.normal(size=40)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gradient_boosting", HistGradientBoostingRegressor(
            max_iter=10,
            random_state=42,
        )),
    ])

    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    assert np.allclose(
        model.predict(X),
        loaded_model.predict(X),
    )
