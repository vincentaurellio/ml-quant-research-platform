import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.utils import FEATURES, TARGET, prepare_data


def test_prepare_data_removes_infinite_features():
    data = pd.DataFrame({
        feature: [1.0, 2.0, 3.0]
        for feature in FEATURES
    })
    data[TARGET] = [0.1, 0.2, 0.3]

    data.loc[0, FEATURES[0]] = np.inf
    prepared = prepare_data(data)

    assert np.isnan(prepared.iloc[0][FEATURES[0]])


def test_gradient_boosting_pipeline_fits_and_predicts():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, len(FEATURES))),
        columns=FEATURES,
    )
    y = (
        0.5 * X["momentum_5"]
        - 0.3 * X["volatility_20"]
        + rng.normal(scale=0.1, size=100)
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gradient_boosting", HistGradientBoostingRegressor(
            max_iter=10,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_gradient_boosting_handles_missing_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, len(FEATURES))),
        columns=FEATURES,
    )
    y = rng.normal(size=100)

    X.loc[0, FEATURES[0]] = np.nan

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gradient_boosting", HistGradientBoostingRegressor(
            max_iter=10,
            random_state=42,
        )),
    ])

    model.fit(X, y)
    predictions = model.predict(X)

    assert np.isfinite(predictions).all()
