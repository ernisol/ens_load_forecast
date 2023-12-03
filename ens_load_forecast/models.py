"""Module used for model training"""

from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator

import ens_load_forecast.constants as cst


class NaiveModel(BaseEstimator):
    """Naive model that returns directly the input load forecast."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y) -> None:
        pass

    def predict(self, X):
        return X[cst.LOAD_FORECAST]


def initialize_models() -> Dict[str, BaseEstimator]:
    """Initialize models.

    Returns
    -------
    Dict[str, Any]
        Dictionnary of models, with keys:
        - naive_model
        - linear_model
        - polynomial_model
        - radom_forest_model
    """
    naive_model = NaiveModel()
    linear_model = LinearRegression(fit_intercept=True)
    polynomial_model = Pipeline(
        steps=[
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear_regression", LinearRegression(fit_intercept=True)),
        ]
    )
    random_forest_model = RandomForestRegressor(n_estimators=100)
    return {
        cst.NAIVE_MODEL: naive_model,
        cst.LINEAR_MODEL: linear_model,
        cst.POLYNOMIAL_MODEL: polynomial_model,
        cst.RANDOM_FOREST_MODEL: random_forest_model,
    }


def train_models_for_each_zone(df_features):
    models = {}
    scores = {}
    for zone in df_features[cst.ZONE].unique():
        df = df_features[df_features[cst.ZONE] == zone]
        zone_models, zone_scores = train_models(df_features=df)
        models[zone] = zone_models
        scores[zone] = zone_scores
    return models, scores


def train_models(
    df_features: pd.DataFrame,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Any]]:
    """Train and score all defined models on the given Dataset (preferably one zone at a time)

    Parameters
    ----------
    df_features : pd.DataFrame
        Features DataFrame, preferably only one zone

    Returns
    -------
    Tuple[Dict[str, BaseEstimator], Dict[str, Any]]
        Two Dictionaries:
        - Trained models, keys are model kind
        - Scores
    """
    # split in train and test set (the 25% last data points are used for test)
    df_train, df_test = train_test_split(df_features, test_size=0.25, shuffle=False)

    # initialize models
    models = initialize_models()

    scores = {}
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X=df_train[cst.FEATURES_LIST], y=df_train[cst.LOAD])
        scores[model_name] = score_model(
            df_train=df_train, df_test=df_test, model=model
        )
        trained_models[model_name] = model
    return trained_models, scores


def score_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame, model: BaseEstimator
) -> Dict[str, Any]:
    """Score a given model on both the training and test set.

    Parameters
    ----------
    df_train : pd.DataFrame
        Train set
    df_test : pd.DataFrame
        Test set
    model : BaseEstimator
        Model to score

    Returns
    -------
    Dict[str, Any]
        Dictionary, with keys:
        - train: train RMSE and MAE
        - test: test RMSE and MAE
    """
    scores = {}
    for kind, df in zip([cst.TRAIN, cst.TEST], [df_train, df_test]):
        y_pred = model.predict(X=df[cst.FEATURES_LIST])
        y_true = df[cst.LOAD]
        scores[kind] = {
            cst.MAE: mean_absolute_error(y_true=y_true, y_pred=y_pred),
            cst.RMSE: mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        }
    return scores
