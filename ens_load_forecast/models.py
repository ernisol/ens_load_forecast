"""Module used for model training."""

from typing import Any, Dict, Tuple
import joblib
import json

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import ens_load_forecast.constants as cst
from ens_load_forecast.paths import PATH_SAVED_MODELS


class NaiveModel(BaseEstimator):
    """Naive model that returns directly the input load forecast."""

    def __init__(self) -> None:  # noqa: D107 (disable ruff: missing docstring)
        super().__init__()

    def fit(self, X, y) -> None:  # noqa: D102, N803
        pass

    def predict(self, X):  # noqa: D102, N803
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
        - gradient_boosting_model
        - radom_forest_model
    """
    naive_model = NaiveModel()  # Scaling unwanted here
    linear_model = Pipeline(
        steps=[
            # ("standard_scaler", StandardScaler()),  # Scaling unnecessary
            ("linear_model", LinearRegression(fit_intercept=True)),
        ]
    )
    polynomial_model = Pipeline(
        steps=[
            # ("standard_scaler", StandardScaler()),  # Scaling does not change much
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear_regression", LinearRegression(fit_intercept=True)),
        ]
    )
    gradient_boosting_model = Pipeline(
        steps=[
            # ("standard_scaler", StandardScaler()),  # Scaling does not change much
            ("gradient_boosting_model", GradientBoostingRegressor(n_estimators=100)),
        ]
    )
    random_forest_model = Pipeline(
        steps=[
            # ("standard_scaler", StandardScaler()),  # Scaling does not change much
            ("random_forest_model", RandomForestRegressor(n_estimators=100)),
        ]
    )
    return {
        cst.NAIVE_MODEL: naive_model,
        cst.LINEAR_MODEL: linear_model,
        cst.POLYNOMIAL_MODEL: polynomial_model,
        cst.GRADIENT_BOOSTING_MODEL: gradient_boosting_model,
        cst.RANDOM_FOREST_MODEL: random_forest_model,
    }


def train_models_for_each_zone(
    df_features: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train each model on each zone.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing features for all zones

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        Two dictionaries:
        - trained models
        - scores
    """
    models = {}
    scores = {}
    for zone in df_features[cst.ZONE].unique():
        df = df_features[df_features[cst.ZONE] == zone]
        zone_models, zone_scores = train_models(df_features=df)
        models[zone] = zone_models
        scores[zone] = zone_scores
    save_models(models=models, scores=scores)
    return models, scores


def train_models(
    df_features: pd.DataFrame,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Any]]:
    """Train and score all defined models on the given Dataset.

    Preferably `df_features` should only contain one zone.

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


def save_models(models: Dict[str, Any], scores: Dict[str, Any]) -> None:
    """Save models and scores in sub-folders.

    Parameters
    ----------
    models : Dict[str, Any]
        Models dictionary (one key per zone, then one key per model type)
    scores : Dict[str, Any]
        Scores dictionary (one key per zone, then one key per model type then train/test)
    """
    if not PATH_SAVED_MODELS.exists():
        PATH_SAVED_MODELS.mkdir()
    for zone, zone_models in models.items():
        if not (PATH_SAVED_MODELS / zone).exists():
            (PATH_SAVED_MODELS / zone).mkdir()
        for model_name, model in zone_models.items():
            joblib.dump(
                value=model, filename=PATH_SAVED_MODELS / zone / f"{model_name}.joblib"
            )
        with open(
            PATH_SAVED_MODELS / zone / "scores.json", mode="w", encoding="utf-8"
        ) as file:
            json.dump(obj=scores[zone], fp=file, indent=4)
