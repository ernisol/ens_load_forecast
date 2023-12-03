"""Module for feature engineering."""


import numpy as np
import pandas as pd

import ens_load_forecast.constants as cst


def process_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Split wind direction in cos and sin of wind direction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a column `wdr`.

    Returns
    -------
    pd.DataFrame
        DataFrame with split wind direction.
    """
    df_copy = df.copy()
    # first convert in rad
    df_copy[cst.WDR] = df_copy[cst.WDR] * np.pi / 180

    # add cos and sin
    df_copy[cst.COS_WDR] = np.cos(df_copy[cst.WDR])
    df_copy[cst.SIN_WDR] = np.sin(df_copy[cst.WDR])
    df_copy = df_copy.drop(labels=[cst.WDR], axis="columns")
    return df_copy


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot-encoded day of the week.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, index is the target date.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot-encoded day of the week.
    """
    df_copy = df.copy()
    # Create a column with day names (Monday, Tuesday, ...)
    df_copy[cst.MONTH] = df_copy.index.month_name()

    # One-hot-encode
    df_copy = pd.get_dummies(
        data=df_copy, columns=[cst.MONTH], prefix_sep="_", dtype=int
    )
    return df_copy


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot-encoded day of the week.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, index is the target date.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot-encoded day of the week.
    """
    df_copy = df.copy()
    # Create a column with day names (Monday, Tuesday, ...)
    df_copy[cst.DAY_OF_WEEK] = df_copy.index.day_name()

    # One-hot-encode
    df_copy = pd.get_dummies(
        data=df_copy, columns=[cst.DAY_OF_WEEK], prefix_sep="_", dtype=int
    )
    return df_copy


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot-encoded time of day (Morning, working_hours, Evening).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, index is the target date.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot-encoded time of day.
    """
    df_copy = df.copy()

    def get_time_of_day(hour):
        if hour < 7:
            return cst.MORNING
        if hour >= 7 and hour < 19:
            return cst.WORKING_HOURS
        return cst.EVENING

    # Create a column with time of day names (Morning, working_hours, Evening)
    df_copy[f"{cst.TIME_OF_DAY}_as_int"] = df_copy.index.hour
    df_copy[cst.TIME_OF_DAY] = df_copy[f"{cst.TIME_OF_DAY}_as_int"].apply(
        get_time_of_day
    )
    df_copy = df_copy.drop(labels=[f"{cst.TIME_OF_DAY}_as_int"], axis="columns")

    # One-hot-encode
    df_copy = pd.get_dummies(
        data=df_copy, columns=[cst.TIME_OF_DAY], prefix_sep="_", dtype=int
    )
    return df_copy


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with some columns normalized
    """
    df_copy = df.copy()
    for column in cst.PERCENT_FEATURES:
        df_copy[column] = df_copy[column] / 100
    return df_copy


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features
    """
    df_copy = df.copy()
    df_copy = add_month(df=df_copy)
    df_copy = add_time_of_day(df=df_copy)
    df_copy = add_day_of_week(df=df_copy)
    df_copy = process_wind_direction(df=df_copy)
    df_copy = normalize_columns(df=df_copy)
    return df_copy
