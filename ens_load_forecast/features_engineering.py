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
