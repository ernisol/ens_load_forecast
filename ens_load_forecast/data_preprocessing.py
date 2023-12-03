"""Module to load and pre-process data (handle index, timezones, etc.)."""
import numpy as np
import pandas as pd
import pytz

import ens_load_forecast.constants as cst
from ens_load_forecast.paths import (
    PATH_LOAD_ACTUAL,
    PATH_LOAD_FORECAST,
    PATH_PREPROCESSED_WEATHER,
    PATH_WEATHER,
    PATH_ZONES_AND_STATIONS,
)

eastern_tz = pytz.timezone("EST")


def get_load_actual() -> pd.DataFrame:
    """Get actual load data. Time zone is `EST`.

    Parameters
    ----------
    path : Path
        Path to the data

    Returns
    -------
    pd.DataFrame
        Loaded data
        - index: date (EST)
        - columns:
            - zone: the zone
            - load: the load (MW)
    """
    df = pd.read_csv(
        PATH_LOAD_ACTUAL, index_col=0
    )  # using first column as index (date)
    # Localizing to Eastern Standard Time
    df.index = pd.to_datetime(df.index).tz_localize(tz=eastern_tz)
    return df


def get_load_forecast() -> pd.DataFrame:
    """Get forecast load data. Time zone is `EST`.

    Parameters
    ----------
    path : Path
        Path to the data

    Returns
    -------
    pd.DataFrame
        Loaded data
        - index: date (EST)
        - columns:
            - zone: the zone
            - load: the load (MW)
            - vintage_date: issued date (around 11:30 AM)
    """
    df = pd.read_csv(
        PATH_LOAD_FORECAST, index_col=0
    )  # use first column as index (target date of the forecast)

    # Handle dates
    df.index = pd.to_datetime(df.index)
    df[cst.VINTAGE_DATE] = pd.to_datetime(df[cst.VINTAGE_DATE])
    # Add 11:30 AM to issued date, and localizing to eastern time
    df[cst.VINTAGE_DATE] = (
        pd.to_datetime(df[cst.VINTAGE_DATE]) + pd.Timedelta(hours=11, minutes=30)
    ).dt.tz_localize(tz=eastern_tz)
    # localize index
    df.index = df.index.tz_localize(tz=eastern_tz, ambiguous=True)

    # Capitalize zone
    df[cst.ZONE] = df[cst.ZONE].apply(lambda x: x.upper())

    # Remove forbidden forecasts (They must be issued before 5AM on the previous day)
    df = remove_forbidden_forecasts(df=df, duplicates_key=cst.ZONE)

    return df


def get_weather(force_recompute: bool) -> pd.DataFrame:
    """Get weather forecast data.

    Parameters
    ----------
    force_recompute : bool
        Recompute the weather dataframe instead of using saved one.

    Returns
    -------
    pd.DataFrame
        - index: date (EST)
        - columns:
            - zone
            - weight: weight of the station in the zone
            - a column per weather feature
    """
    if PATH_PREPROCESSED_WEATHER.exists() and not force_recompute:
        return get_preprocessed_weather()

    df = pd.read_csv(
        PATH_WEATHER, index_col=1, low_memory=False
    )  # using second column as index (target date of the forecast)
    df_zones_and_stations = pd.read_csv(
        PATH_ZONES_AND_STATIONS, index_col=1
    )  # using station code as index

    # Localize in UTC
    df.index = pd.to_datetime(df.index, utc=True)
    df[cst.VINTAGE_DATE] = pd.to_datetime(df[cst.VINTAGE_DATE], utc=True)

    # Convert to EST time (original timezone is UTC)
    df.index = df.index.tz_convert(tz=eastern_tz)
    df[cst.VINTAGE_DATE] = df[cst.VINTAGE_DATE].dt.tz_convert(tz=eastern_tz)

    # Remove forbidden forecasts (They must be issued before 5AM on the previous day)
    df = remove_forbidden_forecasts(df=df, duplicates_key=cst.STATION_CODE)

    # Add zone. Note: index gets duplicated here because some stations are used for
    # multiple zones.
    df = df.join(other=df_zones_and_stations, on=cst.STATION_CODE, how="left")

    aggregated_df = aggregate_weather_record(df=df)

    aggregated_df.to_csv(path_or_buf=PATH_PREPROCESSED_WEATHER)

    return aggregated_df


def get_preprocessed_weather() -> pd.DataFrame:
    """Get weather data from a preprocessed csv file.

    Returns
    -------
    pd.DataFrame
        The preprocessed weather data.
    """
    df = pd.read_csv(PATH_PREPROCESSED_WEATHER, index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index)
    return df


def remove_forbidden_forecasts(df: pd.DataFrame, duplicates_key: str) -> pd.DataFrame:
    """Remove forecasts that are not available the previous day at 5 AM.

    Also drop duplicates, keeping most recent forecast.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    duplicates_key : str
        Key used to remove duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame without forbidden forecasts.
    """
    # Remove forbidden forecasts (must be available the day before at 5 AM)
    last_valid_date = df.index - pd.Timedelta(value=1, unit="days")
    last_valid_date = last_valid_date.round(freq="D") + pd.Timedelta(
        value=5, unit="hours"
    )

    # Drop forbidden previsions
    df = df[df[cst.VINTAGE_DATE] < last_valid_date]

    # Only keep most recent prevision
    df["tmp_col"] = df.index
    df = df.drop_duplicates(subset=["tmp_col", duplicates_key], keep="last").drop(
        "tmp_col", axis="columns"
    )

    return df


def aggregate_weather_record(df: pd.DataFrame) -> pd.DataFrame:
    """Weighs forecasts according to column `weight`.

    Then sum and divide by sum of weights.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to aggregate.

    Returns
    -------
    pd.DataFrame
        Dataframe with aggregated values.
    """
    # Cast wind speed to float
    # (missing values are indicated with a string, therefore the column has object type)
    df[df[cst.WSP] == cst.NG] = np.nan
    df[cst.WSP] = df[cst.WSP].astype(float)

    # Weigh and sum over stations
    df[cst.SELECTED_WEATHER_FEATURES] = df[cst.SELECTED_WEATHER_FEATURES].mul(
        other=df[cst.WEIGHT], axis="index"
    )

    def func(x):
        return x[cst.SELECTED_WEATHER_FEATURES].sum() / (x[cst.WEIGHT].sum())

    return df.groupby(by=[cst.DELIVERY_TS, cst.ZONE]).apply(func)


def get_merged_dataset(
    df_weather: pd.DataFrame,
    df_load_actual: pd.DataFrame,
    df_load_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all datasets in one.

    Parameters
    ----------
    df_weather : pd.DataFrame
        Weather data
    df_load_actual : pd.DataFrame
        Actual load data
    df_load_forecast : pd.DataFrame
        Forecast load data

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    # Merge load and load forecast
    df_merged = pd.merge(
        left=df_load_forecast,
        right=df_load_actual,
        on=[cst.DELIVERY_TS, cst.ZONE],
        how="inner",
    )
    # Two columns have the same name, we need to rename them after the merging operation
    renaming_dict = {f"{cst.LOAD}_x": cst.LOAD_FORECAST, f"{cst.LOAD}_y": cst.LOAD}
    df_merged = df_merged.rename(columns=renaming_dict)

    # Merge the result with weather data
    df_merged = pd.merge(
        left=df_merged, right=df_weather, on=[cst.DELIVERY_TS, cst.ZONE], how="inner"
    )

    # Drop nan values
    df_merged = df_merged.dropna(how="any", axis="index")
    return df_merged
