"""Module implementing graphs."""

from typing import Any
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import ens_load_forecast.constants as cst
from ens_load_forecast.paths import PATH_MAP_DATA
from ens_load_forecast.plot_params import (
    MAP_LAYOUT,
    EQUAL_ASPECT_RATIO_LAYOUT,
    TRANSPARENT_RED_MARKERS,
)


def plot_load_per_zone(df: pd.DataFrame, **kwargs: Any) -> None:
    """Plot load evolution for each zone.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing load values for all zones.
        Expected columns:
        - "zone"
        - "load"
    """
    fig = go.Figure()
    for zone in df[cst.ZONE].unique():
        zone_df = df[df[cst.ZONE] == zone]
        fig.add_trace(
            go.Scatter(
                x=zone_df.index,
                y=zone_df[cst.LOAD],
                name=zone,
            )
        )
    fig.update_layout(yaxis={"ticksuffix": "MW"}, **kwargs)
    fig.show()


def plot_load_seasonal(df: pd.DataFrame, zone: str, **kwargs: Any) -> None:
    """Plot a heatmap showing averaged data on a grid.

    With
    - x-axis: day of year
    - y-axis: hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        The load data
    zone : str
        The zone
    """
    load_df = df[df[cst.ZONE] == zone].drop(labels="zone", axis="columns")
    load_df[cst.DAY_OF_YEAR] = load_df.index.dayofyear
    load_df[cst.HOUR] = load_df.index.hour
    average_seasonal_load = load_df.groupby([cst.DAY_OF_YEAR, cst.HOUR]).mean()
    load_array = average_seasonal_load[cst.LOAD].to_numpy().reshape((366, 24)).T
    # Create a day of year axis (from a leap year)
    x_axis_date = pd.Series(pd.date_range(start="2000-01-01", end="2000-12-31")).apply(
        lambda x: f"{x.month_name()} {x.day}"
    )
    fig = px.imshow(
        img=load_array,
        x=x_axis_date,
        labels={
            "x": "Day of year",
            "y": "Hour of day",
        },
    )
    fig.update_layout(**kwargs)
    fig.show()


def plot_on_map(df: pd.DataFrame, quantity_key: str, **kwargs: Any) -> None:
    """Represent a quantity on a map (NYISO) using a color gradient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a `zone` column.
    quantity_key : str
        Name of DataFrame column used to color the map.
    """
    geo_data = gpd.read_file(PATH_MAP_DATA)
    fig = px.choropleth(
        data_frame=df,
        geojson=geo_data,
        locations=cst.ZONE,
        featureidkey="properties.id",
        color=quantity_key,
        **kwargs,
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(**MAP_LAYOUT)
    fig.show()


def correlation_heatmap(df: pd.DataFrame, **kwargs: Any) -> None:
    """Plots a correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame
    """
    correlations = df[[*cst.FEATURES_LIST, cst.LOAD]].corr(method="pearson")

    heatmap = go.Heatmap(
        z=correlations,
        x=correlations.columns,
        y=correlations.columns,
        colorscale=px.colors.diverging.RdBu,
        zmin=-1,
        zmax=1,
    )
    fig = go.Figure(heatmap)
    fig.update_layout(**EQUAL_ASPECT_RATIO_LAYOUT, **kwargs)
    fig.show()


def scatter_matrix(df: pd.DataFrame, **kwargs: Any) -> None:
    """Plots a scatter matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame.
    """
    features_to_plot = [
        cst.LOAD_FORECAST,
        cst.TMP,
        cst.DPT,
        cst.SKY,
        cst.WSP,
        cst.GST,
        cst.PSN,
        cst.COS_WDR,
        cst.SIN_WDR,
        cst.LOAD,
    ]
    fig = px.scatter_matrix(
        df[features_to_plot],
    )
    fig.update_traces(marker=TRANSPARENT_RED_MARKERS)
    fig.update_layout(**EQUAL_ASPECT_RATIO_LAYOUT, **kwargs)
    fig.show()
