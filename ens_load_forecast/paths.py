"""Module containing paths for the project."""

from pathlib import Path

# Folders

PATH_REPO = Path("").resolve()
PATH_MODULE = PATH_REPO / "ens_load_forecast"
PATH_DATA = PATH_MODULE / "data"

# Files

PATH_LOAD_ACTUAL = PATH_DATA / "load_actual.csv"
PATH_LOAD_FORECAST = PATH_DATA / "load_forecast.csv"
PATH_WEATHER = PATH_DATA / "weather.csv"
PATH_PREPROCESSED_WEATHER = PATH_DATA / "preprocessed_weather.csv"
PATH_ZONES_AND_STATIONS = PATH_DATA / "zones_and_stations.csv"
PATH_MAP_DATA = PATH_DATA / "map_data.geojson"
