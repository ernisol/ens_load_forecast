"""Module containing constants for the project."""

# Data-related constants
ZONE = "zone"
LOAD = "load"
LOAD_FORECAST = "load_forecast"
HOUR = "hour"
MONTH = "month"
DAY_OF_YEAR = "day_of_year"
VINTAGE_DATE = "vintage_date"
FORECAST_HORIZON = "forecast_horizon"
STATION_CODE = "station_code"
DELIVERY_TS = "delivery_ts"
WEIGHT = "weight"

# weather
TMP = "tmp"  # Temperature, deg F
DPT = "dpt"  # Dew point temperature, deg F
SKY = "sky"  # Sky cover, percent
WDR = "wdr"  # Wind drection, deg
# WSP = "wsp"  # Wind speed, knots
GST = "gst"  # Wind gust, knots
PSN = "psn"  # Probability of snow, percent

SELECTED_WEATHER_FEATURES = [TMP, DPT, SKY, WDR, GST, PSN]
