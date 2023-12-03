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

# wind speed has value "NG" sometimes
NG = "NG"

# weather
TMP = "tmp"  # Temperature, deg F
DPT = "dpt"  # Dew point temperature, deg F
SKY = "sky"  # Sky cover, percent
WDR = "wdr"  # Wind drection, deg
WSP = "wsp"  # Wind speed, knots
GST = "gst"  # Wind gust, knots
PSN = "psn"  # Probability of snow, percent

SELECTED_WEATHER_FEATURES = [TMP, DPT, SKY, WDR, WSP, GST, PSN]

# Features
COS_WDR = "cos_wdr"
SIN_WDR = "sin_wdr"
DAY_OF_WEEK = "day_of_week"
TIME_OF_DAY = "time_of_day"
DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MORNING = "Morning"
WORKING_HOURS = "working_hours"
EVENING = "Evening"
DAY_TIMES = [
    MORNING,
    WORKING_HOURS,
    EVENING,
]
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
PERCENT_FEATURES = [
    SKY,
    PSN,
]
FEATURES_LIST = [
    LOAD_FORECAST,
    TMP,
    DPT,
    SKY,
    WSP,
    GST,
    PSN,
    COS_WDR,
    SIN_WDR,
    *[f"{MONTH}_{name}" for name in MONTH_NAMES],
    *[f"{DAY_OF_WEEK}_{name}" for name in DAY_NAMES],
    *[f"{TIME_OF_DAY}_{name}" for name in DAY_TIMES],
]


# Models

NAIVE_MODEL = "naive_model"
LINEAR_MODEL = "linear_model"
POLYNOMIAL_MODEL = "polynomial_model"
GRADIENT_BOOSTING_MODEL = "gradient_boosting_model"
RANDOM_FOREST_MODEL = "radom_forest_model"
TRAIN = "train"
TEST = "test"
MAE = "mae"
RMSE = "rmse"


# Files
JSON = ".json"
JOBLIB = ".joblib"
