"""Main module"""
from ens_load_forecast.data_preprocessing import (
    get_merged_dataset,
    get_load_actual,
    get_load_forecast,
    get_weather,
)
from ens_load_forecast.features_engineering import extract_features
from ens_load_forecast.models import train_models_for_each_zone


def main() -> None:
    # Load and preprocess data
    df_load_actual = get_load_actual()
    df_load_forecast = get_load_forecast()
    df_weather = get_weather(
        force_recompute=False
    )  # Allow up to 5 minutes the first time

    # Merge data
    df_merged = get_merged_dataset(
        df_load_actual=df_load_actual,
        df_load_forecast=df_load_forecast,
        df_weather=df_weather,
    )

    # Extract features
    df_features = extract_features(df=df_merged)

    # Train and save models
    train_models_for_each_zone(df_features=df_features, force_retrain=False)


if __name__ == "__main__":
    main()
