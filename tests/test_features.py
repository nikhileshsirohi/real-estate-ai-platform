"""Tests for the feature engineering step."""

import pandas as pd

from src.features.feature_engineering import (
    TARGET_COLUMN,
    create_engineered_features,
    split_features_and_target,
)


def test_create_engineered_features_adds_expected_columns() -> None:
    cleaned_df = pd.DataFrame(
        {
            "median_income": [8.3],
            "house_age": [41.0],
            "average_rooms": [6.9],
            "average_bedrooms": [1.0],
            "population": [322.0],
            "average_occupancy": [2.5],
            "latitude": [37.88],
            "longitude": [-122.23],
            "median_house_value": [4.526],
        }
    )

    feature_df = create_engineered_features(cleaned_df)
    X, y = split_features_and_target(feature_df)

    assert "bedroom_ratio" in feature_df.columns
    assert "rooms_per_person" in feature_df.columns
    assert "log_population" in feature_df.columns
    assert "log_average_occupancy" in feature_df.columns
    assert "population_capped" in feature_df.columns
    assert "rooms_per_person_capped" in feature_df.columns
    assert TARGET_COLUMN not in X.columns
    assert y.iloc[0] == 4.526


def test_create_engineered_features_adds_capped_and_log_features() -> None:
    cleaned_df = pd.DataFrame(
        {
            "median_income": [8.3, 7.1, 6.2],
            "house_age": [41.0, 20.0, 30.0],
            "average_rooms": [6.9, 7.2, 8.5],
            "average_bedrooms": [1.0, 1.1, 1.2],
            "population": [322.0, 1200.0, 5000.0],
            "average_occupancy": [2.5, 3.0, 10.0],
            "latitude": [37.88, 37.2, 36.7],
            "longitude": [-122.23, -121.8, -119.5],
            "median_house_value": [4.526, 3.5, 2.7],
        }
    )

    feature_df = create_engineered_features(cleaned_df)

    assert (feature_df["log_population"] > 0).all()
    assert (feature_df["log_average_rooms"] > 0).all()
    assert (feature_df["population_capped"] <= feature_df["population"]).all()
    assert (feature_df["average_occupancy_capped"] <= feature_df["average_occupancy"]).all()
