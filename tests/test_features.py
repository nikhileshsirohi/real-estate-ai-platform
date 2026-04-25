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
    assert TARGET_COLUMN not in X.columns
    assert y.iloc[0] == 4.526
