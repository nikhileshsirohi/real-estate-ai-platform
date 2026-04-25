"""Tests for the data cleaning step."""

import pandas as pd

from src.data.cleaning import clean_dataset


def test_clean_dataset_standardizes_columns_and_removes_nulls() -> None:
    raw_df = pd.DataFrame(
        {
            "MedInc": [8.3, 8.3],
            "HouseAge": [41.0, 41.0],
            "AveRooms": [6.9, 6.9],
            "AveBedrms": [1.0, 1.0],
            "Population": [322.0, 322.0],
            "AveOccup": [2.5, 2.5],
            "Latitude": [37.88, 37.88],
            "Longitude": [-122.23, -122.23],
            "MedHouseVal": [4.526, None],
        }
    )

    cleaned_df = clean_dataset(raw_df)

    assert list(cleaned_df.columns) == [
        "median_income",
        "house_age",
        "average_rooms",
        "average_bedrooms",
        "population",
        "average_occupancy",
        "latitude",
        "longitude",
        "median_house_value",
    ]
    assert cleaned_df.shape == (1, 9)
