"""Build model-ready features from the cleaned housing dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


CLEANED_DATA_PATH = Path("data/processed/california_housing_cleaned.csv")
FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
FEATURE_METADATA_PATH = Path("data/processed/california_housing_feature_metadata.json")
TARGET_COLUMN = "median_house_value"
CAP_QUANTILE = 0.99
SKEWED_COLUMNS = [
    "population",
    "average_occupancy",
    "average_rooms",
    "average_bedrooms",
    "rooms_per_person",
]


def load_cleaned_dataset(input_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    return pd.read_csv(input_path)


def add_base_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio features that later steps depend on."""
    feature_df = df.copy()
    feature_df["bedroom_ratio"] = feature_df["average_bedrooms"] / feature_df["average_rooms"]
    feature_df["rooms_per_person"] = feature_df["average_rooms"] / feature_df["average_occupancy"]
    return feature_df


def compute_clip_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """Compute upper clipping thresholds for the skewed columns."""
    return {
        column_name: float(df[column_name].quantile(CAP_QUANTILE))
        for column_name in SKEWED_COLUMNS
    }


def create_engineered_features(
    df: pd.DataFrame,
    clip_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Create additional model features guided by EDA findings."""
    feature_df = add_base_ratio_features(df)

    thresholds = clip_thresholds or compute_clip_thresholds(feature_df)
    for column_name in SKEWED_COLUMNS:
        feature_df[f"log_{column_name}"] = np.log1p(feature_df[column_name])
        feature_df[f"{column_name}_capped"] = feature_df[column_name].clip(upper=thresholds[column_name])

    return feature_df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the prediction target."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def save_feature_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the engineered dataset for downstream training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_feature_metadata(clip_thresholds: dict[str, float], output_path: Path) -> None:
    """Persist feature-engineering metadata needed for consistent inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cap_quantile": CAP_QUANTILE,
        "clip_thresholds": clip_thresholds,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Run feature engineering end to end for the V1 dataset."""
    cleaned_df = load_cleaned_dataset(CLEANED_DATA_PATH)
    base_feature_df = add_base_ratio_features(cleaned_df)
    clip_thresholds = compute_clip_thresholds(base_feature_df)
    feature_df = create_engineered_features(cleaned_df, clip_thresholds=clip_thresholds)
    save_feature_dataset(feature_df, FEATURE_DATA_PATH)
    save_feature_metadata(clip_thresholds, FEATURE_METADATA_PATH)
    X, y = split_features_and_target(feature_df)
    print(f"Saved feature dataset to {FEATURE_DATA_PATH} with shape: {feature_df.shape}")
    print(f"Saved feature metadata to {FEATURE_METADATA_PATH}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")


if __name__ == "__main__":
    main()
