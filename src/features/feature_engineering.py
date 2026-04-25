"""Build model-ready features from the cleaned housing dataset."""

from pathlib import Path

import pandas as pd


CLEANED_DATA_PATH = Path("data/processed/california_housing_cleaned.csv")
FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
TARGET_COLUMN = "median_house_value"


def load_cleaned_dataset(input_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    return pd.read_csv(input_path)


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal set of additional model features for V1."""
    feature_df = df.copy()
    feature_df["bedroom_ratio"] = feature_df["average_bedrooms"] / feature_df["average_rooms"]
    feature_df["rooms_per_person"] = feature_df["average_rooms"] / feature_df["average_occupancy"]
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


def main() -> None:
    """Run feature engineering end to end for the V1 dataset."""
    cleaned_df = load_cleaned_dataset(CLEANED_DATA_PATH)
    feature_df = create_engineered_features(cleaned_df)
    save_feature_dataset(feature_df, FEATURE_DATA_PATH)
    X, y = split_features_and_target(feature_df)
    print(f"Saved feature dataset to {FEATURE_DATA_PATH} with shape: {feature_df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")


if __name__ == "__main__":
    main()
