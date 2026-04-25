"""Clean the raw housing dataset for downstream modeling."""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/california_housing.csv")
PROCESSED_DATA_PATH = Path("data/processed/california_housing_cleaned.csv")


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    """Load the raw housing dataset from disk."""
    return pd.read_csv(input_path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataset columns to a consistent snake_case style."""
    renamed_df = df.copy()
    renamed_df.columns = [
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
    return renamed_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a minimal V1 cleaning routine."""
    cleaned_df = standardize_column_names(df)
    cleaned_df = cleaned_df.drop_duplicates().dropna()
    return cleaned_df


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned dataset into the processed data folder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the cleaning step end to end."""
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    save_processed_dataset(cleaned_df, PROCESSED_DATA_PATH)
    print(f"Saved cleaned dataset to {PROCESSED_DATA_PATH} with shape: {cleaned_df.shape}")


if __name__ == "__main__":
    main()
