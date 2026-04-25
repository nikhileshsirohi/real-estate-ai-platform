"""Download and save the raw housing dataset for V1."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


RAW_DATA_PATH = Path("data/raw/california_housing.csv")


def download_california_housing() -> pd.DataFrame:
    """Fetch the California housing dataset as a Pandas DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame.copy()


def save_raw_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the raw dataset to disk as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Download the dataset and save it into the raw data folder."""
    df = download_california_housing()
    save_raw_dataset(df, RAW_DATA_PATH)
    print(f"Saved dataset to {RAW_DATA_PATH} with shape: {df.shape}")


if __name__ == "__main__":
    main()
