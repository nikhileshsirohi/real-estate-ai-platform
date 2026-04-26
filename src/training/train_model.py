"""Train a baseline regression model for V1."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import TARGET_COLUMN, split_features_and_target
from src.training.evaluate import evaluate_regression_model


FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_feature_dataset(input_path: Path) -> pd.DataFrame:
    """Load the engineered feature dataset."""
    return pd.read_csv(input_path)


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a simple baseline regressor."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Run the baseline training flow end to end."""
    feature_df = load_feature_dataset(FEATURE_DATA_PATH)
    X, y = split_features_and_target(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = train_baseline_model(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_regression_model(y_test, predictions)

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print("Baseline evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
