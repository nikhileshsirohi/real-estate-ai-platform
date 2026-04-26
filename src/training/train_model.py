"""Train a baseline regression model for V1."""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import TARGET_COLUMN, split_features_and_target
from src.training.evaluate import evaluate_regression_model
from src.utils.config_loader import load_yaml_config


FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")


def setup_mlflow_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the configured backend and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_training_run(model_name: str, test_size: float, random_state: int, metrics: dict[str, float]) -> None:
    """Log the training configuration and evaluation metrics to MLflow."""
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)


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
    model_config = load_yaml_config(MODEL_CONFIG_PATH)
    experiment_name = str(model_config["experiment_name"])
    model_name = str(model_config["model_name"])
    test_size = float(model_config["test_size"])
    random_state = int(model_config["random_state"])
    tracking_uri = str(model_config["tracking_uri"])

    setup_mlflow_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)

    feature_df = load_feature_dataset(FEATURE_DATA_PATH)
    X, y = split_features_and_target(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    with mlflow.start_run():
        model = train_baseline_model(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_regression_model(y_test, predictions)
        log_training_run(
            model_name=model_name,
            test_size=test_size,
            random_state=random_state,
            metrics=metrics,
        )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print("Baseline evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
