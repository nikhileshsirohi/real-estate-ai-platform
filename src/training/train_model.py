"""Train a baseline regression model for V1."""

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import TARGET_COLUMN, split_features_and_target
from src.training.evaluate import evaluate_regression_model
from src.utils.config_loader import load_yaml_config


FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
MODEL_OUTPUT_PATH = Path("models/linear_regression_model.joblib")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")


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


def save_model_artifacts(model: LinearRegression, feature_columns: list[str]) -> None:
    """Persist the trained model and feature column order for inference."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def is_tracking_server_available(tracking_uri: str) -> bool:
    """Check whether an HTTP MLflow tracking server is reachable quickly."""
    if not tracking_uri.startswith(("http://", "https://")):
        return True

    try:
        with urlopen(tracking_uri, timeout=1):
            return True
    except (URLError, TimeoutError, OSError):
        return False


def main() -> None:
    """Run the baseline training flow end to end."""
    model_config = load_yaml_config(MODEL_CONFIG_PATH)
    experiment_name = str(model_config["experiment_name"])
    model_name = str(model_config["model_name"])
    test_size = float(model_config["test_size"])
    random_state = int(model_config["random_state"])
    tracking_uri = str(model_config["tracking_uri"])

    feature_df = load_feature_dataset(FEATURE_DATA_PATH)
    X, y = split_features_and_target(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = train_baseline_model(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_regression_model(y_test, predictions)
    save_model_artifacts(model=model, feature_columns=X.columns.tolist())

    mlflow_logging_status = "not attempted"
    if is_tracking_server_available(tracking_uri):
        try:
            setup_mlflow_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
            with mlflow.start_run():
                log_training_run(
                    model_name=model_name,
                    test_size=test_size,
                    random_state=random_state,
                    metrics=metrics,
                )
                mlflow.log_artifact(str(MODEL_OUTPUT_PATH))
                mlflow.log_artifact(str(FEATURE_COLUMNS_PATH))
            mlflow_logging_status = "success"
        except Exception as exc:
            mlflow_logging_status = f"skipped ({exc})"
    else:
        mlflow_logging_status = "skipped (tracking server unavailable)"

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow logging status: {mlflow_logging_status}")
    print(f"Saved model path: {MODEL_OUTPUT_PATH}")
    print(f"Saved feature columns path: {FEATURE_COLUMNS_PATH}")
    print("Baseline evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
