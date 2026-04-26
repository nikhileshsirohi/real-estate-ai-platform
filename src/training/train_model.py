"""Train a configurable regression model for the real estate project."""

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.features.feature_engineering import TARGET_COLUMN, split_features_and_target
from src.training.evaluate import evaluate_regression_model
from src.utils.config_loader import load_yaml_config


FEATURE_DATA_PATH = Path("data/processed/california_housing_features.csv")
FEATURE_METADATA_PATH = Path("data/processed/california_housing_feature_metadata.json")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
MODEL_OUTPUT_PATH = Path("models/xgboost_price_model_tuned_clean.joblib")
MODEL_METADATA_PATH = Path("models/xgboost_price_model_features.json")
MODEL_METRICS_PATH = Path("models/xgboost_price_model_metrics.json")


def setup_mlflow_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the configured backend and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_training_run(
    model_name: str,
    test_size: float,
    random_state: int,
    metrics: dict[str, float],
    model_params: dict[str, Any],
) -> None:
    """Log the training configuration and evaluation metrics to MLflow."""
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    for param_name, param_value in model_params.items():
        mlflow.log_param(param_name, param_value)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)


def load_feature_dataset(input_path: Path) -> pd.DataFrame:
    """Load the engineered feature dataset."""
    return pd.read_csv(input_path)


def load_feature_metadata(input_path: Path) -> dict[str, Any]:
    """Load feature-engineering metadata for consistent inference."""
    with input_path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def build_model(model_name: str, model_config: dict[str, Any], random_state: int) -> Any:
    """Build the configured regression model."""
    if model_name == "linear_regression":
        return LinearRegression()

    if model_name == "random_forest":
        random_forest_config = model_config.get("random_forest", {})
        return RandomForestRegressor(
            n_estimators=int(random_forest_config.get("n_estimators", 200)),
            max_depth=int(random_forest_config.get("max_depth", 16)),
            min_samples_split=int(random_forest_config.get("min_samples_split", 4)),
            min_samples_leaf=int(random_forest_config.get("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        xgboost_config = model_config.get("xgboost", {})
        return XGBRegressor(
            n_estimators=int(xgboost_config.get("n_estimators", 900)),
            max_depth=int(xgboost_config.get("max_depth", 5)),
            learning_rate=float(xgboost_config.get("learning_rate", 0.07)),
            subsample=float(xgboost_config.get("subsample", 1.0)),
            colsample_bytree=float(xgboost_config.get("colsample_bytree", 0.7)),
            reg_alpha=float(xgboost_config.get("reg_alpha", 1.0)),
            reg_lambda=float(xgboost_config.get("reg_lambda", 3.0)),
            min_child_weight=float(xgboost_config.get("min_child_weight", 7.0)),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def extract_model_params(model_name: str, model_config: dict[str, Any], random_state: int) -> dict[str, Any]:
    """Return the model-specific parameters that should be logged."""
    if model_name != "random_forest":
        if model_name != "xgboost":
            return {}

        xgboost_config = model_config.get("xgboost", {})
        return {
            "n_estimators": int(xgboost_config.get("n_estimators", 900)),
            "max_depth": int(xgboost_config.get("max_depth", 5)),
            "learning_rate": float(xgboost_config.get("learning_rate", 0.07)),
            "subsample": float(xgboost_config.get("subsample", 1.0)),
            "colsample_bytree": float(xgboost_config.get("colsample_bytree", 0.7)),
            "reg_alpha": float(xgboost_config.get("reg_alpha", 1.0)),
            "reg_lambda": float(xgboost_config.get("reg_lambda", 3.0)),
            "min_child_weight": float(xgboost_config.get("min_child_weight", 7.0)),
            "model_random_state": random_state,
        }

    random_forest_config = model_config.get("random_forest", {})
    return {
        "n_estimators": int(random_forest_config.get("n_estimators", 200)),
        "max_depth": int(random_forest_config.get("max_depth", 16)),
        "min_samples_split": int(random_forest_config.get("min_samples_split", 4)),
        "min_samples_leaf": int(random_forest_config.get("min_samples_leaf", 2)),
        "model_random_state": random_state,
    }


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit the configured regressor."""
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(
    model: Any,
    model_name: str,
    feature_columns: list[str],
    feature_metadata: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Persist the trained model and metadata for inference."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    model_metadata = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "feature_metadata_path": str(FEATURE_METADATA_PATH),
        "clip_thresholds": feature_metadata["clip_thresholds"],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


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
    """Run the model training flow end to end."""
    model_config = load_yaml_config(MODEL_CONFIG_PATH)
    experiment_name = str(model_config["experiment_name"])
    model_name = str(model_config["model_name"])
    test_size = float(model_config["test_size"])
    random_state = int(model_config["random_state"])
    tracking_uri = str(model_config["tracking_uri"])

    feature_df = load_feature_dataset(FEATURE_DATA_PATH)
    feature_metadata = load_feature_metadata(FEATURE_METADATA_PATH)
    X, y = split_features_and_target(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model(model_name=model_name, model_config=model_config, random_state=random_state)
    model_params = extract_model_params(model_name=model_name, model_config=model_config, random_state=random_state)
    model = train_model(model=model, X_train=X_train, y_train=y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_regression_model(y_test, predictions)
    save_model_artifacts(
        model=model,
        model_name=model_name,
        feature_columns=X.columns.tolist(),
        feature_metadata=feature_metadata,
        metrics=metrics,
    )

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
                    model_params=model_params,
                )
                mlflow.log_artifact(str(MODEL_OUTPUT_PATH))
                mlflow.log_artifact(str(MODEL_METADATA_PATH))
                mlflow.log_artifact(str(MODEL_METRICS_PATH))
            mlflow_logging_status = "success"
        except Exception as exc:
            mlflow_logging_status = f"skipped ({exc})"
    else:
        mlflow_logging_status = "skipped (tracking server unavailable)"

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Selected model: {model_name}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow logging status: {mlflow_logging_status}")
    print(f"Saved model path: {MODEL_OUTPUT_PATH}")
    print(f"Saved model metadata path: {MODEL_METADATA_PATH}")
    print(f"Saved model metrics path: {MODEL_METRICS_PATH}")
    print("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
