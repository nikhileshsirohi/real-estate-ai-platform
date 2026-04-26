"""Tests for the model training step."""

import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.training.evaluate import evaluate_regression_model
from src.training.train_model import build_model, save_model_artifacts, train_model


def test_train_model_returns_fitted_model() -> None:
    X_train = pd.DataFrame(
        {
            "median_income": [2.0, 3.0, 4.0, 5.0],
            "house_age": [10.0, 15.0, 20.0, 25.0],
            "average_rooms": [4.0, 5.0, 6.0, 7.0],
        }
    )
    y_train = pd.Series([1.2, 1.8, 2.4, 3.0])

    model = build_model("linear_regression", {}, random_state=42)
    model = train_model(model, X_train, y_train)
    predictions = model.predict(X_train)
    metrics = evaluate_regression_model(y_train, predictions)

    assert len(predictions) == len(y_train)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


def test_build_model_returns_random_forest_when_configured() -> None:
    model = build_model(
        "random_forest",
        {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
        },
        random_state=42,
    )

    assert isinstance(model, RandomForestRegressor)


def test_save_model_artifacts_creates_model_and_metadata_files(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "median_income": [2.0, 3.0, 4.0, 5.0],
            "house_age": [10.0, 15.0, 20.0, 25.0],
            "average_rooms": [4.0, 5.0, 6.0, 7.0],
        }
    )
    y_train = pd.Series([1.2, 1.8, 2.4, 3.0])

    model = build_model("linear_regression", {}, random_state=42)
    model = train_model(model, X_train, y_train)

    from src.training import train_model as train_model_module

    original_model_path = train_model_module.MODEL_OUTPUT_PATH
    original_metadata_path = train_model_module.MODEL_METADATA_PATH

    train_model_module.MODEL_OUTPUT_PATH = tmp_path / "trained_model.joblib"
    train_model_module.MODEL_METADATA_PATH = tmp_path / "model_metadata.json"
    try:
        save_model_artifacts(model, "linear_regression", X_train.columns.tolist())
        assert train_model_module.MODEL_OUTPUT_PATH.exists()
        assert train_model_module.MODEL_METADATA_PATH.exists()
        saved_metadata = json.loads(train_model_module.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        assert saved_metadata["model_name"] == "linear_regression"
        assert saved_metadata["feature_columns"] == X_train.columns.tolist()
    finally:
        train_model_module.MODEL_OUTPUT_PATH = original_model_path
        train_model_module.MODEL_METADATA_PATH = original_metadata_path
