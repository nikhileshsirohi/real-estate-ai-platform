"""Helpers for loading the trained model and generating predictions."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_PATH = Path("models/trained_model.joblib")
MODEL_METADATA_PATH = Path("models/model_metadata.json")


def load_trained_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the saved regression model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found at {model_path}")

    return joblib.load(model_path)


def load_model_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Load the saved model metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata file not found at {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_feature_columns(metadata_path: Path = MODEL_METADATA_PATH) -> list[str]:
    """Load the saved feature order used during training."""
    metadata = load_model_metadata(metadata_path)
    return list(metadata["feature_columns"])


def load_model_name(metadata_path: Path = MODEL_METADATA_PATH) -> str:
    """Load the saved model name used during training."""
    metadata = load_model_metadata(metadata_path)
    return str(metadata["model_name"])


def create_inference_features(input_data: dict[str, float]) -> dict[str, float]:
    """Create the full feature dictionary expected by the trained model."""
    average_rooms = input_data["average_rooms"]
    average_bedrooms = input_data["average_bedrooms"]
    average_occupancy = input_data["average_occupancy"]

    features = dict(input_data)
    features["bedroom_ratio"] = average_bedrooms / average_rooms
    features["rooms_per_person"] = average_rooms / average_occupancy
    return features


def prepare_features_for_inference(input_data: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to the trained feature order."""
    full_feature_dict = create_inference_features(input_data)
    return pd.DataFrame(
        [[full_feature_dict[column] for column in feature_columns]],
        columns=feature_columns,
    )


def predict_price(input_data: dict[str, float]) -> float:
    """Run an end-to-end price prediction using the saved local artifacts."""
    model = load_trained_model()
    feature_columns = load_feature_columns()
    inference_df = prepare_features_for_inference(input_data=input_data, feature_columns=feature_columns)
    prediction = model.predict(inference_df)[0]
    return float(prediction)
