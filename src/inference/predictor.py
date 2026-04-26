"""Helpers for loading the trained model and generating predictions."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/xgboost_price_model_tuned_clean.joblib")
MODEL_METADATA_PATH = Path("models/xgboost_price_model_features.json")


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
    features["log_population"] = float(np.log1p(features["population"]))
    features["log_average_occupancy"] = float(np.log1p(features["average_occupancy"]))
    features["log_average_rooms"] = float(np.log1p(features["average_rooms"]))
    features["log_average_bedrooms"] = float(np.log1p(features["average_bedrooms"]))
    features["log_rooms_per_person"] = float(np.log1p(features["rooms_per_person"]))
    model_metadata = load_model_metadata()
    clip_thresholds = model_metadata.get("clip_thresholds", {})
    features["population_capped"] = min(features["population"], float(clip_thresholds.get("population", features["population"])))
    features["average_occupancy_capped"] = min(
        features["average_occupancy"],
        float(clip_thresholds.get("average_occupancy", features["average_occupancy"])),
    )
    features["average_rooms_capped"] = min(
        features["average_rooms"],
        float(clip_thresholds.get("average_rooms", features["average_rooms"])),
    )
    features["average_bedrooms_capped"] = min(
        features["average_bedrooms"],
        float(clip_thresholds.get("average_bedrooms", features["average_bedrooms"])),
    )
    features["rooms_per_person_capped"] = min(
        features["rooms_per_person"],
        float(clip_thresholds.get("rooms_per_person", features["rooms_per_person"])),
    )
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
