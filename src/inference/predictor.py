"""Helpers for loading the trained model and generating predictions."""

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from src.utils.config_loader import get_project_root, resolve_project_path

MODEL_PATH = Path("models/xgboost_price_model_tuned_clean.joblib")
MODEL_METADATA_PATH = Path("models/xgboost_price_model_features.json")


@lru_cache(maxsize=1)
def load_trained_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the saved regression model from disk."""
    resolved_model_path = resolve_project_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Trained model file not found at {resolved_model_path}")

    return joblib.load(resolved_model_path)


@lru_cache(maxsize=1)
def load_model_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Load the saved model metadata."""
    resolved_metadata_path = resolve_project_path(metadata_path)
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata file not found at {resolved_metadata_path}")

    return json.loads(resolved_metadata_path.read_text(encoding="utf-8"))


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


def predict_price_direct(input_data: dict[str, float]) -> float:
    """Run a direct in-process price prediction using the saved local artifacts."""
    model = load_trained_model()
    feature_columns = load_feature_columns()
    inference_df = prepare_features_for_inference(input_data=input_data, feature_columns=feature_columns)
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except ValueError:
            pass
    prediction = model.predict(inference_df)[0]
    return float(prediction)


def predict_price(input_data: dict[str, float]) -> float:
    """Run prediction in a subprocess to isolate native-model crashes."""
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.predict_cli",
            json.dumps(input_data),
        ],
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    if process.returncode != 0:
        stderr = process.stderr.strip() or "No stderr output captured."
        stdout = process.stdout.strip() or "No stdout output captured."
        raise RuntimeError(
            "Prediction subprocess failed. "
            f"returncode={process.returncode}. stderr={stderr} stdout={stdout}"
        )

    try:
        response = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Prediction subprocess returned invalid JSON: {process.stdout}") from exc

    return float(response["predicted_price"])
