"""Tests for the baseline training step."""

import pandas as pd

from src.training.evaluate import evaluate_regression_model
from src.training.train_model import train_baseline_model


def test_train_baseline_model_returns_fitted_model() -> None:
    X_train = pd.DataFrame(
        {
            "median_income": [2.0, 3.0, 4.0, 5.0],
            "house_age": [10.0, 15.0, 20.0, 25.0],
            "average_rooms": [4.0, 5.0, 6.0, 7.0],
        }
    )
    y_train = pd.Series([1.2, 1.8, 2.4, 3.0])

    model = train_baseline_model(X_train, y_train)
    predictions = model.predict(X_train)
    metrics = evaluate_regression_model(y_train, predictions)

    assert len(predictions) == len(y_train)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
