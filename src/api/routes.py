"""API route definitions."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)
from src.db.repository import list_recent_prediction_records, save_prediction_record
from src.db.session import get_db
from src.inference.predictor import load_model_name, predict_price


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for quick service checks."""
    return {"status": "ok"}


@router.get("/predictions", response_model=PredictionHistoryResponse)
def list_predictions_route(
    limit: int = 10,
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Return recent prediction history from PostgreSQL."""
    try:
        records = list_recent_prediction_records(db=db, limit=limit)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail=f"Database read failed: {exc}")

    items = [
        PredictionHistoryItem(
            id=record.id,
            model_name=record.model_name,
            predicted_price=record.predicted_price,
            median_income=record.median_income,
            house_age=record.house_age,
            average_rooms=record.average_rooms,
            average_bedrooms=record.average_bedrooms,
            population=record.population,
            average_occupancy=record.average_occupancy,
            latitude=record.latitude,
            longitude=record.longitude,
            created_at=record.created_at,
        )
        for record in records
    ]
    return PredictionHistoryResponse(items=items, count=len(items))


@router.post("/predict-price", response_model=PricePredictionResponse)
def predict_price_route(
    payload: PricePredictionRequest,
    db: Session = Depends(get_db),
) -> PricePredictionResponse:
    """Predict a median house value from request features."""
    try:
        prediction = predict_price(payload.model_dump())
        model_name = load_model_name()
        saved_record = save_prediction_record(
            db=db,
            payload=payload,
            predicted_price=prediction,
            model_name=model_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail=f"Database logging failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PricePredictionResponse(
        predicted_price=prediction,
        model_name=model_name,
        prediction_id=saved_record.id,
    )
