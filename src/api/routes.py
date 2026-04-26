"""API route definitions."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import PricePredictionRequest, PricePredictionResponse
from src.db.repository import save_prediction_record
from src.db.session import get_db
from src.inference.predictor import load_model_name, predict_price


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for quick service checks."""
    return {"status": "ok"}


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
