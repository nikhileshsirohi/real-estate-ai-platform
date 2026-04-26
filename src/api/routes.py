"""API route definitions."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import PricePredictionRequest, PricePredictionResponse
from src.inference.predictor import load_model_name, predict_price


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for quick service checks."""
    return {"status": "ok"}


@router.post("/predict-price", response_model=PricePredictionResponse)
def predict_price_route(payload: PricePredictionRequest) -> PricePredictionResponse:
    """Predict a median house value from request features."""
    try:
        prediction = predict_price(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    
    return PricePredictionResponse(
        predicted_price=prediction,
        model_name=load_model_name(),
    )
