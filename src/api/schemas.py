"""Request and response schemas for the prediction API."""

from datetime import datetime

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    median_income: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0)
    average_rooms: float = Field(..., gt=0)
    average_bedrooms: float = Field(..., ge=0)
    population: float = Field(..., ge=0)
    average_occupancy: float = Field(..., gt=0)
    latitude: float
    longitude: float


class PricePredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    prediction_id: int | None = None


class PredictionHistoryItem(BaseModel):
    id: int
    model_name: str
    predicted_price: float
    median_income: float
    house_age: float
    average_rooms: float
    average_bedrooms: float
    population: float
    average_occupancy: float
    latitude: float
    longitude: float
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: list[PredictionHistoryItem]
    count: int


class PredictionDetailResponse(PredictionHistoryItem):
    """Detailed response for a single stored prediction."""
