"""Request and response schemas for the prediction API."""

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
