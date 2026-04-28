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


class MarketQuestionRequest(BaseModel):
    question: str = Field(..., min_length=5)


class MarketAdviceSource(BaseModel):
    chunk_id: str
    source_path: str
    title: str
    content: str
    score: float


class MarketAdviceResponse(BaseModel):
    answer: str
    model_name: str
    sources: list[MarketAdviceSource]


class PropertyAdviceRequest(PricePredictionRequest):
    question: str = Field(..., min_length=5)


class PropertyAdviceResponse(BaseModel):
    answer: str
    model_name: str
    predicted_price: float
    predicted_price_usd: float
    sources: list[MarketAdviceSource]


class PropertySearchFilters(BaseModel):
    city: str | None = None
    locality: str | None = None
    property_type: str | None = None
    min_price_usd: float | None = Field(default=None, ge=0)
    max_price_usd: float | None = Field(default=None, ge=0)
    min_bedrooms: int | None = Field(default=None, ge=0)
    max_bedrooms: int | None = Field(default=None, ge=0)
    min_bathrooms: float | None = Field(default=None, ge=0)
    max_bathrooms: float | None = Field(default=None, ge=0)
    min_area_sqft: float | None = Field(default=None, ge=0)
    max_area_sqft: float | None = Field(default=None, ge=0)
    limit: int = Field(default=10, ge=1, le=50)
    sort_by: str = Field(default="asking_price_usd")
    sort_order: str = Field(default="asc")


class PropertyListingItem(BaseModel):
    id: int
    listing_code: str
    title: str
    city: str
    locality: str
    property_type: str
    bedrooms: int
    bathrooms: float
    area_sqft: float
    asking_price_usd: float
    description: str
    latitude: float
    longitude: float
    created_at: datetime


class PropertySearchResponse(BaseModel):
    items: list[PropertyListingItem]
    count: int
    applied_filters: PropertySearchFilters


class PropertySearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=4)
    limit: int = Field(default=10, ge=1, le=50)


class PropertySearchQueryResponse(PropertySearchResponse):
    parser_model_name: str
