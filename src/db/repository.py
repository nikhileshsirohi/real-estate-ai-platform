"""Repository helpers for prediction persistence."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.schemas import PricePredictionRequest
from src.db.models import PredictionRecord


def save_prediction_record(
    db: Session,
    payload: PricePredictionRequest,
    predicted_price: float,
    model_name: str,
) -> PredictionRecord:
    """Persist a prediction request and response to PostgreSQL."""
    record = PredictionRecord(
        model_name=model_name,
        predicted_price=predicted_price,
        median_income=payload.median_income,
        house_age=payload.house_age,
        average_rooms=payload.average_rooms,
        average_bedrooms=payload.average_bedrooms,
        population=payload.population,
        average_occupancy=payload.average_occupancy,
        latitude=payload.latitude,
        longitude=payload.longitude,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_recent_prediction_records(db: Session, limit: int = 20) -> list[PredictionRecord]:
    """Fetch recent prediction records ordered from newest to oldest."""
    stmt = (
        select(PredictionRecord)
        .order_by(PredictionRecord.id.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())
