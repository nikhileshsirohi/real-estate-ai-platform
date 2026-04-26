"""Database models for persisted prediction records."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class PredictionRecord(Base):
    """Stored record for each prediction request."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    median_income: Mapped[float] = mapped_column(Float, nullable=False)
    house_age: Mapped[float] = mapped_column(Float, nullable=False)
    average_rooms: Mapped[float] = mapped_column(Float, nullable=False)
    average_bedrooms: Mapped[float] = mapped_column(Float, nullable=False)
    population: Mapped[float] = mapped_column(Float, nullable=False)
    average_occupancy: Mapped[float] = mapped_column(Float, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
