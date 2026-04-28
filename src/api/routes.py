"""API route definitions."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import (
    MarketAdviceResponse,
    MarketQuestionRequest,
    PropertyAdviceRequest,
    PropertyAdviceResponse,
    PredictionDetailResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)
from src.db.repository import (
    filter_prediction_records,
    get_prediction_record_by_id,
    list_recent_prediction_records,
    save_prediction_record,
)
from src.db.session import get_db
from src.inference.predictor import load_model_name, predict_price
from src.rag.service import ask_market_question, ask_property_question
from src.utils.logger import get_logger, log_event


router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
def root() -> dict[str, str]:
    """Friendly root endpoint for quick browser checks."""
    return {
        "message": "Real Estate AI Platform API is running",
        "docs": "/docs",
        "health": "/health",
        "predictions": "/predictions?limit=10",
    }


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for quick service checks."""
    return {"status": "ok"}


@router.get("/predictions", response_model=PredictionHistoryResponse)
def list_predictions_route(
    limit: int = 10,
    model_name: str | None = None,
    min_predicted_price: float | None = None,
    max_predicted_price: float | None = None,
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Return recent prediction history from PostgreSQL."""
    try:
        if model_name or min_predicted_price is not None or max_predicted_price is not None:
            records = filter_prediction_records(
                db=db,
                limit=limit,
                model_name=model_name,
                min_predicted_price=min_predicted_price,
                max_predicted_price=max_predicted_price,
            )
        else:
            records = list_recent_prediction_records(db=db, limit=limit)
    except SQLAlchemyError as exc:
        log_event(
            logger,
            logging.ERROR,
            "prediction_history_failed",
            limit=limit,
            model_name=model_name,
            min_predicted_price=min_predicted_price,
            max_predicted_price=max_predicted_price,
            error=str(exc),
        )
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
    log_event(
        logger,
        logging.INFO,
        "prediction_history_fetched",
        limit=limit,
        count=len(items),
        model_name=model_name,
        min_predicted_price=min_predicted_price,
        max_predicted_price=max_predicted_price,
    )
    return PredictionHistoryResponse(items=items, count=len(items))


@router.get("/predictions/{prediction_id}", response_model=PredictionDetailResponse)
def get_prediction_detail_route(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> PredictionDetailResponse:
    """Return a single stored prediction by id."""
    try:
        record = get_prediction_record_by_id(db=db, prediction_id=prediction_id)
    except SQLAlchemyError as exc:
        log_event(logger, logging.ERROR, "prediction_detail_failed", prediction_id=prediction_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Database read failed: {exc}")

    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction with id {prediction_id} not found")

    log_event(logger, logging.INFO, "prediction_detail_fetched", prediction_id=prediction_id)
    return PredictionDetailResponse(
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
        log_event(logger, logging.ERROR, "prediction_failed_missing_file", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    except SQLAlchemyError as exc:
        log_event(logger, logging.ERROR, "prediction_failed_database", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Database logging failed: {exc}")
    except Exception as exc:
        log_event(logger, logging.ERROR, "prediction_failed_runtime", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    log_event(
        logger,
        logging.INFO,
        "prediction_created",
        prediction_id=saved_record.id,
        model_name=model_name,
        predicted_price=prediction,
    )

    return PricePredictionResponse(
        predicted_price=prediction,
        model_name=model_name,
        prediction_id=saved_record.id,
    )


@router.post("/ask-market", response_model=MarketAdviceResponse)
def ask_market_route(payload: MarketQuestionRequest) -> MarketAdviceResponse:
    """Answer a market question using local retrieval plus Ollama."""
    try:
        result = ask_market_question(payload.question)
    except Exception as exc:
        log_event(logger, logging.ERROR, "market_advice_failed", question=payload.question, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Market advice failed: {exc}")

    log_event(
        logger,
        logging.INFO,
        "market_advice_created",
        question=payload.question,
        model_name=result["model_name"],
        source_count=len(result["sources"]),
    )
    return MarketAdviceResponse(**result)


@router.post("/advise-property", response_model=PropertyAdviceResponse)
def advise_property_route(payload: PropertyAdviceRequest) -> PropertyAdviceResponse:
    """Generate property-level advisory output using prediction plus RAG."""
    property_features = {
        "median_income": payload.median_income,
        "house_age": payload.house_age,
        "average_rooms": payload.average_rooms,
        "average_bedrooms": payload.average_bedrooms,
        "population": payload.population,
        "average_occupancy": payload.average_occupancy,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
    }
    log_event(logger, logging.INFO, "property_advice_started", question=payload.question)
    try:
        result = ask_property_question(payload.question, property_features)
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "property_advice_failed",
            question=payload.question,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Property advice failed: {exc}")

    log_event(
        logger,
        logging.INFO,
        "property_advice_created",
        question=payload.question,
        model_name=result["model_name"],
        predicted_price=result["predicted_price"],
        source_count=len(result["sources"]),
    )
    return PropertyAdviceResponse(**result)
