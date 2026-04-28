"""Repository helpers for prediction persistence and property search."""

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from src.api.schemas import PricePredictionRequest, PropertySearchFilters
from src.db.models import PredictionRecord, PropertyListing


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


def get_prediction_record_by_id(db: Session, prediction_id: int) -> PredictionRecord | None:
    """Fetch a single prediction record by its identifier."""
    stmt = select(PredictionRecord).where(PredictionRecord.id == prediction_id)
    return db.scalar(stmt)


def filter_prediction_records(
    db: Session,
    limit: int = 20,
    model_name: str | None = None,
    min_predicted_price: float | None = None,
    max_predicted_price: float | None = None,
) -> list[PredictionRecord]:
    """Fetch prediction records using simple filters."""
    stmt = select(PredictionRecord)

    if model_name:
        stmt = stmt.where(PredictionRecord.model_name == model_name)
    if min_predicted_price is not None:
        stmt = stmt.where(PredictionRecord.predicted_price >= min_predicted_price)
    if max_predicted_price is not None:
        stmt = stmt.where(PredictionRecord.predicted_price <= max_predicted_price)

    stmt = stmt.order_by(PredictionRecord.id.desc()).limit(limit)
    return list(db.scalars(stmt).all())


def upsert_property_listing(
    db: Session,
    *,
    listing_code: str,
    title: str,
    city: str,
    locality: str,
    property_type: str,
    bedrooms: int,
    bathrooms: float,
    area_sqft: float,
    asking_price_usd: float,
    description: str,
    latitude: float,
    longitude: float,
) -> PropertyListing:
    """Insert or update a property listing using listing_code as the stable key."""
    stmt = select(PropertyListing).where(PropertyListing.listing_code == listing_code)
    record = db.scalar(stmt)

    if record is None:
        record = PropertyListing(
            listing_code=listing_code,
            title=title,
            city=city,
            locality=locality,
            property_type=property_type,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            area_sqft=area_sqft,
            asking_price_usd=asking_price_usd,
            description=description,
            latitude=latitude,
            longitude=longitude,
        )
        db.add(record)
    else:
        record.title = title
        record.city = city
        record.locality = locality
        record.property_type = property_type
        record.bedrooms = bedrooms
        record.bathrooms = bathrooms
        record.area_sqft = area_sqft
        record.asking_price_usd = asking_price_usd
        record.description = description
        record.latitude = latitude
        record.longitude = longitude

    db.commit()
    db.refresh(record)
    return record


def _apply_property_filters(stmt: Select, filters: PropertySearchFilters) -> Select:
    """Apply structured property filters to a SQLAlchemy select."""
    if filters.city:
        stmt = stmt.where(func.lower(PropertyListing.city) == filters.city.lower())
    if filters.locality:
        stmt = stmt.where(func.lower(PropertyListing.locality).contains(filters.locality.lower()))
    if filters.property_type:
        stmt = stmt.where(func.lower(PropertyListing.property_type) == filters.property_type.lower())
    if filters.min_price_usd is not None:
        stmt = stmt.where(PropertyListing.asking_price_usd >= filters.min_price_usd)
    if filters.max_price_usd is not None:
        stmt = stmt.where(PropertyListing.asking_price_usd <= filters.max_price_usd)
    if filters.min_bedrooms is not None:
        stmt = stmt.where(PropertyListing.bedrooms >= filters.min_bedrooms)
    if filters.max_bedrooms is not None:
        stmt = stmt.where(PropertyListing.bedrooms <= filters.max_bedrooms)
    if filters.min_bathrooms is not None:
        stmt = stmt.where(PropertyListing.bathrooms >= filters.min_bathrooms)
    if filters.max_bathrooms is not None:
        stmt = stmt.where(PropertyListing.bathrooms <= filters.max_bathrooms)
    if filters.min_area_sqft is not None:
        stmt = stmt.where(PropertyListing.area_sqft >= filters.min_area_sqft)
    if filters.max_area_sqft is not None:
        stmt = stmt.where(PropertyListing.area_sqft <= filters.max_area_sqft)
    return stmt


def search_property_listings(
    db: Session,
    filters: PropertySearchFilters,
) -> list[PropertyListing]:
    """Search property listings using structured filters and sorting."""
    stmt = _apply_property_filters(select(PropertyListing), filters)

    sort_column = PropertyListing.asking_price_usd
    if filters.sort_by == "area_sqft":
        sort_column = PropertyListing.area_sqft
    elif filters.sort_by == "bedrooms":
        sort_column = PropertyListing.bedrooms
    elif filters.sort_by == "created_at":
        sort_column = PropertyListing.created_at

    if filters.sort_order.lower() == "desc":
        stmt = stmt.order_by(sort_column.desc())
    else:
        stmt = stmt.order_by(sort_column.asc())

    stmt = stmt.limit(filters.limit)
    return list(db.scalars(stmt).all())


def search_property_listings_with_fallback(
    db: Session,
    filters: PropertySearchFilters,
) -> tuple[list[PropertyListing], str, str | None]:
    """Search exactly first, then relax the budget cap to return closest matches if needed."""
    exact_matches = search_property_listings(db=db, filters=filters)
    if exact_matches:
        return exact_matches, "exact", None

    if filters.max_price_usd is None:
        return [], "exact", "No listings matched the current non-price filters."

    relaxed_filters = filters.model_copy(deep=True)
    budget_cap = relaxed_filters.max_price_usd
    relaxed_filters.max_price_usd = None

    stmt = _apply_property_filters(select(PropertyListing), relaxed_filters)
    stmt = stmt.order_by(func.abs(PropertyListing.asking_price_usd - budget_cap).asc()).limit(filters.limit)
    fallback_matches = list(db.scalars(stmt).all())

    if not fallback_matches:
        return [], "exact", "No listings matched even after relaxing the budget cap."

    cheapest_match = min(listing.asking_price_usd for listing in fallback_matches)
    advisory_note = (
        "No exact matches were found under the requested budget. "
        f"Showing closest matches instead. Cheapest nearby match is ${cheapest_match:,.0f} "
        f"versus requested max ${budget_cap:,.0f}."
    )
    return fallback_matches, "closest_match", advisory_note
