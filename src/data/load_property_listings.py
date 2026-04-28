"""Load sample property listings into PostgreSQL."""

import csv
from pathlib import Path

from src.db.repository import upsert_property_listing
from src.db.session import SessionLocal, create_db_tables
from src.utils.config_loader import resolve_project_path


LISTINGS_CSV_PATH = Path("data/sample/property_listings_seed.csv")


def main() -> None:
    """Upsert the sample property listings into the database."""
    # Keep notebook/local loading smooth even if the migration wasn't run yet.
    create_db_tables()

    csv_path = resolve_project_path(LISTINGS_CSV_PATH)
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    db = SessionLocal()
    try:
        for row in rows:
            upsert_property_listing(
                db=db,
                listing_code=row["listing_code"],
                title=row["title"],
                city=row["city"],
                locality=row["locality"],
                property_type=row["property_type"],
                bedrooms=int(row["bedrooms"]),
                bathrooms=float(row["bathrooms"]),
                area_sqft=float(row["area_sqft"]),
                asking_price_usd=float(row["asking_price_usd"]),
                description=row["description"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
    finally:
        db.close()

    print(f"Loaded {len(rows)} property listings from {csv_path}")


if __name__ == "__main__":
    main()
