"""FastAPI application entry point."""

from fastapi import FastAPI

from src.api.routes import router
from src.db.session import create_db_tables


app = FastAPI(
    title="Real Estate AI Platform",
    version="0.1.0",
    description="V1 API for real estate price prediction.",
)


@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on application startup."""
    create_db_tables()


app.include_router(router)
