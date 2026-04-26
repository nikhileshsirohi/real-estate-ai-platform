"""FastAPI application entry point."""

import time

from fastapi import FastAPI
from fastapi import Request

from src.api.routes import router
from src.db.session import AUTO_CREATE_TABLES, create_db_tables
from src.utils.logger import get_logger, log_event, setup_logging


setup_logging()
logger = get_logger(__name__)


app = FastAPI(
    title="Real Estate AI Platform",
    version="0.1.0",
    description="V1 API for real estate price prediction.",
)


@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on application startup."""
    if AUTO_CREATE_TABLES:
        create_db_tables()
    log_event(logger, 20, "application_startup", event="startup", auto_create_tables=AUTO_CREATE_TABLES)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request and response details with timing."""
    start_time = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_event(
            logger,
            20,
            "http_request",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            status_code=response.status_code if response is not None else 500,
            duration_ms=duration_ms,
        )


app.include_router(router)
