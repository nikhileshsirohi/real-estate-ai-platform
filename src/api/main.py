"""FastAPI application entry point."""

from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Real Estate AI Platform",
    version="0.1.0",
    description="V1 API for real estate price prediction.",
)

app.include_router(router)
