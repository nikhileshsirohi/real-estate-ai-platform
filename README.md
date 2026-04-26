# AI-Powered Real Estate Price Intelligence & Advisory Platform

## Overview
Production-style AI/ML project for real estate price prediction and advisory.

## Current Scope
- Data ingestion from a real housing dataset
- Data cleaning
- Feature engineering
- Configurable model training
- MLflow experiment tracking
- Local model artifact saving
- FastAPI prediction endpoint

## Tech Stack
- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- MLflow
- PostgreSQL later

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline
```bash
python -m src.data.ingestion
python -m src.data.cleaning
python -m src.features.feature_engineering
```

## Training
Choose the model in [configs/model_config.yaml](/Volumes/NIKHILESH/Projects/real-estate-ai-advisor/real-estate-ai-platform/configs/model_config.yaml).

Supported values:
- `linear_regression`
- `random_forest`

Run training:

```bash
python -m src.training.train_model
```

Saved artifacts:
- `models/trained_model.joblib`
- `models/model_metadata.json`

## API
Start the API:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict-price" \
  -H "Content-Type: application/json" \
  -d '{
    "median_income": 8.3252,
    "house_age": 41.0,
    "average_rooms": 6.984127,
    "average_bedrooms": 1.02381,
    "population": 322.0,
    "average_occupancy": 2.555556,
    "latitude": 37.88,
    "longitude": -122.23
  }'
```

## Next Improvement Ideas
- Compare `linear_regression` vs `random_forest` in MLflow
- Add XGBoost after the tree baseline is stable
- Store prediction history in PostgreSQL
- Package the app with Docker
