# AI-Powered Real Estate Price Intelligence & Advisory Platform

## Overview
Production-style AI/ML project for real estate price prediction and advisory.

## Current Scope
- Data ingestion from a real housing dataset
- Data cleaning
- feature engineering with log/capped transforms
- tuned XGBoost model training
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

## PostgreSQL 14 Local Setup
If you are using Homebrew PostgreSQL 14 on macOS:

```bash
brew services start postgresql@14
createdb real_estate_db
```

Create a local `.env` file from [.env.example](/Volumes/NIKHILESH/Projects/real-estate-ai-advisor/real-estate-ai-platform/.env.example) and set your macOS username in `DATABASE_URL`.

Example:

```env
DATABASE_URL=postgresql+psycopg2://YOUR_MAC_USERNAME@localhost:5432/real_estate_db
```

## Data Pipeline
```bash
python -m src.data.ingestion
python -m src.data.cleaning
python -m src.features.feature_engineering
```

Current feature engineering adds:
- `bedroom_ratio`
- `rooms_per_person`
- log-transformed versions of skewed columns
- capped versions of skewed columns using the 99th percentile from the training dataset
- saved feature-engineering metadata for consistent inference

## Training
Choose the model in [configs/model_config.yaml](/Volumes/NIKHILESH/Projects/real-estate-ai-advisor/real-estate-ai-platform/configs/model_config.yaml).

Supported values:
- `linear_regression`
- `random_forest`
- `xgboost`

Current best tuned XGBoost params:

```json
{
  "subsample": 1.0,
  "reg_lambda": 3,
  "reg_alpha": 1,
  "n_estimators": 900,
  "min_child_weight": 7,
  "max_depth": 5,
  "learning_rate": 0.07,
  "colsample_bytree": 0.7
}
```

Run training:

```bash
python -m src.training.train_model
```

Saved artifacts:
- `models/xgboost_price_model_tuned_clean.joblib`
- `models/xgboost_price_model_features.json`
- `models/xgboost_price_model_metrics.json`

## API
Start the API:

```bash
uvicorn src.api.main:app --reload
```

Make sure PostgreSQL is running and `DATABASE_URL` is set in your environment or `.env`.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Root API summary:

```bash
curl http://127.0.0.1:8000/
```

Prediction history:

```bash
curl "http://127.0.0.1:8000/predictions?limit=10"
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

Successful responses now include a saved `prediction_id` from PostgreSQL.

## Prediction Logging
Each `/predict-price` request is stored in PostgreSQL with:
- input property features
- predicted price
- model name
- creation timestamp

Table name:
- `prediction_records`

History endpoint:
- `GET /predictions?limit=10`
- `GET /predictions/{id}`
- `GET /predictions?limit=10&model_name=xgboost`
- `GET /predictions?limit=10&min_predicted_price=3.0&max_predicted_price=5.0`

## Observability
The API now emits structured JSON logs for:
- application startup
- every HTTP request
- prediction creation events
- prediction history fetches
- prediction/database failures

Typical logged fields include:
- `method`
- `path`
- `status_code`
- `duration_ms`
- `prediction_id`
- `model_name`

## Next Improvement Ideas
- Add a dedicated PostgreSQL prediction log
- Add geospatial features or neighborhood clustering
- Try stricter clipping or additional geo features if XGBoost plateaus
- Package the app with Docker

## Docker
Run the full API + PostgreSQL stack with Docker:

```bash
docker compose up --build
```

This starts:
- `api` on `http://127.0.0.1:8000`
- `db` on `localhost:5432`

The Docker image disables Uvicorn's default access log so the structured JSON application logs are easier to read.

Inside Docker, the API uses:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/real_estate_db
```

Useful commands:

```bash
docker compose up --build
docker compose down
docker compose down -v
```

Use `docker compose down -v` only if you want to remove the Postgres volume and reset stored prediction history.

## Alembic Migrations
Use Alembic for production-style schema management.

Create or upgrade the database schema:

```bash
alembic upgrade head
```

Create a new migration later:

```bash
alembic revision -m "describe_change"
```

For local convenience, the API can still auto-create tables on startup. To disable that and rely only on migrations:

```env
AUTO_CREATE_TABLES=false
```

## CI/CD
A minimal GitHub Actions workflow is included at:
- `.github/workflows/ci.yml`

It currently:
- installs dependencies
- runs `pytest`
- builds the Docker image

## Notebooks
Keep notebook work separated by purpose:
- `notebooks/01_eda.ipynb`: data understanding and skew/outlier analysis
- `notebooks/02_model_comparison.ipynb`: baseline model comparison
- `notebooks/03_tuned_xgboost.ipynb`: XGBoost tuning and best-model export

## RAG Starter
The project now includes a starter local knowledge corpus built from official California sources so you can begin RAG work without a private dataset.

Starter source documents live in:
- `data/knowledge/raw`

Initial official sources used:
- U.S. Census QuickFacts California
- California Department of Finance population and housing estimates
- California Department of Finance population projections

Build the local vector index:

```bash
python -m src.rag.build_index
```

Run a simple retrieval test:

```bash
python -m src.rag.retrieve
```

Core RAG files:
- `configs/rag_config.yaml`
- `src/rag/document_loader.py`
- `src/rag/chunking.py`
- `src/rag/embeddings.py`
- `src/rag/build_index.py`
- `src/rag/retrieve.py`
