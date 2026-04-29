# AI-Powered Real Estate Price Intelligence & Advisory Platform

Production-style AI/ML + RAG project for real estate price prediction, property search, and market/property advisory.

## What This Project Does
- Predicts property prices with a tuned XGBoost model
- Stores prediction history in PostgreSQL
- Answers market questions using RAG + Ollama
- Explains property prices using prediction + retrieved context
- Searches seeded property listings with structured filters or natural language
- Recommends closest matches when exact search results are unavailable
- Shows monitoring and evaluation snapshots in a simple frontend

## Tech Stack
- Python 3.10+
- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Pandas / NumPy / Scikit-learn / XGBoost
- MLflow
- Ollama
- FAISS
- HTML / CSS / JavaScript frontend
- Docker

## Current Best Model
- Model: tuned XGBoost regressor
- RMSE: `0.4370`
- MAE: `0.2847`
- R²: `0.8543`

Model artifacts:
- `models/xgboost_price_model_tuned_clean.joblib`
- `models/xgboost_price_model_features.json`
- `models/xgboost_price_model_metrics.json`

## Project Structure
```text
real-estate-ai-platform/
├── frontend/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── knowledge/
├── models/
├── notebooks/
├── src/
│   ├── api/
│   ├── data/
│   ├── db/
│   ├── features/
│   ├── inference/
│   ├── monitoring/
│   ├── rag/
│   ├── search/
│   ├── training/
│   └── utils/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup
From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## PostgreSQL Local Setup
For Homebrew PostgreSQL 14 on macOS:

```bash
brew services start postgresql@14
createdb real_estate_db
```

Create a local `.env` file from [.env.example]

Example:

```env
APP_ENV=development
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
DATABASE_URL=postgresql+psycopg2://YOUR_USERNAME_PASSWORD@localhost:5432/real_estate_db
AUTO_CREATE_TABLES=true
```

## Data Pipeline
Run the end-to-end tabular pipeline:

```bash
python -m src.data.ingestion
python -m src.data.cleaning
python -m src.features.feature_engineering
python -m src.training.train_model
```

Feature engineering currently includes:
- `bedroom_ratio`
- `rooms_per_person`
- log-transformed skewed features
- capped feature variants
- saved feature metadata for consistent inference

## MLflow
Start MLflow UI:

```bash
mlflow ui
```

Then train the model:

```bash
python -m src.training.train_model
```

## Database Migrations
Create or upgrade schema:

```bash
alembic upgrade head
```

## Load Seed Property Listings
The project includes demo property inventory for local search and recommendation.

```bash
python -m src.data.load_property_listings
```

The seeded inventory covers multiple cities, including:
- Oakland
- San Francisco
- San Jose
- Sacramento
- San Diego

## Build the RAG Index
```bash
python -m src.rag.build_index
```

This builds:
- `data/knowledge/index/knowledge.index`
- `data/knowledge/index/chunks.json`
- `data/knowledge/index/index_metadata.json`

## Run the API
Always run from the project root:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Root summary:

```bash
curl http://127.0.0.1:8000/
```

## Frontend
Open the interactive app:

```bash
open http://127.0.0.1:8000/app
```

The frontend supports:
- Predict Price
- Ask Market
- Advise Property
- Search Inventory
- Recommend Listings
- Monitoring Snapshot
- Evaluation Snapshot

## API Endpoints

### Prediction
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

Prediction history:

```bash
curl "http://127.0.0.1:8000/predictions?limit=10"
curl "http://127.0.0.1:8000/predictions/1"
```

### Ask Market
```bash
curl -X POST "http://127.0.0.1:8000/ask-market" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which demo city looks more affordable for a lower-entry buyer, Sacramento or San Francisco?"
  }'
```

### Advise Property
```bash
curl -X POST "http://127.0.0.1:8000/advise-property" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How should I interpret this predicted price for a buyer who wants Oakland city access?",
    "median_income": 8.3252,
    "house_age": 41.0,
    "average_rooms": 6.984127,
    "average_bedrooms": 1.02381,
    "population": 322.0,
    "average_occupancy": 2.555556,
    "latitude": 37.809,
    "longitude": -122.257
  }'
```

### Structured Property Search
```bash
curl "http://127.0.0.1:8000/search-properties?city=Oakland&max_price_usd=900000&limit=5"
```

### Natural-Language Property Search
```bash
curl -X POST "http://127.0.0.1:8000/search-properties/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me a 2 bedroom condo in Oakland under 800000",
    "limit": 5
  }'
```

### Listing Recommendation
```bash
curl -X POST "http://127.0.0.1:8000/search-properties/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me a 2 BHK in Oakland under 80 lakh",
    "limit": 5
  }'
```

## How Search Works
- User writes a natural-language search query
- Query is normalized:
  - `2 BHK` -> `2 bedroom`
  - `sq ft` -> `sqft`
  - `lakh` / `crore` -> approximate USD
- Ollama parses the query into structured filters
- PostgreSQL runs the actual search
- If exact matches do not exist, fallback logic returns closest matches

This keeps search deterministic because the database, not the LLM, is the source of truth.

## Monitoring Snapshot
Endpoint:

```bash
curl http://127.0.0.1:8000/monitoring/summary
```

It shows:
- total API requests
- average response time
- prediction record count
- property listing count
- saved model RMSE
- RAG document count

How `total cities` is computed in evaluation:
- the backend groups the `property_listings` table by `city`
- builds one row per city
- returns `len(cities)` as `total_cities`

This logic lives in [src/monitoring/service.py](/Volumes/NIKHILESH/Projects/real-estate-ai-advisor/real-estate-ai-platform/src/monitoring/service.py), inside `build_inventory_evaluation_summary()`.

## Evaluation Snapshot
Endpoint:

```bash
curl http://127.0.0.1:8000/evaluation/summary
```

It shows:
- model MAE
- model R²
- number of cities covered by demo inventory
- number of RAG chunks available

This gives a quick “how trustworthy is the system right now?” view.

## Logging and Observability
The API emits structured JSON logs for:
- startup
- every HTTP request
- prediction creation
- prediction history fetches
- failures

Typical fields include:
- `method`
- `path`
- `status_code`
- `duration_ms`
- `prediction_id`
- `model_name`

## Docker
Run API + PostgreSQL:

```bash
docker compose up --build
```

Services:
- API: `http://127.0.0.1:8000`
- Postgres: `localhost:5432`

Useful commands:

```bash
docker compose down
docker compose down -v
```

## Testing
Run tests:

```bash
python -m pytest
```

Or just API tests:

```bash
python -m pytest tests/test_api.py
```

## Current Status
This project includes:
- ML training and evaluation
- experiment tracking
- inference API
- PostgreSQL persistence
- natural-language property search
- RAG + Ollama advisory flows
- monitoring/evaluation views
- frontend demo
- Docker support

## Next Improvements
- Replace seed listings with a more realistic external/open dataset
- Expand local area knowledge for stronger property advice
- Add richer monitoring dashboards or persistence
- Add stronger evaluation reports for search/RAG quality
- Polish deployment and production setup
