# Agri Price Forecast

> AI/ML-powered price forecasting for agricultural commodities — beating ARIMA baselines using deep learning, weather data, and satellite crop indices.

Built as part of the Smart India Hackathon 2024 problem statement on AI-ML based models for predicting prices of agri-horticultural commodities. Targets the Department of Consumer Affairs' mandate to stabilise essential food commodity prices through data-driven market interventions.

---

## What this does

- Scrapes daily wholesale prices from Agmarknet (Government of India)
- Fuses price data with weather (Open-Meteo) and crop health indices (NDVI via Google Earth Engine)
- Trains and compares multiple forecasting models: ARIMA baseline → Prophet → LSTM/GRU → Temporal Fusion Transformer → ensemble
- Exposes a FastAPI REST API for predictions
- Visualises forecasts and model comparisons in a Streamlit dashboard

**Current scope:** Onion prices across major markets (Lasalgaon, Pimpalgaon, Delhi, etc.)
**Extending to:** Potato, Tomato, and 5 pulses (Tur, Gram, Urad, Moong, Masur)

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Monisha1313/agri-price-forecast.git
cd agri-price-forecast

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env if needed (defaults work for local SQLite dev)

# 5. Initialise database
python -c "from src.data.database import init_db; init_db()"

# 6. Fetch onion price data (last 30 days)
python -m src.data.scraper_agmarknet --commodity onion --days 30

# 7. Run historical backfill (3 years — takes ~10 min, run once)
python -m src.data.scraper_agmarknet --commodity onion --history

# 8. Launch dashboard
streamlit run dashboard/app.py

# 9. Launch API (separate terminal)
uvicorn src.api.main:app --reload
```

---

## Project structure

```
agri-price-forecast/
├── src/
│   ├── data/          # Scrapers, database layer, scheduler
│   ├── features/      # Feature engineering pipeline
│   ├── models/        # ARIMA, LSTM, TFT, XGBoost, ensemble
│   ├── api/           # FastAPI endpoints
│   └── utils/         # Config, logging
├── dashboard/         # Streamlit app
├── notebooks/         # EDA and model experiments
├── tests/             # Unit and integration tests
└── docs/              # Architecture docs, paper notes
```

---

## Models and results

| Model | RMSE (₹/q) | MAE (₹/q) | MAPE (%) |
|-------|-----------|-----------|----------|
| ARIMA (baseline) | 285.561 | 117.211 | 5.543 |
| Prophet | — | — | — |
| LSTM | 337.817 | 252.722 | 14.602 |
| Temporal Fusion Transformer | — | — | — |
| Ensemble | — | — | — |

*Results will be updated as training completes.*

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Data | pandas, SQLAlchemy, SQLite/PostgreSQL |
| Scraping | requests, BeautifulSoup4 |
| ML | PyTorch, pytorch-forecasting, XGBoost, Prophet, statsmodels |
| Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly, Folium |
| Explainability | SHAP |

---

## Paper

This project accompanies an IEEE paper submission:

> *Multi-modal price forecasting for essential food commodities using Temporal Fusion Transformers, weather data, and satellite crop indices*

Draft in `docs/paper_notes.md`.

---

## Data sources

- **Agmarknet** — Daily mandi prices (government, free, public domain)
- **Open-Meteo** — Weather data (free, no API key required)
- **Google Earth Engine** — NDVI crop health via Sentinel-2 (free academic account)

---

## License

MIT
