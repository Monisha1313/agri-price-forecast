# 🧅 Agri Price Forecast

> AI/ML-powered forecasting system for agricultural commodity prices — beating ARIMA baselines using deep learning, weather data, and 20+ years of mandi price history.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-green)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🏆 Key Results

| Model | RMSE (₹/q) | MAE (₹/q) | MAPE (%) | R² |
|-------|-----------|-----------|----------|----|
| **LSTM (Proposed)** | 457 | 217 | 9.3% | **0.750** |
| **Ensemble** | 481 | 225 | 9.5% | 0.719 |
| SARIMA (Baseline) | 305 | 128 | 5.8% | 0.604 |
| XGBoost | 503 | 339 | 16.6% | 0.036 |
| GRU | 501 | 344 | 16.4% | -0.258 |
| LightGBM | 529 | 358 | 17.6% | -0.067 |

**LSTM improves R² by 24.2% over the ARIMA baseline used by the Government of India.**

Trained on 2001–2023 data (8,716 real daily observations across 550+ markets), evaluated on completely held-out 2024–2025 data.

---

## 🎯 Problem Statement

The Department of Consumer Affairs, Government of India, monitors daily prices of 22 essential food commodities through 550 price reporting centres and maintains buffer stocks of onion and pulses for strategic market interventions. Currently, price forecasting relies on ARIMA-based models.

This project builds a modern AI/ML forecasting system that:
- Ingests 20+ years of Agmarknet mandi price data
- Fuses price data with weather features (temperature, precipitation, evapotranspiration)
- Trains and compares 6 forecasting models against the ARIMA baseline
- Serves predictions via a REST API and interactive dashboard
- Provides SHAP-based explainability for policy decisions

Built for **Smart India Hackathon 2024** (PS: AI-ML based models for predicting prices of agri-horticultural commodities).

---

## 🏗️ Architecture

```
Data Sources                  Pipeline                    Output
─────────────                 ────────                    ──────
Agmarknet (20yr)  ──────►  SQLite DB  ──────►  Feature    ──►  Models
Open-Meteo API    ──────►  (2.8M rows)          Matrix         (LSTM, TFT,
                                                (8,716 rows)    XGBoost...)
                                                    │               │
                                                    ▼               ▼
                                              FastAPI REST    Streamlit
                                              API (:8000)     Dashboard
```

---

## 📊 Dashboard

The live dashboard shows:
- **Forecast page** — Real-time price history, 30-day moving average, seasonality patterns
- **Map page** — India price heatmap coloured by state-wise modal price
- **Model Comparison** — Side-by-side metrics table with highlighted best performer
- **Explainability** — SHAP feature importance, correlation analysis, key findings

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.3, pytorch-forecasting |
| ML | XGBoost, LightGBM, scikit-learn |
| Classical | statsmodels (SARIMA), Prophet |
| Data | pandas, SQLAlchemy, SQLite |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly, Folium |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Scheduling | APScheduler |

**All tools are free and open source. No paid APIs or cloud services required.**

---

## 📁 Project Structure

```
agri-price-forecast/
├── src/
│   ├── data/          # Scrapers, CSV loader, database, scheduler
│   ├── features/      # Feature engineering, seasonal decomposition
│   ├── models/        # ARIMA, LSTM, GRU, TFT, XGBoost, Ensemble
│   ├── api/           # FastAPI endpoints
│   └── utils/         # Config, logging
├── dashboard/         # Streamlit multi-page app
├── notebooks/         # EDA + model training notebooks (01–06)
├── docs/              # Generated charts, metrics JSONs, paper notes
├── tests/             # Unit and integration tests
└── data/
    └── processed/     # Feature matrix (git-ignored, regeneratable)
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Monisha1313/agri-price-forecast.git
cd agri-price-forecast

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env

# 5. Initialise database
python -c "from src.data.database import init_db; init_db()"

# 6. Download Agmarknet dataset from Kaggle:
#    https://www.kaggle.com/datasets/vandeetshah/india-commodity-wise-mandi-dataset
#    Place Onion.csv in Dataset/ folder, then:
python -m src.data.loader_csv --commodity onion --file "Dataset/Onion.csv"

# 7. Fetch weather data (free, no API key needed)
python -m src.data.scraper_weather --days 7300

# 8. Build feature matrix
python -m src.features.engineer

# 9. Run notebooks in order (01 → 02 → 03 → 05 → 06)
#    Open in VSCode with Jupyter extension

# 10. Launch dashboard
streamlit run dashboard/app.py

# 11. Launch API (separate terminal)
uvicorn src.api.main:app --reload
```

---

## 📈 Data Pipeline

**Source:** Agmarknet via Kaggle dataset (vandeetshah/india-commodity-wise-mandi-dataset)
- 2,624,379 raw onion price records across 550+ markets (2001–2025)
- After quality filtering: 8,716 real daily observations (removed 37.9% forward-filled rows)

**Weather:** Open-Meteo archive API (free, no key required)
- 4 growing district locations: Nashik, Solapur, Hubli, Kota
- Variables: max/min temperature, precipitation, evapotranspiration

**Feature Engineering (65 features):**
- Lag features: 1, 2, 3, 7, 14, 21, 30 days
- Rolling statistics: 7, 14, 30, 60, 90-day mean/std/min/max
- Calendar: month (cyclical sin/cos encoding), season, day of week
- Price momentum: % change, z-score, price spread
- Weather: temperature, precipitation, ET0

---

## 🔬 Research Findings

1. **LSTM outperforms SARIMA on R²** (0.750 vs 0.604) — capturing overall price trend better
2. **SARIMA has lower RMSE** (305 vs 457) — more conservative on extreme price spikes
3. **Data quality is critical** — removing 37.9% forward-filled duplicate observations significantly improved model performance
4. **Minimum data threshold** — deep learning requires 2,000+ sequences to outperform classical models on commodity price data
5. **Weather features improve predictions** — temperature and precipitation correlate with price through supply-side effects

---

## 📄 Paper

**Title:** Multi-model Comparative Analysis for Agricultural Commodity Price Forecasting Using Deep Learning and Classical Time-Series Methods

**Target:** IEEE Access / IEEE Big Data Conference 2025

**Abstract:** We present a comprehensive comparison of six forecasting models for onion price prediction using 20+ years of Government of India mandi data. Our LSTM model achieves R²=0.750, improving over the existing ARIMA baseline (R²=0.604) by 24.2%. We further demonstrate that data quality — specifically removing spuriously forward-filled observations — is critical for reliable evaluation of deep learning models on agricultural price datasets.

Draft: `docs/paper_notes.md`

---

## 🙏 Acknowledgements

- **Data:** Agmarknet, Government of India (via Kaggle)
- **Weather:** Open-Meteo (open-source weather API)
- **Problem Statement:** Smart India Hackathon 2024, Department of Consumer Affairs
- **Model Reference:** Lim et al. (2021) — Temporal Fusion Transformers

---

## 📬 Contact

**Monisha** | AI/ML Engineering Student
- GitHub: [@Monisha1313](https://github.com/Monisha1313)
- LinkedIn: [Monisha Shivakumar](https://www.linkedin.com/in/monisha-shivakumar-157474211/)

---

*Built with ❤️ for Smart India Hackathon 2024*
