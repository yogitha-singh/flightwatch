# FLIGHTWATCH

### *Know before you board.*

ML-powered flight delay predictor with a mission-control UI. Pick any airport pair (DEL→BOM, JFK→LAX, LHR→DXB…) and FLIGHTWATCH scores the delay risk of upcoming departures with a Random Forest, layers in weather at both ends, and shows how the route and each airline have historically behaved.

> **Screenshots**
> `docs/landing.png` · `docs/results.png` · `docs/mobile.png` *(add yours here)*

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python 3.11+, FastAPI, uvicorn, Pydantic v2 |
| ML | scikit-learn (RandomForestClassifier + RandomForestRegressor), pandas, numpy, joblib |
| Weather | [Open-Meteo](https://open-meteo.com) (free, no API key) |
| Frontend | HTML5, CSS3 (custom properties, glassmorphism), vanilla ES6+, Chart.js (CDN) |
| Fonts | Inter, JetBrains Mono, Space Grotesk |
| Deploy | Render (API + static in one service) or Render + Vercel; Dockerfile included |

## Features

- **Route search** with airport autocomplete (40 airports), swap button, quick chips and recent searches
- **Delay-risk gauge**: animated needle, green < 30 % / amber 30–60 % / red > 60 %
- **Expected delay** in minutes plus a model-confidence bar
- **Top 3 risk factors** per flight (e.g. "Evening departure (peak hour)", "Weather at DEL: Fog")
- **Upcoming flights** with predicted delay badge, status (ON TIME / AT RISK / DELAYED), click-to-inspect and a flip animation for details
- **Weather cards** for departure and arrival, scored at the *scheduled hour* of each flight
- **Historical patterns**: delay by hour and by weekday (worst bar red, best bar green)
- **Airline comparison** table with a "Safest bet" badge and a filter by carrier
- **Best time to fly** panel
- **Radar landing page** where the 40 airports are plotted on a real azimuthal projection and light up as the sweep passes
- Shareable URLs (`?from=DEL&to=BOM`), optional sound alert on elevated risk (off by default), UTC clock, live/estimated data indicator, loading + error states, keyboard-accessible, responsive, reduced-motion aware

## ML pipeline

```
data.generate_flights(10 000)  ->  80/20 stratified split  ->  RandomForestClassifier  (delayed > 15 min)
                                                           ->  RandomForestRegressor   (delay minutes)
                                ->  metrics + importances  ->  model.pkl (joblib)
```

**Features (11):** hour of day, day of week, month, airline id, origin id, destination id, distance (km), origin weather score, destination weather score, is_weekend, is_holiday.

**Synthetic data** (`backend/data.py`) is built from a hidden ground-truth log-odds model so the patterns are learnable but noisy:

| Effect | Behaviour |
|---|---|
| Time of day | peaks at 07–09h and 17–20h, knock-on delays late evening, calm early morning |
| Weekday | Friday and Sunday worst, Tuesday / Wednesday best |
| Season | Dec–Feb worst, Jul–Aug elevated, plus holiday windows |
| Weather | wind, visibility, precipitation, icing / heat → 0–1 score; origin weighs more than destination |
| Distance | longer sectors slightly worse |
| Airline & airport | hidden punctuality / congestion effects the model must learn from encoded IDs |

**Weather score.** The same function (`compute_weather_score`) turns raw weather into a 0–1 severity score for training data *and* for live Open-Meteo readings, so the model sees identical semantics at inference.

**Evaluation (seed 42, 2 000 held-out flights):** ROC-AUC ≈ 0.72, accuracy ≈ 79 % (majority baseline 78 %), MAE ≈ 23 min. The ceiling is deliberately modest: delays are noisy, and the model is honest about it. `/api/model-info` exposes all metrics and feature importances.

**Confidence** is the agreement between the forest's individual trees: low spread = high confidence.

**Risk factors** use counterfactual attribution: for each feature group the flight's value is swapped with values from a background sample, and the change in predicted probability is that factor's contribution.

**Route stats** come from a larger deterministic simulated history per route (3 000 flights), so charts are stable between requests.

### Honest limits (read before showing this to a recruiter)

- The **training data and route history are synthetic**. Patterns are realistic by construction, not measured.
- The **flight timetable is simulated** (deterministic per route). Weather is real when Open-Meteo is reachable; otherwise a climatology estimate is used and the header shows **WX ESTIMATED**.
- To go real: replace `generate_flights()` with a loader for the [US BTS On-Time dataset](https://www.transtats.bts.gov/) (same columns), and swap `route_schedule()` for AviationStack / OpenSky / FlightAware.

## Run locally

```bash
# 1. Backend (serves the frontend too)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# first start trains and saves model.pkl (~5 s)

# 2. Open http://localhost:8000
```

Prefer a separate static server for the frontend? Run `python -m http.server 5500` inside `frontend/` and open `http://localhost:5500`. The frontend auto-targets `http://localhost:8000` when served from another localhost port.

Docker: `docker build -t flightwatch . && docker run -p 8000:8000 flightwatch`

## Deployment

### Option A: everything on Render (simplest)

1. Push the repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. **Root directory:** `backend` · **Runtime:** Python 3 (set env `PYTHON_VERSION=3.11.9`)
4. **Build command:** `pip install -r requirements.txt && python model.py`
5. **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

FastAPI serves `../frontend` automatically when it exists, so no extra config. If your Render root directory is `backend` and the frontend folder is not found, use the Docker option (Render → Runtime: Docker, root = repo root).

### Option B: Render (API) + Vercel (frontend)

1. Deploy the backend on Render as above.
2. In `frontend/script.js` set `CONFIG.API_BASE = "https://YOUR-SERVICE.onrender.com"`.
3. Vercel → **New Project** → import the repo → **Root directory:** `frontend` → framework preset *Other* → Deploy.
4. Optional hardening: in `backend/main.py` replace `allow_origins=["*"]` with your Vercel domain.

> Free Render instances sleep when idle; the first request can take ~30 s. The UI shows a "waking the server" message while it waits.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/airports` | 40 airports: code, name, city, country, lat, lon, timezone |
| GET | `/api/flights?from=DEL&to=BOM` | 8 upcoming departures with ML delay probability, expected minutes, confidence, status, risk factors; current weather at both airports |
| GET | `/api/predict?from=&to=&airline=&hour=&dayOfWeek=` | Single prediction + top risk factors (`dayOfWeek` 0 = Mon … 6 = Sun; optional `date=YYYY-MM-DD`) |
| GET | `/api/route-stats?from=&to=` | Avg delay, on-time %, worst/best day & hour, by-hour and by-day series, airline comparison |
| GET | `/api/weather?iata=DEL` | Temperature, wind, visibility, precipitation, delay-risk contribution |
| GET | `/api/model-info` | Training metrics + feature importance |
| GET | `/api/health` | Liveness, model status, weather API status |

Interactive docs at `/docs`. Errors are JSON: `{"detail": "Airport 'XXX' is not in the FLIGHTWATCH directory."}`.

## Project layout

```
flightwatch/
├── backend/   main.py · model.py · data.py · weather.py · requirements.txt · README.md
├── frontend/  index.html · style.css · script.js
├── Dockerfile
├── README.md
└── .gitignore
```

## Credits

- Weather data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0)
- ML: [scikit-learn](https://scikit-learn.org)
- API: [FastAPI](https://fastapi.tiangolo.com) · charts: [Chart.js](https://www.chartjs.org)
- Typefaces: Inter, JetBrains Mono, Space Grotesk (Google Fonts)

Built as Day 4 of a portfolio series. FLIGHTWATCH v1.0
