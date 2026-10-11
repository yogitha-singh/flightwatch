# FLIGHTWATCH backend

FastAPI service + scikit-learn model. See the [root README](../README.md) for the full guide.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000                # trains model.pkl on first start (~5 s)
```

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, Pydantic schemas, endpoints, static frontend mount |
| `model.py` | Train / load / predict, tree-agreement confidence, per-flight risk factors |
| `data.py` | Airports, airlines, weather scoring, synthetic data simulator |
| `weather.py` | Open-Meteo client with cache + offline climatology fallback |

Retrain manually: `python model.py` · Inspect the data generator: `python data.py` · API docs: <http://localhost:8000/docs>
