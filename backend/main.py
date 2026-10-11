"""
FLIGHTWATCH · main.py
=====================
FastAPI application.

    GET /api/airports       airport directory
    GET /api/flights        upcoming flights on a route + ML delay predictions
    GET /api/predict        single-flight prediction with risk factors
    GET /api/route-stats    historical delay patterns for a route
    GET /api/weather        current weather + delay-risk contribution
    GET /api/model-info     training metrics + feature importance
    GET /api/health         liveness + data-source status

Run:   uvicorn main:app --reload --port 8000
The same process also serves ../frontend (if present), so one Render service is enough.
"""
from __future__ import annotations

import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import date as _date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import weather as wx
from data import (AIRLINE_BY_CODE, AIRPORT_BY_CODE, AIRPORTS, DAY_NAMES, distance_km,
                  flight_duration_min, is_holiday, route_airlines, route_history, route_schedule)
from model import (FlightDelayModel, build_feature_row, load_or_train, risk_level, status_label)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("flightwatch")

APP_VERSION = "1.0.0"
STATE: dict = {"model": None, "error": None, "started": _time.time()}


# --------------------------------------------------------------------------- #
# App lifecycle: load the cached model, or train it on first boot
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        STATE["model"] = load_or_train()
        log.info("Model ready (v%s, AUC %.3f)", STATE["model"].version, STATE["model"].metrics["roc_auc"])
    except Exception as exc:  # keep the API up so /api/health can report the problem
        STATE["error"] = str(exc)
        log.exception("Model failed to load")
    yield


app = FastAPI(title="FLIGHTWATCH API", version=APP_VERSION,
              description="Know before you board. ML-powered flight delay prediction.",
              lifespan=lifespan)

# CORS: lets a static frontend on Vercel call this API on Render. Tighten
# `allow_origins` to your Vercel domain in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Try again shortly."})


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #
class AirportOut(BaseModel):
    code: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    timezone: str


class WeatherOut(BaseModel):
    iata: str
    airport: str
    city: str
    temperature_c: float
    wind_kmh: float
    wind_gust_kmh: float | None = None
    visibility_km: float
    precipitation_mm: float
    humidity_pct: int | None = None
    condition: str
    weather_code: int | None = None
    score: float
    delay_risk_contribution: float
    severity: str
    source: str
    valid_at: str


class RiskFactor(BaseModel):
    factor: str
    category: str
    key: str
    direction: str          # "increases" | "reduces"
    impact_pts: float       # percentage-point effect on delay probability
    weight: float           # 0-1, relative to the strongest factor shown


class AirlineOut(BaseModel):
    code: str
    name: str


class RouteInfo(BaseModel):
    origin: AirportOut
    destination: AirportOut
    distance_km: float
    duration_min: int
    airlines: list[AirlineOut]


class FlightOut(BaseModel):
    flight_number: str
    airline_code: str
    airline_name: str
    aircraft: str
    scheduled_departure_local: str
    scheduled_departure_utc: str
    scheduled_arrival_local: str
    departs_in_min: int
    hour: int
    day_of_week: int
    delay_probability: float
    predicted_delay_minutes: float
    confidence: float
    risk_level: str
    status: str
    weather_origin_condition: str
    weather_destination_condition: str
    risk_factors: list[RiskFactor]


class FlightsSummary(BaseModel):
    avg_delay_probability: float
    avg_predicted_delay_minutes: float
    risk_level: str
    safest_flight: str
    riskiest_flight: str


class FlightsResponse(BaseModel):
    route: RouteInfo
    generated_at: str
    schedule_source: str
    weather: dict[str, WeatherOut]
    flights: list[FlightOut]
    summary: FlightsSummary


class PredictResponse(BaseModel):
    route: str
    airline: str
    airline_name: str
    hour: int
    day_of_week: int
    day_name: str
    delay_probability: float
    predicted_delay_minutes: float
    confidence: float
    risk_level: str
    status: str
    risk_factors: list[RiskFactor]
    weather: dict[str, WeatherOut]


class HourStat(BaseModel):
    hour: int
    flights: int
    delay_rate_pct: float | None
    avg_delay_minutes: float | None


class DayStat(BaseModel):
    day_of_week: int
    day: str
    flights: int
    delay_rate_pct: float
    avg_delay_minutes: float


class AirlineStat(BaseModel):
    airline_code: str
    airline: str
    on_time_pct: float
    avg_delay_minutes: float
    flights_tracked: int
    rank: int


class RouteStatsResponse(BaseModel):
    route: str
    flights_tracked: int
    average_delay_minutes: float
    on_time_percentage: float
    worst_day_of_week: DayStat
    best_day_of_week: DayStat
    worst_hour: HourStat
    best_hour: HourStat
    by_hour: list[HourStat]
    by_day: list[DayStat]
    airline_comparison: list[AirlineStat]
    best_airline: str
    data_source: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _airport(code: str, label: str):
    c = (code or "").strip().upper()
    if len(c) != 3 or not c.isalpha():
        raise HTTPException(400, f"'{code}' is not a valid IATA code (expected 3 letters) for {label}.")
    ap = AIRPORT_BY_CODE.get(c)
    if ap is None:
        raise HTTPException(404, f"Airport '{c}' is not in the FLIGHTWATCH directory. See /api/airports.")
    return ap


def _route(origin: str, dest: str):
    a, b = _airport(origin, "origin"), _airport(dest, "destination")
    if a.iata == b.iata:
        raise HTTPException(400, "Origin and destination must be different airports.")
    return a, b


def _model() -> FlightDelayModel:
    m = STATE["model"]
    if m is None:
        raise HTTPException(503, f"Prediction model unavailable: {STATE['error'] or 'still loading'}")
    return m


def _airport_out(a) -> AirportOut:
    return AirportOut(code=a.iata, name=a.name, city=a.city, country=a.country,
                      lat=a.lat, lon=a.lon, timezone=a.tz)


def _iso_local(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    s = wx.status
    if s["last_ok"] and (not s["last_error_at"] or s["last_ok"] >= s["last_error_at"]):
        weather_api = "online"
    elif s["last_error_at"]:
        weather_api = "degraded"
    else:
        weather_api = "idle"
    return {
        "status": "ok" if STATE["model"] else "degraded",
        "version": APP_VERSION,
        "model_loaded": STATE["model"] is not None,
        "model_error": STATE["error"],
        "weather_api": weather_api,
        "uptime_s": round(_time.time() - STATE["started"]),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/api/airports", response_model=list[AirportOut])
def airports():
    return [_airport_out(a) for a in AIRPORTS]


@app.get("/api/model-info")
def model_info():
    m = _model()
    return {"version": m.version, "trained_at": m.trained_at, "algorithm": "RandomForest (classifier + regressor)",
            "metrics": m.metrics, "feature_importance": m.importance}


@app.get("/api/weather", response_model=WeatherOut)
def weather_endpoint(iata: str = Query(..., min_length=3, max_length=3, description="Airport IATA code")):
    ap = _airport(iata, "iata")
    return wx.get_weather(ap.iata)


@app.get("/api/flights", response_model=FlightsResponse)
def flights(origin: str = Query(..., alias="from", description="Origin IATA"),
            dest: str = Query(..., alias="to", description="Destination IATA")):
    ao, ad = _route(origin, dest)
    model = _model()
    wx.prefetch([ao.iata, ad.iata])                        # parallel Open-Meteo calls

    tz_o, tz_d = ZoneInfo(ao.tz), ZoneInfo(ad.tz)
    now_utc = datetime.now(timezone.utc)
    now_o = now_utc.astimezone(tz_o)
    dist = distance_km(ao.iata, ad.iata)
    dur = flight_duration_min(dist)

    # 1) Build the upcoming departures (next occurrence of each daily slot)
    items = []
    for slot in route_schedule(ao.iata, ad.iata):
        t = time(slot["hour"], slot["minute"])
        dep_local = datetime.combine(now_o.date(), t, tzinfo=tz_o)
        if dep_local < now_o + timedelta(minutes=20):
            dep_local = datetime.combine(now_o.date() + timedelta(days=1), t, tzinfo=tz_o)
        dep_utc = dep_local.astimezone(timezone.utc)
        arr_utc = dep_utc + timedelta(minutes=dur)
        items.append((dep_utc, dep_local, arr_utc, slot))
    items.sort(key=lambda x: x[0])

    # 2) Features: weather at the *scheduled* departure / arrival hour
    rows, ctxs, wxs = [], [], []
    for dep_utc, dep_local, arr_utc, slot in items:
        w_o, w_d = wx.get_weather(ao.iata, dep_utc), wx.get_weather(ad.iata, arr_utc)
        dow = dep_local.weekday()
        rows.append(build_feature_row(ao.iata, ad.iata, slot["airline"], dep_local.hour, dow, dep_local.month,
                                      is_holiday(dep_local.month, dep_local.day), w_o["score"], w_d["score"]))
        ctxs.append({"hour": dep_local.hour, "dow": dow, "month": dep_local.month, "origin": ao.iata,
                     "dest": ad.iata, "airline": slot["airline"], "distance": dist,
                     "wx_origin": w_o["condition"], "wx_dest": w_d["condition"]})
        wxs.append((w_o, w_d))

    # 3) One batched ML call + per-flight explanations
    try:
        preds = model.predict_batch(rows)
        factors = [model.explain(r, c) for r, c in zip(rows, ctxs)]
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(500, f"Model inference failed: {exc}") from exc

    out = []
    for (dep_utc, dep_local, arr_utc, slot), row, p, f, (w_o, w_d) in zip(items, rows, preds, factors, wxs):
        out.append(FlightOut(
            flight_number=slot["flight_number"], airline_code=slot["airline"],
            airline_name=AIRLINE_BY_CODE[slot["airline"]].name, aircraft=slot["aircraft"],
            scheduled_departure_local=_iso_local(dep_local), scheduled_departure_utc=_iso_utc(dep_utc),
            scheduled_arrival_local=_iso_local(arr_utc.astimezone(tz_d)),
            departs_in_min=max(0, int((dep_utc - now_utc).total_seconds() // 60)),
            hour=dep_local.hour, day_of_week=dep_local.weekday(),
            delay_probability=round(p.probability, 4), predicted_delay_minutes=round(p.minutes, 1),
            confidence=round(p.confidence, 3), risk_level=risk_level(p.probability),
            status=status_label(p.probability),
            weather_origin_condition=w_o["condition"], weather_destination_condition=w_d["condition"],
            risk_factors=f))

    avg_p = sum(f.delay_probability for f in out) / len(out)
    summary = FlightsSummary(
        avg_delay_probability=round(avg_p, 4),
        avg_predicted_delay_minutes=round(sum(f.predicted_delay_minutes for f in out) / len(out), 1),
        risk_level=risk_level(avg_p),
        safest_flight=min(out, key=lambda f: f.delay_probability).flight_number,
        riskiest_flight=max(out, key=lambda f: f.delay_probability).flight_number)

    return FlightsResponse(
        route=RouteInfo(origin=_airport_out(ao), destination=_airport_out(ad), distance_km=dist, duration_min=dur,
                        airlines=[AirlineOut(code=c, name=AIRLINE_BY_CODE[c].name)
                                  for c in route_airlines(ao.iata, ad.iata)]),
        generated_at=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        schedule_source="simulated",
        weather={"origin": WeatherOut(**wx.get_weather(ao.iata)), "destination": WeatherOut(**wx.get_weather(ad.iata))},
        flights=out, summary=summary)


@app.get("/api/predict", response_model=PredictResponse)
def predict(origin: str = Query(..., alias="from"),
            dest: str = Query(..., alias="to"),
            airline: str = Query(..., min_length=2, max_length=2, description="Airline IATA code, e.g. AI"),
            hour: int = Query(..., ge=0, le=23, description="Local departure hour"),
            day_of_week: int = Query(..., alias="dayOfWeek", ge=0, le=6, description="0=Mon ... 6=Sun"),
            date: str | None = Query(None, description="Optional YYYY-MM-DD for month/holiday context")):
    ao, ad = _route(origin, dest)
    model = _model()
    code = airline.strip().upper()
    if code not in AIRLINE_BY_CODE:
        raise HTTPException(404, f"Unknown airline code '{code}'.")

    tz_o, tz_d = ZoneInfo(ao.tz), ZoneInfo(ad.tz)
    now_o = datetime.now(timezone.utc).astimezone(tz_o)
    if date:
        try:
            d0 = _date.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, "date must be formatted YYYY-MM-DD.")
        dep_local = datetime.combine(d0, time(hour), tzinfo=tz_o)
    else:  # next occurrence of that local hour
        dep_local = datetime.combine(now_o.date(), time(hour), tzinfo=tz_o)
        if dep_local < now_o:
            dep_local += timedelta(days=1)
    dep_utc = dep_local.astimezone(timezone.utc)
    dist = distance_km(ao.iata, ad.iata)
    arr_utc = dep_utc + timedelta(minutes=flight_duration_min(dist))

    wx.prefetch([ao.iata, ad.iata])
    w_o, w_d = wx.get_weather(ao.iata, dep_utc), wx.get_weather(ad.iata, arr_utc)
    row = build_feature_row(ao.iata, ad.iata, code, hour, day_of_week, dep_local.month,
                            is_holiday(dep_local.month, dep_local.day), w_o["score"], w_d["score"])
    ctx = {"hour": hour, "dow": day_of_week, "month": dep_local.month, "origin": ao.iata, "dest": ad.iata,
           "airline": code, "distance": dist, "wx_origin": w_o["condition"], "wx_dest": w_d["condition"]}
    try:
        p = model.predict_one(row)
        factors = model.explain(row, ctx)
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(500, f"Model inference failed: {exc}") from exc

    return PredictResponse(
        route=f"{ao.iata}-{ad.iata}", airline=code, airline_name=AIRLINE_BY_CODE[code].name, hour=hour,
        day_of_week=day_of_week, day_name=DAY_NAMES[day_of_week],
        delay_probability=round(p.probability, 4), predicted_delay_minutes=round(p.minutes, 1),
        confidence=round(p.confidence, 3), risk_level=risk_level(p.probability), status=status_label(p.probability),
        risk_factors=factors, weather={"origin": WeatherOut(**w_o), "destination": WeatherOut(**w_d)})


@lru_cache(maxsize=512)
def _route_stats(o: str, d: str) -> dict:
    h = route_history(o, d)
    n_min = 20                                              # ignore thin hours when ranking

    def hour_stat(hr: int) -> dict:
        s = h[h.hour_of_day == hr]
        if len(s) == 0:
            return {"hour": hr, "flights": 0, "delay_rate_pct": None, "avg_delay_minutes": None}
        return {"hour": hr, "flights": int(len(s)), "delay_rate_pct": round(float(s.delayed.mean() * 100), 1),
                "avg_delay_minutes": round(float(s.delay_minutes.mean()), 1)}

    by_hour = [hour_stat(hr) for hr in range(24)]
    ranked_hours = [x for x in by_hour if x["flights"] >= n_min]

    by_day = []
    for dow in range(7):
        s = h[h.day_of_week == dow]
        by_day.append({"day_of_week": dow, "day": DAY_NAMES[dow], "flights": int(len(s)),
                       "delay_rate_pct": round(float(s.delayed.mean() * 100), 1),
                       "avg_delay_minutes": round(float(s.delay_minutes.mean()), 1)})

    comp = []
    for code, s in h.groupby("airline"):
        comp.append({"airline_code": code, "airline": AIRLINE_BY_CODE[code].name,
                     "on_time_pct": round(float((1 - s.delayed.mean()) * 100), 1),
                     "avg_delay_minutes": round(float(s.delay_minutes.mean()), 1),
                     "flights_tracked": int(len(s))})
    comp.sort(key=lambda x: (-x["on_time_pct"], x["avg_delay_minutes"]))
    for i, c in enumerate(comp, 1):
        c["rank"] = i

    return {
        "route": f"{o}-{d}",
        "flights_tracked": int(len(h)),
        "average_delay_minutes": round(float(h.delay_minutes.mean()), 1),
        "on_time_percentage": round(float((1 - h.delayed.mean()) * 100), 1),
        "worst_day_of_week": max(by_day, key=lambda x: x["delay_rate_pct"]),
        "best_day_of_week": min(by_day, key=lambda x: x["delay_rate_pct"]),
        "worst_hour": max(ranked_hours, key=lambda x: x["delay_rate_pct"]),
        "best_hour": min(ranked_hours, key=lambda x: x["delay_rate_pct"]),
        "by_hour": by_hour, "by_day": by_day, "airline_comparison": comp,
        "best_airline": comp[0]["airline"],
        "data_source": "simulated historical flights (see README to plug in BTS / OAG data)",
    }


@app.get("/api/route-stats", response_model=RouteStatsResponse)
def route_stats(origin: str = Query(..., alias="from"), dest: str = Query(..., alias="to")):
    ao, ad = _route(origin, dest)
    return _route_stats(ao.iata, ad.iata)


# --------------------------------------------------------------------------- #
# Static frontend (single-service deploys). API routes above take precedence.
# --------------------------------------------------------------------------- #
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"name": "FLIGHTWATCH API", "tagline": "Know before you board.", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
