"""
FLIGHTWATCH · weather.py
========================
Weather layer. Pulls current conditions + a 3-day hourly forecast from Open-Meteo
(free, no API key) so we can score weather at the actual scheduled departure /
arrival time of each flight.

Resilience
----------
* 10-minute in-memory cache per airport
* 2-minute negative cache after a failed call (don't hammer a dead API)
* 4 s timeout
* If Open-Meteo is unreachable we fall back to a deterministic climatology
  estimate (same generator the model trained on) and flag `source: "climatology"`
  so the UI can show "WX ESTIMATED" instead of "LIVE".
"""
from __future__ import annotations

import logging
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import numpy as np

from data import AIRPORT_BY_CODE, compute_weather_score, describe_weather, sample_weather

log = logging.getLogger("flightwatch.weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_S = 600
FAIL_TTL_S = 120
TIMEOUT_S = 4.0

_cache: dict[str, tuple[float, dict | None]] = {}
_lock = threading.Lock()
status: dict = {"last_ok": None, "last_error": None, "last_error_at": None}


def _fetch_open_meteo(iata: str) -> dict:
    ap = AIRPORT_BY_CODE[iata]
    params = {
        "latitude": ap.lat,
        "longitude": ap.lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_gusts_10m,visibility",
        "hourly": "temperature_2m,precipitation,weather_code,wind_speed_10m,visibility",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
        "forecast_days": 3,
    }
    r = httpx.get(OPEN_METEO_URL, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _forecast(iata: str) -> dict | None:
    """Cached Open-Meteo payload or None when the API is unavailable."""
    now = time.time()
    with _lock:
        hit = _cache.get(iata)
        if hit and hit[0] > now:
            return hit[1]
    try:
        payload = _fetch_open_meteo(iata)
        with _lock:
            _cache[iata] = (now + CACHE_TTL_S, payload)
        status["last_ok"] = now
        return payload
    except Exception as exc:
        log.warning("Open-Meteo unavailable for %s: %s", iata, exc)
        status["last_error"] = f"{type(exc).__name__}: {exc}"
        status["last_error_at"] = now
        with _lock:
            _cache[iata] = (now + FAIL_TTL_S, None)
        return None


def prefetch(codes: list[str]) -> None:
    """Warm the cache for several airports in parallel."""
    with ThreadPoolExecutor(max_workers=max(1, len(codes))) as pool:
        list(pool.map(_forecast, set(codes)))


def _num(v, default: float) -> float:
    return default if v is None else float(v)


def _severity(score: float) -> str:
    return "LOW" if score < 0.25 else "MODERATE" if score < 0.50 else "HIGH"


def _pack(iata: str, temp, wind, vis_km, precip, code, humidity, gust, valid_at: datetime, source: str) -> dict:
    ap = AIRPORT_BY_CODE[iata]
    score = compute_weather_score(temp, wind, vis_km, precip)
    return {
        "iata": iata,
        "airport": ap.name,
        "city": ap.city,
        "temperature_c": round(temp, 1),
        "wind_kmh": round(wind, 1),
        "wind_gust_kmh": None if gust is None else round(gust, 1),
        "visibility_km": round(vis_km, 1),
        "precipitation_mm": round(precip, 1),
        "humidity_pct": None if humidity is None else int(round(humidity)),
        "condition": describe_weather(temp, wind, vis_km, precip, code),
        "weather_code": code,
        "score": round(score, 4),
        "delay_risk_contribution": round(score, 4),
        "severity": _severity(score),
        "source": source,
        "valid_at": valid_at.strftime("%Y-%m-%dT%H:%MZ"),
    }


def _climatology(iata: str, when: datetime) -> dict:
    """Deterministic offline estimate: same climate model the ML data used."""
    ap = AIRPORT_BY_CODE[iata]
    seed = zlib.crc32(f"{iata}-{when:%Y%m%d%H}".encode())
    temp, wind, vis, precip = sample_weather(np.random.default_rng(seed), ap, when.month)
    return _pack(iata, temp, wind, vis, precip, None, None, None, when, "climatology")


def get_weather(iata: str, when_utc: datetime | None = None) -> dict:
    """
    Weather at an airport. `when_utc=None` -> current conditions; otherwise the
    hourly forecast slot nearest to that moment.
    """
    now = datetime.now(timezone.utc)
    target = (when_utc or now).astimezone(timezone.utc)
    data = _forecast(iata)
    if not data:
        return _climatology(iata, target.replace(minute=0, second=0, microsecond=0))
    try:
        if when_utc is None and data.get("current"):
            c = data["current"]
            vis_km = _num(c.get("visibility"), 10000.0) / 1000.0
            return _pack(iata, _num(c.get("temperature_2m"), 15), _num(c.get("wind_speed_10m"), 0),
                         vis_km, _num(c.get("precipitation"), 0), c.get("weather_code"),
                         c.get("relative_humidity_2m"), c.get("wind_gusts_10m"), now, "open-meteo")
        h = data["hourly"]
        stamps = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h["time"]]
        idx = min(range(len(stamps)), key=lambda i: abs((stamps[i] - target).total_seconds()))
        return _pack(iata, _num(h["temperature_2m"][idx], 15), _num(h["wind_speed_10m"][idx], 0),
                     _num(h["visibility"][idx], 10000.0) / 1000.0, _num(h["precipitation"][idx], 0),
                     h["weather_code"][idx], None, None, stamps[idx], "open-meteo")
    except Exception as exc:  # malformed payload -> degrade gracefully
        log.warning("Bad Open-Meteo payload for %s: %s", iata, exc)
        return _climatology(iata, target.replace(minute=0, second=0, microsecond=0))
