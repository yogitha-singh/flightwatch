"""
FLIGHTWATCH · data.py
=====================
Synthetic-but-realistic data layer. Everything the ML pipeline needs lives here:

  * AIRPORTS / AIRLINES reference tables (40 airports, ~45 carriers)
  * Weather scoring (the SAME formula is used for training data and for live
    Open-Meteo observations, so the model sees identical feature semantics)
  * `generate_flights()` - ground-truth simulator that produces historical rows
    with realistic delay structure:
        - peak hours (07-09h, 17-20h) are worse
        - Friday & Sunday are worse
        - Dec-Feb (winter) is worse, plus holiday spikes
        - bad weather (wind / low visibility / precipitation / extreme temp)
          raises delay odds, origin weather matters more than destination
        - longer sectors are slightly worse
        - airline punctuality + airport congestion are baked in as hidden
          effects that the model has to *learn* from encoded IDs
  * `route_history()` - a larger deterministic history for one route, used for
    the Route Stats charts.

Day-of-week convention everywhere in FLIGHTWATCH: 0 = Monday ... 6 = Sunday.

NOTE: This is simulated data. To go "real", replace `generate_flights()` with
a loader for the US BTS On-Time dataset (or any OAG/Cirium export) that emits
the same FEATURE_COLUMNS + `delayed` + `delay_minutes`.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Feature schema (single source of truth - imported by model.py)
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "airline_id",
    "origin_airport_id",
    "destination_airport_id",
    "distance_km",
    "weather_origin_score",
    "weather_destination_score",
    "is_weekend",
    "is_holiday",
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    city: str
    country_code: str
    country: str
    lat: float
    lon: float
    tz: str
    climate: str        # key into CLIMATES
    congestion: float   # hidden delay effect (logit units) used by the simulator
    weight: float       # relative traffic volume (sampling weight)


# (iata, name, city, cc, country, lat, lon, tz, climate, congestion, weight)
_AIRPORT_ROWS = [
    # India
    ("DEL", "Indira Gandhi International", "Delhi", "IN", "India", 28.5562, 77.1000, "Asia/Kolkata", "monsoon_fog", 0.30, 10),
    ("BOM", "Chhatrapati Shivaji Maharaj International", "Mumbai", "IN", "India", 19.0896, 72.8656, "Asia/Kolkata", "monsoon", 0.35, 9),
    ("BLR", "Kempegowda International", "Bengaluru", "IN", "India", 13.1986, 77.7066, "Asia/Kolkata", "monsoon", 0.10, 7),
    ("MAA", "Chennai International", "Chennai", "IN", "India", 12.9941, 80.1709, "Asia/Kolkata", "monsoon", 0.05, 5),
    ("HYD", "Rajiv Gandhi International", "Hyderabad", "IN", "India", 17.2403, 78.4294, "Asia/Kolkata", "monsoon", 0.00, 5),
    ("CCU", "Netaji Subhas Chandra Bose International", "Kolkata", "IN", "India", 22.6547, 88.4467, "Asia/Kolkata", "monsoon_fog", 0.10, 4),
    # United States
    ("JFK", "John F. Kennedy International", "New York", "US", "United States", 40.6413, -73.7781, "America/New_York", "continental", 0.45, 9),
    ("LAX", "Los Angeles International", "Los Angeles", "US", "United States", 33.9416, -118.4085, "America/Los_Angeles", "mediterranean", 0.15, 9),
    ("ORD", "O'Hare International", "Chicago", "US", "United States", 41.9742, -87.9073, "America/Chicago", "continental", 0.45, 9),
    ("SFO", "San Francisco International", "San Francisco", "US", "United States", 37.6213, -122.3790, "America/Los_Angeles", "seafog", 0.35, 6),
    ("ATL", "Hartsfield-Jackson Atlanta International", "Atlanta", "US", "United States", 33.6407, -84.4277, "America/New_York", "subtropical", 0.20, 10),
    ("DFW", "Dallas/Fort Worth International", "Dallas", "US", "United States", 32.8998, -97.0403, "America/Chicago", "subtropical", 0.15, 8),
    ("SEA", "Seattle-Tacoma International", "Seattle", "US", "United States", 47.4502, -122.3088, "America/Los_Angeles", "oceanic", 0.05, 6),
    ("MIA", "Miami International", "Miami", "US", "United States", 25.7959, -80.2870, "America/New_York", "tropical", 0.20, 6),
    ("BOS", "Logan International", "Boston", "US", "United States", 42.3656, -71.0096, "America/New_York", "continental", 0.30, 6),
    ("DEN", "Denver International", "Denver", "US", "United States", 39.8561, -104.6737, "America/Denver", "continental", 0.15, 7),
    # Europe
    ("LHR", "Heathrow", "London", "GB", "United Kingdom", 51.4700, -0.4543, "Europe/London", "oceanic", 0.40, 10),
    ("CDG", "Charles de Gaulle", "Paris", "FR", "France", 49.0097, 2.5479, "Europe/Paris", "oceanic", 0.30, 9),
    ("AMS", "Schiphol", "Amsterdam", "NL", "Netherlands", 52.3105, 4.7683, "Europe/Amsterdam", "oceanic", 0.25, 8),
    ("FRA", "Frankfurt am Main", "Frankfurt", "DE", "Germany", 50.0379, 8.5622, "Europe/Berlin", "temperate", 0.25, 9),
    ("MUC", "Munich", "Munich", "DE", "Germany", 48.3537, 11.7750, "Europe/Berlin", "temperate", -0.05, 6),
    ("MAD", "Adolfo Suarez Madrid-Barajas", "Madrid", "ES", "Spain", 40.4983, -3.5676, "Europe/Madrid", "mediterranean", 0.00, 7),
    ("FCO", "Leonardo da Vinci-Fiumicino", "Rome", "IT", "Italy", 41.8003, 12.2389, "Europe/Rome", "mediterranean", 0.10, 6),
    ("IST", "Istanbul Airport", "Istanbul", "TR", "Turkey", 41.2753, 28.7519, "Europe/Istanbul", "temperate", 0.15, 8),
    # Middle East & Asia
    ("DXB", "Dubai International", "Dubai", "AE", "United Arab Emirates", 25.2532, 55.3657, "Asia/Dubai", "desert", -0.05, 10),
    ("DOH", "Hamad International", "Doha", "QA", "Qatar", 25.2731, 51.6081, "Asia/Qatar", "desert", -0.10, 7),
    ("SIN", "Changi", "Singapore", "SG", "Singapore", 1.3644, 103.9915, "Asia/Singapore", "tropical", -0.30, 9),
    ("HKG", "Hong Kong International", "Hong Kong", "HK", "Hong Kong", 22.3080, 113.9185, "Asia/Hong_Kong", "typhoon", -0.05, 8),
    ("NRT", "Narita International", "Tokyo", "JP", "Japan", 35.7720, 140.3929, "Asia/Tokyo", "typhoon", -0.25, 7),
    ("ICN", "Incheon International", "Seoul", "KR", "South Korea", 37.4602, 126.4407, "Asia/Seoul", "temperate", -0.25, 7),
    ("BKK", "Suvarnabhumi", "Bangkok", "TH", "Thailand", 13.6900, 100.7501, "Asia/Bangkok", "tropical", 0.10, 8),
    ("KUL", "Kuala Lumpur International", "Kuala Lumpur", "MY", "Malaysia", 2.7456, 101.7099, "Asia/Kuala_Lumpur", "tropical", 0.00, 6),
    ("PEK", "Beijing Capital International", "Beijing", "CN", "China", 40.0799, 116.5844, "Asia/Shanghai", "continental", 0.30, 9),
    # Oceania
    ("SYD", "Kingsford Smith", "Sydney", "AU", "Australia", -33.9399, 151.1753, "Australia/Sydney", "oceanic", 0.15, 6),
    ("MEL", "Melbourne Tullamarine", "Melbourne", "AU", "Australia", -37.6690, 144.8410, "Australia/Melbourne", "oceanic", 0.10, 6),
    ("AKL", "Auckland", "Auckland", "NZ", "New Zealand", -37.0082, 174.7850, "Pacific/Auckland", "oceanic", 0.00, 4),
    # South America
    ("GRU", "Sao Paulo-Guarulhos International", "Sao Paulo", "BR", "Brazil", -23.4356, -46.4731, "America/Sao_Paulo", "subtropical", 0.20, 7),
    ("EZE", "Ministro Pistarini International", "Buenos Aires", "AR", "Argentina", -34.8222, -58.5358, "America/Argentina/Buenos_Aires", "subtropical", 0.05, 4),
    # Africa
    ("JNB", "O. R. Tambo International", "Johannesburg", "ZA", "South Africa", -26.1392, 28.2460, "Africa/Johannesburg", "subtropical", 0.05, 5),
    ("CPT", "Cape Town International", "Cape Town", "ZA", "South Africa", -33.9715, 18.6021, "Africa/Johannesburg", "mediterranean", 0.10, 4),
]
AIRPORTS: list[Airport] = [Airport(*row) for row in _AIRPORT_ROWS]

AIRPORT_BY_CODE: dict[str, Airport] = {a.iata: a for a in AIRPORTS}
AIRPORT_INDEX: dict[str, int] = {a.iata: i for i, a in enumerate(AIRPORTS)}


@dataclass(frozen=True)
class Airline:
    code: str
    name: str
    country_code: str
    bias: float  # hidden punctuality effect (logit units): negative = punctual


AIRLINES: list[Airline] = [
    # India
    Airline("AI", "Air India", "IN", 0.25),
    Airline("6E", "IndiGo", "IN", -0.10),
    Airline("SG", "SpiceJet", "IN", 0.35),
    Airline("IX", "Air India Express", "IN", 0.20),
    Airline("QP", "Akasa Air", "IN", -0.05),
    # United States
    Airline("AA", "American Airlines", "US", 0.10),
    Airline("DL", "Delta Air Lines", "US", -0.25),
    Airline("UA", "United Airlines", "US", 0.10),
    Airline("WN", "Southwest Airlines", "US", 0.15),
    Airline("B6", "JetBlue", "US", 0.30),
    Airline("AS", "Alaska Airlines", "US", -0.15),
    # Europe
    Airline("BA", "British Airways", "GB", 0.10),
    Airline("VS", "Virgin Atlantic", "GB", 0.00),
    Airline("AF", "Air France", "FR", 0.10),
    Airline("KL", "KLM", "NL", 0.05),
    Airline("LH", "Lufthansa", "DE", 0.15),
    Airline("EW", "Eurowings", "DE", 0.10),
    Airline("IB", "Iberia", "ES", -0.05),
    Airline("VY", "Vueling", "ES", 0.20),
    Airline("AZ", "ITA Airways", "IT", 0.05),
    Airline("TK", "Turkish Airlines", "TR", 0.05),
    # Middle East
    Airline("EK", "Emirates", "AE", -0.20),
    Airline("QR", "Qatar Airways", "QA", -0.25),
    # Asia
    Airline("SQ", "Singapore Airlines", "SG", -0.35),
    Airline("CX", "Cathay Pacific", "HK", -0.10),
    Airline("NH", "ANA", "JP", -0.40),
    Airline("JL", "Japan Airlines", "JP", -0.35),
    Airline("KE", "Korean Air", "KR", -0.15),
    Airline("TG", "Thai Airways", "TH", 0.05),
    Airline("MH", "Malaysia Airlines", "MY", 0.00),
    Airline("AK", "AirAsia", "MY", 0.15),
    Airline("CA", "Air China", "CN", 0.30),
    Airline("MU", "China Eastern", "CN", 0.25),
    # Oceania
    Airline("QF", "Qantas", "AU", 0.05),
    Airline("VA", "Virgin Australia", "AU", 0.10),
    Airline("JQ", "Jetstar", "AU", 0.20),
    Airline("NZ", "Air New Zealand", "NZ", -0.05),
    # South America
    Airline("LA", "LATAM Airlines", "BR", -0.05),
    Airline("AR", "Aerolineas Argentinas", "AR", 0.25),
    # Africa
    Airline("SA", "South African Airways", "ZA", 0.30),
    Airline("FA", "FlySafair", "ZA", -0.10),
]
AIRLINE_BY_CODE: dict[str, Airline] = {a.code: a for a in AIRLINES}
AIRLINE_INDEX: dict[str, int] = {a.code: i for i, a in enumerate(AIRLINES)}

# Global carriers used to pad international routes that have too few home carriers
_FILLER_CARRIERS = ["EK", "QR", "TK", "SQ", "LH", "BA", "AF", "KL", "CX"]


# --------------------------------------------------------------------------- #
# Geometry + route helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


@lru_cache(maxsize=None)
def distance_km(origin: str, dest: str) -> float:
    a, b = AIRPORT_BY_CODE[origin], AIRPORT_BY_CODE[dest]
    return round(haversine_km(a.lat, a.lon, b.lat, b.lon), 1)


def flight_duration_min(dist_km: float) -> int:
    """Rough block time: cruise ~820 km/h plus 30 min taxi/climb/descent."""
    return int(round(dist_km / 820 * 60 + 30))


@lru_cache(maxsize=None)
def route_airlines(origin: str, dest: str) -> tuple[str, ...]:
    """Carriers 'operating' a route (deterministic, symmetric A<->B)."""
    ao, ad = AIRPORT_BY_CODE[origin], AIRPORT_BY_CODE[dest]
    domestic = ao.country_code == ad.country_code
    pool = [a.code for a in AIRLINES if a.country_code in (ao.country_code, ad.country_code)]
    rng = random.Random(f"routes-{min(origin, dest)}-{max(origin, dest)}")
    if not domestic and len(pool) < 4:
        extras = [c for c in _FILLER_CARRIERS if c not in pool]
        rng.shuffle(extras)
        pool += extras[: 4 - len(pool)]
    if len(pool) > 6:
        keep = set(rng.sample(pool, 6))
        pool = [c for c in pool if c in keep]
    return tuple(pool)


_AIRCRAFT = {
    "short": ["A320neo", "B737-800", "A321", "B737 MAX 8", "E190"],
    "medium": ["A321neo", "B737 MAX 8", "A330-300", "B787-8", "A320neo"],
    "long": ["B777-300ER", "A350-900", "B787-9", "B777-200LR", "A330-900"],
}


@lru_cache(maxsize=None)
def route_schedule(origin: str, dest: str, n: int = 8) -> tuple[dict, ...]:
    """
    Deterministic daily timetable for a route (local departure time at origin).
    Schedules are SIMULATED - swap for AviationStack / OAG / FlightAware AeroAPI
    in production.
    """
    rng = random.Random(f"sched-{origin}-{dest}")
    airlines = list(route_airlines(origin, dest))
    cycle = (airlines * n)[:n]
    rng.shuffle(cycle)
    hours = sorted(rng.sample(range(5, 23), n))
    dist = distance_km(origin, dest)
    bucket = "short" if dist < 1800 else "medium" if dist < 5000 else "long"
    used: set[str] = set()
    slots = []
    for code, hour in zip(cycle, hours):
        number = f"{code}{rng.randint(101, 989)}"
        while number in used:
            number = f"{code}{rng.randint(101, 989)}"
        used.add(number)
        aircraft = "A380-800" if (code == "EK" and bucket == "long") else rng.choice(_AIRCRAFT[bucket])
        slots.append({
            "airline": code,
            "flight_number": number,
            "hour": hour,
            "minute": rng.choice(range(0, 60, 5)),
            "aircraft": aircraft,
        })
    return tuple(slots)


# --------------------------------------------------------------------------- #
# Weather model
# --------------------------------------------------------------------------- #
def compute_weather_score(temp_c, wind_kmh, visibility_km, precip_mm):
    """
    Collapse raw weather into a single 0-1 'operational severity' score.

        wind  : 0 below 25 km/h  -> 1 at 70 km/h
        vis   : 0 above 8 km     -> 1 at 0.5 km (fog / heavy snow)
        precip: 0 dry            -> 1 at >= 6 mm/h
        temp  : icing (< -2C) and extreme heat (> 38C) penalties

    Combined as a noisy-OR so independent hazards stack but never exceed 1.
    Works on scalars and numpy arrays.
    """
    wind_s = np.clip((np.asarray(wind_kmh, float) - 25.0) / 45.0, 0, 1)
    vis_s = np.clip((8.0 - np.asarray(visibility_km, float)) / 7.5, 0, 1)
    precip_s = np.clip(np.asarray(precip_mm, float) / 6.0, 0, 1)
    t = np.asarray(temp_c, float)
    temp_s = np.maximum(np.clip((-2.0 - t) / 18.0, 0, 1) * 0.8, np.clip((t - 38.0) / 10.0, 0, 1) * 0.5)
    score = 1.0 - (1 - 0.85 * wind_s) * (1 - 0.90 * vis_s) * (1 - 0.70 * precip_s) * (1 - temp_s)
    score = np.clip(score, 0, 1)
    return float(score) if np.ndim(score) == 0 else score


_WMO = {
    0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


def describe_weather(temp_c: float, wind_kmh: float, visibility_km: float, precip_mm: float,
                     code: int | None = None) -> str:
    """Human-readable condition string (WMO code when we have it, heuristics otherwise)."""
    if code is not None and code in _WMO:
        text = _WMO[code]
    elif visibility_km < 1.5:
        text = "Dense fog"
    elif precip_mm >= 4:
        text = "Heavy snow" if temp_c < 1 else "Heavy rain"
    elif precip_mm >= 0.5:
        text = "Snow" if temp_c < 1 else "Rain"
    elif precip_mm > 0:
        text = "Light rain"
    elif visibility_km < 5:
        text = "Mist / haze"
    else:
        text = "Fair"
    if wind_kmh >= 45 and "storm" not in text.lower():
        text += ", strong winds"
    return text


# name -> baseline climate (annual mean C, seasonal amplitude, rain prob, fog prob, mean wind km/h)
CLIMATES = {
    "monsoon":       dict(mean=27, amp=4,  rain=0.08, fog=0.02, wind=12),
    "monsoon_fog":   dict(mean=24, amp=9,  rain=0.08, fog=0.03, wind=11),
    "tropical":      dict(mean=28, amp=1.5, rain=0.30, fog=0.01, wind=12),
    "desert":        dict(mean=28, amp=10, rain=0.02, fog=0.04, wind=18),
    "oceanic":       dict(mean=11, amp=7,  rain=0.33, fog=0.06, wind=20),
    "continental":   dict(mean=10, amp=15, rain=0.20, fog=0.03, wind=20),
    "temperate":     dict(mean=14, amp=10, rain=0.20, fog=0.04, wind=15),
    "mediterranean": dict(mean=17, amp=8,  rain=0.12, fog=0.03, wind=15),
    "subtropical":   dict(mean=21, amp=7,  rain=0.25, fog=0.05, wind=15),
    "seafog":        dict(mean=15, amp=3,  rain=0.12, fog=0.10, wind=22),
    "typhoon":       dict(mean=22, amp=9,  rain=0.25, fog=0.03, wind=18),
}


def sample_weather(rng: np.random.Generator, ap: Airport, month: int) -> tuple[float, float, float, float]:
    """
    Draw one plausible (temp_c, wind_kmh, visibility_km, precip_mm) for an airport
    in a given month. Used for training data and as the offline fallback when the
    Open-Meteo API is unreachable.
    """
    p = CLIMATES[ap.climate]
    north = ap.lat >= 0
    season = math.cos(2 * math.pi * (month - 7) / 12)      # +1 in July, -1 in January
    season = season if north else -season
    eff = month if north else (month + 5) % 12 + 1          # hemisphere-adjusted month

    rain, fog, wind, storm = p["rain"], p["fog"], p["wind"], 0.0
    if ap.climate.startswith("monsoon") and 6 <= month <= 9:
        rain, storm = 0.55, 0.04
    if ap.climate == "monsoon_fog":
        fog = 0.28 if month in (12, 1) else 0.12 if month in (11, 2) else fog
    if ap.climate in ("tropical", "typhoon") and 5 <= month <= 10:
        rain += 0.20
        if ap.lat > 10 and 7 <= month <= 10:
            storm = 0.05
    if ap.climate in ("oceanic", "continental", "temperate") and eff in (12, 1, 2):
        fog += 0.05
        wind += 3
    if ap.climate in ("continental", "subtropical") and 6 <= eff <= 8:
        storm = 0.06                                           # convective summer storms
    if ap.climate == "seafog" and 6 <= eff <= 8:
        fog = 0.22

    temp = p["mean"] + p["amp"] * season + rng.normal(0, 3.5)
    raining = rng.random() < rain
    stormy = rng.random() < storm
    precip = (rng.gamma(2.0, 1.4) if raining else 0.0) + (rng.uniform(4, 12) if stormy else 0.0)
    vis = rng.uniform(8.0, 10.0)
    if raining:
        vis = min(vis, rng.uniform(3.5, 8.0))
    if rng.random() < fog:
        vis = rng.uniform(0.2, 3.0)
    wnd = rng.gamma(4.0, wind / 4.0) + (rng.uniform(25, 45) if stormy else 0.0)
    return float(temp), float(wnd), float(vis), float(precip)


# --------------------------------------------------------------------------- #
# Holidays (synthetic travel-peak windows)
# --------------------------------------------------------------------------- #
_HOLIDAY_WINDOWS = [
    ((1, 1), (1, 2)),      # New Year
    ((1, 25), (1, 26)),    # Republic Day (IN) / long weekend
    ((3, 13), (3, 15)),    # Spring festival
    ((7, 3), (7, 5)),      # Independence Day (US)
    ((8, 14), (8, 16)),    # Independence Day (IN) / summer break
    ((10, 28), (11, 3)),   # Festival-of-lights window
    ((11, 22), (11, 28)),  # Thanksgiving week
    ((12, 20), (12, 31)),  # Winter holidays
]


def is_holiday(month: int, day: int) -> int:
    for (m1, d1), (m2, d2) in _HOLIDAY_WINDOWS:
        if (month, day) >= (m1, d1) and (month, day) <= (m2, d2):
            return 1
    return 0


# --------------------------------------------------------------------------- #
# Ground-truth delay simulator
# --------------------------------------------------------------------------- #
_HOUR_WEIGHTS = np.array([0.3, 0.2, 0.15, 0.15, 0.3, 1.2, 3.0, 4.5, 5.0, 4.5, 4.0, 4.0,
                          4.0, 4.0, 4.0, 4.0, 4.5, 5.0, 5.5, 5.0, 4.5, 3.5, 2.5, 1.0])
_HOUR_WEIGHTS = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()
_DOW_EFFECT = np.array([0.05, -0.20, -0.20, 0.10, 0.35, -0.15, 0.30])        # Mon..Sun
_MONTH_EFFECT = np.array([0.35, 0.30, 0.05, -0.05, 0.00, 0.10, 0.15, 0.15, -0.10, -0.15, 0.00, 0.35])
_BASE_LOGIT = -3.1


def delay_logit(hour, dow, month, holiday, w_o, w_d, dist, airline_bias, cong_o, cong_d):
    """Hidden 'true' log-odds of a >15 min delay. Vectorised over numpy arrays."""
    hour = np.asarray(hour, float)
    peak_am = 0.45 * np.exp(-(((hour - 8.0) / 1.6) ** 2))
    peak_pm = 0.80 * np.exp(-(((hour - 18.5) / 2.2) ** 2))
    late = 0.30 * np.clip((hour - 20.0) / 3.0, 0, 1)
    early = -0.45 * (hour <= 6)
    hour_eff = peak_am + peak_pm + late + early
    return (
        _BASE_LOGIT
        + hour_eff
        + _DOW_EFFECT[np.asarray(dow, int)]
        + _MONTH_EFFECT[np.asarray(month, int) - 1]
        + 0.45 * np.asarray(holiday, float)
        + 2.6 * np.asarray(w_o, float)
        + 1.4 * np.asarray(w_d, float)
        + 0.30 * np.minimum(np.asarray(dist, float) / 10000.0, 1.0)
        + np.asarray(airline_bias, float)
        + 1.0 * np.asarray(cong_o, float)
        + 0.5 * np.asarray(cong_d, float)
    )


def generate_flights(n: int = 10_000, seed: int = 42,
                     origin: str | None = None, dest: str | None = None) -> pd.DataFrame:
    """
    Simulate `n` historical flights.

    With origin/dest given, every row is on that route (used for route history).
    Otherwise routes are sampled proportional to airport traffic weight.
    Returns FEATURE_COLUMNS + [airline, origin, destination, delayed, delay_minutes].
    """
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    if origin and dest:
        o_idx = np.full(n, AIRPORT_INDEX[origin])
        d_idx = np.full(n, AIRPORT_INDEX[dest])
    else:
        w = np.array([a.weight for a in AIRPORTS], float)
        w /= w.sum()
        o_idx = rng.choice(len(AIRPORTS), size=n, p=w)
        d_idx = rng.choice(len(AIRPORTS), size=n, p=w)
        same = o_idx == d_idx
        while same.any():
            d_idx[same] = rng.choice(len(AIRPORTS), size=int(same.sum()), p=w)
            same = o_idx == d_idx

    hour = rng.choice(24, size=n, p=_HOUR_WEIGHTS)
    dow = rng.integers(0, 7, size=n)
    month = rng.integers(1, 13, size=n)
    day = rng.integers(1, 29, size=n)
    holiday = np.array([is_holiday(int(m), int(d)) for m, d in zip(month, day)])

    airline_codes, wx_o, wx_d, dists = [], np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        ao, ad = AIRPORTS[o_idx[i]], AIRPORTS[d_idx[i]]
        airline_codes.append(pyrng.choice(route_airlines(ao.iata, ad.iata)))
        wx_o[i] = compute_weather_score(*sample_weather(rng, ao, int(month[i])))
        wx_d[i] = compute_weather_score(*sample_weather(rng, ad, int(month[i])))
        dists[i] = distance_km(ao.iata, ad.iata)

    bias = np.array([AIRLINE_BY_CODE[c].bias for c in airline_codes])
    cong_o = np.array([AIRPORTS[i].congestion for i in o_idx])
    cong_d = np.array([AIRPORTS[i].congestion for i in d_idx])

    logit = delay_logit(hour, dow, month, holiday, wx_o, wx_d, dists, bias, cong_o, cong_d)
    logit = logit + rng.normal(0, 0.25, size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    delayed = (rng.random(n) < prob).astype(int)

    # Delay magnitude: heavy-tailed when delayed (worse weather / worse odds -> longer),
    # tightly clustered around on-time otherwise.
    scale = 20.0 + 45.0 * wx_o + 25.0 * prob
    late_min = 15.0 + rng.gamma(1.5, scale)
    ok_min = np.clip(rng.normal(-1.0, 5.0, size=n), -10, 14)
    minutes = np.where(delayed == 1, late_min, ok_min)

    df = pd.DataFrame({
        "hour_of_day": hour,
        "day_of_week": dow,
        "month": month,
        "airline_id": [AIRLINE_INDEX[c] for c in airline_codes],
        "origin_airport_id": o_idx,
        "destination_airport_id": d_idx,
        "distance_km": dists,
        "weather_origin_score": wx_o.round(4),
        "weather_destination_score": wx_d.round(4),
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": holiday,
        "airline": airline_codes,
        "origin": [AIRPORTS[i].iata for i in o_idx],
        "destination": [AIRPORTS[i].iata for i in d_idx],
        "delayed": delayed,
        "delay_minutes": np.round(np.clip(minutes, 0, None), 1),
    })
    return df


@lru_cache(maxsize=256)
def route_history(origin: str, dest: str, n: int = 3000) -> pd.DataFrame:
    """Deterministic simulated history for one route (cached; do not mutate)."""
    seed = zlib.crc32(f"{origin}>{dest}".encode()) & 0x7FFFFFFF
    return generate_flights(n=n, seed=seed, origin=origin, dest=dest)


if __name__ == "__main__":  # quick sanity check:  python data.py
    d = generate_flights(10_000)
    print(d.head())
    print(f"delay rate: {d.delayed.mean():.1%}   mean delay: {d.delay_minutes.mean():.1f} min")
    print(d.groupby("hour_of_day").delayed.mean().round(2).to_dict())
    print(d.groupby("day_of_week").delayed.mean().round(2).to_dict())
