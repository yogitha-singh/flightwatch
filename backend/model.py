"""
FLIGHTWATCH · model.py
======================
ML pipeline
-----------
1. DATA      data.generate_flights(10_000) -> historical flights (11 features)
2. SPLIT     80 / 20 stratified train / test split
3. TRAIN     RandomForestClassifier  -> P(delay > 15 min)
             RandomForestRegressor   -> expected delay minutes
4. EVALUATE  accuracy, ROC-AUC, F1 (classifier) and MAE, R^2 (regressor)
5. EXPLAIN   Gini feature importances (global) + counterfactual attribution (per flight)
6. PERSIST   joblib bundle -> model.pkl (auto-trained on first start if missing)

Inference extras
----------------
* confidence  : agreement between the individual trees of the forest
                (low spread across trees -> high confidence)
* risk factors: for each feature group we swap the flight's value for values drawn
                from a background sample and measure how the probability moves.
                A positive delta means that factor is pushing delay risk UP.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data import (AIRLINE_BY_CODE, AIRLINE_INDEX, AIRPORT_BY_CODE, AIRPORT_INDEX, DAY_NAMES,
                  FEATURE_COLUMNS, MONTH_NAMES, distance_km, generate_flights)

log = logging.getLogger("flightwatch.model")

MODEL_PATH = Path(__file__).with_name("model.pkl")
MODEL_VERSION = "1.0.0"

# Feature groups used for per-flight attribution + importance roll-up
FEATURE_GROUPS: dict[str, list[str]] = {
    "hour": ["hour_of_day"],
    "day": ["day_of_week", "is_weekend"],
    "season": ["month"],
    "holiday": ["is_holiday"],
    "airline": ["airline_id"],
    "origin": ["origin_airport_id"],
    "destination": ["destination_airport_id"],
    "distance": ["distance_km"],
    "weather_origin": ["weather_origin_score"],
    "weather_destination": ["weather_destination_score"],
}
GROUP_CATEGORY = {
    "hour": "time", "day": "calendar", "season": "season", "holiday": "calendar",
    "airline": "airline", "origin": "airport", "destination": "airport",
    "distance": "distance", "weather_origin": "weather", "weather_destination": "weather",
}

# Risk bands (shared with the frontend gauge: green < 30 %, amber 30-60 %, red > 60 %)
def risk_level(prob: float) -> str:
    return "low" if prob < 0.30 else "medium" if prob < 0.60 else "high"


def status_label(prob: float) -> str:
    return {"low": "ON TIME", "medium": "AT RISK", "high": "DELAYED"}[risk_level(prob)]


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_feature_row(origin: str, dest: str, airline: str, hour: int, dow: int, month: int,
                      holiday: int, wx_origin: float, wx_dest: float) -> dict:
    return {
        "hour_of_day": int(hour),
        "day_of_week": int(dow),
        "month": int(month),
        "airline_id": AIRLINE_INDEX[airline],
        "origin_airport_id": AIRPORT_INDEX[origin],
        "destination_airport_id": AIRPORT_INDEX[dest],
        "distance_km": distance_km(origin, dest),
        "weather_origin_score": float(wx_origin),
        "weather_destination_score": float(wx_dest),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(holiday),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_save(path: Path = MODEL_PATH, n_rows: int = 10_000, seed: int = 42) -> dict:
    t0 = time.time()
    log.info("Training FLIGHTWATCH models on %s synthetic flights ...", f"{n_rows:,}")
    df = generate_flights(n=n_rows, seed=seed)
    # numpy arrays (not DataFrames) so training and inference share one calling convention
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_cls, y_reg = df["delayed"], df["delay_minutes"]

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.20, random_state=seed, stratify=y_cls
    )

    clf = RandomForestClassifier(
        n_estimators=160, max_depth=12, min_samples_leaf=8, max_features="sqrt",
        n_jobs=-1, random_state=seed,
    ).fit(X_tr, yc_tr)

    reg = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=8, max_features=0.6,
        n_jobs=-1, random_state=seed,
    ).fit(X_tr, yr_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    baseline_acc = max(yc_te.mean(), 1 - yc_te.mean())
    metrics = {
        "rows": int(n_rows),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "delay_rate": round(float(y_cls.mean()), 4),
        "accuracy": round(float(accuracy_score(yc_te, pred)), 4),
        "majority_baseline_accuracy": round(float(baseline_acc), 4),
        "roc_auc": round(float(roc_auc_score(yc_te, proba)), 4),
        "f1": round(float(f1_score(yc_te, pred)), 4),
        "mae_minutes": round(float(mean_absolute_error(yr_te, reg.predict(X_te))), 2),
        "r2": round(float(r2_score(yr_te, reg.predict(X_te))), 4),
    }

    # Global importance, rolled up to human-readable groups
    raw = dict(zip(FEATURE_COLUMNS, clf.feature_importances_))
    importance = {g: float(sum(raw[c] for c in cols)) for g, cols in FEATURE_GROUPS.items()}
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    # Background sample for counterfactual attribution
    background = X_tr[np.random.default_rng(seed).choice(len(X_tr), size=48, replace=False)]

    bundle = {
        "clf": clf, "reg": reg, "metrics": metrics, "importance": importance,
        "feature_importance_raw": {k: float(v) for k, v in raw.items()},
        "background": background, "features": FEATURE_COLUMNS,
        "version": MODEL_VERSION, "sklearn": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    joblib.dump(bundle, path, compress=3)
    log.info("Model saved -> %s  (AUC %.3f, acc %.3f, MAE %.1f min) in %.1fs",
             path, metrics["roc_auc"], metrics["accuracy"], metrics["mae_minutes"], time.time() - t0)
    return bundle


def load_or_train(path: Path = MODEL_PATH) -> "FlightDelayModel":
    """Load the cached model; (re)train if the file is missing, stale or corrupt."""
    bundle = None
    if path.exists():
        try:
            bundle = joblib.load(path)
            if bundle.get("version") != MODEL_VERSION or bundle.get("sklearn") != sklearn.__version__ \
                    or bundle.get("features") != FEATURE_COLUMNS:
                log.warning("model.pkl is from a different version / schema - retraining")
                bundle = None
        except Exception as exc:  # corrupt pickle, incompatible sklearn, etc.
            log.warning("Could not load model.pkl (%s) - retraining", exc)
            bundle = None
    if bundle is None:
        bundle = train_and_save(path)
    return FlightDelayModel(bundle)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    probability: float
    minutes: float
    confidence: float


class FlightDelayModel:
    def __init__(self, bundle: dict):
        self.clf: RandomForestClassifier = bundle["clf"]
        self.reg: RandomForestRegressor = bundle["reg"]
        self.clf.n_jobs = 1     # tiny batches: threads cost more than they save
        self.reg.n_jobs = 1
        self.metrics: dict = bundle["metrics"]
        self.importance: dict = bundle["importance"]
        self.background: np.ndarray = bundle["background"]
        self.version: str = bundle["version"]
        self.trained_at: str = bundle["trained_at"]
        self._col = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

    # ---- core predictions ------------------------------------------------- #
    @staticmethod
    def _to_matrix(rows: list[dict]) -> np.ndarray:
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_numpy(dtype=np.float32)

    def predict_batch(self, rows: list[dict]) -> list[Prediction]:
        X = self._to_matrix(rows)
        # per-tree probabilities -> mean = P(delay), spread = (dis)agreement
        tree_p = np.stack([t.predict_proba(X)[:, 1] for t in self.clf.estimators_])
        prob = tree_p.mean(axis=0)
        spread = tree_p.std(axis=0)
        confidence = np.clip(1.0 - 1.6 * spread, 0.35, 0.99)
        minutes = np.clip(self.reg.predict(X), 0, None)
        return [Prediction(float(p), float(m), float(c)) for p, m, c in zip(prob, minutes, confidence)]

    def predict_one(self, row: dict) -> Prediction:
        return self.predict_batch([row])[0]

    # ---- per-flight explanation ------------------------------------------- #
    def explain(self, row: dict, ctx: dict, top_k: int = 3) -> list[dict]:
        """
        Counterfactual attribution. For every feature group g:
            delta_g = P(flight) - mean_b P(flight with group g replaced by background row b)
        delta > 0  => factor increases delay risk.
        """
        x = self._to_matrix([row])[0]
        bg = self.background
        n_groups = len(FEATURE_GROUPS)
        batch = np.tile(x, (n_groups * len(bg) + 1, 1))
        batch[0] = x
        for gi, (_, cols) in enumerate(FEATURE_GROUPS.items()):
            lo = 1 + gi * len(bg)
            for c in cols:
                batch[lo:lo + len(bg), self._col[c]] = bg[:, self._col[c]]
        probs = self.clf.predict_proba(batch)[:, 1]
        base = probs[0]
        deltas = {}
        for gi, g in enumerate(FEATURE_GROUPS):
            lo = 1 + gi * len(bg)
            deltas[g] = float(base - probs[lo:lo + len(bg)].mean())

        ranked = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
        up = [kv for kv in ranked if kv[1] > 0.004]
        chosen = up[:top_k]
        if len(chosen) < top_k:                      # pad with strongest protective factors
            down = sorted((kv for kv in ranked if kv[1] <= 0.004), key=lambda kv: kv[1])
            chosen += down[: top_k - len(chosen)]

        biggest = max((abs(d) for _, d in chosen), default=1e-9) or 1e-9
        out = []
        for g, d in chosen:
            out.append({
                "factor": describe_factor(g, d, ctx),
                "category": GROUP_CATEGORY[g],
                "key": g,
                "direction": "increases" if d > 0 else "reduces",
                "impact_pts": round(d * 100, 1),
                "weight": round(min(abs(d) / biggest, 1.0), 3),
            })
        return out


# --------------------------------------------------------------------------- #
# Human-readable factor text
# --------------------------------------------------------------------------- #
def describe_factor(group: str, delta: float, ctx: dict) -> str:
    up = delta > 0
    h, dow, month = ctx["hour"], ctx["dow"], ctx["month"]
    o, d = ctx["origin"], ctx["dest"]
    if group == "hour":
        if 7 <= h <= 9:
            return f"Morning rush departure ({h:02d}:00)" if up else f"Morning departure ({h:02d}:00)"
        if 16 <= h <= 20:
            return f"Evening departure (peak hour, {h:02d}:00)" if up else f"Evening departure ({h:02d}:00)"
        if h >= 21:
            return f"Late departure ({h:02d}:00) - accumulated knock-on delays" if up else f"Late departure ({h:02d}:00)"
        if h <= 6:
            return f"Early departure ({h:02d}:00)" if up else f"Early departure ({h:02d}:00) - quiet airspace"
        return f"Midday departure ({h:02d}:00)" if up else f"Off-peak departure ({h:02d}:00)"
    if group == "day":
        name = DAY_NAMES[dow]
        if up:
            return {4: "Friday departure (busiest travel day)", 6: "Sunday departure (return-travel surge)"}.get(dow, f"{name} departure")
        return f"{name} departure (lighter traffic)"
    if group == "season":
        if month in (12, 1, 2):
            return f"Winter operations ({MONTH_NAMES[month - 1]})" if up else f"Mild-season operations ({MONTH_NAMES[month - 1]})"
        if month in (6, 7, 8):
            return f"Summer storm & peak-travel season ({MONTH_NAMES[month - 1]})" if up else f"Summer season ({MONTH_NAMES[month - 1]})"
        return f"Seasonal pattern ({MONTH_NAMES[month - 1]})" if up else f"Calm season ({MONTH_NAMES[month - 1]})"
    if group == "holiday":
        return "Holiday travel period" if up else "No holiday congestion"
    if group == "airline":
        name = AIRLINE_BY_CODE[ctx["airline"]].name
        return f"{name} - below-average punctuality" if up else f"{name} - strong punctuality record"
    if group == "origin":
        return f"{o} departure congestion" if up else f"{o} runs efficient departures"
    if group == "destination":
        return f"{d} arrival congestion" if up else f"{d} handles arrivals smoothly"
    if group == "distance":
        km = f"{ctx['distance']:,.0f} km"
        return f"Long sector ({km}) - more exposure to knock-ons" if up else f"Short sector ({km})"
    if group == "weather_origin":
        return f"Weather at {o}: {ctx['wx_origin']}" if up else f"Benign weather at {o}"
    if group == "weather_destination":
        return f"Weather at {d}: {ctx['wx_dest']}" if up else f"Benign weather at {d}"
    return group


if __name__ == "__main__":  # python model.py  -> (re)train and print metrics
    logging.basicConfig(level=logging.INFO)
    b = train_and_save()
    print(b["metrics"])
    print({k: round(v, 3) for k, v in b["importance"].items()})
