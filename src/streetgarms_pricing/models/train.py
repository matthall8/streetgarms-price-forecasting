"""Fit the price model on ALL available data and persist it locally for serving.

compare.py evaluates on a time split; here we retrain on the full dataset so the
saved model uses every sale (including the most recent). The model trains on
log(price) internally but — via TransformedTargetRegressor — its predict()
returns actual £, so callers don't need to know about the log transform.

Run:  PYTHONPATH=src python -m streetgarms_pricing.models.train
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from streetgarms_pricing.features.build import build_preprocessor, load_xy
from streetgarms_pricing.models.split import time_split

DATASET = "data/interim/sales_combined.csv"
MODEL_PATH = Path("models/price_model.joblib")
CONF_ALPHA = 0.10   # 90% conformal band


def build_model() -> TransformedTargetRegressor:
    """Preprocess -> gradient boosting, with log/exp handled around the target."""
    inner = Pipeline(
        [("pre", build_preprocessor()), ("model", HistGradientBoostingRegressor(random_state=0))]
    )
    return TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp)


def conformal_factor(X, y_log, meta, alpha: float = CONF_ALPHA) -> float:
    """Multiplicative band factor. Split-conformal: fit on the time-split TRAIN, take the
    (1-alpha) quantile of |log residual| on the held-out TEST, return exp(q) — so a band
    is simply price / factor .. price * factor (keeps the log<->£ coupling explicit)."""
    tr = time_split(meta)
    cal = Pipeline([("pre", build_preprocessor()),
                    ("model", HistGradientBoostingRegressor(random_state=0))])
    cal.fit(X[tr], y_log[tr])
    resid = np.abs(y_log[~tr].to_numpy() - cal.predict(X[~tr]))
    return float(np.exp(np.quantile(resid, 1 - alpha)))


def train(out_path: Path = MODEL_PATH) -> TransformedTargetRegressor:
    X, y_log, meta = load_xy(DATASET)
    factor = conformal_factor(X, y_log, meta)   # calibrate BEFORE the full-data fit
    model = build_model()
    model.fit(X, np.exp(y_log))                 # deploy model fit on ALL rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "conformal_factor": factor, "alpha": CONF_ALPHA}, out_path)
    print(f"trained on {len(X)} rows -> {out_path}  (predict() returns £; 90% band ×/÷ {factor:.3f})")
    return model


if __name__ == "__main__":
    train()
