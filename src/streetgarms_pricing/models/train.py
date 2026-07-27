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

MODEL_PATH = Path("models/price_model.joblib")


def build_model() -> TransformedTargetRegressor:
    """Preprocess -> gradient boosting, with log/exp handled around the target."""
    inner = Pipeline(
        [("pre", build_preprocessor()), ("model", HistGradientBoostingRegressor(random_state=0))]
    )
    return TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp)


def train(out_path: Path = MODEL_PATH) -> TransformedTargetRegressor:
    X, y_log, _ = load_xy()
    y_price = np.exp(y_log)              # load_xy returns log target; undo for the wrapper
    model = build_model()
    model.fit(X, y_price)               # fit on ALL rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"trained on {len(X)} rows -> {out_path}  (predict() returns £)")
    return model


if __name__ == "__main__":
    train()
