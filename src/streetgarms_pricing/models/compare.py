"""Benchmark bake-off: linear baseline vs gradient boosting, on the time split.

Run from the repo root:
    PYTHONPATH=src python -m streetgarms_pricing.models.compare
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from streetgarms_pricing.features.build import build_preprocessor, load_xy
from streetgarms_pricing.models.metrics import report
from streetgarms_pricing.models.split import time_split

MODELS = {
    "naive (median)": DummyRegressor(strategy="median"),
    "ridge (baseline)": Ridge(alpha=1.0),
    "hist gradient boosting": HistGradientBoostingRegressor(random_state=0),
}


def run() -> None:
    X, y, meta = load_xy()
    train = time_split(meta)
    print(f"train {int(train.sum())} | test {int((~train).sum())}  "
          f"(time split, log target)\n")

    header = f"{'model':24s} {'RMSE(log)':>10s} {'MAPE':>8s} {'MedAPE':>8s}"
    print(header)
    print("-" * len(header))
    for name, model in MODELS.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("model", model)])
        pipe.fit(X[train], y[train])
        m = report(y[~train].to_numpy(), pipe.predict(X[~train]))
        print(f"{name:24s} {m['rmse_log']:10.3f} {m['mape']:8.1%} {m['median_ape']:8.1%}")


if __name__ == "__main__":
    run()
