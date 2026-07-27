"""Price a single item from its parameters using the persisted model.

As a function:
    from streetgarms_pricing.models.predict import predict
    predict(product_type="jacket", brand="Stone Island", condition_grade="Fantastic")

As a CLI (from repo root):
    PYTHONPATH=src python -m streetgarms_pricing.models.predict \
        --product-type jacket --brand "Stone Island" --size L \
        --colour Black --condition Fantastic

The model returns a point estimate in £. TYPICAL_ERROR is the benchmark median
error (~14%) — shown as a rough band so the number reads as an estimate. It is
NOT calibrated per-item; treat wide-ranging / unusual pieces as low-confidence.
"""
import argparse
from functools import lru_cache

import joblib
import pandas as pd

from streetgarms_pricing.features.build import NOMINAL
from streetgarms_pricing.models.train import MODEL_PATH

TYPICAL_ERROR = 0.14  # benchmark MedAPE; rough band only

# Sensible defaults for a men's-default streetwear shop (fill any omitted field).
DEFAULTS = {
    "brand": "Stone Island",
    "product_type": "jacket",
    "size": "L",
    "colour": "Black",
    "gender": "Mens",
    "condition_grade": "Fantastic",
}


@lru_cache(maxsize=1)
def _load_model(path: str = str(MODEL_PATH)):
    return joblib.load(path)


def predict(**features) -> float:
    """Predict sale price (£) for one item. Unspecified fields use DEFAULTS."""
    row = {col: features.get(col) or DEFAULTS[col] for col in NOMINAL}
    row["product_type"] = str(row["product_type"]).strip().lower()  # match training casing
    return float(_load_model().predict(pd.DataFrame([row]))[0])


def _cli() -> None:
    p = argparse.ArgumentParser(description="Predict a resale price from item parameters.")
    p.add_argument("--brand", default=DEFAULTS["brand"])
    p.add_argument("--product-type", dest="product_type", default=DEFAULTS["product_type"])
    p.add_argument("--size", default=DEFAULTS["size"])
    p.add_argument("--colour", default=DEFAULTS["colour"])
    p.add_argument("--gender", default=DEFAULTS["gender"])
    p.add_argument("--condition", dest="condition_grade", default=DEFAULTS["condition_grade"])
    args = vars(p.parse_args())

    price = predict(**args)
    lo, hi = price * (1 - TYPICAL_ERROR), price * (1 + TYPICAL_ERROR)
    desc = ", ".join(f"{k}={v}" for k, v in args.items())
    print(f"≈ £{price:.0f}   (typically £{lo:.0f}–£{hi:.0f})")
    print(f"   {desc}")


if __name__ == "__main__":
    _cli()
