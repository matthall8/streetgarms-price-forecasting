"""Build model-ready features from cleaned interim data.

interim (data/interim/) -> feature matrix X + log target y.

Leakage discipline is structural:
- The sklearn preprocessor (rare-category collapse + encoders) is FIT ON TRAIN
  ONLY, then applied to test. Rare categories and unseen test values collapse
  into an "infrequent" bucket rather than leaking or crashing.
- ColumnTransformer(remainder="drop") means only the declared feature columns
  reach the model — list_price, days-on-market, tags, etc. can't leak in.
- Deterministic row-wise prep (product_type coalesce, measurement parsing) runs
  before the split and is leakage-safe.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "sold_price"
TIME_COL = "sold_at"

# --- feature groups ----------------------------------------------------------
# condition_grade is ONE-HOT, not ordinal: verified non-monotonic vs price
# (Brand New n=8 sits BELOW Great/Fantastic/Like New), so no clean order to impose.
NOMINAL = ["brand", "product_type", "size", "colour", "gender", "condition_grade"]
MEASUREMENT_KEYS = {   # raw label in `measurements` -> numeric feature name
    "pit to pit": "pit_to_pit",
    "shoulder hem to bottom": "shoulder_to_hem",
    "waist": "waist",
    "inside leg": "inside_leg",
}
# release_year is sparse (~22% of rows carry a season); NaN elsewhere.
NUMERIC = list(MEASUREMENT_KEYS.values()) + ["release_year"]

# --- product_type fallback parser (coalesce Shopify field w/ product_name) ---
PRODUCT_TYPES = {
    "sweatshirt": ["sweatshirt", "crewneck"],
    "overshirt": ["overshirt"],
    "hoodie": ["hoodie", "hooded"],
    "jumper": ["jumper", "sweater", "knit", "cardigan"],
    "gilet": ["gilet"],
    "jacket": ["jacket", "coat", "parka", "puffer"],
    "tshirt": ["t-shirt", "t shirt", "tee", "jersey"],
    "shirt": ["shirt"],
    "shorts": ["shorts"],
    "trousers": ["cargo", "trousers", "pants", "sweatpants", "joggers"],
}


def _parse_type(name) -> str:
    n = str(name).lower()
    for canonical, kws in PRODUCT_TYPES.items():
        if any(k in n for k in kws):
            return canonical
    return "other"


def add_product_type(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce Shopify productType with a product_name parse fallback."""
    pt = df["product_type"].astype("string").str.strip().str.lower()
    missing = pt.isna() | (pt == "")
    pt = pt.where(~missing, df["product_name"].map(_parse_type))
    df["product_type"] = pt.fillna("unknown")
    return df


def add_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'pit to pit=22;waist=32' -> numeric cols. Row-wise, deterministic."""
    def parse(s: object) -> dict:
        out = {}
        for part in str(s).split(";"):
            key, sep, val = part.partition("=")
            if sep and key.strip().lower() in MEASUREMENT_KEYS:
                try:
                    out[MEASUREMENT_KEYS[key.strip().lower()]] = float(val)
                except ValueError:
                    pass
        return out

    parsed = df["measurements"].map(parse).apply(pd.Series)
    for col in NUMERIC:
        df[col] = parsed[col] if col in parsed.columns else np.nan
    return df


def add_season_year(df: pd.DataFrame) -> pd.DataFrame:
    """Season 'A/W 16' -> release_year 2016 (numeric recency). NaN if no season."""
    yr = df["season"].astype("string").str.extract(r"(\d{2})")[0]
    df["release_year"] = pd.to_numeric(yr, errors="coerce") + 2000
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise feature prep (safe to run before the split)."""
    df = df.copy()
    df = add_product_type(df)
    df = add_measurements(df)
    df = add_season_year(df)
    df[NOMINAL] = df[NOMINAL].fillna("Unknown")
    return df


def build_preprocessor(min_frequency: int = 5) -> ColumnTransformer:
    """sklearn preprocessor — FIT ON TRAIN ONLY.

    Categories rarer than `min_frequency` in train collapse into one bucket;
    unseen test categories route there too. Missing measurements -> median.
    """
    nominal = OneHotEncoder(
        min_frequency=min_frequency,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("nominal", nominal, NOMINAL),
            ("numeric", numeric, NUMERIC),
        ],
        remainder="drop",
    )


def load_xy(in_path: str = "data/interim/sales_clean.csv"):
    """Load interim -> (X features, y = log(sold_price), meta with sold_at).

    Returns the raw feature frame X (un-encoded) so the caller can time-split
    first, THEN fit the preprocessor on train only. meta[TIME_COL] drives the
    split (see models/split.py).
    """
    df = pd.read_csv(in_path, parse_dates=[TIME_COL])
    df = prepare(df)
    X = df[NOMINAL + NUMERIC]
    y = np.log(df[TARGET])
    meta = df[[TIME_COL]]
    return X, y, meta


if __name__ == "__main__":
    X, y, meta = load_xy()
    print(f"X: {X.shape}  y: {y.shape}")
    print("feature columns:", list(X.columns))
    print("date range:", meta[TIME_COL].min(), "->", meta[TIME_COL].max())
