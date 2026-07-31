"""Build model-ready features from cleaned interim data.

interim (data/interim/) -> feature matrix X + log target y.

Leakage discipline is structural:
- The sklearn preprocessor (rare-category collapse + one-hot) is FIT ON TRAIN
  ONLY, then applied to test. Rare categories and unseen test values collapse
  into an "infrequent" bucket rather than leaking or crashing.
- ColumnTransformer(remainder="drop") means only the declared feature columns
  reach the model — list_price, days-on-market, tags, etc. can't leak in.
- Deterministic row-wise prep (product_type coalesce) runs before the split.

Features are all categorical. Measurements and release_year were dropped after
permutation importance showed them to be noise (zero/negative on the test set).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

TARGET = "sold_price"
TIME_COL = "sold_at"

# condition_grade is one-hot, not ordinal: verified non-monotonic vs price
# (Brand New n=8 sits BELOW Great/Fantastic/Like New), so no clean order to impose.
NOMINAL = ["brand", "product_type", "size", "colour", "gender", "condition_grade", "platform"]
# free-text product name -> tokenised; fabric/model terms carry price signal.
TEXT_COL = "product_name"

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise feature prep (safe to run before the split)."""
    df = df.copy()
    for col in NOMINAL:                         # tolerate sources missing a column
        if col not in df.columns:
            df[col] = pd.NA
    df = add_product_type(df)
    df[NOMINAL] = df[NOMINAL].fillna("Unknown")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def build_preprocessor(min_frequency: int = 5, min_title_df: int = 10) -> ColumnTransformer:
    """sklearn preprocessor — FIT ON TRAIN ONLY.

    Categories rarer than `min_frequency` in train collapse into one bucket;
    unseen test categories route there too. product_name tokens appearing in
    fewer than `min_title_df` train rows are dropped — the guard against
    overfitting the near-unique names.
    """
    nominal = OneHotEncoder(
        min_frequency=min_frequency,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    title = CountVectorizer(binary=True, min_df=min_title_df)
    return ColumnTransformer(
        [
            ("nominal", nominal, NOMINAL),
            ("title", title, TEXT_COL),
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
    X = df[NOMINAL + [TEXT_COL]]
    y = np.log(df[TARGET])
    meta = df[[TIME_COL]]
    return X, y, meta


if __name__ == "__main__":
    X, y, meta = load_xy()
    print(f"X: {X.shape}  y: {y.shape}")
    print("feature columns:", list(X.columns))
    print("date range:", meta[TIME_COL].min(), "->", meta[TIME_COL].max())
