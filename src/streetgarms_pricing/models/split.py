"""Time-based train/test split — train on the past, test on the future.

NOT random k-fold: EDA found ~25-30% price drift across the window, so a random
split would leak future price levels into training. Boundary is the 80th-pct
sold_at from the full EDA set.
"""
import pandas as pd

DEFAULT_BOUNDARY = "2025-12-14"
TIME_COL = "sold_at"


def time_split(meta: pd.DataFrame, boundary: str = DEFAULT_BOUNDARY) -> pd.Series:
    """Boolean TRAIN mask: rows on/before `boundary` are train, after are test."""
    return meta[TIME_COL] <= pd.Timestamp(boundary, tz="UTC")
