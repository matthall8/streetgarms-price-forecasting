"""Consolidates the Vinted and Shopify data"""
from pathlib import Path

import pandas as pd

SOURCES = [
    "data/interim/shopify_clean.csv",
    "data/interim/vinted_clean.csv",
    "data/interim/depop_clean.csv",
]
OUT = "data/interim/sales_combined.csv"


def combine(sources: list[str] = SOURCES, out: str = OUT) -> pd.DataFrame:
    dfs = [pd.read_csv(s) for s in sources if Path(s).exists()]
    df = pd.concat(dfs, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"combined {len(df)} rows from {len(dfs)} sources -> {out}")
    return df


if __name__ == "__main__":
    combine()
