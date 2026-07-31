"""Concatenate the cleaned per-source interim files into one training dataset.

Each source (Shopify, Vinted, …) is cleaned to the shared schema separately and
carries a `platform` column; this stacks them into the file the models read.
"""
from pathlib import Path

import pandas as pd

SOURCES = [
    "data/interim/sales_clean.csv",     # shopify
    "data/interim/vinted_clean.csv",    # vinted
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
