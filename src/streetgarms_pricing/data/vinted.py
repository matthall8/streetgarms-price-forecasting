"""Parse a Vinted sales export into cleaned interim rows in the shared schema.

Vinted CSV columns: Date, Brand, Product, Category, Sold for (GBP).
Maps to: sold_at, brand, product_name, product_type, sold_price, platform=vinted.
Fields Vinted doesn't provide (size/colour/gender/condition_grade) -> NA -> Unknown
downstream. Titles are the same rich style as Shopify, so the token feature and
brand/product_type carry the signal.
"""
from pathlib import Path

import pandas as pd

from streetgarms_pricing.features.build import _parse_type  # shared garment-type parser

PLATFORM = "vinted"


def parse_vinted(
    in_path: str = "data/raw/vinted_sales_data.csv",
    out_path: str = "data/interim/vinted_clean.csv",
) -> pd.DataFrame:
    v = pd.read_csv(in_path)
    out = pd.DataFrame({
        "sold_at": pd.to_datetime(v["Date"], dayfirst=True, errors="coerce", utc=True),
        "sold_price": v["Sold for (GBP)"].str.replace(r"[£,]", "", regex=True).astype(float),
        "brand": v["Brand"].astype("string").str.strip(),
        "product_name": v["Product"].astype("string").str.strip(),
        "product_type": v["Category"].map(_parse_type),
        "platform": PLATFORM,
    })
    for col in ("size", "colour", "gender", "condition_grade"):
        out[col] = pd.NA                       # not provided by Vinted
    out = out[out["sold_price"].notna() & (out["sold_price"] > 0)].copy()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} vinted rows -> {out_path}")
    return out


if __name__ == "__main__":
    parse_vinted()
