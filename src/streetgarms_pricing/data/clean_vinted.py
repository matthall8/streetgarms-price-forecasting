"""Cleans the Vinted sales export into cleaned interim rows in the shared schema."""
from pathlib import Path

import pandas as pd

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
        "product_type": v["Category"],          # canonicalised centrally in add_product_type
        "platform": PLATFORM,
        "gender": "Mens",                       # men's-default shop
    })
    for col in ("size", "colour", "condition_grade"):
        out[col] = pd.NA                       # not provided by Vinted
    out = out[out["sold_price"].notna() & (out["sold_price"] > 0)].copy()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} vinted rows -> {out_path}")
    return out


if __name__ == "__main__":
    parse_vinted()
