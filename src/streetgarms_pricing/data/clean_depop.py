"""Parse the Depop GDPR export into cleaned interim rows in the shared schema.

Joins purchase_items_as_seller (product + price) to completed_purchases_as_seller
(purchase_id -> completed_at date). product_name = the FIRST LINE of the snapshot
description (the rest is SEO tag-spam and must not be tokenised). Depop provides no
structured size/colour/condition; brand is prefix-matched from the title, and
product_type is derived from the title centrally in the feature step.
"""
import json
from pathlib import Path

import pandas as pd

PLATFORM = "depop"

# Canonical brand roster (stable) — titles start with the brand.
BRANDS = [
    "The North Face", "Stone Island", "Canada Goose", "Palm Angels", "CP Company",
    "Arc'teryx", "Supreme", "Moncler", "Palace", "Stussy", "Prada", "Nike", "RAB",
    "Burberry",
]
ALIASES = {"north face": "The North Face", "arcteryx": "Arc'teryx"}


def _brand_of(title: str):
    t = title.strip().lower().replace("’", "'")
    for b in sorted(BRANDS, key=len, reverse=True):
        if t.startswith(b.lower()):
            return b
    for alias, canon in ALIASES.items():
        if t.startswith(alias):
            return canon
    return pd.NA


def parse_depop(
    in_dir: str = "data/raw/DEPOP_TOTAL_DATA",
    out_path: str = "data/interim/depop_clean.csv",
) -> pd.DataFrame:
    items = json.load(open(f"{in_dir}/purchase_items_as_seller.json"))
    completed = json.load(open(f"{in_dir}/completed_purchases_as_seller.json"))
    date_by_pid = {r["purchase_id"]: r["completed_at"] for r in completed}

    rows = []
    for r in items:
        pid = r["purchase_id"]
        if pid not in date_by_pid:            # keep only completed (dated) sales
            continue
        desc = json.loads(r["product_snapshot"]).get("description", "")
        title = desc.split("\n")[0].strip()
        rows.append({
            "sold_at": date_by_pid[pid],
            "sold_price": r["product_price"],
            "product_name": title,
            "brand": _brand_of(title),
            "platform": PLATFORM,
            "gender": "Mens",                 # men's-default shop
        })

    out = pd.DataFrame(rows)
    out["sold_at"] = pd.to_datetime(out["sold_at"], utc=True, errors="coerce")
    out = out[out["sold_price"].notna() & (out["sold_price"] > 1)]     # drop £1 junk
    out = out.drop_duplicates(["sold_at", "product_name", "sold_price"]).copy()
    for col in ("product_type", "size", "colour", "condition_grade"):
        out[col] = pd.NA                       # not provided; product_type from title later

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} depop rows -> {out_path}")
    return out


if __name__ == "__main__":
    parse_depop()
