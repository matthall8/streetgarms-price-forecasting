"""Clean raw Shopify sales -> tidy interim dataset. Moves the CSV from raw (data/raw/) -> interim (data/interim/)"""
import re
from pathlib import Path

import pandas as pd

DATE_COLS = ["sold_at", "listed_at", "published_at"]

# Known-misspelling / variant fixes
BRAND_FIXES: dict[str, str] = {
    "stone isand": "Stone Island",
    "stone island ": "Stone Island",
    "stoen island": "Stone Island",
    "Stoen Island": "Stone Island",
    "Arc‚Äôteryx": "Arc'teryx",
    "rab": "RAB",                 
}

# Known condition_grade misspellings. 
CONDITION_FIXES: dict[str, str] = {
    "fanatstic": "Fantastic",
    "fantatsic": "Fantastic",
}

# Consolidate granular / rare colours (<5 quantity) into base group
COLOUR_MAP: dict[str, str] = {
    "burgandy": "Burgundy", "biege": "Beige",
    # blues (+ navy variant)
    "royal blue": "Blue", "light blue": "Blue", "pale blue": "Blue",
    "electric blue": "Blue", "denim blue": "Blue", "mint blue": "Blue",
    "baby blue": "Blue", "aqua": "Blue", "teal": "Blue", "dark navy": "Navy",
    # greens
    "pale green": "Green", "neon green": "Green", "mint green": "Green",
    "olive green": "Green", "lime green": "Green", "sage green": "Green",
    "dull green": "Green", "matcha green": "Green", "military green": "Green",
    "mint": "Green", "olive": "Green", "pistachio": "Green",
    # khaki family
    "khaki green": "Khaki", "khaki brown": "Khaki", "light khaki": "Khaki",
    # browns / earth neutrals
    "coal brown": "Brown", "dark brown": "Brown", "rust": "Brown",
    "tan": "Beige", "sand": "Beige", "stone": "Beige",
    # greys
    "dark grey": "Grey", "pearl grey": "Grey", "slate grey": "Grey",
    "charcoal black": "Grey", "silver": "Grey",
    # whites
    "off white": "White", "pearl": "White",
    # pinks
    "dusty pink": "Pink", "peach": "Pink", "pale peach": "Pink", "coral": "Pink",
    # purples
    "violet": "Purple", "lilac": "Purple", "plum": "Purple",
    # reds
    "maroon": "Burgundy", "brick red": "Red",
    # oranges / yellows
    "burnt orange": "Orange", "lemon yellow": "Yellow", "mustard": "Yellow",
    # multi / junk
    "camo": "Multi", "labryinth": "Multi", "blue/white": "Multi", "medium": "Multi",
}

# Canonical clothing sizes + splits out gender 
CANON_SIZE: dict[str, str] = {
    "xs": "XS", "x-small": "XS",
    "s": "S", "small": "S",
    "m": "M", "medium": "M",
    "l": "L", "large": "L",
    "xl": "XL", "x-large": "XL",
    "xxl": "2XL", "2xl": "2XL", "xx-large": "2XL",
    "xxxl": "3XL", "3xl": "3XL", "xxx-large": "3XL",
    "4xl": "4XL",
    "one size": "One Size", "os": "One Size",
}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sku": str, "art_number": str})
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df

def blanks_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """Empty / whitespace-only strings -> NA, so missingness is consistent."""
    obj = df.select_dtypes("object").columns
    df[obj] = df[obj].apply(lambda s: s.str.strip().replace("", pd.NA))
    return df


def drop_invalid_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target must be a real positive price (EDA: floor ~£18, so this is a guard)."""
    before = len(df)
    df = df[df["sold_price"].notna() & (df["sold_price"] > 0)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} rows with null/<=0 sold_price")
    return df


def _apply_map(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> None:
    """Case-insensitive deterministic remap; unmapped values pass through."""
    if col in df.columns:
        s = df[col].astype("string").str.strip().str.replace("’", "'", regex=False)
        df[col] = s.str.lower().map(mapping).fillna(s)


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fix misspellings / consolidate variants."""
    _apply_map(df, "brand", BRAND_FIXES)
    _apply_map(df, "condition_grade", CONDITION_FIXES)
    _apply_map(df, "colour", COLOUR_MAP)
    return df


def parse_gender(raw) -> str:
    """Men's-default shop: Womens only if 'women' is explicit, else Mens."""
    if pd.notna(raw) and "women" in str(raw).lower():
        return "Womens"
    return "Mens"


def parse_size(raw):
    """'Womens Large' -> 'L'; '52 (XL)' -> 'XL'; '30W' -> 'other'. Deterministic."""
    if pd.isna(raw):
        return pd.NA
    t = str(raw).strip().lower()

    # prefer an alpha size inside parens, e.g. "52 (XL)" -> XL
    paren = re.search(r"\(([^)]*)\)", t)
    if paren and paren.group(1).strip() in CANON_SIZE:
        return CANON_SIZE[paren.group(1).strip()]

    # strip gender words / junior markers / parens, then look up
    cleaned = re.sub(r"\b(mens|womens|women|men|junior|jr)\b|/", " ", t)
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return pd.NA
    return CANON_SIZE.get(cleaned, "other")


def normalise_size(df: pd.DataFrame) -> pd.DataFrame:
    """Split size into gender (men's-default) + canonical size."""
    if "size" in df.columns:
        df["gender"] = df["size"].map(parse_gender)
        df["size"] = df["size"].map(parse_size)
    return df


def neutralise_import_spike(df: pd.DataFrame, min_share: float = 0.10) -> pd.DataFrame:
    """OPTIONAL: blank listed_at on the bulk-import date (untrustworthy).

    Detects a single date holding an outsized share of listed_at values and
    sets those to NaT. Skip/remove if you dropped the listed-at signal.
    """
    if "listed_at" not in df.columns:
        return df
    by_date = df["listed_at"].dt.date.value_counts(normalize=True)
    if len(by_date) and by_date.iloc[0] >= min_share:
        spike = by_date.index[0]
        df.loc[df["listed_at"].dt.date == spike, "listed_at"] = pd.NaT
        print(f"neutralised import-spike listed_at date: {spike}")
    return df


def clean(
    in_path: str = "data/raw/shopify_sales_data_complete_data.csv",
    out_path: str = "data/interim/shopify_clean.csv",
) -> pd.DataFrame:
    """Full clean function to apply all helper functions"""
    df = load_raw(in_path)
    df = blanks_to_na(df)
    df = drop_invalid_target(df)
    df = normalise_categoricals(df)
    df = normalise_size(df)
    df = neutralise_import_spike(df)   # optional — comment out if unwanted
    df["platform"] = "shopify"         # source tag for multi-platform training

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{len(df)} clean rows -> {out_path}")
    return df


if __name__ == "__main__":
    clean()
