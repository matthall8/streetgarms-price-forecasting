import os
import re
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

STORE = os.getenv("SHOPIFY_STORE")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

MEASURE_RE = re.compile(
    r"([A-Za-z][A-Za-z ]{2,30}?):\s*([\d.]+)\s*(?:inch|in\b|\")", re.I
)

SALES_QUERY = """
query Sales($cursor: String) {
  orders(first: 100, after: $cursor, sortKey: CREATED_AT, reverse: true,
         query: "financial_status:paid") {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        name
        createdAt
        lineItems(first: 25) {
          edges {
            node {
              title
              sku
              discountedUnitPriceSet { shopMoney { amount } }
              variant {
                price
                selectedOptions { name value }
              }
              product {
                productType
                createdAt
                publishedAt
                vendor
                tags
                metafields(first: 20, namespace: "custom") {
                  edges { node { key value } }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def get_token() -> str:
    resp = requests.post(
        f"https://{STORE}.myshopify.com/admin/oauth/access_token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def gql(query: str, variables: dict, token: str) -> dict:
    """One GraphQL call with retry on throttling."""
    for attempt in range(5):
        resp = requests.post(
            f"https://{STORE}.myshopify.com/admin/api/2025-07/graphql.json",
            json={"query": query, "variables": variables},
            headers={
                "X-Shopify-Access-Token": token,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            if any("THROTTLED" in str(e) for e in payload["errors"]):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        return payload["data"]
    raise RuntimeError("Retries exhausted")


def normalise_condition(text: str) -> str:
    return re.sub(r"\bcondition\b", "", text, flags=re.I).strip(" .-").lower()


def parse_title(title: str, vendor: str = "") -> dict:
    out = {"season": "", "product_name": ""}
    work = title.strip()
    if vendor and work.lower().startswith(vendor.lower()):
        work = work[len(vendor):].strip()
    m = re.search(r"\b[AS]/[SW]\s?\d{2}\b", work)
    if m:
        out["season"] = m.group(0)
        work = (work[: m.start()] + work[m.end():]).strip()
    if " - " in work and "/" in work.rpartition(" - ")[2]:
        work = work.rpartition(" - ")[0].strip()
    out["product_name"] = re.sub(r"\s{2,}", " ", work)
    return out


def line_item_to_row(order_node: dict, li_node: dict) -> dict:
    prod = li_node.get("product") or {}
    variant = li_node.get("variant") or {}
    mf = {
        e["node"]["key"]: e["node"]["value"] or ""
        for e in (prod.get("metafields") or {}).get("edges", [])
    }
    cond_full = mf.get("condition", "")
    cond_grade = mf.get("short_condition", "")
    meas = ";".join(
        f"{name.strip().lower()}={val}"
        for name, val in MEASURE_RE.findall(mf.get("short_description", ""))
        if name.strip().lower() not in ("art number", "season released")
    )
    row = {
        "order": order_node["name"],
        "sold_at": order_node["createdAt"],
        "sold_price": float(li_node["discountedUnitPriceSet"]["shopMoney"]["amount"]),
        "list_price": float(variant["price"]) if variant.get("price") else None,
        "title": li_node["title"],
        "sku": li_node.get("sku") or "",
        "brand": prod.get("vendor", "") or "",
        "product_type": prod.get("productType") or "",
        **parse_title(li_node["title"], prod.get("vendor", "") or ""),
        "listed_at": prod.get("createdAt") or "",
        "published_at": prod.get("publishedAt") or "",
        "condition_grade": cond_grade,
        "condition_full": cond_full,
        "has_defect_note": bool(
            normalise_condition(cond_full)
            and normalise_condition(cond_grade)
            and normalise_condition(cond_full) != normalise_condition(cond_grade)
        ),
        "art_number": mf.get("art_numbers", ""),
        "measurements": meas,
        "tags": ",".join(prod.get("tags") or []),
        "size": "",
        "colour": "",
    }
    for opt in variant.get("selectedOptions") or []:
        name = opt["name"].strip().lower()
        if name == "size":
            row["size"] = opt["value"]
        elif name in ("colour", "color"):
            row["colour"] = opt["value"]
    if not row["colour"] and mf.get("colour"):
        try:
            row["colour"] = json.loads(mf["colour"])[0]
        except (ValueError, IndexError):
            pass
    return row


def generate_sales_data(out_path: str = "./data/sales.csv") -> pd.DataFrame:
    token = get_token()
    rows = []
    cursor = None
    with tqdm(desc="Pulling orders", unit=" pages") as bar:
        while True:
            data = gql(SALES_QUERY, {"cursor": cursor}, token)
            conn = data["orders"]
            for edge in conn["edges"]:
                for li in edge["node"]["lineItems"]["edges"]:
                    rows.append(line_item_to_row(edge["node"], li["node"]))
            bar.update(1)
            bar.set_postfix(items=len(rows))
            if not conn["pageInfo"]["hasNextPage"]:
                break
            cursor = conn["pageInfo"]["endCursor"]

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{len(df)} sold items -> {out_path}")
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    pd.set_option("display.max_columns", None)
    print(df.head(15))