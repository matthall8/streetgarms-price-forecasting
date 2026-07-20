# Config

Non-secret project config goes here (e.g. `settings.yaml`).

**Secrets** — Shopify API token, shop domain — belong in a `.env` file at the
repo root (gitignored), not in this folder. Suggested keys:

```
SHOPIFY_SHOP=your-shop.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_...
SHOPIFY_API_VERSION=2025-07
```

Reminder: reading full order history needs the `read_all_orders` scope,
otherwise the API only returns the last 60 days.
