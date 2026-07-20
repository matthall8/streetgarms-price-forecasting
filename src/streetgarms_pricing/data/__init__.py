"""Data acquisition and cleaning.

Responsible for pulling raw sales history from Shopify and turning it into a
clean, tidy dataset. Suggested modules:

- extract.py : pull line-item-level order history from the Shopify GraphQL
               Admin API (Bulk Operations) -> data/raw/
- clean.py   : dedupe, drop bad rows (£0 / outliers), normalise types,
               one row == one line-item sale -> data/interim/
"""
