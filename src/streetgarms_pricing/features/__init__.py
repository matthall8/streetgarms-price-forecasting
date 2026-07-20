"""Feature engineering.

Turn the cleaned line-item data into a model-ready feature matrix
(-> data/processed/). This is where most of the predictive signal comes from.

Candidate features for second-hand streetwear:
- item attributes: brand, category, size, colour, condition
- pricing context: original/RRP, discount, days on market
- temporal: month/season, day-of-week, release-vs-sale gap
- history: recent comparable sale prices for similar items

NOTE: the target is likely log(price); decide the transform here.
"""
