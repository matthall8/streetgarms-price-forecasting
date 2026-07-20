"""Model training, evaluation, and comparison.

Suggested modules:

- split.py   : TIME-BASED train/test split (train past, test future).
               Do NOT use random k-fold — it leaks future info.
- metrics.py : percentage-error metrics (MAPE / RMSLE) alongside RMSE.
- train.py   : fit a model, persist artefact -> models/
- compare.py : bake-off harness — linear baseline vs LightGBM/XGBoost vs
               a third contender, on identical splits + metrics.
"""
