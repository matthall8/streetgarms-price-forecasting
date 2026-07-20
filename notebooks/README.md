# Notebooks

Exploratory work. Keep production logic in `src/streetgarms_pricing/`.

Suggested order:

- `01_eda.ipynb` — **start here.** Understand the target before modelling:
  price distribution (raw vs log), £0 / outliers / duplicates, what one row
  represents, date range & volume over time. Decide metric + transform here.
- `02_features.ipynb` — prototype features before promoting them to `features/`.
- `03_model_bakeoff.ipynb` — compare models on the time-based split.
