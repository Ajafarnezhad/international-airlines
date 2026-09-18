# Changelog

## 1.0.0

Rewrite of three interdependent notebook prototypes into a tested package,
fixing a bug that made the third stage impossible to run:

- **The anomaly-detection/clustering notebook could not run.** It loaded
  `processed_city_pairs_eda.csv` (the *first* notebook's output) and
  immediately referenced `Normalized_Passengers` / `Normalized_Freight`
  columns -- but those were computed by the *second* notebook
  (forecasting), which never saved its augmented dataframe anywhere (only
  the Prophet forecast results were written out). Loading the actual file
  on disk raised `KeyError` on the very first cell that used it. Fixed by
  centralizing all feature engineering in `international_airlines.features`,
  computed once and written to a single processed file every downstream
  stage reads from.
- **KMeans clustering mixed normalized and raw-scale features.** Port
  clustering used `Normalized_Passengers` / `Normalized_Freight` (0-1
  range) alongside a *raw* `Passenger_Diff` column that can run into the
  tens of thousands -- for a distance-based algorithm like KMeans, that one
  unnormalized column would dominate every cluster assignment.
  `Passenger_Diff` is now normalized alongside the rest before clustering.
- **Forecasts were never checked for accuracy.** Prophet was fit once on
  the full series and plotted beyond the last date, with no held-out
  comparison. `international_airlines.forecasting.backtest` now holds out
  the last N months, forecasts them, and reports MAE/MAPE.
- **Dead code removed**: `LabelEncoder`-encoded columns
  (`Country_Encoded`, `ForeignPort_Encoded`, `AustralianPort_Encoded`) were
  computed in the forecasting notebook but never used anywhere.
- **Wrong forecast date frequency**: `freq='M'` (the deprecated alias for
  month-*end*) generated future dates like `2023-07-31` that don't align
  with this dataset's month-*start* date convention (`2023-07-01`),
  silently breaking any downstream join or lookup by date. Fixed to
  `freq='MS'` (month-start).
- Split three monolithic, single-cell notebooks into a typed, tested
  `src/international_airlines/` package (`features`, `eda`, `forecasting`,
  `anomaly`, `pipeline`, `cli`), with narrative notebooks that call into it.
