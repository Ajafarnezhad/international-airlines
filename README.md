# International Air Traffic Analysis

[![CI](https://github.com/Ajafarnezhad/international-airlines/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajafarnezhad/international-airlines/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

EDA, Prophet-based forecasting, and anomaly/cluster analysis of ~89k rows of
monthly passenger, freight, and mail traffic between Australian and foreign
ports (1985–2025).

## Why this project

A three-stage pipeline (EDA → forecasting → anomaly detection) is only as
reliable as its handoffs between stages. Rewriting this from three
interdependent notebooks turned up a handoff that was completely broken:
see ["What changed"](#what-changed-from-the-original-notebooks) below.

## What changed from the original notebooks

- **Stage 3 could not run at all.** The anomaly-detection notebook loaded
  stage 1's output CSV and immediately referenced `Normalized_Passengers`
  columns that only ever existed in stage 2's in-memory dataframe -- never
  saved to disk. Running it raised `KeyError` on the first real line of
  code. Fixed by computing every engineered feature once, centrally
  (`international_airlines.features.build_features`), so every stage reads
  a single, complete, consistent dataset.
- **Clustering mixed normalized and raw-scale features.** Port clustering
  fed `KMeans` a `Passenger_Diff` column in the tens of thousands alongside
  0-1 normalized columns -- a distance-based algorithm would have let that
  one column dominate every cluster assignment. `Passenger_Diff` is now
  normalized like everything else before clustering.
- **Forecasts were never checked for accuracy.** The original notebook fit
  Prophet once on the entire series and plotted the forecast -- there was
  no way to know if it was any good. `international_airlines.forecasting.backtest`
  now holds out the last N months, forecasts them, and reports MAE/MAPE.
- **Dead code removed**: label-encoded columns computed in the forecasting
  notebook were never referenced anywhere.
- **Wrong forecast date frequency.** The original `freq='M'` (deprecated
  pandas alias for month-*end*) generated future dates like `2023-07-31`
  that don't align with this dataset's month-*start* convention
  (`2023-07-01`). Fixed to `freq='MS'`.
- Three monolithic single-cell notebooks → a typed, tested
  `src/international_airlines/` package with narrative notebooks that call
  into it.

## Installation

```bash
git clone https://github.com/Ajafarnezhad/international-airlines.git
cd international-airlines
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Dataset

See [`data/README.md`](data/README.md) — this project analyzes the
[International Airlines Traffic by City Pairs](https://www.kaggle.com/datasets/imtkaggleteam/international-airlines-traffic-by-city-pairs)
dataset (imtkaggleteam, Kaggle; sourced from Australia's BITRE statistics).

## Usage

```bash
international-airlines --data data/city_pairs.csv --results-dir results
```

Writes to `results/`:

- `processed_city_pairs.csv`, `final_processed_city_pairs.csv` — fully engineered dataset (regions, derived metrics, normalized columns, anomaly flags)
- `passenger_forecast.csv`, `freight_forecast.csv` — 12-month-ahead Prophet forecasts with confidence intervals
- `port_clusters.csv` — KMeans cluster assignment per foreign port
- `pipeline_summary.json` — row counts, forecast backtest MAE/MAPE, anomaly/cluster counts
- `top_countries.png`, `region_distribution.png`, `yearly_trends.png` — EDA plots

Run `international-airlines --help` for all options. See
[`notebooks/`](notebooks/) for a step-by-step, narrative walkthrough of each
stage.

## Project layout

```
international-airlines/
├── src/international_airlines/
│   ├── config.py        # paths, region map, random state
│   ├── features.py       # data loading + all feature engineering (single source of truth)
│   ├── eda.py             # aggregations + plots
│   ├── forecasting.py     # Prophet fit/forecast + backtest evaluation
│   ├── anomaly.py         # IsolationForest + KMeans port clustering
│   ├── pipeline.py        # end-to-end orchestration
│   └── cli.py             # `international-airlines` command-line entry point
├── tests/                 # pytest suite (synthetic 3-year, 2-port dataset -- fast, no Kaggle download needed)
├── notebooks/             # narrative walkthrough notebooks
├── data/README.md         # dataset provenance and setup instructions
└── .github/workflows/ci.yml
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=international_airlines --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Amirhossein Jafarnezhad** — [github.com/Ajafarnezhad](https://github.com/Ajafarnezhad)
