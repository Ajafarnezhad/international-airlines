"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import DEFAULT_PATHS
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="international-airlines",
        description="EDA, Prophet forecasting, and anomaly/cluster analysis of Australian international air traffic.",
    )
    parser.add_argument("--data", default=DEFAULT_PATHS.raw_data_path, help="Path to the raw city_pairs CSV.")
    parser.add_argument("--results-dir", default=DEFAULT_PATHS.results_dir, help="Directory for outputs.")
    parser.add_argument("--forecast-periods", type=int, default=12, help="Months to forecast beyond the last observation.")
    parser.add_argument("--backtest-periods", type=int, default=12, help="Months held out to score forecast accuracy.")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of KMeans clusters for port grouping.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction for IsolationForest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_pipeline(
        raw_data_path=args.data,
        results_dir=args.results_dir,
        forecast_periods=args.forecast_periods,
        backtest_periods=args.backtest_periods,
        n_clusters=args.n_clusters,
        contamination=args.contamination,
    )
    print(f"Rows processed: {result.n_rows}")
    print(f"Passenger forecast backtest: {result.passenger_forecast_backtest}")
    print(f"Freight forecast backtest: {result.freight_forecast_backtest}")
    print(f"Anomalies detected: {result.n_anomalies}")
    print(f"Port clusters: {result.n_clusters}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
