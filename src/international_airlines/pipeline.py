"""End-to-end pipeline orchestration: features -> EDA -> forecasting -> anomaly detection."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .anomaly import cluster_ports, detect_anomalies
from .config import DEFAULT_PATHS
from .eda import plot_region_distribution, plot_top_countries, plot_yearly_trends
from .features import build_features
from .forecasting import backtest, fit_forecast, to_prophet_frame

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    n_rows: int
    passenger_forecast_backtest: dict[str, float]
    freight_forecast_backtest: dict[str, float]
    n_anomalies: int
    n_clusters: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_pipeline(
    raw_data_path: str = DEFAULT_PATHS.raw_data_path,
    results_dir: str = DEFAULT_PATHS.results_dir,
    forecast_periods: int = 12,
    backtest_periods: int = 12,
    n_clusters: int = 3,
    contamination: float = 0.05,
) -> PipelineResult:
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building features from %s", raw_data_path)
    df = build_features(raw_data_path)
    df.to_csv(results_dir_path / "processed_city_pairs.csv", index=False)

    logger.info("Running EDA plots")
    plot_top_countries(df, results_dir_path)
    plot_region_distribution(df, results_dir_path)
    plot_yearly_trends(df, results_dir_path)

    logger.info("Forecasting passenger and freight traffic")
    monthly = df.groupby("Date", as_index=False)[
        ["Passengers_Total", "Freight_Total_(tonnes)"]
    ].sum()

    passenger_ts = to_prophet_frame(monthly, "Date", "Passengers_Total")
    passenger_backtest = backtest(passenger_ts, holdout_periods=backtest_periods)
    _, passenger_forecast = fit_forecast(passenger_ts, periods=forecast_periods)
    passenger_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        results_dir_path / "passenger_forecast.csv", index=False
    )

    freight_ts = to_prophet_frame(monthly, "Date", "Freight_Total_(tonnes)")
    freight_backtest = backtest(freight_ts, holdout_periods=backtest_periods)
    _, freight_forecast = fit_forecast(freight_ts, periods=forecast_periods)
    freight_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        results_dir_path / "freight_forecast.csv", index=False
    )

    logger.info("Detecting anomalies and clustering ports")
    df = detect_anomalies(df, contamination=contamination)
    port_clusters = cluster_ports(df, n_clusters=n_clusters)

    df.to_csv(results_dir_path / "final_processed_city_pairs.csv", index=False)
    port_clusters.to_csv(results_dir_path / "port_clusters.csv", index=False)

    result = PipelineResult(
        n_rows=len(df),
        passenger_forecast_backtest=passenger_backtest,
        freight_forecast_backtest=freight_backtest,
        n_anomalies=int((df["Anomaly"] == -1).sum()),
        n_clusters=port_clusters["Cluster"].nunique(),
    )
    (results_dir_path / "pipeline_summary.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


__all__ = ["PipelineResult", "run_pipeline"]
