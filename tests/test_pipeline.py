from __future__ import annotations

from pathlib import Path

from international_airlines.pipeline import run_pipeline


def test_run_pipeline_end_to_end(raw_csv: Path, tmp_path: Path):
    results_dir = tmp_path / "results"
    result = run_pipeline(
        raw_data_path=str(raw_csv),
        results_dir=str(results_dir),
        forecast_periods=6,
        backtest_periods=6,
        n_clusters=2,
    )

    assert result.n_rows > 0
    assert "mae" in result.passenger_forecast_backtest
    assert "mae" in result.freight_forecast_backtest
    assert result.n_clusters <= 2

    for filename in (
        "processed_city_pairs.csv",
        "passenger_forecast.csv",
        "freight_forecast.csv",
        "final_processed_city_pairs.csv",
        "port_clusters.csv",
        "pipeline_summary.json",
        "top_countries.png",
        "region_distribution.png",
        "yearly_trends.png",
    ):
        assert (results_dir / filename).exists(), f"missing {filename}"

    # Regression guard for the original cross-notebook bug: the file that
    # anomaly detection/clustering reads from must already contain the
    # normalized columns it needs.
    import pandas as pd

    final_df = pd.read_csv(results_dir / "final_processed_city_pairs.csv")
    assert "Normalized_Passengers" in final_df.columns
    assert "Anomaly" in final_df.columns
