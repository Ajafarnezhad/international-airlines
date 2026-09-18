from __future__ import annotations

from pathlib import Path

import pytest

from international_airlines.features import build_features
from international_airlines.forecasting import backtest, fit_forecast, to_prophet_frame


@pytest.fixture
def passenger_ts(raw_csv: Path):
    df = build_features(str(raw_csv))
    monthly = df.groupby("Date", as_index=False)["Passengers_Total"].sum()
    return to_prophet_frame(monthly, "Date", "Passengers_Total")


def test_fit_forecast_extends_beyond_history(passenger_ts):
    _, forecast = fit_forecast(passenger_ts, periods=6)
    assert len(forecast) == len(passenger_ts) + 6
    assert forecast["ds"].max() > passenger_ts["ds"].max()


def test_backtest_returns_error_metrics(passenger_ts):
    metrics = backtest(passenger_ts, holdout_periods=6)
    assert metrics["mae"] >= 0
    assert metrics["mape"] >= 0


def test_backtest_rejects_insufficient_history(passenger_ts):
    with pytest.raises(ValueError):
        backtest(passenger_ts, holdout_periods=len(passenger_ts))
