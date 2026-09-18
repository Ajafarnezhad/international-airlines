"""Prophet-based forecasting, with a held-out backtest so accuracy is actually measured.

The original notebook only ever fit Prophet on the *entire* series and
plotted the forecast beyond the last observed date -- there was no way to
tell whether the model's predictions were any good, since nothing was ever
compared against real held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def to_prophet_frame(monthly: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    return monthly[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})


def fit_forecast(ts_data: pd.DataFrame, periods: int = 12) -> tuple[Prophet, pd.DataFrame]:
    """Fit Prophet on `ts_data` (columns `ds`, `y`) and forecast `periods` months beyond it.

    Uses the month-*start* frequency alias ``"MS"``, matching how this
    project's dates are built (``features.load_raw`` always constructs
    ``Date`` as the first of the month). The original notebook used the
    deprecated ``"M"`` alias, which -- like its modern replacement ``"ME"``
    -- means month-*end*: `make_future_dataframe` would have generated
    future dates like ``2023-07-31`` that don't line up with the
    month-start convention (``2023-07-01``) used everywhere else in the
    data, silently misaligning any downstream join or lookup by date.
    """
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(ts_data)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return model, forecast


def backtest(ts_data: pd.DataFrame, holdout_periods: int = 12) -> dict[str, float]:
    """Fit on all but the last `holdout_periods` rows, forecast, and score against the holdout.

    Returns MAE and MAPE on the held-out actuals -- the accuracy check the
    original notebook never performed.
    """
    if len(ts_data) <= holdout_periods:
        raise ValueError(
            f"Need more than {holdout_periods} observations to backtest; got {len(ts_data)}."
        )

    train = ts_data.iloc[:-holdout_periods]
    holdout = ts_data.iloc[-holdout_periods:]

    _, forecast = fit_forecast(train, periods=holdout_periods)
    predicted = forecast.set_index("ds").loc[holdout["ds"], "yhat"].to_numpy()
    actual = holdout["y"].to_numpy()

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(mean_absolute_percentage_error(actual, np.where(actual == 0, 1e-9, actual))),
    }


__all__ = ["backtest", "fit_forecast", "to_prophet_frame"]
