from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """A small synthetic dataset matching the real schema, spanning 36 months
    across two foreign ports so both forecasting (needs enough history for a
    12-month backtest) and clustering (needs >1 port) have enough to work
    with."""
    rng = np.random.default_rng(0)
    rows = []
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            for port, country in [("Auckland", "New Zealand"), ("Unmapped City", "Nowhereland")]:
                passengers_in = int(rng.integers(100, 1000))
                passengers_out = int(rng.integers(100, 1000))
                rows.append(
                    {
                        "Month": f"{month:02d}-{year}",
                        "AustralianPort": "Sydney",
                        "ForeignPort": port,
                        "Country": country,
                        "Passengers_In": passengers_in,
                        "Freight_In_(tonnes)": float(rng.uniform(0, 50)),
                        "Mail_In_(tonnes)": float(rng.uniform(0, 5)),
                        "Passengers_Out": passengers_out,
                        "Freight_Out_(tonnes)": float(rng.uniform(0, 50)),
                        "Mail_Out_(tonnes)": float(rng.uniform(0, 5)),
                        "Passengers_Total": passengers_in + passengers_out,
                        "Freight_Total_(tonnes)": float(rng.uniform(0, 100)),
                        "Mail_Total_(tonnes)": float(rng.uniform(0, 10)),
                        "Year": year,
                        "Month_num": month,
                    }
                )
    df = pd.DataFrame(rows)
    path = tmp_path / "city_pairs.csv"
    df.to_csv(path, index=False)
    return path
