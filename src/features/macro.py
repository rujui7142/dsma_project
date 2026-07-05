"""Macro-financial features: dollar index + S&P 500, as a proxy for NYC
riders' ability/willingness to pay for a cab (wealth effect / consumer
sentiment), tested as wide-net candidates alongside everything else.

Data: daily closes from FRED (Federal Reserve Economic Data) --
  DTWEXBGS -- Nominal Broad U.S. Dollar Index
  SP500    -- S&P 500 daily close
Both cover 2024-01-02 through 2026-02-27 (fetched via FRED's public CSV
endpoint), i.e. our full training + test window.

Leakage note: mapped using the LAST CLOSE STRICTLY BEFORE the pickup date
(shifted by one calendar day), not same-day close -- same-day close isn't
known until market close, so using it would leak future-of-day information
into a booking-time feature. Weekends/holidays forward-fill from the last
trading day, which is the realistic "most recently known value" a live
system would have.

Production note: unlike engineered features computed purely from the trip
itself, this requires an ongoing daily market-data feed at inference time --
a real operational dependency, not a one-time computation like the holiday
calendar. Flagging this explicitly so it isn't taken for granted.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATHS

_RETURN_WINDOW_DAYS = 21  # ~1 trading month


def _load_series(path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.set_index("observation_date")[value_col].astype(float)

    # Reindex to every CALENDAR day in range and forward-fill weekends/holidays
    # so any pickup date maps to "the most recently known close".
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range).ffill()

    # Shift by 1 day: use the PRIOR day's close, never same-day (leakage guard).
    df = df.shift(1).ffill()
    return df


def _build_macro_table() -> pd.DataFrame:
    dxy = _load_series(DATA_PATHS["training"].parent / "macro_data" / "dxy.csv", "DTWEXBGS")
    sp500 = _load_series(DATA_PATHS["training"].parent / "macro_data" / "sp500.csv", "SP500")

    table = pd.DataFrame({"dxy_level": dxy, "sp500_level": sp500})
    table["dxy_change_1m"] = table["dxy_level"].pct_change(_RETURN_WINDOW_DAYS)
    table["sp500_return_1m"] = table["sp500_level"].pct_change(_RETURN_WINDOW_DAYS)
    table[["dxy_change_1m", "sp500_return_1m"]] = table[["dxy_change_1m", "sp500_return_1m"]].fillna(0.0)
    return table


_MACRO_TABLE = _build_macro_table()
_MACRO_MIN_DATE = _MACRO_TABLE.index.min()
_MACRO_MAX_DATE = _MACRO_TABLE.index.max()


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map pickup_year/month/day to the prior day's DXY / S&P 500 level and
    1-month change. Dates outside the fetched range clip to the nearest edge
    (holds the boundary value) rather than producing NaN.
    """
    df = df.copy()
    dates = pd.to_datetime(
        {"year": df["pickup_year"], "month": df["pickup_month"], "day": df["pickup_day"]}
    )
    dates_clipped = dates.clip(lower=_MACRO_MIN_DATE, upper=_MACRO_MAX_DATE)

    joined = _MACRO_TABLE.reindex(dates_clipped.values)
    for col in _MACRO_TABLE.columns:
        df[col] = joined[col].to_numpy()
    return df
