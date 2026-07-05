"""US federal + major religious/cultural holiday features for NYC trip data.

Three sources of holiday dates, all generalizing to ANY year (no hardcoded
per-year date tables):

  - RULE-BASED (pure formula): fixed calendar dates (Jan 1, Jul 4, ...) and
    "nth weekday of month" rules (MLK Day, Thanksgiving, Election Day, ...).
  - EASTER / GOOD FRIDAY: the Anonymous Gregorian (Meeus/Jones/Butcher)
    algorithm — a closed-form Computus formula, valid for any Gregorian year.
  - MOVABLE / LUNAR (Jewish, Islamic, Hindu, Chinese calendars): computed via
    the `holidays` package, which implements the actual Hebrew, (tabular)
    Hijri, and Chinese lunisolar calendar arithmetic — not lookup tables — so
    it is correct for any year, not just the ones we happened to verify.
    Cross-checked against manually-sourced 2024-2026 dates (web search) before
    switching to this library; all matched exactly except Diwali 2024, which
    is a known ±1-day ambiguity between regional calendar conventions.
"""

from typing import Dict, List, Set, Tuple

import holidays as _holidays_lib
import numpy as np
import pandas as pd

DateTuple = Tuple[int, int, int]  # (year, month, day)

# Wide year buffer around our actual data range (>= 2024 per CLEANING) so the
# module-level precomputation below covers any plausible past/future data.
# Capped at 2030: the `holidays` package's India calendar only computes
# Diwali/Holi through 2030 and warns beyond that; our real data never gets
# close to this boundary.
_YEARS = list(range(2015, 2031))


# ---------------------------------------------------------------------------
# Easter / Good Friday — closed-form Computus formula (any Gregorian year)
# ---------------------------------------------------------------------------

def _easter_sunday(year: int) -> DateTuple:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return (year, month, day)


def _good_friday(year: int) -> DateTuple:
    d = pd.Timestamp(*_easter_sunday(year)) - pd.Timedelta(days=2)
    return (d.year, d.month, d.day)


# ---------------------------------------------------------------------------
# Rule-based federal / civil holidays — pure formulas (any year)
# ---------------------------------------------------------------------------

FIXED_MONTH_DAY: List[Tuple[int, int]] = [
    (1, 1),    # New Year's Day
    (2, 14),   # Valentine's Day
    (3, 17),   # St. Patrick's Day
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (10, 31),  # Halloween
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
]

# name, month, weekday (0=Mon..6=Sun), n (nth occurrence; -1 = last in month)
RULE_HOLIDAYS = [
    ("MLK Day", 1, 0, 3),
    ("Presidents Day", 2, 0, 3),
    ("Memorial Day", 5, 0, -1),
    ("Labor Day", 9, 0, 1),
    ("Columbus Day", 10, 0, 2),
    ("Thanksgiving", 11, 3, 4),
    ("Mother's Day", 5, 6, 2),
]

MAJOR_HOLIDAY_MONTH_DAY: Set[Tuple[int, int]] = {(1, 1), (12, 25), (12, 31), (7, 4), (10, 31)}
MAJOR_HOLIDAY_RULE_NAMES = {"Thanksgiving"}

FEDERAL_HOLIDAY_MONTH_DAY: Set[Tuple[int, int]] = {(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)}
FEDERAL_RULE_NAMES = {"MLK Day", "Presidents Day", "Memorial Day", "Labor Day", "Columbus Day"}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> DateTuple:
    """weekday: 0=Monday..6=Sunday. n>0 = nth occurrence; n=-1 = last occurrence."""
    first = pd.Timestamp(year, month, 1)
    first_weekday_offset = (weekday - first.dayofweek) % 7
    first_occurrence = first + pd.Timedelta(days=first_weekday_offset)
    if n > 0:
        d = first_occurrence + pd.Timedelta(weeks=n - 1)
        return (d.year, d.month, d.day)
    next_month = pd.Timestamp(year, month, 1) + pd.DateOffset(months=1)
    last_day = next_month - pd.Timedelta(days=1)
    last_weekday_offset = (last_day.dayofweek - weekday) % 7
    d = last_day - pd.Timedelta(days=last_weekday_offset)
    return (d.year, d.month, d.day)


def _election_day(year: int) -> DateTuple:
    """Tuesday after the first Monday in November (federal rule)."""
    first_monday = _nth_weekday(year, 11, 0, 1)
    d = pd.Timestamp(*first_monday) + pd.Timedelta(days=1)
    return (d.year, d.month, d.day)


# ---------------------------------------------------------------------------
# Movable religious/cultural calendars — computed via the `holidays` package
# (real calendar arithmetic, not per-year lookup tables)
# ---------------------------------------------------------------------------

def _lib_dates(country_holidays, name_filter=None) -> Set[DateTuple]:
    """Extract (year, month, day) tuples from a `holidays` calendar object,
    optionally keeping only entries whose name contains one of name_filter
    (case-insensitive) — used to pull just the religious holidays out of a
    country calendar that also includes secular/national ones.
    """
    out = set()
    for d, name in country_holidays.items():
        if name_filter is None or any(kw.lower() in name.lower() for kw in name_filter):
            out.add((d.year, d.month, d.day))
    return out


def _build_jewish_dates(years: List[int]) -> Set[DateTuple]:
    il = _holidays_lib.IL(years=years, language="en_US",
                           categories=("public", "optional", "school"))
    return _lib_dates(il)  # all Israeli religious/civil observances -> "jewish"


def _build_muslim_dates(years: List[int]) -> Set[DateTuple]:
    sa = _holidays_lib.SaudiArabia(years=years, language="en_US")
    return _lib_dates(sa, name_filter=["eid"])  # exclude Founding/National Day


def _build_other_cultural_dates(years: List[int]) -> Set[DateTuple]:
    india = _holidays_lib.India(years=years, language="en_US")
    china = _holidays_lib.China(years=years, language="en_US")
    diwali = _lib_dates(india, name_filter=["diwali", "deepavali"])
    lunar_ny = _lib_dates(china, name_filter=["chinese new year", "spring festival"])
    return diwali | lunar_ny


def _build_christian_dates(years: List[int]) -> Set[DateTuple]:
    dates = {(y, 12, 25) for y in years}  # Christmas (also in FIXED_MONTH_DAY)
    for y in years:
        dates.add(_easter_sunday(y))
        dates.add(_good_friday(y))
    return dates


# ---------------------------------------------------------------------------
# Assemble all categories
# ---------------------------------------------------------------------------

def build_holiday_sets(years: List[int]) -> Dict[str, Set[DateTuple]]:
    fixed_all, major_all, federal_all, rule_by_name = [], [], [], {}

    for yr in years:
        for mo, day in FIXED_MONTH_DAY:
            fixed_all.append((yr, mo, day))
            if (mo, day) in MAJOR_HOLIDAY_MONTH_DAY:
                major_all.append((yr, mo, day))
            if (mo, day) in FEDERAL_HOLIDAY_MONTH_DAY:
                federal_all.append((yr, mo, day))

        for name, mo, wd, n in RULE_HOLIDAYS:
            d = _nth_weekday(yr, mo, wd, n)
            rule_by_name.setdefault(name, []).append(d)
            if name in MAJOR_HOLIDAY_RULE_NAMES:
                major_all.append(d)
            if name in FEDERAL_RULE_NAMES:
                federal_all.append(d)

        federal_all.append(_election_day(yr))

    all_rule_dates = [d for dates in rule_by_name.values() for d in dates]
    all_rule_dates += [_election_day(yr) for yr in years]

    christian = _build_christian_dates(years)
    jewish = _build_jewish_dates(years)
    muslim = _build_muslim_dates(years)
    other_cultural = _build_other_cultural_dates(years)

    return {
        "any": set(fixed_all) | set(all_rule_dates) | christian | jewish | muslim | other_cultural,
        "major": set(major_all),
        "federal": set(federal_all),
        "christian": christian,
        "jewish": jewish,
        "muslim": muslim,
        "other_cultural": other_cultural,
    }


_HOLIDAY_SETS = build_holiday_sets(_YEARS)
_ALL_HOLIDAY_ORDINALS = np.array(sorted(
    pd.Timestamp(y, m, d).toordinal() for (y, m, d) in _HOLIDAY_SETS["any"]
))


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag holidays and holiday proximity from pickup_year/month/day.

    Requires pickup_year, pickup_month, pickup_day (see cleaner.add_datetime_features).
    """
    df = df.copy()
    keys = list(zip(df["pickup_year"], df["pickup_month"], df["pickup_day"]))

    df["is_holiday"] = np.fromiter((k in _HOLIDAY_SETS["any"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_major_holiday"] = np.fromiter((k in _HOLIDAY_SETS["major"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_federal_holiday"] = np.fromiter((k in _HOLIDAY_SETS["federal"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_christian_holiday"] = np.fromiter((k in _HOLIDAY_SETS["christian"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_jewish_holiday"] = np.fromiter((k in _HOLIDAY_SETS["jewish"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_muslim_holiday"] = np.fromiter((k in _HOLIDAY_SETS["muslim"] for k in keys), dtype=np.int8, count=len(keys))
    df["is_other_cultural_holiday"] = np.fromiter(
        (k in _HOLIDAY_SETS["other_cultural"] for k in keys), dtype=np.int8, count=len(keys)
    )

    ordinals = pd.to_datetime(
        {"year": df["pickup_year"], "month": df["pickup_month"], "day": df["pickup_day"]}
    ).map(pd.Timestamp.toordinal).to_numpy()
    idx = np.searchsorted(_ALL_HOLIDAY_ORDINALS, ordinals)
    idx_clip = np.clip(idx, 1, len(_ALL_HOLIDAY_ORDINALS) - 1)
    dist_left = np.abs(ordinals - _ALL_HOLIDAY_ORDINALS[idx_clip - 1])
    dist_right = np.abs(_ALL_HOLIDAY_ORDINALS[idx_clip] - ordinals)
    df["days_to_nearest_holiday"] = np.minimum(dist_left, dist_right)

    return df
