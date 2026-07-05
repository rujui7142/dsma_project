"""US federal + major religious/cultural holiday features for NYC trip data.

Two kinds of holiday dates:
  - RULE-BASED (computed for any year): fixed calendar dates (Jan 1, Jul 4, ...)
    and "nth weekday of month" rules (MLK Day, Thanksgiving, ...).
  - MOVABLE / LUNAR (hardcoded per year): Easter, Jewish holidays (Hebrew
    lunisolar calendar), Islamic holidays (Hijri lunar calendar, drifts ~11
    days earlier each year), Diwali, Lunar New Year. These cannot be derived
    by a simple formula, so exact dates are hardcoded for 2024-2026 (the span
    of our actual training/test data) from published religious calendars.
    Islamic dates additionally depend on local moon sighting and can shift by
    a day; treat as accurate to within ±1 day.

Only 2024-2026 movable dates are populated (that's our real data range); a
row from an earlier/garbage year simply won't match any movable holiday,
which is a harmless fallback for the handful of corrupt-timestamp rows the
2014 floor doesn't already remove.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

DateTuple = Tuple[int, int, int]  # (year, month, day)


def _expand_range(start: DateTuple, n_days: int) -> List[DateTuple]:
    """Expand a (year, month, day) start date into n_days consecutive dates."""
    ts = pd.Timestamp(*start)
    return [(d.year, d.month, d.day) for d in pd.date_range(ts, periods=n_days)]


# ---------------------------------------------------------------------------
# Movable / lunar holidays — hardcoded, verified per-year dates (2024-2026)
# ---------------------------------------------------------------------------

CHRISTIAN_MOVABLE: List[DateTuple] = [
    (2024, 3, 29), (2025, 4, 18), (2026, 4, 3),    # Good Friday
    (2024, 3, 31), (2025, 4, 20), (2026, 4, 5),    # Easter Sunday
]

JEWISH_MOVABLE: List[DateTuple] = (
    [(2024, 10, 3), (2024, 10, 4)]                  # Rosh Hashanah
    + [(2025, 9, 23), (2025, 9, 24)]
    + [(2026, 9, 12), (2026, 9, 13)]
    + [(2024, 10, 12)]                              # Yom Kippur
    + [(2025, 10, 2)]
    + [(2026, 9, 21)]
    + _expand_range((2024, 4, 23), 8)                # Passover
    + _expand_range((2025, 4, 13), 8)
    + _expand_range((2026, 4, 2), 8)
    + _expand_range((2024, 12, 26), 8)               # Hanukkah
    + _expand_range((2025, 12, 15), 8)
    + _expand_range((2026, 12, 5), 8)                # approximate, unverified
)

MUSLIM_MOVABLE: List[DateTuple] = [
    (2024, 4, 10),                                   # Eid al-Fitr
    (2025, 3, 30),
    (2026, 3, 20),
    (2024, 6, 16),                                   # Eid al-Adha
    (2025, 6, 7),
    (2026, 5, 27),                                   # approximate
]

OTHER_CULTURAL_MOVABLE: List[DateTuple] = [
    (2024, 10, 31),                                  # Diwali (coincides with Halloween in 2024)
    (2025, 10, 20),
    (2026, 11, 8),
    (2024, 2, 10),                                   # Lunar New Year
    (2025, 1, 29),
    (2026, 2, 17),
]

# Fixed-date, non-religious holidays with a clear NYC demand/traffic signature
_FIXED_DATES: List[DateTuple] = []  # filled per-year in build_holiday_sets()

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

# Federal holidays computed by rule (nth weekday of month); "n" negative = from month end
RULE_HOLIDAYS = [
    ("MLK Day", 1, 0, 3),          # 3rd Monday of January
    ("Presidents Day", 2, 0, 3),   # 3rd Monday of February
    ("Memorial Day", 5, 0, -1),    # last Monday of May
    ("Labor Day", 9, 0, 1),        # 1st Monday of September
    ("Columbus Day", 10, 0, 2),    # 2nd Monday of October
    ("Thanksgiving", 11, 3, 4),    # 4th Thursday of November
    ("Mother's Day", 5, 6, 2),     # 2nd Sunday of May
]

# Subset with the biggest NYC travel/demand-shift signature
MAJOR_HOLIDAY_MONTH_DAY: Set[Tuple[int, int]] = {
    (1, 1), (12, 25), (12, 31), (7, 4), (10, 31),
}
MAJOR_HOLIDAY_RULE_NAMES = {"Thanksgiving"}

FEDERAL_HOLIDAY_MONTH_DAY: Set[Tuple[int, int]] = {
    (1, 1), (6, 19), (7, 4), (11, 11), (12, 25),
}
FEDERAL_RULE_NAMES = {"MLK Day", "Presidents Day", "Memorial Day", "Labor Day", "Columbus Day"}
# Election Day and Veterans Day are federal but computed/listed separately below.


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> DateTuple:
    """weekday: 0=Monday..6=Sunday. n>0 = nth occurrence; n=-1 = last occurrence."""
    first = pd.Timestamp(year, month, 1)
    first_weekday_offset = (weekday - first.dayofweek) % 7
    first_occurrence = first + pd.Timedelta(days=first_weekday_offset)
    if n > 0:
        d = first_occurrence + pd.Timedelta(weeks=n - 1)
        return (d.year, d.month, d.day)
    # last occurrence in month
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


def build_holiday_sets(years: List[int]) -> Dict[str, Set[DateTuple]]:
    """Build the full set of (year, month, day) tuples per holiday category
    for the given years, combining rule-based/fixed dates with the hardcoded
    movable-holiday tables above.
    """
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

        elec = _election_day(yr)
        federal_all.append(elec)

    all_rule_dates = [d for dates in rule_by_name.values() for d in dates]
    all_rule_dates += [_election_day(yr) for yr in years]

    return {
        "any": set(fixed_all) | set(all_rule_dates) | set(CHRISTIAN_MOVABLE)
               | set(JEWISH_MOVABLE) | set(MUSLIM_MOVABLE) | set(OTHER_CULTURAL_MOVABLE),
        "major": set(major_all),
        "federal": set(federal_all),
        "christian": set(CHRISTIAN_MOVABLE) | {(y, 12, 25) for y in years},
        "jewish": set(JEWISH_MOVABLE),
        "muslim": set(MUSLIM_MOVABLE),
        "other_cultural": set(OTHER_CULTURAL_MOVABLE),
    }


# Pre-built for the span our data actually covers (with margin either side).
_YEARS = list(range(2014, 2028))
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

    # Continuous proximity signal (days to nearest holiday, either direction) —
    # captures elevated demand on eves/after-days (Christmas Eve, Black Friday)
    # that the exact-day flags alone miss.
    ordinals = pd.to_datetime(
        {"year": df["pickup_year"], "month": df["pickup_month"], "day": df["pickup_day"]}
    ).map(pd.Timestamp.toordinal).to_numpy()
    idx = np.searchsorted(_ALL_HOLIDAY_ORDINALS, ordinals)
    idx_clip = np.clip(idx, 1, len(_ALL_HOLIDAY_ORDINALS) - 1)
    dist_left = np.abs(ordinals - _ALL_HOLIDAY_ORDINALS[idx_clip - 1])
    dist_right = np.abs(_ALL_HOLIDAY_ORDINALS[idx_clip] - ordinals)
    df["days_to_nearest_holiday"] = np.minimum(dist_left, dist_right)

    return df
