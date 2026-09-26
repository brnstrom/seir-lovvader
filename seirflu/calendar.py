"""Swedish school holidays and public holidays as daily exposures.

The municipal school calendars (Skolporten, school years 2016/17-2026/27) are
turned into a population-weighted fraction of the population whose schools
are closed for each holiday type on each day:

* ``host``   höstlov, the ISO week given in the calendar (almost always v.44)
* ``jul``    from the day after the last autumn-term school day to the day
             before the first spring-term school day
* ``sport``  sportlov, the ISO week given (v.7-10 depending on municipality)
* ``pask``   påsklov, the ISO week given (the week before or after Easter)
* ``sommar`` from the day after the last spring-term school day to the day
             before the next autumn term starts

Missing municipal entries are filled from the same municipality in the
nearest earlier school year (week-type holidays; for påsklov the position
relative to Easter is carried over from the earlier year with the most similar
Easter date) or from the county median for that year (term dates). Later
school years are used only when no earlier year exists. School year 2015/16
is not published by Skolporten and is filled entirely in this way.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
HOLIDAY_TYPES = ("host", "jul", "sport", "pask", "sommar")
FIRST_SCHOOL_YEAR = 2015
LAST_SCHOOL_YEAR = 2026
DATE_FIELDS = ("hoststart", "hostslut", "varstart", "varslut")
WEEK_FIELDS = ("hostlov_weeks", "sportlov_weeks", "pasklov_weeks")


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def easter_week(year: int) -> int:
    """ISO week that ends on Easter Sunday (the week before Easter)."""
    return easter_sunday(year).isocalendar()[1]


def _parse_weeks(text) -> list[int] | None:
    if text is None or (isinstance(text, float) and np.isnan(text)) or text == "":
        return None
    return [int(part) for part in str(text).split("+")]


@lru_cache(maxsize=1)
def _raw_calendar() -> pd.DataFrame:
    table = pd.read_csv(DATA / "school_calendar_municipal.csv", dtype=str, keep_default_na=False)
    table["y1"] = table["school_year"].str[:4].astype(int)
    return table


@lru_cache(maxsize=1)
def _population() -> pd.DataFrame:
    table = pd.read_csv(DATA / "scb_municipal_population.csv", dtype={"municipality_code": str})
    table = table[table["year"] == 2024][["municipality_code", "municipality", "population"]]
    table["county_code"] = table["municipality_code"].str[:2]
    return table.reset_index(drop=True)


def _fill_order(years: list[int], target: int) -> list[int]:
    """Earlier years first (nearest first), later years only as a fallback."""
    earlier = sorted((y for y in years if y < target), reverse=True)
    later = sorted(y for y in years if y > target)
    return earlier + later


def _shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:  # 29 February
        return value.replace(year=value.year + years, day=28)


@lru_cache(maxsize=1)
def complete_calendar() -> pd.DataFrame:
    """One fully populated row per municipality and school year."""
    raw = _raw_calendar()
    municipal = raw[raw["source"] == "municipal"]
    summary = raw[raw["source"] == "county_summary"]
    population = _population()
    known = {
        (row.municipality_code, row.y1): row
        for row in municipal.itertuples(index=False)
    }
    years_available = sorted(municipal["y1"].unique())

    # County medians of term dates per school year (municipal rows only).
    county_dates: dict[tuple[str, int, str], date] = {}
    for (county, y1), group in municipal.groupby(["county_code", "y1"]):
        for field in DATE_FIELDS:
            values = [date.fromisoformat(v) for v in group[field] if v]
            if values:
                ordinals = sorted(v.toordinal() for v in values)
                county_dates[(county, y1, field)] = date.fromordinal(ordinals[len(ordinals) // 2])
    summary_dates: dict[tuple[str, int, str], date] = {}
    for row in summary.itertuples(index=False):
        for field in DATE_FIELDS:
            value = getattr(row, field)
            if value:
                summary_dates[(row.county_code, row.y1, field)] = date.fromisoformat(value)

    records = []
    for y1 in range(FIRST_SCHOOL_YEAR, LAST_SCHOOL_YEAR + 1):
        y2 = y1 + 1
        for muni in population.itertuples(index=False):
            code, county = muni.municipality_code, muni.county_code
            own = known.get((code, y1))
            record = {
                "school_year": f"{y1}/{y2}",
                "y1": y1,
                "municipality_code": code,
                "county_code": county,
                "population": muni.population,
            }
            filled = []
            # Term dates
            for field in DATE_FIELDS:
                value = date.fromisoformat(getattr(own, field)) if own is not None and getattr(own, field) else None
                if value is None:
                    value = county_dates.get((county, y1, field)) or summary_dates.get((county, y1, field))
                    if value is None:
                        # Nearest year for the same county, shifted in calendar year.
                        for other in _fill_order(years_available, y1):
                            candidate = county_dates.get((county, other, field))
                            if candidate is not None:
                                value = _shift_year(candidate, y1 - other)
                                break
                    filled.append(field)
                record[field] = value
            # Höstlov and sportlov: same municipality, nearest year
            for field, default in (("hostlov_weeks", [44]), ("sportlov_weeks", None)):
                weeks = _parse_weeks(getattr(own, field)) if own is not None else None
                if weeks is None:
                    for other in _fill_order(years_available, y1):
                        row = known.get((code, other))
                        if row is not None and _parse_weeks(getattr(row, field)):
                            weeks = _parse_weeks(getattr(row, field))
                            break
                    filled.append(field)
                if weeks is None:
                    weeks = default
                record[field] = weeks
            # Påsklov: position relative to Easter from the most similar Easter
            weeks = _parse_weeks(own.pasklov_weeks) if own is not None else None
            if weeks is None:
                target_easter = easter_sunday(y2)
                pool = [y for y in years_available if y < y1] or list(years_available)
                ranked = sorted(
                    pool,
                    key=lambda y: (
                        abs(easter_sunday(y + 1).timetuple().tm_yday - target_easter.timetuple().tm_yday),
                        abs(y - y1),
                    ),
                )
                for other in ranked:
                    row = known.get((code, other))
                    other_weeks = _parse_weeks(row.pasklov_weeks) if row is not None else None
                    if other_weeks:
                        offsets = [w - easter_week(other + 1) for w in other_weeks]
                        if all(o in (0, 1) for o in offsets):
                            weeks = [easter_week(y2) + o for o in offsets]
                        else:
                            weeks = other_weeks
                        break
                filled.append("pasklov_weeks")
            record["pasklov_weeks"] = weeks
            record["filled_fields"] = "+".join(filled)
            records.append(record)
    table = pd.DataFrame.from_records(records)
    # Any still-missing week fields fall back to the county's population-weighted mode.
    for field in WEEK_FIELDS:
        missing = table[field].isna()
        if missing.any():
            for idx in table.index[missing]:
                row = table.loc[idx]
                peers = table[(table["county_code"] == row["county_code"]) & (table["y1"] == row["y1"]) & table[field].notna()]
                if peers.empty:
                    peers = table[(table["y1"] == row["y1"]) & table[field].notna()]
                counts: dict[tuple, float] = {}
                for weeks, pop in zip(peers[field], peers["population"]):
                    counts[tuple(weeks)] = counts.get(tuple(weeks), 0.0) + pop
                table.at[idx, field] = list(max(counts, key=counts.get))
    return table


def _series_filter(series: str, county_codes: pd.Series) -> pd.Series:
    if series == "SE":
        return pd.Series(True, index=county_codes.index)
    if series == "SE-M":
        return county_codes == "12"
    if series == "SE-O":
        return county_codes == "14"
    if series == "SE-rest":
        return ~county_codes.isin(["12", "14"])
    raise ValueError(f"Unknown series {series!r}")


def _iso_week_days(year: int, week: int) -> tuple[date, date]:
    start = date.fromisocalendar(year, week, 1)
    return start, start + timedelta(days=6)


@lru_cache(maxsize=16)
def daily_holiday_exposure(series: str) -> pd.DataFrame:
    """Daily population fraction on each holiday type, 2015-06-01 to 2027-08-31."""
    table = complete_calendar()
    table = table[_series_filter(series, table["county_code"])].copy()
    index = pd.date_range("2015-06-01", "2027-08-31", freq="D")
    exposure = pd.DataFrame(0.0, index=index, columns=list(HOLIDAY_TYPES))
    day0 = index[0].date()
    n_days = len(index)

    def add(column: str, start: date, end: date, weight: float) -> None:
        a = max((start - day0).days, 0)
        b = min((end - day0).days, n_days - 1)
        if b >= a:
            exposure.iloc[a : b + 1, exposure.columns.get_loc(column)] += weight

    total = {}
    for y1, group in table.groupby("y1"):
        pop_total = group["population"].sum()
        total[y1] = pop_total
        next_year = table[table["y1"] == y1 + 1].set_index("municipality_code")
        for row in group.itertuples(index=False):
            w = row.population / pop_total
            for column, weeks, iso_year in (
                ("host", row.hostlov_weeks, y1),
                ("sport", row.sportlov_weeks, y1 + 1),
                ("pask", row.pasklov_weeks, y1 + 1),
            ):
                for week in weeks:
                    start, end = _iso_week_days(iso_year, week)
                    add(column, start, end, w / len(weeks))
            add("jul", row.hostslut + timedelta(days=1), row.varstart - timedelta(days=1), w)
            if row.municipality_code in next_year.index:
                next_start = next_year.at[row.municipality_code, "hoststart"]
            else:
                next_start = _shift_year(row.hoststart, 1)
            add("sommar", row.varslut + timedelta(days=1), next_start - timedelta(days=1), w)
            if y1 == FIRST_SCHOOL_YEAR:
                # Summer 2015 before the first school year in the table; the
                # spring term 2015 is assumed to have ended on 12 June.
                add("sommar", date(2015, 6, 13), row.hoststart - timedelta(days=1), w)
    return exposure.clip(upper=1.0)


@lru_cache(maxsize=1)
def red_days() -> pd.Series:
    """Indicator of weekday public holidays and de facto holidays (1/0), daily."""
    index = pd.date_range("2015-01-01", "2027-12-31", freq="D")
    swedish = holidays_lib.Sweden(years=range(2015, 2028), language="sv")
    days = set()
    for day, name in swedish.items():
        names = [part.strip() for part in name.split(";")]
        if all(part == "Söndag" for part in names):
            continue
        days.add(day)
    for year in range(2015, 2028):
        days.update({date(year, 12, 24), date(year, 12, 31)})
        midsummer_eve = next(date(year, 6, d) for d in range(19, 26) if date(year, 6, d).weekday() == 4)
        days.add(midsummer_eve)
    values = [1.0 if (day.date() in days and day.weekday() < 5) else 0.0 for day in index]
    return pd.Series(values, index=index, name="red_day")
