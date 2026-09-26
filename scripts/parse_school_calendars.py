"""Parse Skolporten's municipal school calendars into one tidy CSV.

Input: text files produced with ``pdftotext -layout`` from Skolporten's
"Grundskolans läsårstider" PDFs, stored in ``data/raw/``.
Output: ``data/school_calendar_municipal.csv`` with one row per municipality
and school year.

The PDF layouts differ between years, so fields are recognised by content
rather than by column position. Dates are assigned by month (August = autumn
term start, December = last day before Christmas, January = spring term start,
June = last day of spring term). Week numbers are assigned by range (40-46 =
höstlov, 6-11 = sportlov, 12-18 = påsklov).

Run from ``seir-model/``::

    python scripts/parse_school_calendars.py
"""

from __future__ import annotations

import csv
import re
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
POPULATION = ROOT / "data" / "scb_municipal_population.csv"
OUTPUT = ROOT / "data" / "school_calendar_municipal.csv"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6, "juni": 6,
    "jul": 7, "juli": 7, "aug": 8, "sep": 9, "sept": 9, "okt": 10, "nov": 11,
    "dec": 12,
}

COUNTY_CODES = {
    "stockholms": "01", "uppsala": "03", "södermanlands": "04",
    "östergötlands": "05", "jönköpings": "06", "kronobergs": "07",
    "kalmar": "08", "gotlands": "09", "blekinge": "10", "skåne": "12",
    "hallands": "13", "västra götalands": "14", "värmlands": "17",
    "örebro": "18", "västmanlands": "19", "dalarnas": "20",
    "gävleborgs": "21", "västernorrlands": "22", "jämtlands": "23",
    "västerbottens": "24", "norrbottens": "25",
}
# County names as they appear in the county summary table ("Läsårstiderna länsvis").
SUMMARY_COUNTIES = {
    "stockholm": "01", "uppsala": "03", "södermanland": "04",
    "östergötland": "05", "jönköping": "06", "kronoberg": "07", "kalmar": "08",
    "gotland": "09", "blekinge": "10", "skåne": "12", "halland": "13",
    "västra götaland": "14", "värmland": "17", "örebro": "18",
    "västmanland": "19", "dalarna": "20", "gävleborg": "21",
    "västernorrland": "22", "jämtland": "23", "västerbotten": "24",
    "norrbotten": "25",
}

SKIP_MARKERS = (
    "Höststart", "Kommun", "Länsintervall", "skolporten", "läsårstider",
    "Publicerad", "Läsårstiderna", "Län ", "v. 20", "Här finner", "samlar och",
    "ännu inte", "info@", "För mer",
)

FIELDS = ("hoststart", "hostlov_weeks", "hostslut", "varstart",
          "sportlov_weeks", "pasklov_weeks", "varslut")


def normalise_name(name: str) -> str:
    text = unicodedata.normalize("NFC", name).lower().strip()
    text = text.replace("–", "-")
    text = re.sub(r"[\s\-]+", "", text)
    text = re.sub(r"[^a-zåäöé]", "", text)
    for suffix in ("skommun", "kommun", "stad"):
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            text = text[: -len(suffix)]
    return text


def _date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_tokens(rest: str, school_year: tuple[int, int]) -> dict:
    """Recognise dates and week numbers in the value part of a row."""
    y1, y2 = school_year
    text = rest.replace("Vecka", " W").replace("vecka", " W")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split(" ")
    dates: list[tuple[int, int, int | None]] = []  # (month, day, explicit year)
    weeks: list[list[int]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip().rstrip(",")
        nxt = tokens[i + 1].strip().rstrip(".").lower() if i + 1 < len(tokens) else ""
        # YY-MM-DD
        m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", tok)
        if m:
            dates.append((int(m.group(2)), int(m.group(3)), 2000 + int(m.group(1))))
            i += 1
            continue
        # D-D-mon.  or  D-mon.  (range keeps the first day)
        m = re.fullmatch(r"(\d{1,2})(?:-\d{1,2})?-([a-zåäö]+)\.?", tok.lower())
        if m and m.group(2) in MONTHS:
            dates.append((MONTHS[m.group(2)], int(m.group(1)), None))
            i += 1
            continue
        # D/M or D-D/M
        m = re.fullmatch(r"(\d{1,2})(?:-\d{1,2})?/(\d{1,2})", tok)
        if m:
            dates.append((int(m.group(2)), int(m.group(1)), None))
            i += 1
            continue
        # D mon  or  D-D mon
        m = re.fullmatch(r"(\d{1,2})(?:-\d{1,2})?", tok)
        if m and nxt in MONTHS:
            dates.append((MONTHS[nxt], int(m.group(1)), None))
            i += 2
            continue
        # Week tokens: W 9, 9, 7-8, 43-44
        if tok == "W" and i + 1 < len(tokens):
            tok = tokens[i + 1]
            i += 1
        m = re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))?", tok)
        if m:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            if 1 <= a <= 53 and 1 <= b <= 53 and b >= a and b - a <= 3:
                weeks.append(list(range(a, b + 1)))
        i += 1

    out: dict = {field: None for field in FIELDS}
    for month, day, explicit_year in dates:
        if month in (7, 8, 9):
            year = explicit_year or y1
            if year == y1:
                out["hoststart"] = out["hoststart"] or _date(year, month, day)
        elif month in (11, 12):
            out["hostslut"] = out["hostslut"] or _date(explicit_year or y1, month, day)
        elif month in (1, 2):
            out["varstart"] = out["varstart"] or _date(explicit_year or y2, month, day)
        elif month in (5, 6):
            year = explicit_year or y2
            if year == y2:
                out["varslut"] = out["varslut"] or _date(year, month, day)
            # A June date in the first calendar year is a source typo for the
            # autumn start; it is ignored and filled later from the county.
    for week_list in weeks:
        first = week_list[0]
        if 40 <= first <= 46 and out["hostlov_weeks"] is None:
            out["hostlov_weeks"] = week_list
        elif 6 <= first <= 11 and out["sportlov_weeks"] is None:
            out["sportlov_weeks"] = week_list
        elif 12 <= first <= 18 and out["pasklov_weeks"] is None:
            out["pasklov_weeks"] = week_list
    return out


def parse_file(path: Path) -> tuple[list[dict], list[dict]]:
    m = re.search(r"(\d{4})-(\d{4})", path.name)
    school_year = (int(m.group(1)), int(m.group(2)))
    municipal: list[dict] = []
    summary: list[dict] = []
    county = None
    in_summary = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.replace("*", " ").rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Läsårstiderna länsvis"):
            in_summary = True
            continue
        header = re.fullmatch(r"(.+?) län, [A-Z]{1,2}", stripped)
        if header:
            key = header.group(1).lower()
            county = COUNTY_CODES.get(key)
            in_summary = False
            continue
        if any(marker in stripped for marker in SKIP_MARKERS):
            continue
        name_match = re.match(r"([A-Za-zÅÄÖåäöÉé][A-Za-zÅÄÖåäöÉé\s\-\.]*?)\s{2,}(.*)$", stripped)
        if not name_match:
            name_match = re.match(r"([A-Za-zÅÄÖåäöÉé][A-Za-zÅÄÖåäöÉé\-\.]*)\s+(\d.*|Vecka.*)$", stripped)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        rest = name_match.group(2)
        values = parse_tokens(rest, school_year)
        if sum(v is not None for v in values.values()) < 3:
            continue
        record = {"school_year": f"{school_year[0]}/{school_year[1]}", "name": name, **values}
        if in_summary:
            code = SUMMARY_COUNTIES.get(name.lower())
            if code:
                record["county_code"] = code
                summary.append(record)
        elif county is not None:
            record["county_code"] = county
            municipal.append(record)
    return municipal, summary


def main() -> None:
    population: dict[str, tuple[str, int]] = {}
    with POPULATION.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["year"] == "2024":
                population[row["municipality_code"]] = (row["municipality"], int(row["population"]))
    by_county_name: dict[tuple[str, str], str] = {}
    for code, (name, _) in population.items():
        by_county_name[(code[:2], normalise_name(name))] = code

    rows: list[dict] = []
    unmatched: list[str] = []
    for path in sorted(RAW.glob("skolporten_lasarstider_*.txt")):
        municipal, summary = parse_file(path)
        m = re.search(r"(\d{4})-(\d{4})", path.name)
        school_year = f"{m.group(1)}/{m.group(2)}"
        seen: dict[str, dict] = {}
        for record in municipal:
            key = (record["county_code"], normalise_name(record["name"]))
            code = by_county_name.get(key)
            if code is None:
                # Tolerate small spelling differences within the county.
                candidates = [
                    c for (county, norm), c in by_county_name.items()
                    if county == record["county_code"]
                    and (norm.startswith(key[1]) or key[1].startswith(norm))
                ]
                code = candidates[0] if len(candidates) == 1 else None
            if code is None:
                # A few rows are printed under the wrong county heading in the
                # source (for example Krokom under Halland); use a unique
                # national name match in that case.
                candidates = [
                    c for (_, norm), c in by_county_name.items() if norm == key[1]
                ]
                code = candidates[0] if len(candidates) == 1 else None
            if code is None:
                unmatched.append(f"{school_year}: {record['name']} ({record['county_code']})")
                continue
            seen[code] = record

        def emit(record: dict, code: str, name: str, source: str, pop: int | str) -> None:
            out = {
                "school_year": school_year,
                "municipality_code": code,
                "municipality": name,
                "county_code": code[:2],
                "population_2024": pop,
                "source": source,
            }
            for field in FIELDS:
                value = record.get(field)
                if isinstance(value, list):
                    value = "+".join(str(v) for v in value)
                elif isinstance(value, date):
                    value = value.isoformat()
                out[field] = "" if value is None else value
            rows.append(out)

        for code, record in sorted(seen.items()):
            name, pop = population[code]
            emit(record, code, name, "municipal", pop)
        for record in summary:
            emit(record, record["county_code"], f"{record['name']} (county summary)",
                 "county_summary", "")

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    if unmatched:
        print(f"{len(unmatched)} municipal rows could not be matched to SCB names:")
        for item in unmatched:
            print("  ", item)


if __name__ == "__main__":
    main()
