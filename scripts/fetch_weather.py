"""Download daily mean temperature from SMHI open data for the weather points.

Source: SMHI Meteorologiska observationer, parameter 2 (Lufttemperatur,
medelvärde 1 dygn), CC BY 4.0. For each point the quality-controlled
``corrected-archive`` is combined with ``latest-months`` (the last four
months). The result is cached in ``data/weather_points_daily.csv``; model
fitting never calls the network.

Run from ``seir-model/``::

    python scripts/fetch_weather.py
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seirflu.weather import WEATHER_POINTS  # noqa: E402

OUTPUT = ROOT / "data" / "weather_points_daily.csv"
BASE = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/{station}/period/{period}/data.csv"
FIRST_DATE = "2000-01-01"


def download(station: int, period: str) -> dict[str, float]:
    url = BASE.format(station=station, period=period)
    response = None
    for attempt in range(6):
        try:
            response = requests.get(url, timeout=120)
        except requests.RequestException as exc:  # transient network errors
            print(f"  retrying after {type(exc).__name__}")
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))
    else:
        status = response.status_code if response is not None else "no response"
        raise RuntimeError(f"SMHI request failed: {url} ({status})")
    text = response.content.decode("utf-8-sig")
    values: dict[str, float] = {}
    started = False
    for line in io.StringIO(text):
        parts = line.strip().split(";")
        if not started:
            started = parts[0].startswith("Från Datum")
            continue
        if len(parts) < 4 or not parts[3]:
            continue
        day = parts[2]
        if day >= FIRST_DATE:
            values[day] = float(parts[3])
    return values


def main() -> None:
    rows = []
    for point in WEATHER_POINTS:
        series: dict[str, float] = {}
        for period in ("corrected-archive", "latest-months"):
            series.update(download(point.smhi_station, period))
            time.sleep(0.5)
        print(f"{point.name:<13} station {point.smhi_station:>6}: {len(series)} days, "
              f"{min(series)}..{max(series)}")
        rows.extend(
            {"point": point.name, "station": point.smhi_station, "date": day, "temperature": value}
            for day, value in sorted(series.items())
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["point", "station", "date", "temperature"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
