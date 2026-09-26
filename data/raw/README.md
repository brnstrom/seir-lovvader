# Skolporten source files

`scripts/parse_school_calendars.py` builds `data/school_calendar_municipal.csv`
from Skolporten's annual compilation "Grundskolans läsårstider" (one PDF per
school year, published at skolporten.se). The compilations are not openly
licensed, so the text versions are not included in the public repository;
the parsed municipal calendar in `data/` is enough to run the model.

To rebuild the calendar, download the PDFs for the school years you need and
convert each with

```bash
pdftotext -layout "Grundskolans läsårstider 2026-2027.pdf" skolporten_lasarstider_2026-2027.txt
```

into this directory (file names `skolporten_lasarstider_YYYY-YYYY.txt`), then
run `python scripts/parse_school_calendars.py` from the model directory. The
model uses the school years 2016/17 to 2026/27; add the next year's file
before the 2027/28 season.
