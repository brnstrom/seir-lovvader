"""Assemble the self-contained report page from the template and the data file.

Run from ``seir-model/`` after ``scripts/build_report_data.py``::

    python scripts/render_report.py            # writes report/lov-vader-influensa.html
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
template = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
data = (ROOT / "results" / "report" / "report_data.json").read_text(encoding="utf-8")
page = template.replace("/*__DATA__*/", data.replace("</", "<\\/"))
out = ROOT / "report" / "lov-vader-influensa.html"
out.write_text(page, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
