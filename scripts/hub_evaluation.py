"""Run the hub's own evaluation pipeline with the SEIR model included.

The committed ``evaluation-output/`` files are left untouched; all outputs go
to ``--output-root`` (default ``results/hub-evaluation`` in the model
directory). Requires ``pip install -e evaluation-pipeline`` in the hub
checkout (``SEIRFLU_HUB_ROOT`` if the model directory is not inside it).

Run from anywhere::

    python seir-model/scripts/hub_evaluation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from seirflu.data import HUB_ROOT  # noqa: E402


def run(*command: str) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=HUB_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=MODEL_ROOT / "results" / "hub-evaluation")
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    matched = out / "matched-forecasts.csv"
    run("match-retrospective-forecasts", "--output", str(matched), "--report", str(out / "match-report.json"))
    run("score-retrospective-point-forecasts", "--input", str(matched), "--output-root", str(out))
    run("build-retrospective-qra", "--input", str(matched), "--output-root", str(out))
    run("score-retrospective-qra", "--qra-input", str(out / "qra-forecasts.csv"),
        "--matched-input", str(matched), "--output-root", str(out))
    run("compare-retrospective-probabilistic-forecasts", "--qra-input", str(out / "qra-forecasts.csv"),
        "--baseline-input", "evaluation-output/2025-2026/normal-ma3-baseline-forecasts.csv",
        "--matched-input", str(matched), "--output-root", str(out))


if __name__ == "__main__":
    main()
