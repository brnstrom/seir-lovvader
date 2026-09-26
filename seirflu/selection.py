"""Model selection on the training seasons by Laplace marginal likelihood.

Each configuration is fitted jointly to Skåne, Västra Götaland and the rest of
Sweden over the training seasons, and its log evidence and effect estimates
are written to ``results/historical/model_selection.csv`` (one row per
configuration, replaced when rerun). The posterior mode of each fit is kept in
``results/historical/selection_fits/<name>.npy``.

Run from ``seir-model/``::

    python -m seirflu.selection                 # all configurations
    python -m seirflu.selection sigmoid linear  # some of them
    python -m seirflu.selection --steps-check linear

``--steps-check`` evaluates the log likelihood of a stored fit with 1, 2, 4
and 8 Runge-Kutta steps per day, which shows how large the ODE
discretisation error is at the fitted parameters.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from .fit import Problem, shared_names
from .historical import ESTIMATION_SERIES, RESULTS, TRAINING_SEASONS, estimate_effects, load_history
from .units import historical_units
from .variants import MAIN_CONFIG, with_delay, without_summer

# The comparisons are made around the best-fitting configuration: sigmoid
# temperature effect, 11-day reporting delay, random walk 0.07 per week. The
# main (forecasting) model is the same with a log-linear temperature effect,
# entry "linear". Every other entry differs from the reference in one respect
# (the delay entries scale the SD with the mean; the longest two allow delays
# up to 35 days). Because of this, an entry here can differ from the entry of
# the same name in ``variants.VARIANTS``, which are built around the main
# model (e.g. "no-holidays" is sigmoid-based here and log-linear there).
REFERENCE = MAIN_CONFIG.with_(temperature="sigmoid")
LINEAR = REFERENCE.with_(temperature="linear")
assert LINEAR == MAIN_CONFIG

CONFIGS = {
    "sigmoid": ("Sigmoid temperature, delay 11 days, random walk 0.07 (reference)", REFERENCE),
    "linear": ("Log-linear temperature effect (main forecasting model)", LINEAR),
    "no-temperature": ("Without temperature", REFERENCE.with_(temperature="none")),
    "delay6.5": ("Reporting delay 6.5 days", with_delay(REFERENCE, report_mean=6.5, report_sd=3.0)),
    "delay8": ("Reporting delay 8 days", with_delay(REFERENCE, report_mean=8.0, report_sd=3.5)),
    "delay9.5": ("Reporting delay 9.5 days", with_delay(REFERENCE, report_mean=9.5, report_sd=4.0)),
    "delay12.5": ("Reporting delay 12.5 days",
                  with_delay(REFERENCE, report_mean=12.5, report_sd=5.0, max_report=35)),
    "delay14": ("Reporting delay 14 days",
                with_delay(REFERENCE, report_mean=14.0, report_sd=5.5, max_report=35)),
    "rw0.08": ("Random-walk SD 0.08 per week", REFERENCE.with_(sigma_rw=0.08)),
    "rw0.09": ("Random-walk SD 0.09 per week", REFERENCE.with_(sigma_rw=0.09)),
    "no-summer": ("Without the summer-holiday term", without_summer(REFERENCE)),
    "linear-no-summer": ("Log-linear temperature without the summer-holiday term", without_summer(LINEAR)),
    "no-holidays": ("Without holiday effects", REFERENCE.with_(holidays=())),
    "no-holidays-no-temperature": ("Without holidays and temperature",
                                   REFERENCE.with_(holidays=(), temperature="none")),
    "no-red-days": ("Without the holiday reporting shift", REFERENCE.with_(red_day_reporting=False)),
    # Same settings as the earlier renewal model with gamma-distributed stages
    # (log-linear temperature, 9.5-day delay), for comparing the two.
    "linear-delay9.5": ("Log-linear temperature, reporting delay 9.5 days",
                        with_delay(LINEAR, report_mean=9.5, report_sd=4.0)),
}

OUTPUT = RESULTS / os.environ.get("SELECTION_FILE", "model_selection.csv")
FITS = RESULTS / "selection_fits"


def run(names: list[str]) -> None:
    history = load_history()
    FITS.mkdir(parents=True, exist_ok=True)
    for name in names:
        label, config = CONFIGS[name]
        started = time.time()
        fit = estimate_effects(config, wide=history)
        np.save(FITS / f"{name}.npy", np.asarray(fit["flat"]))
        row = {"config": name, "label": label, "log_evidence": fit["log_evidence"],
               "log_likelihood": fit["loglik"], "grad_max": fit["grad_max"]}
        row.update({n: v for n, v in zip(shared_names(), fit["theta"])})
        row.update({f"se_{n}": v for n, v in zip(shared_names(), fit["se"])})
        frame = pd.DataFrame([row])
        if OUTPUT.exists():
            existing = pd.read_csv(OUTPUT)
            frame = pd.concat([existing[existing["config"] != name], frame])
        frame.to_csv(OUTPUT, index=False)
        extra = ""
        if config.temperature == "sigmoid":
            extra = (f" A={row['theta_temp']:.3f} mid={row['temp_mid']:.1f} "
                     f"width={np.exp(row['temp_log_width']):.1f}")
        elif config.temperature == "linear":
            extra = f" slope={row['theta_temp']:.4f}"
        print(f"{name}: log evidence {fit['log_evidence']:.2f} (|g| {fit['grad_max']:.1e}, "
              f"{time.time() - started:.0f}s){extra}", flush=True)


def steps_check(name: str) -> pd.DataFrame:
    _, config = CONFIGS[name]
    flat = np.load(FITS / f"{name}.npy")
    history = load_history()
    rows = []
    for steps in (1, 2, 4, 8):
        cfg = with_delay(config, steps_per_day=steps)
        units = historical_units(history, ESTIMATION_SERIES, TRAINING_SEASONS, cfg)
        problem = Problem(units, cfg)
        rows.append({"config": name, "steps_per_day": steps,
                     "log_likelihood": float(problem.log_likelihood_terms(flat).sum()),
                     "objective": problem.objective_value(flat)})
    table = pd.DataFrame(rows)
    reference = table["log_likelihood"].iloc[-1]
    table["difference_to_8_steps"] = table["log_likelihood"] - reference
    table.to_csv(RESULTS / "rk4_steps_check.csv", index=False)
    print(table.to_string(index=False))
    return table


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--steps-check"]:
        steps_check(args[1] if len(args) > 1 else "linear")
        return
    run(args or list(CONFIGS))


if __name__ == "__main__":
    main()
