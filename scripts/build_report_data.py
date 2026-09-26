"""Collect the numbers shown in the results report into one JSON file.

Run from the model directory after the historical estimation, the model
selection, the retrospective runs, the backtests and
``scripts/hub_evaluation.py``::

    python scripts/build_report_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seirflu.data import FINAL_OUTCOMES, HUB_ROOT, ROUND_ROOT, read_target_file  # noqa: E402
from seirflu.historical import TRAINING_SEASONS, temperature_multiplier  # noqa: E402
from seirflu.model import generation_summary, report_delay  # noqa: E402
from seirflu.variants import MAIN_DELAYS  # noqa: E402
from seirflu.weather import series_weather  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "report" / "report_data.json"
LOCATIONS = {"SE": "Sverige", "SE-M": "Region Skåne", "SE-O": "Västra Götaland"}
EFFECT_LABELS = {
    "theta_jul": "Jullov", "theta_sport": "Sportlov", "theta_pask": "Påsklov",
    "theta_host": "Höstlov", "theta_sommar": "Sommarlov",
}
# Log evidence of the earlier renewal model with gamma-distributed stages
# (log-linear temperature, 9.5-day delay; commit 20a9848,
# results/historical/model_selection.csv, row "main").
GAMMA_MODEL_LOG_EVIDENCE = -3794.66


def pct(x: float) -> float:
    return float(100 * (np.exp(x) - 1))


def load_prior(variant: str) -> dict:
    return json.loads((RESULTS / "historical" / f"forecast_prior_{variant}.json").read_text())


def temperature_curves(grid: np.ndarray) -> dict:
    out = {}
    rng = np.random.default_rng(11)
    for key, variant, form in (("linear", "main", "linear"), ("sigmoid", "sigmoid-temperature", "sigmoid")):
        prior = load_prior(variant)
        mean, cov = np.asarray(prior["mean"]), np.asarray(prior["cov"])
        draws = rng.multivariate_normal(mean, cov, size=4000)
        curves = np.array([temperature_multiplier(d, grid, form) for d in draws])
        out[key] = {
            "est": temperature_multiplier(mean, grid, form).round(4).tolist(),
            "lo": np.quantile(curves, 0.025, axis=0).round(4).tolist(),
            "hi": np.quantile(curves, 0.975, axis=0).round(4).tolist(),
        }
    return out


def season_temperatures(bins: np.ndarray) -> list[int]:
    """Days per temperature bin in weeks 40 to 20 of the training seasons."""
    counts = np.zeros(len(bins) - 1, dtype=int)
    for series in ("SE-M", "SE-O", "SE-rest"):
        weather = series_weather(series)["temperature"]
        for season in TRAINING_SEASONS:
            start = pd.Timestamp(date.fromisocalendar(season, 40, 1))
            end = pd.Timestamp(date.fromisocalendar(season + 1, 20, 7))
            if season == 2019:
                end = pd.Timestamp("2020-03-15")
            values = weather[(weather.index >= start) & (weather.index <= end)].to_numpy()
            counts += np.histogram(np.clip(values, bins[0], bins[-1] - 1e-9), bins=bins)[0]
    return counts.tolist()


EFFECT_COLUMNS = [f"theta_{h}" for h in ("host", "jul", "sport", "pask", "sommar")] + [
    "theta_temp", "temp_mid", "temp_log_width", "gamma_red"]


def effects() -> dict:
    hist = RESULTS / "historical"
    table = pd.read_csv(hist / "effects.csv").set_index("parameter")
    loso = pd.read_csv(hist / "leave_one_season_out.csv")
    sigmoid = pd.read_csv(hist / "effects_sigmoid-temperature.csv").set_index("parameter")
    loso_sigmoid = pd.read_csv(hist / "leave_one_season_out_sigmoid-temperature.csv")
    no_temp = pd.read_csv(hist / "effects_no-temperature.csv").set_index("parameter")
    rows = []
    for key, label in EFFECT_LABELS.items():
        est, se = table.at[key, "estimate"], table.at[key, "se"]
        rows.append({
            "key": key, "label": label,
            "est": pct(est), "lo": pct(est - 1.96 * se), "hi": pct(est + 1.96 * se),
            "loso": [pct(v) for v in loso[key]],
            "noTemp": pct(no_temp.at[key, "estimate"]),
            "sigmoid": pct(sigmoid.at[key, "estimate"]),
        })
    grid = np.arange(-15.0, 22.01, 0.5)
    bins = np.arange(-16.0, 23.0, 1.0)
    points = {}
    for temp in (-15, -10, -5, 0, 10, 15, 20):
        name = f"R_at_{temp}C_vs_5C"
        points[str(temp)] = {
            form: {"est": float(t.at[name, "relative_change_in_R_percent"]),
                   "lo": float(t.at[name, "ci_low_percent"]), "hi": float(t.at[name, "ci_high_percent"])}
            for form, t in (("linear", table), ("sigmoid", sigmoid))
        }
    loso_curves = [temperature_multiplier(row, np.array([-15.0, -5.0]), "sigmoid")
                   for row in loso_sigmoid[EFFECT_COLUMNS].to_numpy()]
    lin = table.loc["theta_temp"]
    temperature = {
        "grid": grid.tolist(),
        "curves": temperature_curves(grid),
        "bins": bins.tolist(),
        "days": season_temperatures(bins),
        "points": points,
        "perDegreeColder": pct(-lin["estimate"]),
        "perDegreeLo": pct(-lin["estimate"] - 1.96 * lin["se"]),
        "perDegreeHi": pct(-lin["estimate"] + 1.96 * lin["se"]),
        "losoPerDegree": [pct(-v) for v in loso["theta_temp"]],
        "amplitude": float(sigmoid.at["theta_temp", "estimate"]),
        "amplitudeLo": float(sigmoid.at["theta_temp", "ci_low"]),
        "amplitudeHi": float(sigmoid.at["theta_temp", "ci_high"]),
        "mid": float(sigmoid.at["temp_mid", "estimate"]),
        "midLo": float(sigmoid.at["temp_mid", "ci_low"]),
        "midHi": float(sigmoid.at["temp_mid", "ci_high"]),
        "width": float(np.exp(sigmoid.at["temp_log_width", "estimate"])),
        "widthLo": float(np.exp(sigmoid.at["temp_log_width", "ci_low"])),
        "widthHi": float(np.exp(sigmoid.at["temp_log_width", "ci_high"])),
        "losoSigmoidCold": [100 * (c[1] - 1) for c in loso_curves],
        "losoSigmoidFreeze": [100 * (c[0] - 1) for c in loso_curves],
    }
    red = table.loc["gamma_red"]
    return {"holidays": rows, "temperature": temperature, "postponedShare": float(red["postponed_share"])}


REFERENCE = "sigmoid"
SELECTION_LABELS = {
    "sigmoid": "Sigmoid, 11 dagar, 0,07 per vecka (referens)",
    "linear": "Log-linjär temperatur (huvudmodell)",
    "rw0.08": "Slumpvandring 0,08 per vecka",
    "rw0.09": "Slumpvandring 0,09 per vecka",
    "delay9.5": "Rapportfördröjning 9,5 dagar",
    "delay12.5": "Rapportfördröjning 12,5 dagar",
    "delay8": "Rapportfördröjning 8 dagar",
    "delay14": "Rapportfördröjning 14 dagar",
    "delay6.5": "Rapportfördröjning 6,5 dagar",
    "no-summer": "Utan sommarlovsterm",
    "no-red-days": "Utan fördröjd rapportering på helgdagar",
    "no-holidays": "Utan lov",
    "no-temperature": "Utan temperatur",
    "no-holidays-no-temperature": "Utan lov och temperatur",
}


def evidence() -> dict:
    table = pd.read_csv(RESULTS / "historical" / "model_selection.csv").set_index("config")
    base = float(table.at[REFERENCE, "log_evidence"])
    rows = []
    for key, label in SELECTION_LABELS.items():
        if key in table.index:
            value = float(table.at[key, "log_evidence"])
            rows.append({"key": key, "label": label, "logZ": value, "delta": value - base})
    delays = [{"mean": m, "delta": float(table.at[k, "log_evidence"]) - base}
              for m, k in ((6.5, "delay6.5"), (8.0, "delay8"), (9.5, "delay9.5"), (11.0, REFERENCE),
                           (12.5, "delay12.5"), (14.0, "delay14")) if k in table.index]
    out = {"rows": rows, "delays": delays, "reference": REFERENCE, "referenceLogZ": base}
    if "linear-delay9.5" in table.index:
        out["gammaComparison"] = {
            "gamma": GAMMA_MODEL_LOG_EVIDENCE,
            "ode": float(table.at["linear-delay9.5", "log_evidence"]),
        }
    steps = RESULTS / "historical" / "rk4_steps_check.csv"
    if steps.exists():
        s = pd.read_csv(steps)
        out["steps"] = [{"steps": int(r.steps_per_day), "diff": float(r.difference_to_8_steps)}
                        for r in s.itertuples(index=False)]
    return out


def delays() -> dict:
    t = np.arange(0.0, 30.01, 0.25)
    sigma, gamma = 1.0 / MAIN_DELAYS.latent_mean, 1.0 / MAIN_DELAYS.infectious_mean
    generation = sigma * gamma / (sigma - gamma) * (np.exp(-gamma * t) - np.exp(-sigma * t))
    h = report_delay(MAIN_DELAYS)
    return {
        "t": t.tolist(),
        "generation": generation.round(5).tolist(),
        "generationSummary": generation_summary(MAIN_DELAYS),
        "report": h.round(5).tolist(),
        "reportMean": MAIN_DELAYS.report_mean,
        "reportSd": MAIN_DELAYS.report_sd,
    }


def example_season(series: str = "SE-M", season: int = 2024) -> dict:
    fitted = pd.read_csv(RESULTS / "historical" / "fitted_training_seasons.csv")
    d = fitted[(fitted["series"] == series) & (fitted["season"] == season)]
    d = d[(d["week_end"] >= f"{season}-09-01") & (d["week_end"] <= f"{season + 1}-05-31")]
    return {
        "series": series, "season": f"{season}/{str(season + 1)[2:]}",
        "weeks": d["week_end"].tolist(),
        "observed": [None if pd.isna(v) else float(v) for v in d["observed"]],
        "fitted": d["fitted"].round(2).tolist(),
        "rEff": d["r_eff"].round(3).tolist(),
        "rT": d["r_t"].round(3).tolist(),
        "temperature": d["temperature"].round(1).tolist(),
        **{h: d[f"hol_{h}"].round(3).tolist() for h in ("host", "jul", "sport", "pask")},
    }


def retrospective() -> dict:
    forecasts = pd.read_csv(RESULTS / "retrospective" / "forecasts_main.csv")
    forecasts = forecasts[forecasts["future_rw_scale"] == 1.0]
    outcomes = read_target_file(FINAL_OUTCOMES)
    history = read_target_file(ROUND_ROOT / "2025-10-05.csv")
    observed = pd.concat([history.loc["2025-08-01":], outcomes]).groupby(level=0).last()
    out = {}
    for loc in LOCATIONS:
        rounds = []
        for ref, g in forecasts[forecasts["location"] == loc].groupby("reference_date"):
            cutoff = (pd.Timestamp(ref) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            g = g.sort_values("horizon")
            rounds.append({
                "ref": ref, "cutoff": cutoff,
                "last": float(observed.at[pd.Timestamp(cutoff), loc]),
                "dates": g["target_end_date"].tolist(),
                "point": g["map_mean"].round(2).tolist(),
                "mean": g["mean"].round(2).tolist(),
                "q10": g["q0.1"].tolist(), "q90": g["q0.9"].tolist(),
                "q025": g["q0.025"].tolist(), "q975": g["q0.975"].tolist(),
            })
        series = observed[loc].dropna()
        out[loc] = {
            "label": LOCATIONS[loc],
            "observed": [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in series.items()],
            "rounds": rounds,
        }
    return out


def hub_scores() -> dict:
    metrics = pd.read_csv(RESULTS / "hub-evaluation" / "point-metrics-by-location-horizon.csv")
    keep = {"brannstrom-seirlovvader": "SEIR (denna modell)", "hubdemo-persistence": "Persistens",
            "hubdemo-damped4": "Dämpad trend", "hubdemo-mean3": "Medel av 3 veckor"}
    out = {}
    for loc in LOCATIONS:
        m = metrics[metrics["location"] == loc]
        out[loc] = {
            label: {
                "mae": m[m["model_id"] == model].sort_values("horizon")["mae"].round(2).tolist(),
                "rmse": m[m["model_id"] == model].sort_values("horizon")["rmse"].round(2).tolist(),
                "bias": m[m["model_id"] == model].sort_values("horizon")["bias"].round(2).tolist(),
            }
            for model, label in keep.items()
        }
    comparison = pd.read_csv(RESULTS / "hub-evaluation" / "probabilistic-comparison-by-location-horizon.csv")
    before = pd.read_csv(HUB_ROOT / "evaluation-output" / "2025-2026" / "probabilistic-comparison-by-location-horizon.csv")
    qra = {}
    for loc in LOCATIONS:
        qra[loc] = {
            "before": before[before["location"] == loc].sort_values("horizon")["wis_skill"].round(3).tolist(),
            "after": comparison[comparison["location"] == loc].sort_values("horizon")["wis_skill"].round(3).tolist(),
        }
    return {"points": out, "qra": qra}


def probabilistic_wis() -> dict:
    """WIS of the model's own quantiles against the hub's normal-MA3 baseline."""
    from seirflu.evaluate import wis

    forecasts = pd.read_csv(RESULTS / "retrospective" / "forecasts_main.csv")
    forecasts = forecasts[forecasts["future_rw_scale"] == 1.0]
    outcomes = pd.read_csv(FINAL_OUTCOMES)
    outcomes = outcomes[outcomes["status"] == "available"][["location", "target_end_date", "value"]]
    f = forecasts.merge(outcomes, on=["location", "target_end_date"]).rename(columns={"value": "observed"})
    f["wis"] = f.apply(wis, axis=1)
    f["cov95"] = ((f["observed"] >= f["q0.025"]) & (f["observed"] <= f["q0.975"])).astype(float)
    f["cov50"] = ((f["observed"] >= f["q0.25"]) & (f["observed"] <= f["q0.75"])).astype(float)
    base = pd.read_csv(RESULTS / "hub-evaluation" / "normal-ma3-baseline-scores.csv")
    m = f.merge(base[["reference_date", "location", "horizon", "wis"]].rename(columns={"wis": "wis_base"}),
                on=["reference_date", "location", "horizon"])
    out = {}
    for loc in LOCATIONS:
        g = m[m["location"] == loc].groupby("horizon")
        out[loc] = {
            "skill": (1 - g["wis"].mean() / g["wis_base"].mean()).round(3).tolist(),
            "wis": g["wis"].mean().round(2).tolist(),
            "wisBase": g["wis_base"].mean().round(2).tolist(),
            "cov50": g["cov50"].mean().round(3).tolist(),
            "cov95": g["cov95"].mean().round(3).tolist(),
        }
    return out


_PERSISTENCE_BACKTEST: pd.Series | None = None


def _persistence_backtest() -> pd.Series:
    """Persistence MAE per (season, location, horizon) on the backtest rounds."""
    global _PERSISTENCE_BACKTEST
    if _PERSISTENCE_BACKTEST is None:
        from seirflu.backtest import reference_dates
        from seirflu.backtest_eval import baseline_forecasts
        from seirflu.historical import load_history

        history = load_history()
        frames = []
        for season in (2022, 2023, 2024):
            base = baseline_forecasts(reference_dates(season), history)
            base = base[base["model"] == "hubdemo-persistence"].copy()
            base["season"] = season
            frames.append(base)
        base = pd.concat(frames)
        obs = history[list(LOCATIONS)].stack().rename("observed").reset_index()
        obs.columns = ["target_end_date", "location", "observed"]
        obs["target_end_date"] = obs["target_end_date"].dt.strftime("%Y-%m-%d")
        base = base.merge(obs, on=["target_end_date", "location"])
        base["ae"] = (base["point"] - base["observed"]).abs()
        _PERSISTENCE_BACKTEST = base.groupby(["season", "location", "horizon"])["ae"].mean()
    return _PERSISTENCE_BACKTEST


def backtest_relative(variant: str, column: str = "map_mean") -> dict | None:
    paths = sorted((RESULTS / "backtest").glob(f"backtest_{variant}_*.csv"))
    if len(paths) < 3:
        return None
    table = pd.concat([pd.read_csv(p) for p in paths])
    if "future_rw_scale" in table:
        table = table[table["future_rw_scale"] == 1.0]
    table = table.dropna(subset=["observed"])
    table["ae"] = (table[column] - table["observed"]).abs()
    mae = table.groupby(["season", "location", "horizon"])["ae"].mean()
    rel = (mae / _persistence_backtest().reindex(mae.index)).groupby(level="location").mean()
    return {loc: float(rel[loc]) for loc in LOCATIONS}


def ablations() -> list[dict]:
    outcomes = pd.read_csv(FINAL_OUTCOMES)
    outcomes = outcomes[outcomes["status"] == "available"][["location", "target_end_date", "value"]]
    persistence = pd.read_csv(RESULTS / "hub-evaluation" / "point-errors.csv")
    persistence = persistence[persistence["model_id"] == "hubdemo-persistence"]
    pers_mae = persistence.groupby(["location", "horizon"])["absolute_error"].mean()
    variants = [
        ("main", "main", "Huvudmodell (log-linjär temperatur)", "map_mean"),
        ("main", "main", "Huvudmodell, prediktivt medelvärde", "mean"),
        ("sigmoid-temperature", "sigmoid-temperature", "Sigmoid för temperaturen", "map_mean"),
        ("no-temperature", "no-temperature", "Utan temperatur", "map_mean"),
        ("no-holidays", "no-holidays", "Utan lov", "map_mean"),
        ("main-oracle-weather", None, "Huvudmodell med faktiskt väder efter cutoff", "map_mean"),
        ("no-holidays-no-temperature", None, "Utan lov och temperatur", "map_mean"),
    ]
    rows = []
    for tag, backtest_tag, label, column in variants:
        path = RESULTS / "retrospective" / f"forecasts_{tag}.csv"
        if not path.exists():
            continue
        f = pd.read_csv(path)
        if "future_rw_scale" in f:
            f = f[f["future_rw_scale"] == 1.0]
        f = f.merge(outcomes, on=["location", "target_end_date"])
        f["abs_error"] = (f[column] - f["value"]).abs()
        mae = f.groupby(["location", "horizon"])["abs_error"].mean()
        rel = (mae / pers_mae).groupby(level=0).mean()
        row = {"tag": tag, "label": label, "point": column}
        for loc in LOCATIONS:
            row[loc] = float(rel.get(loc, np.nan))
            row[f"{loc}_mae"] = [float(v) for v in mae.loc[loc].sort_index()]
        row["backtest"] = backtest_relative(backtest_tag, column) if backtest_tag else None
        rows.append(row)
    return rows


def form_comparison() -> dict:
    """Season-location pairs where the log-linear main beats the sigmoid variant.

    Mean absolute error of the submitted point forecast, averaged over
    horizons; the three backtest seasons and 2025/26.
    """
    def mae(table: pd.DataFrame, observed: str) -> pd.Series:
        if "future_rw_scale" in table:
            table = table[table["future_rw_scale"] == 1.0]
        table = table.dropna(subset=[observed])
        err = (table["map_mean"] - table[observed]).abs()
        return err.groupby([table["location"], table["horizon"]]).mean().groupby(level=0).mean()

    wins = total = 0
    for season in (2022, 2023, 2024):
        a = RESULTS / "backtest" / f"backtest_main_{season}.csv"
        b = RESULTS / "backtest" / f"backtest_sigmoid-temperature_{season}.csv"
        if not (a.exists() and b.exists()):
            continue
        ma, mb = mae(pd.read_csv(a), "observed"), mae(pd.read_csv(b), "observed")
        wins += int((ma < mb).sum()); total += len(ma)
    outcomes = pd.read_csv(FINAL_OUTCOMES)
    outcomes = outcomes[outcomes["status"] == "available"][["location", "target_end_date", "value"]]
    retro = {}
    for tag in ("main", "sigmoid-temperature"):
        f = pd.read_csv(RESULTS / "retrospective" / f"forecasts_{tag}.csv").merge(outcomes, on=["location", "target_end_date"])
        retro[tag] = mae(f, "value")
    wins += int((retro["main"] < retro["sigmoid-temperature"]).sum()); total += len(retro["main"])
    return {"linearWins": wins, "total": total}


def backtest() -> dict:
    path = RESULTS / "backtest" / "scored_main.csv"
    if not path.exists():
        return {}
    s = pd.read_csv(path)
    labels = {"seir-map@rw1": "Förväntat antal vid posteriormoden", "seir-median@rw1": "Prediktiv median",
              "seir-mean@rw1": "Prediktivt medelvärde", "hubdemo-damped4": "Dämpad trend",
              "hubdemo-persistence": "Persistens"}
    s = s[s["model"].isin(labels)]
    mae = s.groupby(["model", "season", "location", "horizon"])["abs_error"].mean().reset_index()
    pers = mae[mae["model"] == "hubdemo-persistence"].set_index(["season", "location", "horizon"])["abs_error"]
    mae["rel"] = mae.apply(lambda r: r["abs_error"] / pers.loc[(r["season"], r["location"], r["horizon"])], axis=1)
    out = {"seasons": sorted(int(x) for x in s["season"].unique()), "rows": []}
    for model, label in labels.items():
        m = mae[mae["model"] == model]
        row = {"label": label, "model": model}
        for loc in LOCATIONS:
            row[loc] = float(m[m["location"] == loc]["rel"].mean())
        out["rows"].append(row)
    return out


def weather_2025() -> dict:
    """Weekly temperature anomalies in the 2025/26 season (population-weighted, Sweden)."""
    from seirflu.weather import climatology

    weather = series_weather("SE")["temperature"]
    days = pd.date_range("2025-09-29", "2026-05-24", freq="D")
    observed = weather.reindex(days)
    clim = pd.Series(climatology("SE", "temperature", days, 2025), index=days)
    weekly = (observed - clim).resample("W-SUN").mean()
    return {"weeks": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "anomaly": weekly.round(2).tolist(),
            "temperature": observed.resample("W-SUN").mean().round(2).tolist()}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "effects": effects(),
        "evidence": evidence(),
        "delays": delays(),
        "example": example_season(),
        "retro": retrospective(),
        "hub": hub_scores(),
        "probWis": probabilistic_wis(),
        "ablations": ablations(),
        "backtest": backtest(),
        "backtestBySeason": form_comparison(),
        "weather2025": weather_2025(),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
