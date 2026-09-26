# SEIR model with school holidays and temperature

This directory contains a mechanistic forecasting model for the Swedish
Influenza Forecast Hub (model id `brannstrom-seirlovvader`), the data it uses
besides the hub target data, the code that estimates holiday and temperature
effects from earlier seasons, a retrospective evaluation on the 33 fixed
2025/26 rounds, and a weekly workflow for the live season. A Swedish results
report with figures is in `report/lov-vader-influensa.html`.

## Model

The model is the classical SEIR model written as ordinary differential
equations, with exponentially distributed latent (mean 1.6 days) and
infectious (mean 2.6 days) periods. Infections are measured in reported-case
units:

```text
dX/dt = (R(t) * gamma * I + iota) * (K - X) / K
dE/dt = dX/dt - sigma * E              1/sigma = 1.6 days
dI/dt = sigma * E - gamma * I          1/gamma = 2.6 days
```

`X` is cumulative infections since the start of the window, `K` the initial
susceptible pool expressed in reported units and `iota` a small importation
rate. `R(t)` is the reproduction number at the season's initial
susceptibility; the effective reproduction number is `R(t) (K - X) / K`. The
generation interval implied by the two stages has mean 4.2 days and SD 3.1
days.

The system is solved with the classical fourth-order Runge-Kutta method at
two fixed steps per day, with `R(t)` and the covariates held constant within
a day. Daily new infections are the increments of `X`. They are reported
after a gamma-distributed delay (mean 11 days, SD 4.5; the continuous delay
is mapped to whole days with a triangular kernel, which keeps its mean),
summed over ISO weeks (Monday to Sunday) and compared with the weekly counts
through a negative binomial likelihood. The model is therefore continuous in
time and evaluated in whole weeks; the step size is free. At the fitted
parameters the log likelihood with one, two or four steps per day differs
from that with eight steps by less than 0.01 (`python -m seirflu.selection
--steps-check`, `results/historical/rk4_steps_check.csv`).

Transmission varies as

```text
log R(t) = w(t) + theta_host h_host(t) + theta_jul h_jul(t) + theta_sport h_sport(t)
         + theta_pask h_pask(t) + theta_sommar h_sommar(t) + theta_T (T(t) - 5 °C)
```

`w(t)` is a weekly random walk (SD 0.07 per week, knots on Mondays, linearly
interpolated to days) that absorbs strain replacement and other slow change.
`h_*(t)` is the population-weighted fraction of municipalities whose schools
are closed for each holiday on day `t`, and `T(t)` the population-weighted
daily mean temperature. As an alternative to the log-linear temperature
effect, the model can use a sigmoid that is zero at 5 °C and levels off at
both cold and warm temperatures (`temperature="sigmoid"` in `ModelConfig`,
variant `sigmoid-temperature`):

```text
f(T) = A * [s((m - T) / b) - s((m - 5 °C) / b)],   s(z) = 1 / (1 + exp(-z))
```

`A` is the largest possible difference in log R between cold and warm
weather, `m` the midpoint and `b` the width. The sigmoid fits the training
seasons better, but the log-linear effect forecast better in every backtest
season and in 2025/26, so the main model uses it (see below).

On weekday public holidays a share of the day's reports is moved to the next
ordinary weekday; the share is estimated (about 29 %). The code is in
`seirflu/model.py` (simulation) and `seirflu/fit.py` (posterior mode with
L-BFGS and a Newton-Krylov polish, Laplace approximation, sampling), written
in JAX.

## Data besides the hub target data

`data/school_calendar_municipal.csv` holds the municipal school calendars
(term start and end dates, höstlov, sportlov and påsklov weeks) for school
years 2016/17 to 2026/27, parsed from Skolporten's annual compilation
"Grundskolans läsårstider" (parser in `scripts/parse_school_calendars.py`;
the compilations themselves are not openly licensed and are not included, see
`data/raw/README.md` for how to obtain them). Missing municipalities are filled from
the same municipality in the nearest earlier year (for påsklov, the position
relative to Easter is carried over from the earlier year with the most similar
Easter date) or from the county median; 2015/16 has no earlier year and is
filled from later ones. Municipal weights come from Statistics Sweden's 2024
population (`data/scb_municipal_population.csv`).

`data/weather_points_daily.csv` holds SMHI daily mean temperature (open data,
CC BY 4.0) from one station per county, with three stations in Skåne and
four in Västra Götaland, weighted by county population. Station gaps are
filled from the anomaly of the other stations, using station climatologies
fitted before July 2025. Refresh the file with `scripts/fetch_weather.py`.

## Estimated effects

The effects are estimated jointly from three non-overlapping series (Region
Skåne, Västra Götaland and the rest of Sweden) over the seasons 2015/16 to
2024/25, leaving out 2020/21 and 2021/22 because of COVID-19 and truncating
2019/20 on 15 March 2020. Each season window runs from ISO week 26 (four
unobserved burn-in weeks) to week 23. Estimates are posterior modes with 95 %
intervals from a Laplace approximation (`results/historical/effects.csv`);
the last column gives the range over fits that leave out one season at a time
(`results/historical/leave_one_season_out.csv`).

| Effect | Change in R | 95 % interval | Leave one season out |
| --- | --- | --- | --- |
| Jullov | -27.5 % | -30.1 to -24.9 % | -28.5 to -24.1 % |
| Sportlov | -18.8 % | -25.5 to -11.5 % | -20.1 to -16.6 % |
| Påsklov | -15.2 % | -23.5 to -5.9 % | -20.3 to -9.5 % |
| Höstlov | -6.1 % | -18.7 to +8.4 % | -11.6 to +2.7 % |
| Sommarlov | +17.3 % | +7.8 to +27.7 % | +11.3 to +21.5 % |
| Temperature, per 1 °C colder | +2.5 % | +2.1 to +2.9 % | +2.4 to +2.6 % |

With the sigmoid temperature effect the holiday estimates are almost the
same (jullov -27.5 %, sportlov -18.3 %, påsklov -15.0 %). They are larger in
terms of R than in the earlier version of this model with gamma-distributed
stages (jullov -21 %). The exponential stages give a longer generation
interval (4.2 instead of 3.1 days), and the same change in growth rate then
corresponds to a larger change in R; the data determine the growth rate.

The sigmoid has amplitude `A` = 0.73 (0.51 to 0.94), midpoint `m` = 7.5 °C
(5.1 to 9.9) and width `b` = 4.7 °C (3.2 to 6.9). Relative to R at 5 °C:

| Daily mean temperature | Log-linear (main model) | Sigmoid | 95 % interval, sigmoid |
| --- | --- | --- | --- |
| -15 °C | +64 % | +30 % | +21 to +39 % |
| -10 °C | +45 % | +29 % | +21 to +36 % |
| -5 °C | +28 % | +25 % | +19 to +30 % |
| 0 °C | +13 % | +16 % | +13 to +19 % |
| 10 °C | -12 % | -17 % | -20 to -14 % |
| 15 °C | -22 % | -28 % | -33 to -21 % |
| 20 °C | -31 % | -34 % | -40 to -23 % |

The sigmoid levels off in the cold: going from 5 °C to -5 °C raises R by
25 %, going on to -15 °C adds only another 4 %, whereas the log-linear term
keeps growing. The saturation agrees with models in which influenza
transmission depends on absolute humidity, which falls roughly exponentially
with temperature and is already low a few degrees below zero. Between about
1 and 14 °C, where much of the season's weather lies, the sigmoid is steeper
than the log-linear term (up to 3.9 % per degree), and in forecasts it reacts
more strongly to cold spells and to the seasonal cooling in autumn.

The summer estimate should not be read as a transmission increase. Influenza
is nearly absent in summer and the summer holiday coincides with the warmest
weeks. Without the temperature term the summer estimate is +1.5 % (-6.3 to
+9.8 %). The term does not affect forecasts for weeks 40 to 20.

## Model selection

Choices were compared on the training seasons with the Laplace approximation
to the marginal likelihood (`log evidence`, higher is better;
`python -m seirflu.selection`, output `results/historical/model_selection.csv`).
The comparisons are made around the best-fitting configuration (sigmoid
temperature effect, 11-day delay, random walk 0.07 per week), and every entry
differs from it in one respect; the main model is the entry with the
log-linear temperature effect.

| Configuration | log evidence | Difference |
| --- | --- | --- |
| Reference: sigmoid temperature, delay 11 days, random walk 0.07 | -3774.3 | +0.0 |
| Log-linear temperature effect (main model) | -3785.0 | -10.6 |
| Random-walk SD 0.08 per week | -3773.9 | +0.4 |
| Random-walk SD 0.09 per week | -3776.0 | -1.7 |
| Reporting delay 6.5 days | -3843.8 | -69.4 |
| Reporting delay 8 days | -3805.6 | -31.2 |
| Reporting delay 9.5 days | -3780.9 | -6.5 |
| Reporting delay 12.5 days | -3785.4 | -11.0 |
| Reporting delay 14 days | -3808.6 | -34.2 |
| Without the summer-holiday term | -3774.0 | +0.3 |
| Without the holiday reporting shift | -3775.1 | -0.8 |
| Without holiday effects | -3902.8 | -128.5 |
| Without temperature | -3886.3 | -112.0 |
| Without holidays and temperature | -4002.9 | -228.6 |

Holidays and temperature improve the fit by 128 and 112 log units. The
sigmoid temperature effect is 10.6 log units better than the log-linear one,
a clear difference in fit; the main model nevertheless uses the log-linear
effect because it forecast better (next section).

The data prefer an effective infection-to-report delay of about 11 days,
longer than incubation plus care seeking. A likely reason is that reported
cases are dominated by adults and older people, who are infected about one
generation after the school-age chains that holidays interrupt. The
random-walk scale matters little between 0.07 and 0.09 per week.

Exponential stages cost nothing in fit. With the settings of the earlier
version (log-linear temperature, 9.5-day delay) the ODE model has log
evidence -3794.9, against -3794.7 for the version with gamma-distributed
stages (commit 20a9848).

## Retrospective evaluation 2025/26

Scores from the hub's own pipeline (`scripts/hub_evaluation.py`, output in
`results/hub-evaluation/`), for the submitted point forecasts. Mean absolute
error:

| Location | Horizon 0 | 1 | 2 | 3 | Persistence (0 to 3) |
| --- | --- | --- | --- | --- | --- |
| Sweden | 98 | 222 | 372 | 523 | 173, 299, 414, 495 |
| Skåne | 8.8 | 12.9 | 15.9 | 20.1 | 17.8, 30.8, 41.8, 49.3 |
| Västra Götaland | 28.1 | 48.3 | 64.1 | 69.2 | 28.8, 48.5, 58.8, 65.1 |

Averaged over horizons, the MAE relative to persistence is 0.82 for Sweden,
0.43 for Skåne and 1.03 for Västra Götaland (0.87, 0.45 and 1.10 for the
earlier version with gamma-distributed stages). For Sweden the largest
errors come from late November and early December, when growth was explosive
and the model carried it forward, and from Christmas to mid-January, when it
expected more growth and a stronger rebound after the holiday than came. In
Västra Götaland January accounts for about half of the total error, and in
all three locations the second peak in February was missed.

The model's own predictive quantiles (Laplace approximation plus negative
binomial noise) can be compared with the hub's probabilistic normal-MA3
baseline. The weighted-interval-score skill (1 minus the WIS ratio) at
horizons 0 to 3 is 0.53, 0.41, 0.24, 0.07 for Sweden, 0.57, 0.61, 0.60, 0.58
for Skåne and 0.23, 0.07, 0.05, 0.10 for Västra Götaland. For Sweden, Skåne and
Västra Götaland the 95 % intervals cover 84, 100 and 89 % of outcomes and the
50 % intervals 33, 61 and 39 %; the intervals are too narrow for Sweden and
Västra Götaland.

Adding the model to the hub's QRA ensemble improves its WIS skill in
10 of 12 location-horizon combinations; it gets worse for Västra Götaland
at horizons 1 and 3.

The expected count at the posterior mode is used as the point forecast. The
posterior predictive mean from the Laplace approximation has a heavy right
tail, because an uncertain growth rate is compounded over four to five weeks of
infections that are not yet visible in the data. In rolling-origin backtests
on 2022/23 to 2024/25 (`python -m seirflu.backtest`, effects re-estimated
without the season being forecast) the MAE relative to persistence, averaged
over locations, is 0.78 for the posterior-mode count, 0.90 for the
predictive median and 1.08 for the predictive mean; on 2025/26 the predictive mean scores 0.98 against 0.76.

### Ablations

Each variant has its own historical estimation and is run through the same
33 rounds and, except for the last two rows, through the three backtest
seasons. Mean absolute error relative to persistence, averaged over horizons
(lower is better; posterior-mode point forecasts unless stated):

| Variant | 2025/26 Sweden | Skåne | Västra Götaland | Backtest Sweden | Skåne | Västra Götaland |
| --- | --- | --- | --- | --- | --- | --- |
| Main model (log-linear temperature) | 0.82 | 0.43 | 1.03 | 0.69 | 0.63 | 1.01 |
| Main model, predictive mean | 1.07 | 0.60 | 1.26 | 1.03 | 0.87 | 1.34 |
| Sigmoid temperature effect | 0.85 | 0.45 | 1.04 | 0.74 | 0.67 | 1.09 |
| Without temperature | 0.70 | 0.46 | 0.74 | 0.75 | 0.74 | 1.12 |
| Without holidays | 1.04 | 1.14 | 1.21 | 0.94 | 1.07 | 0.99 |
| Main model with observed weather after the cutoff | 0.73 | 0.48 | 0.99 | | | |
| Without holidays and temperature | 0.87 | 1.05 | 0.96 | | | |

Holidays improve the forecasts most: without them the average relative error
rises from 0.76 to 1.13 on 2025/26 and from 0.78 to 1.00 in the
backtests. The log-linear temperature effect forecasts better than the
sigmoid in 12 of 12 season-location combinations (backtests and
2025/26), although the sigmoid fits the training seasons better; the sigmoid
is steeper between about 1 and 14 °C and reacts more strongly to cold spells
and to the seasonal cooling in autumn. Temperature as such helps in the
backtests (0.78 with, 0.87 without) but made the 2025/26 forecasts
worse (0.76 with, 0.63 without), mainly for Sweden and Västra Götaland.
Observed weather after the cutoff gives 0.73, so a perfect weather forecast
removes only part of the gap; the cold spells in late November and in
January (weekly anomalies of about -5 °C) made the model expect more
transmission than came.

## Running

Python 3.11 with `jax`, `numpy`, `scipy`, `pandas`, `holidays` and `requests`
(see `requirements.txt`; `requirements-lock.txt` has the exact versions used).
From this directory:

```bash
python -m seirflu.historical                  # effects, priors, leave-one-season-out
python -m seirflu.historical --variant sigmoid-temperature --skip-loso
python -m seirflu.selection                   # model-selection table
python -m seirflu.retrospective               # 33 rounds, writes ../model-output/brannstrom-seirlovvader/
python -m seirflu.retrospective --variant no-holidays --scales 1 --no-hub-output
python -m seirflu.retrospective --weather oracle --scales 1 --no-hub-output
python -m seirflu.backtest 2022 2023 2024 --scales 1   # training-season backtests
python -m seirflu.backtest_eval main          # backtest scores
python -m seirflu.evaluate main               # 2025/26 scores for mean, median and mode
python scripts/hub_evaluation.py              # hub pipeline with this model included
python scripts/build_report_data.py && python scripts/render_report.py
python -m seirflu.weekly                      # live round: data check, weather, forecast
```

Setting `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
speeds things up when several runs share a machine. On one CPU core a
retrospective run takes about five minutes, one historical estimation about
three minutes and the leave-one-season-out fits another twenty.

The hub checkout is found as the parent directory; set `SEIRFLU_HUB_ROOT` to
use a checkout elsewhere (for example when this directory is its own
repository).

## Live season and weekly workflow

`python -m seirflu.weekly [YYYY-MM-DD]` makes the forecast for a live round
(by default the coming Sunday). It checks that the hub's
`target-data/time-series.csv` contains the week that ends on the data cutoff
(`reference_date - 7`; exit code 2 if not), refreshes the SMHI temperatures,
fits the model and writes `model-output/brannstrom-seirlovvader/<date>-brannstrom-seirlovvader.csv`
in the hub checkout, plus a summary under `results/live/`.

`.github/workflows/weekly-forecast.yml` runs it on Friday and Saturday
mornings, Saturday afternoon and Sunday afternoon (UTC) once this directory is
the root of its own GitHub repository. It validates the file with the hub's
validator, pushes a branch to your fork of the hub and opens or updates a pull
request; later runs push only if the forecast changed. The Sunday run passes
`--allow-missing-last-week`, so a late data release does not cost the round.
If the hub has not yet merged the model metadata, the workflow adds
`submission/model-metadata/brannstrom-seirlovvader.yml` to the pull request. It
also commits the week's summary under `results/live/` to the model
repository, which keeps a record and keeps GitHub from pausing the schedule
for inactivity. Rounds outside ISO weeks 40 to 20 are skipped. Setup is
described at the top of the file (a fork of the hub and a classic token with
the `public_repo` scope stored as the secret `HUB_PR_TOKEN`).

The school calendar for 2026/27 is already in `data/`; before the 2027/28
season add the next school year's Skolporten file to `data/raw/` (see the
README there) and rerun `scripts/parse_school_calendars.py`.

## Submission files

`submission/` has the hub files in the hub's own layout:
`model-metadata/brannstrom-seirlovvader.yml` and the 33 retrospective
forecasts in `model-output/brannstrom-seirlovvader/`. Copy both folders into
a branch of your fork of the hub and open a pull request. The retrospective
files can be regenerated from `results/retrospective/forecasts_main.csv` with
`python -m seirflu.retrospective --from-results --tag main` (writes into the
hub checkout given by `SEIRFLU_HUB_ROOT`).

## Limitations

The model treats influenza A and B as one epidemic; a second strain shows up
as a rise in the random walk rather than as a separate wave. Weather after the
cutoff is climatology plus a decaying anomaly, so cold spells are not
anticipated (`--weather oracle` measures what a perfect forecast would add).
Holiday effects are averages over the whole break and do not separate schools
from other changes in behaviour at the same time, such as travel and adults'
leave. The long reporting delay is an effective quantity that also absorbs
the age structure of transmission. The backtests on earlier seasons
re-estimate the holiday and temperature effects without the season being
forecast, but the prior for the susceptible pool uses all other post-COVID
seasons, including later ones.
