"""SEIR model with exponential stages, solved as an ODE.

Dynamics
--------
In reported-case units (infections times the reporting fraction) the model is

    dX/dt = (R(t) * gamma * I + iota) * (K - X) / K
    dE/dt = dX/dt - sigma * E
    dI/dt = sigma * E - gamma * I

``X`` is cumulative infections since the start of the window, so the
susceptible pool is ``K - X`` with ``K = rho * S_0``. ``1/sigma`` and
``1/gamma`` are the mean latent and infectious periods, and ``iota`` is a small
importation rate. ``R(t)`` is the reproduction number at the season's initial
susceptibility; the effective reproduction number is ``R(t) (K - X) / K``.

The system is integrated with the classical fourth-order Runge-Kutta method
using ``steps_per_day`` fixed steps per day (default 2, i.e. 12 hours; at the
fitted parameters one, four or eight steps change the log likelihood by less
than 0.01, see ``python -m seirflu.selection --steps-check``); ``R(t)`` and
the covariates are held constant within a day. Daily new infections are the
increments of ``X``.

Transmission
------------
    log R(t) = w(t) + sum_h theta_h * holiday_h(t) + f(T(t))

``w`` is a weekly random walk (linearly interpolated to days), ``holiday_h``
the population fraction with school closed for holiday type h, and ``f`` the
temperature effect, zero at 5 °C:

* ``linear``:  f(T) = theta_T * (T - 5), used by the main model
* ``sigmoid``: f(T) = A * [s((m - T) / w) - s((m - 5) / w)], s the logistic
  function; the effect saturates at both cold and warm temperatures.

Observation
-----------
Daily infections are reported after a gamma-distributed delay. On weekday
public holidays a fraction ``sigmoid(gamma_red)`` of the day's expected reports
is postponed to the next ordinary weekday. Reports are summed over ISO weeks
(Monday to Sunday); weekly counts are negative binomial.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy import stats

jax.config.update("jax_enable_x64", True)

HOLIDAY_TYPES = ("host", "jul", "sport", "pask", "sommar")
T_REF = 5.0
W_MAX = 53
TEMPERATURE_FORMS = ("linear", "sigmoid", "none")
# Numerical guards. They are far outside any plausible fit (R between 0.05
# and 20) and only keep the ODE finite at the extreme trial points an
# optimiser may try; inside the bounds they change nothing.
LOG_R_BOUND = 3.0
LOG_SCALE_BOUNDS = (-30.0, 30.0)


# ---------------------------------------------------------------------------
# Delays
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DelayConfig:
    latent_mean: float = 1.6
    infectious_mean: float = 2.6
    report_mean: float = 9.5
    report_sd: float = 4.0
    steps_per_day: int = 2
    max_report: int = 28


def report_delay(cfg: DelayConfig) -> np.ndarray:
    """Infection-to-report distribution h_d, d = 0..max_report (days).

    A continuous gamma delay is mapped to whole days with a triangular kernel
    (the difference between two times uniformly distributed within their
    days), which keeps the mean.
    """
    dt = 0.005
    tau = (np.arange(int((cfg.max_report + 20) / dt)) + 0.5) * dt
    shape = (cfg.report_mean / cfg.report_sd) ** 2
    density = stats.gamma(a=shape, scale=cfg.report_sd**2 / cfg.report_mean).pdf(tau) * dt
    h = np.array([np.sum(density * np.clip(1.0 - np.abs(tau - d), 0.0, None))
                  for d in range(cfg.max_report + 1)])
    return h / h.sum()


def generation_summary(cfg: DelayConfig) -> dict:
    """Mean and SD of the generation interval of the exponential SEIR model."""
    latent, infectious = cfg.latent_mean, cfg.infectious_mean
    return {"mean": latent + infectious, "sd": float(np.hypot(latent, infectious))}


def reproduction_from_growth(r: np.ndarray, cfg: DelayConfig) -> np.ndarray:
    """R implied by a daily growth rate r in the exponential SEIR model."""
    return (1.0 + r * cfg.latent_mean) * (1.0 + r * cfg.infectious_mean)


# ---------------------------------------------------------------------------
# Model configuration and unit data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    delays: DelayConfig = field(default_factory=DelayConfig)
    sigma_rw: float = 0.07  # weekly SD of the random walk in log R
    holidays: tuple[str, ...] = HOLIDAY_TYPES
    temperature: str = "linear"  # "linear", "sigmoid" or "none"
    red_day_reporting: bool = True
    log_r0_prior_sd: float = 0.3
    theta_prior_sd: float = 0.5
    temp_slope_prior_sd: float = 0.1  # linear: log R per °C
    temp_amplitude_prior_sd: float = 0.5  # sigmoid: log R between cold and warm limits
    temp_mid_prior: tuple[float, float] = (5.0, 6.0)  # sigmoid midpoint, °C
    temp_log_width_prior: tuple[float, float] = (float(np.log(3.0)), 0.6)  # sigmoid width, log °C
    red_prior_sd: float = 1.0
    log_phi_prior: tuple[float, float] = (float(np.log(50.0)), 1.5)

    def __post_init__(self) -> None:
        if self.temperature not in TEMPERATURE_FORMS:
            raise ValueError(f"temperature must be one of {TEMPERATURE_FORMS}")

    def holiday_mask(self) -> np.ndarray:
        return np.array([1.0 if h in self.holidays else 0.0 for h in HOLIDAY_TYPES])

    def with_(self, **changes) -> "ModelConfig":
        return replace(self, **changes)


@dataclass
class Unit:
    """Data for one series and one fitting window (numpy arrays, padded)."""

    series: str
    season: int
    start: date
    n_weeks: int  # weeks that belong to the window (<= W_MAX)
    y: np.ndarray  # (W_MAX,) weekly counts, 0 where unobserved
    obs: np.ndarray  # (W_MAX,) 1 if the week enters the likelihood
    hol: np.ndarray  # (7*W_MAX, 5)
    temp: np.ndarray  # (7*W_MAX,)
    red: np.ndarray  # (7*W_MAX,)
    red_dest: np.ndarray  # (7*W_MAX,) index of the next ordinary weekday
    log_k_prior: tuple[float, float]
    log_iota_prior: tuple[float, float]
    init_log_r: np.ndarray  # (W_MAX+1,) initial knot values

    @property
    def week_ends(self) -> list[date]:
        return [self.start + timedelta(days=7 * (w + 1) - 1) for w in range(W_MAX)]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "y": self.y, "obs": self.obs, "hol": self.hol, "temp": self.temp, "red": self.red,
            "red_dest": self.red_dest,
            "k_mu": np.float64(self.log_k_prior[0]), "k_sd": np.float64(self.log_k_prior[1]),
            "i_mu": np.float64(self.log_iota_prior[0]), "i_sd": np.float64(self.log_iota_prior[1]),
        }


def stack_units(units: list[Unit]) -> dict[str, jnp.ndarray]:
    arrays = [u.arrays() for u in units]
    return {key: jnp.asarray(np.stack([a[key] for a in arrays])) for key in arrays[0]}


def initial_log_r(y: np.ndarray, obs: np.ndarray, delays: DelayConfig,
                  report_shift_weeks: int = 1) -> np.ndarray:
    """Rough knot values from smoothed weekly growth rates."""
    n = len(y)
    logy = np.log(np.where(obs > 0, y, np.nan) + 1.0)
    idx = np.arange(n)
    good = np.isfinite(logy)
    if good.sum() < 3:
        return np.zeros(n + 1)
    logy = np.interp(idx, idx[good], logy[good])
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(np.pad(logy, 2, mode="edge"), kernel, mode="valid")
    growth = np.clip(np.gradient(smooth) / 7.0, -0.2, 0.3)  # per day
    log_r = np.log(np.clip(reproduction_from_growth(growth, delays), 0.5, 2.5))
    log_r = np.concatenate([log_r[report_shift_weeks:], np.repeat(log_r[-1], report_shift_weeks)])
    return np.concatenate([[log_r[0]], log_r])


# ---------------------------------------------------------------------------
# Simulation (JAX)
# ---------------------------------------------------------------------------
def daily_rw(knots):
    """Linear interpolation of W_MAX+1 weekly knots to 7*W_MAX days."""
    frac = jnp.tile(jnp.arange(7) / 7.0, W_MAX)
    left = jnp.repeat(knots[:-1], 7)
    right = jnp.repeat(knots[1:], 7)
    return left * (1.0 - frac) + right * frac


def temperature_effect(temp, shared, form: str):
    if form == "linear":
        return shared["theta_temp"] * (temp - T_REF)
    if form == "sigmoid":
        mid = shared["temp_mid"]
        width = jnp.exp(shared["temp_log_width"])
        return shared["theta_temp"] * (jax.nn.sigmoid((mid - temp) / width)
                                       - jax.nn.sigmoid((mid - T_REF) / width))
    return jnp.zeros_like(temp)


def log_reproduction(unit_params, shared, data, hol_mask, temp_form: str):
    log_r = daily_rw(unit_params["log_r"])
    log_r = log_r + data["hol"] @ (shared["theta_hol"] * hol_mask)
    return log_r + temperature_effect(data["temp"], shared, temp_form)


def simulate(unit_params, shared, data, h, sigma, gamma, steps_per_day: int,
             hol_mask, temp_form: str, use_red):
    """Return (weekly expected reports, daily infections, daily R, daily R_eff)."""
    log_r = log_reproduction(unit_params, shared, data, hol_mask, temp_form)
    r_t = jnp.exp(jnp.clip(log_r, -LOG_R_BOUND, LOG_R_BOUND))
    k = jnp.exp(jnp.clip(unit_params["log_k"], *LOG_SCALE_BOUNDS))
    iota = jnp.exp(jnp.clip(unit_params["log_iota"], *LOG_SCALE_BOUNDS))
    x0 = jnp.exp(jnp.clip(unit_params["log_x0"], *LOG_SCALE_BOUNDS))
    dt = 1.0 / steps_per_day

    def day(state, r_day):
        def rhs(s):
            new = (r_day * gamma * s[2] + iota) * jnp.maximum(k - s[0], 0.0) / k
            return jnp.stack([new, new - sigma * s[1], sigma * s[1] - gamma * s[2]])

        start = state
        for _ in range(steps_per_day):
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state, (state[0] - start[0], start[0])

    # Steady state for a constant infection rate x0 before the window.
    state0 = jnp.stack([jnp.asarray(0.0), x0 / sigma, x0 / gamma])
    _, (x, cum_before) = jax.lax.scan(day, state0, r_t)
    x_ext = jnp.concatenate([jnp.full(h.shape[0] - 1, x0), x])
    reports = jnp.convolve(x_ext, h, mode="valid")
    # Reporting on weekday public holidays: a fraction p of the day's reports
    # is postponed to the next ordinary weekday (totals are conserved).
    p_postpone = use_red * jax.nn.sigmoid(shared["gamma_red"])
    moved = reports * data["red"] * p_postpone
    reports = reports - moved + jnp.zeros_like(reports).at[data["red_dest"]].add(moved)
    weekly = reports.reshape(W_MAX, 7).sum(axis=1)
    r_eff = r_t * (k - cum_before) / k
    return weekly, x, r_t, r_eff


def nb_logpmf(y, mu, phi):
    mu = mu + 1e-9
    return (
        gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
        + phi * (jnp.log(phi) - jnp.log(phi + mu))
        + y * (jnp.log(mu) - jnp.log(phi + mu))
    )


def normal_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2 * jnp.pi)
