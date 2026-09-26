"""Posterior mode, Laplace approximation and sampling for the SEIR model.

All JAX functions are compiled once per (model configuration, number of
units, number of series) and take the data and priors as arguments, so a
retrospective run over many rounds compiles only once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy import optimize

from .model import (
    HOLIDAY_TYPES,
    W_MAX,
    ModelConfig,
    Unit,
    nb_logpmf,
    normal_logpdf,
    report_delay,
    simulate,
    stack_units,
)

# Shared effects: theta_hol (5), theta_temp, temp_mid, temp_log_width, gamma_red.
# theta_temp is the slope per °C (linear form) or the amplitude (sigmoid form);
# temp_mid and temp_log_width are only used by the sigmoid form.
N_EFFECTS = 9
SHARED_KEYS = ("theta_temp", "temp_mid", "temp_log_width", "gamma_red")


def shared_names() -> list[str]:
    return [f"theta_{h}" for h in HOLIDAY_TYPES] + list(SHARED_KEYS)


@dataclass
class SharedPrior:
    """Multivariate normal prior on the shared effects (see ``shared_names``).

    Also carries per-series priors for log dispersion and log K, which are
    estimated from the historical seasons.
    """

    mean: np.ndarray
    cov: np.ndarray
    log_phi: dict[str, tuple[float, float]]
    log_k: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict:
        return {
            "mean": np.asarray(self.mean).tolist(),
            "cov": np.asarray(self.cov).tolist(),
            "log_phi": {k: list(v) for k, v in self.log_phi.items()},
            "log_k": None if self.log_k is None else {k: list(v) for k, v in self.log_k.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SharedPrior":
        return cls(
            mean=np.asarray(data["mean"], dtype=float),
            cov=np.asarray(data["cov"], dtype=float),
            log_phi={k: tuple(v) for k, v in data["log_phi"].items()},
            log_k=None if data.get("log_k") is None else {k: tuple(v) for k, v in data["log_k"].items()},
        )


def _template(n_units: int, n_series: int) -> dict:
    return {
        "units": {
            "log_r": jnp.zeros((n_units, W_MAX + 1)),
            "log_k": jnp.zeros(n_units),
            "log_iota": jnp.zeros(n_units),
            "log_x0": jnp.zeros(n_units),
        },
        "shared": {
            "theta_hol": jnp.zeros(5),
            **{key: jnp.asarray(0.0) for key in SHARED_KEYS},
            "log_phi": jnp.zeros(n_series),
        },
    }


@lru_cache(maxsize=64)
def compiled(cfg: ModelConfig, n_units: int, n_series: int) -> SimpleNamespace:
    h = jnp.asarray(report_delay(cfg.delays))
    sigma = 1.0 / cfg.delays.latent_mean
    gamma = 1.0 / cfg.delays.infectious_mean
    steps = int(cfg.delays.steps_per_day)
    hol_mask = jnp.asarray(cfg.holiday_mask())
    use_red = 1.0 if cfg.red_day_reporting else 0.0
    flat_template, unravel = ravel_pytree(_template(n_units, n_series))

    def simulate_all(params, data):
        shared = params["shared"]

        def one(unit_params, unit_data):
            return simulate(unit_params, shared, unit_data, h, sigma, gamma, steps,
                            hol_mask, cfg.temperature, use_red)

        return jax.vmap(one)(params["units"], data)

    def loglik_terms(flat, data, series_idx):
        params = unravel(flat)
        weekly, *_ = simulate_all(params, data)
        phi = jnp.exp(params["shared"]["log_phi"])[series_idx][:, None]
        return data["obs"] * nb_logpmf(data["y"], weekly, phi)

    def log_prior(flat, data, prior):
        params = unravel(flat)
        up = params["units"]
        lp = jnp.sum(normal_logpdf(up["log_r"][:, 0], 0.0, cfg.log_r0_prior_sd))
        lp += jnp.sum(normal_logpdf(jnp.diff(up["log_r"], axis=1), 0.0, cfg.sigma_rw))
        lp += jnp.sum(normal_logpdf(up["log_k"], data["k_mu"], data["k_sd"]))
        lp += jnp.sum(normal_logpdf(up["log_iota"], data["i_mu"], data["i_sd"]))
        lp += jnp.sum(normal_logpdf(up["log_x0"], data["i_mu"], data["i_sd"]))
        sh = params["shared"]
        vec = jnp.concatenate([sh["theta_hol"]] + [sh[key][None] for key in SHARED_KEYS])
        diff = vec - prior["mean"]
        lp += -0.5 * diff @ prior["precision"] @ diff + prior["log_norm"]
        lp += jnp.sum(normal_logpdf(sh["log_phi"], prior["phi_mu"], prior["phi_sd"]))
        return lp

    def objective(flat, data, series_idx, prior):
        return -(jnp.sum(loglik_terms(flat, data, series_idx)) + log_prior(flat, data, prior))

    grad = jax.grad(objective)

    def hvp(flat, vector, data, series_idx, prior):
        return jax.jvp(lambda x: grad(x, data, series_idx, prior), (flat,), (vector,))[1]

    def hvp_batch(flat, vectors, data, series_idx, prior):
        return jax.vmap(lambda v: hvp(flat, v, data, series_idx, prior))(vectors)

    def simulate_flat(flat, data):
        return simulate_all(unravel(flat), data)

    return SimpleNamespace(
        unravel=unravel,
        n_params=int(flat_template.shape[0]),
        value_and_grad=jax.jit(jax.value_and_grad(objective)),
        objective=jax.jit(objective),
        hvp=jax.jit(hvp),
        hvp_batch=jax.jit(hvp_batch),
        loglik_terms=jax.jit(loglik_terms),
        simulate=jax.jit(simulate_flat),
        simulate_batch=jax.jit(jax.vmap(simulate_flat, in_axes=(0, None))),
    )


class Problem:
    """A set of units (series x window) sharing holiday/weather/reporting effects."""

    def __init__(self, units: list[Unit], config: ModelConfig, shared_prior: SharedPrior | None = None,
                 series_names: list[str] | None = None) -> None:
        self.units = units
        self.cfg = config
        self.series_names = series_names or sorted({u.series for u in units})
        self.series_idx = jnp.asarray([self.series_names.index(u.series) for u in units])
        self.data = stack_units(units)
        self.shared_prior = shared_prior
        self.fns = compiled(config, len(units), len(self.series_names))
        self.unravel = self.fns.unravel
        self.n_params = self.fns.n_params
        self.prior = self._prior_arrays()
        self.flat0, _ = ravel_pytree(self.initial_params())

    # ------------------------------------------------------------------
    def _prior_arrays(self) -> dict:
        cfg = self.cfg
        if self.shared_prior is None:
            temp_sd = cfg.temp_amplitude_prior_sd if cfg.temperature == "sigmoid" else cfg.temp_slope_prior_sd
            mean = np.array([0.0] * 5 + [0.0, cfg.temp_mid_prior[0], cfg.temp_log_width_prior[0], 0.0])
            sds = np.array([cfg.theta_prior_sd] * 5 + [temp_sd, cfg.temp_mid_prior[1],
                                                       cfg.temp_log_width_prior[1], cfg.red_prior_sd])
            cov = np.diag(sds**2)
            phi_mu = np.full(len(self.series_names), cfg.log_phi_prior[0])
            phi_sd = np.full(len(self.series_names), cfg.log_phi_prior[1])
        else:
            mean = np.asarray(self.shared_prior.mean, dtype=float)
            cov = np.asarray(self.shared_prior.cov, dtype=float)
            phi_mu = np.array([self.shared_prior.log_phi[s][0] for s in self.series_names])
            phi_sd = np.array([self.shared_prior.log_phi[s][1] for s in self.series_names])
        precision = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        log_norm = -0.5 * (logdet + N_EFFECTS * np.log(2 * np.pi))
        return {
            "mean": jnp.asarray(mean), "precision": jnp.asarray(precision),
            "log_norm": jnp.asarray(log_norm),
            "phi_mu": jnp.asarray(phi_mu), "phi_sd": jnp.asarray(phi_sd),
        }

    def initial_params(self) -> dict:
        log_r = np.stack([unit.init_log_r for unit in self.units])
        log_k = np.array([unit.log_k_prior[0] for unit in self.units])
        log_iota = np.array([unit.log_iota_prior[0] for unit in self.units])
        mean = np.asarray(self.prior["mean"])
        return {
            "units": {
                "log_r": jnp.asarray(log_r),
                "log_k": jnp.asarray(log_k),
                "log_iota": jnp.asarray(log_iota),
                "log_x0": jnp.asarray(log_iota),
            },
            "shared": {
                "theta_hol": jnp.asarray(mean[:5]),
                **{key: jnp.asarray(mean[5 + i]) for i, key in enumerate(SHARED_KEYS)},
                "log_phi": jnp.asarray(self.prior["phi_mu"]),
            },
        }

    # ------------------------------------------------------------------
    def _args(self):
        return (self.data, self.series_idx, self.prior)

    def objective_value(self, flat) -> float:
        return float(self.fns.objective(jnp.asarray(flat), *self._args()))

    def log_likelihood_terms(self, flat) -> np.ndarray:
        return np.asarray(self.fns.loglik_terms(jnp.asarray(flat), self.data, self.series_idx))

    def variable_scale(self) -> np.ndarray:
        """Scale of each parameter for the optimiser (the objective is unchanged).

        The log-linear temperature slope multiplies temperatures of up to
        about 20 °C from the reference, so its curvature is about a hundred
        times that of the other effects; optimising slope / 0.1 instead keeps
        L-BFGS well conditioned.
        """
        scale = np.ones(self.n_params)
        if self.cfg.temperature == "linear":
            scale[self.shared_indices()[5]] = 0.1
        return scale

    def fit(self, flat0=None, maxiter: int = 4000, verbose: bool = False, polish: bool = True):
        """Posterior mode: L-BFGS followed by a Newton-Krylov trust-region polish.

        The search runs in scaled coordinates ``z = x / variable_scale()``;
        the returned result is in the model's own coordinates.
        """
        scale = self.variable_scale()
        x0 = np.asarray(self.flat0 if flat0 is None else flat0, dtype=np.float64) / scale
        args = self._args()

        def fun(z):
            value, grad = self.fns.value_and_grad(jnp.asarray(z * scale), *args)
            return float(value), np.asarray(grad, dtype=np.float64) * scale

        def hessp(z, v):
            return scale * np.asarray(self.fns.hvp(jnp.asarray(z * scale), jnp.asarray(v * scale), *args),
                                      dtype=np.float64)

        started = time.time()
        result = optimize.minimize(
            fun, x0, jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": maxiter * 2, "maxcor": 50,
                     "ftol": 1e-13, "gtol": 1e-7},
        )
        if verbose:
            print(f"  L-BFGS: {result.nit} it, f={result.fun:.4f}, |g|max={np.abs(result.jac).max():.2e}, "
                  f"{time.time() - started:.1f}s")
        if polish:
            # Newton-Krylov polish; repeat only while it still improves the fit
            # and has not converged. A non-finite value counts as +inf.
            for _ in range(3):
                started = time.time()
                previous = float(result.fun) if np.isfinite(result.fun) else np.inf
                polished = optimize.minimize(
                    fun, result.x, jac=True, hessp=hessp, method="trust-krylov",
                    options={"maxiter": 200, "gtol": 1e-6},
                )
                improved = bool(np.isfinite(polished.fun) and polished.fun <= previous + 1e-9)
                if improved:
                    result = polished
                if verbose:
                    print(f"  trust-krylov: {polished.nit} it, f={polished.fun:.4f}, "
                          f"|g|max={np.abs(polished.jac).max():.2e}, {time.time() - started:.1f}s")
                if not improved or polished.success or previous - float(polished.fun) < 1e-6:
                    break
        if not np.isfinite(result.fun):
            raise FloatingPointError("posterior mode search ended at a non-finite objective")
        result.x = np.asarray(result.x) * scale
        result.jac = np.asarray(result.jac) / scale
        result.pop("hess_inv", None)  # would be in the scaled coordinates; use laplace()
        return result

    def hessian(self, flat, chunk: int = 64) -> np.ndarray:
        flat = jnp.asarray(flat)
        n = self.n_params
        eye = np.eye(n)
        rows = []
        for i in range(0, n, chunk):
            block = eye[i:i + chunk]
            if block.shape[0] < chunk:  # keep a fixed shape for the compiled function
                block = np.vstack([block, np.zeros((chunk - block.shape[0], n))])
            out = np.asarray(self.fns.hvp_batch(flat, jnp.asarray(block), *self._args()))
            rows.append(out[: min(chunk, n - i)])
        hess = np.concatenate(rows, axis=0)
        return 0.5 * (hess + hess.T)

    def laplace(self, flat, jitter: float = 1e-6):
        """Return (hessian, cholesky of the hessian, log marginal likelihood)."""
        hess = self.hessian(flat)
        bump = 0.0
        for _ in range(12):
            try:
                chol = np.linalg.cholesky(hess + bump * np.eye(len(hess)))
                break
            except np.linalg.LinAlgError:
                bump = jitter if bump == 0.0 else bump * 10
        else:
            raise np.linalg.LinAlgError("Hessian is not positive definite")
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        log_evidence = -self.objective_value(flat) + 0.5 * self.n_params * np.log(2 * np.pi) - 0.5 * logdet
        return hess + bump * np.eye(len(hess)), chol, log_evidence

    def sample(self, flat, chol, n: int, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((self.n_params, n))
        delta = np.linalg.solve(chol.T, z)  # H = L L^T  =>  L^{-T} z ~ N(0, H^{-1})
        return np.asarray(flat)[None, :] + delta.T

    def simulate_flat(self, flat):
        return self.fns.simulate(jnp.asarray(flat), self.data)

    def simulate_samples(self, samples: np.ndarray, batch: int = 250):
        weekly_parts, reff_parts = [], []
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            keep = chunk.shape[0]
            if keep < batch:
                chunk = np.vstack([chunk, np.repeat(chunk[-1:], batch - keep, axis=0)])
            weekly, _, _, r_eff = self.fns.simulate_batch(jnp.asarray(chunk), self.data)
            weekly_parts.append(np.asarray(weekly)[:keep])
            reff_parts.append(np.asarray(r_eff)[:keep])
        return np.concatenate(weekly_parts), np.concatenate(reff_parts)

    def params(self, flat) -> dict:
        return jax.tree_util.tree_map(np.asarray, self.unravel(jnp.asarray(flat)))

    def shared_vector(self, flat) -> np.ndarray:
        sh = self.params(flat)["shared"]
        return np.concatenate([sh["theta_hol"]] + [[float(sh[key])] for key in SHARED_KEYS])

    def log_r_indices(self, unit: int = 0) -> np.ndarray:
        """Indices of the weekly log R knots of one unit in the flat vector."""
        marker = jax.tree_util.tree_map(jnp.zeros_like, _template(len(self.units), len(self.series_names)))
        values = jnp.zeros((len(self.units), W_MAX + 1)).at[unit].set(jnp.arange(1.0, W_MAX + 2.0))
        marker["units"]["log_r"] = values
        flat, _ = ravel_pytree(marker)
        flat = np.asarray(flat)
        return np.array([int(np.where(flat == k)[0][0]) for k in range(1, W_MAX + 2)])

    def shared_indices(self) -> np.ndarray:
        """Indices of the shared effects (``shared_names`` order) in the flat vector."""
        marker = jax.tree_util.tree_map(jnp.zeros_like, _template(len(self.units), len(self.series_names)))
        marker["shared"]["theta_hol"] = jnp.arange(1.0, 6.0)
        for i, key in enumerate(SHARED_KEYS):
            marker["shared"][key] = jnp.asarray(6.0 + i)
        flat, _ = ravel_pytree(marker)
        flat = np.asarray(flat)
        return np.array([int(np.where(flat == k)[0][0]) for k in range(1, N_EFFECTS + 1)])
