"""SAUCE: sequential uncertainty estimation from agreement and token entropy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

DEFAULT_EPSILON = 1e-8
VARIANCE_FLOOR = 1e-15

# (lambda, process noise, initial mean, initial variance), paper defaults.
_PROTOCOL_DEFAULTS = {
    "debate": (0.95, 0.10, 0.0, 0.25),
    "dylan": (0.70, 0.001, 0.0, 0.25),
}


def _finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def lambda_kalman_forward(
    observations: Sequence[float],
    R_values: Sequence[float],
    *,
    lam: float = 0.95,
    q: float = 0.10,
    mu_0: float = 0.0,
    sigma2_0: float = 0.25,
) -> dict[str, Any]:
    """Filter round-level observations with nonnegative measurement noise.

    Returns the per-round posterior means, variances, gains, innovations,
    and their temporal averages. The recursion and numerical stabilization
    are extracted from the research implementation. Defaults are the
    current paper's Debate parameters. Empty or invalid inputs raise
    ValueError.
    """
    if len(observations) != len(R_values):
        raise ValueError("observations and R_values must have the same length")
    if len(observations) == 0:
        raise ValueError("a trajectory must contain at least one round")

    lam = _finite(lam, "lam")
    q = _finite(q, "q")
    mu_0 = _finite(mu_0, "mu_0")
    sigma2_0 = _finite(sigma2_0, "sigma2_0")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be between 0 and 1")
    if q < 0.0:
        raise ValueError("q must be nonnegative")
    if sigma2_0 <= 0.0:
        raise ValueError("sigma2_0 must be positive")

    mu = mu_0
    sigma2 = sigma2_0
    rounds: list[dict[str, float | int]] = []
    for t, (observation, noise) in enumerate(zip(observations, R_values)):
        o_t = _finite(observation, f"observations[{t}]")
        R_t = _finite(noise, f"R_values[{t}]")
        if R_t < 0.0:
            raise ValueError(f"R_values[{t}] must be nonnegative")
        mu_pred = lam * mu
        P_minus = (lam ** 2) * sigma2 + q
        S_t = P_minus + R_t + DEFAULT_EPSILON
        nu_t = o_t - mu_pred
        K_t = P_minus / S_t
        mu = mu_pred + K_t * nu_t
        sigma2 = (1.0 - K_t) * P_minus
        if sigma2 < VARIANCE_FLOOR:
            sigma2 = VARIANCE_FLOOR
        rounds.append({
            "round_index": t,
            "mu": mu,
            "sigma2": sigma2,
            "S_t": S_t,
            "K_t": K_t,
            "nu_t": nu_t,
        })

    all_mu = [rd["mu"] for rd in rounds]
    all_sigma2 = [rd["sigma2"] for rd in rounds]
    return {
        "lam": lam,
        "q": q,
        "mu_0": mu_0,
        "sigma2_0": sigma2_0,
        "rounds": rounds,
        "final_mu": mu,
        "final_sigma2": sigma2,
        "avg_mu": sum(all_mu) / len(all_mu),
        "avg_sigma2": sum(all_sigma2) / len(all_sigma2),
    }


def sauce(
    agreement: Sequence[float],
    token_entropy: Sequence[float],
    *,
    protocol: str = "debate",
    lam: float | None = None,
    q: float | None = None,
    mu_0: float | None = None,
    sigma2_0: float | None = None,
) -> dict[str, Any]:
    """Score a trajectory using SAUCE; higher scores mean more uncertainty.

    Each sequence must contain one value per round. ``agreement`` is the
    majority fraction M_t; ``token_entropy`` is mean predictive entropy
    across that round's agents, using natural logarithms. ``protocol``
    selects the paper's Debate or DyLAN defaults. Explicit scalar arguments
    override the selected preset.

    Returns ``{"score": -mean(mu_t), "trace": <filter output>}``.
    Missing, empty, nonfinite, or out-of-range inputs raise ValueError.
    """
    if protocol not in _PROTOCOL_DEFAULTS:
        raise ValueError("protocol must be 'debate' or 'dylan'")
    if len(agreement) != len(token_entropy):
        raise ValueError("agreement and token_entropy must have the same length")
    default_lam, default_q, default_mu, default_sigma = _PROTOCOL_DEFAULTS[protocol]
    observations = []
    for t, value in enumerate(agreement):
        margin = _finite(value, f"agreement[{t}]")
        if not 0.0 <= margin <= 1.0:
            raise ValueError(f"agreement[{t}] must be between 0 and 1")
        observations.append(margin)
    noises = []
    for t, value in enumerate(token_entropy):
        entropy = _finite(value, f"token_entropy[{t}]")
        if entropy < 0.0:
            raise ValueError(f"token_entropy[{t}] must be nonnegative")
        noises.append(max(entropy, DEFAULT_EPSILON))
    trace = lambda_kalman_forward(
        observations,
        noises,
        lam=default_lam if lam is None else lam,
        q=default_q if q is None else q,
        mu_0=default_mu if mu_0 is None else mu_0,
        sigma2_0=default_sigma if sigma2_0 is None else sigma2_0,
    )
    return {"score": -trace["avg_mu"], "trace": trace}


def sauce_score(
    agreement: Sequence[float],
    token_entropy: Sequence[float],
    *,
    protocol: str = "debate",
    lam: float | None = None,
    q: float | None = None,
    mu_0: float | None = None,
    sigma2_0: float | None = None,
) -> float:
    """Return only the SAUCE uncertainty score (negative mean confidence)."""
    return sauce(
        agreement, token_entropy, protocol=protocol,
        lam=lam, q=q, mu_0=mu_0, sigma2_0=sigma2_0,
    )["score"]
