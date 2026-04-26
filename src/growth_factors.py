"""
growth_factors.py — Per-label growth factors with analytical Gronwall bound.

Certificate strategy
--------------------
When the flow condition  Ṁ + JᵀM + MJ + 2λM ≼ 0  is certified on Z
with rate λ (via lipschitz_certified_flow_viol returning certified=True),
the Gronwall inequality applied to the variational dynamics gives:

    d/dt ‖δz‖_M²  ≤  -2λ ‖δz‖_M²
    ⟹  ‖δz(t)‖_M  ≤  e^{-λt} ‖δz(0)‖_M   for all z(0) ∈ Z

Therefore:  ρ_flow(l)  ≤  e^{-λ·l·h}                   [PROVEN UPPER BOUND]

This replaces the Monte Carlo estimate used previously, which was only
the observed maximum over 500 samples (not a provable upper bound).

The Monte Carlo integration is retained as a VALIDATION check only —
it is NOT used in the stability certificate.

Public functions
----------------
analytical_rho_flow  — proven upper bound e^{-λlh} from Gronwall
m_norm               — M-weighted norm
compute_rho_l        — MC validation of flow factor (not used in certificate)
compute_all_rho      — primary: analytical; secondary: MC validation
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

import config
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from contraction import f_flow, jacobian_f, build_metric


# ---------------------------------------------------------------------------
# 1.  analytical_rho_flow   ← CERTIFICATE BOUND
# ---------------------------------------------------------------------------

def analytical_rho_flow(lam: float, l: int, h: float) -> float:
    """Provable upper bound on the flow growth factor.

    Proof sketch
    ------------
    Suppose the flow certificate holds: there exists M(z) diagonal polynomial
    with  Ṁ + JᵀM + MJ + 2λM ≼ 0  on Z.

    For any trajectory z(t) ∈ Z and any variation δz(t) satisfying
    the variational equation d(δz)/dt = J(z)·δz:

        d/dt ‖δz‖_M² = δzᵀ(Ṁ + JᵀM + MJ)δz  ≤  -2λ ‖δz‖_M²

    By Gronwall:  ‖δz(t)‖_M  ≤  e^{-λt} ‖δz(0)‖_M   for all t ≥ 0.

    Over one label-l interval of length T = l·h:
        ρ_flow(l) := sup ‖δz(T)‖_M / ‖δz(0)‖_M  ≤  e^{-λ·l·h}

    This bound is valid whenever the flow certificate is certified=True.

    Parameters
    ----------
    lam : contraction rate λ (from find_metric_sos)
    l   : label (number of sampling periods in this interval)
    h   : sampling period (seconds)

    Returns
    -------
    float  — e^{-λ·l·h}  (proven upper bound on ρ_flow(l))
    """
    return float(np.exp(-lam * l * h))


# ---------------------------------------------------------------------------
# 2.  m_norm
# ---------------------------------------------------------------------------

def m_norm(delta_z: np.ndarray, M: np.ndarray) -> float:
    """Return sqrt(δzᵀ M δz).  Returns 0.0 for negative quadratic forms."""
    dz  = np.asarray(delta_z, dtype=float)
    M_  = np.asarray(M,       dtype=float)
    val = float(dz @ M_ @ dz)
    return float(np.sqrt(max(val, 0.0)))


# ---------------------------------------------------------------------------
# 3.  compute_rho_l   (MC validation — NOT the certificate)
# ---------------------------------------------------------------------------

def compute_rho_l(params: np.ndarray, l: int, h: float,
                  n_samples: int, seed: int) -> dict:
    """Integrate the variational equation over T = l·h seconds (MC validation).

    THIS IS A VALIDATION TOOL, NOT A CERTIFICATE.
    Monte Carlo over n_samples random initial conditions gives the maximum
    OBSERVED ratio ‖δz(T)‖_M / ‖δz(0)‖_M — this is not a provable upper bound.
    Use analytical_rho_flow() for the formal certificate.

    Returns
    -------
    dict with keys:
        rho_flow_mc    — float, max observed ratio (not an upper bound)
        worst_z0       — (z, δz) achieving the observed max
        n_failed       — int, number of ODE failures
        n_total        — int, = n_samples
        mc_warning     — str, always present reminding caller this is MC only
    """
    params = np.asarray(params, dtype=float)
    T      = l * h
    x_lo, x_hi = config.DOMAIN_X
    e_lo, e_hi = config.DOMAIN_E
    rng    = np.random.default_rng(seed)

    def _ode(t, y):
        z  = y[:2]
        dz = y[2:]
        return np.concatenate([f_flow(z), jacobian_f(z) @ dz])

    max_ratio = -np.inf
    worst_z0  = None
    n_failed  = 0

    for _ in range(n_samples):
        z0 = np.array([rng.uniform(x_lo, x_hi), rng.uniform(e_lo, e_hi)])
        dz0_raw = rng.standard_normal(2)
        M0  = build_metric(params, z0)
        nrm = m_norm(dz0_raw, M0)
        if nrm < 1e-14:
            n_failed += 1
            continue
        dz0 = dz0_raw / nrm

        y0 = np.concatenate([z0, dz0])
        try:
            sol = solve_ivp(_ode, (0.0, T), y0, method="RK45",
                            rtol=1e-8, atol=1e-10, dense_output=False)
            if not sol.success:
                n_failed += 1
                continue
        except Exception:
            n_failed += 1
            continue

        z_T  = sol.y[:2, -1]
        dz_T = sol.y[2:, -1]
        M_T  = build_metric(params, z_T)
        ratio = m_norm(dz_T, M_T)
        if ratio > max_ratio:
            max_ratio = ratio
            worst_z0  = (z0.copy(), dz0.copy())

    rho_flow_mc = float(max_ratio) if max_ratio > -np.inf else 0.0

    result = {
        "rho_flow_mc": rho_flow_mc,
        "worst_z0":    worst_z0,
        "n_failed":    n_failed,
        "n_total":     n_samples,
        "mc_warning":  ("MC estimate only — not a proven upper bound. "
                        "Use analytical_rho_flow() for certificate."),
    }
    if n_failed > n_samples * 0.1:
        result["integration_warning"] = True
    return result


# ---------------------------------------------------------------------------
# 4.  compute_all_rho   (primary: analytical bound; secondary: MC validation)
# ---------------------------------------------------------------------------

def compute_all_rho(params: np.ndarray,
                    rho_jump: float,
                    max_l: int,
                    h: float,
                    lam: float,
                    flow_certified: bool = False) -> dict:
    """Compute per-label growth factors.

    Certificate chain
    -----------------
    If flow_certified=True (from lipschitz_certified_flow_viol):
        ρ_flow(l) = e^{-λ·l·h}              [PROVEN upper bound]
    Else:
        ρ_flow(l) = MC observed maximum      [numerical estimate only]

    In both cases:
        ρ_total(l) = ρ_flow(l) × ρ_jump

    where ρ_jump is now DERIVED from the actual reset map g(x,ê)=(x,0)
    via compute_rho_jump() — it is NOT a free design parameter.

    Parameters
    ----------
    params        : metric parameters (shape 12)
    rho_jump      : certified jump gain from compute_rho_jump()
    max_l         : largest label in the automaton
    h             : sampling period
    lam           : certified contraction rate
    flow_certified: True iff flow certificate was rigorous

    Returns
    -------
    dict {l: entry_dict} where each entry contains:
        rho_flow       — proven bound (if certified) or MC estimate
        rho_flow_mc    — MC estimate (always computed for validation)
        rho_flow_source— 'analytical_gronwall' or 'monte_carlo_estimate'
        rho_jump       — certified value from reset map (not free parameter)
        rho_total      — rho_flow × rho_jump
        stable         — bool, rho_total < 1
        certificate_valid — bool, True iff rho_flow came from certified chain
    """
    params = np.asarray(params, dtype=float)
    results: dict = {}

    for l in range(1, int(max_l) + 1):
        # ── Analytical bound (from certified contraction rate) ────────────
        rho_flow_analytic = analytical_rho_flow(lam, l, h)

        # ── MC validation ────────────────────────────────────────────────
        mc_res       = compute_rho_l(params, l, h,
                                     n_samples=config.N_SAMPLES_GF,
                                     seed=config.SEED + l)
        rho_flow_mc  = mc_res["rho_flow_mc"]

        # ── Choose certified value ────────────────────────────────────────
        if flow_certified:
            rho_flow_cert   = rho_flow_analytic
            rho_flow_source = "analytical_gronwall"
            certificate_valid = True
        else:
            rho_flow_cert   = rho_flow_mc
            rho_flow_source = "monte_carlo_estimate"
            certificate_valid = False

        rho_total = rho_flow_cert * rho_jump

        entry: dict = {
            "rho_flow":          rho_flow_cert,
            "rho_flow_analytic": rho_flow_analytic,
            "rho_flow_mc":       rho_flow_mc,
            "rho_flow_source":   rho_flow_source,
            "rho_jump":          rho_jump,
            "rho_total":         rho_total,
            "stable":            bool(rho_total < 1.0),
            "certificate_valid": certificate_valid,
            "l":                 l,
        }

        # Consistency check: MC should not substantially exceed analytic bound
        if flow_certified and rho_flow_mc > rho_flow_analytic + 0.05:
            entry["consistency_warning"] = (
                f"MC estimate {rho_flow_mc:.4f} exceeds analytic bound "
                f"{rho_flow_analytic:.4f} by >{0.05:.2f} — "
                f"flow domain invariance or metric validity should be rechecked."
            )

        if rho_total >= 1.0:
            entry["stability_concern"] = True

        results[l] = entry

    return results
