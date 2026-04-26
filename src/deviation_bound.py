"""
deviation_bound.py – Worst-case deviation bound for the WHRT sampled-data system.

One public function
-------------------
compute_deviation_bound – upper bound on M-norm trajectory deviation over SIM_TIME
"""

from __future__ import annotations

import math
import numpy as np


def compute_deviation_bound(params, gf: dict, cfg) -> dict:
    """Compute worst-case upper bound on trajectory deviation.

    For a sampled-data system whose per-period growth factor is rho_total(l),
    the M-norm deviation after k sampling periods is bounded by:

        ||delta_z(k·H)||_M  <=  rho_max^k · ||delta_z(0)||_M

    where  rho_max = max_l { rho_total(l) }.

    The bound is evaluated for an initial unit-M-norm perturbation
    (||delta_z(0)||_M = 1) over the full simulation horizon SIM_TIME.

    Additional quantities:
      - e_folding_time : time for the bound to decay to 1/e  (only when rho_max < 1)
      - half_life      : time for the bound to halve          (only when rho_max < 1)
      - horizon_bound  : the bound value at t = SIM_TIME

    passed = True iff rho_max < 1  (system is contracting → bound → 0).

    Never raises.  Returns a dict with keys:
        passed, bound, max_rho_total, n_steps, horizon_bound,
        e_folding_time, half_life, error (present on failure).
    """
    if not gf:
        return {
            "passed":        False,
            "bound":         None,
            "max_rho_total": None,
            "n_steps":       0,
            "horizon_bound": None,
            "e_folding_time": None,
            "half_life":     None,
            "error":         "no growth factors provided",
        }

    rho_totals   = [v["rho_total"] for v in gf.values()]
    max_rho      = float(max(rho_totals))
    n_steps      = int(cfg.SIM_TIME / cfg.H)

    # Worst-case bound at every time step (array for histogram / plotting)
    steps        = np.arange(n_steps + 1)
    bound_series = max_rho ** steps          # shape (n_steps+1,)
    horizon_bound = float(bound_series[-1])

    passed = max_rho < 1.0

    # Decay characteristics (only meaningful when converging)
    e_fold = None
    half_life = None
    if passed and max_rho > 0.0:
        log_rho   = math.log(max_rho)        # negative
        e_fold    = -cfg.H / log_rho         # seconds per e-fold
        half_life = -cfg.H * math.log(2.0) / log_rho

    result: dict = {
        "passed":         passed,
        "bound":          float(max_rho),    # per-period growth factor
        "max_rho_total":  max_rho,
        "n_steps":        n_steps,
        "horizon_bound":  horizon_bound,
        "e_folding_time": e_fold,
        "half_life":      half_life,
        "bound_series":   bound_series,      # full time series
    }
    if not passed:
        result["error"] = (
            f"max rho_total={max_rho:.4f} >= 1.0 -- "
            f"deviation bound grows unboundedly (x{horizon_bound:.2e} at t={cfg.SIM_TIME}s)"
        )
    return result
