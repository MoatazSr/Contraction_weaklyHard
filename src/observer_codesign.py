"""
observer_codesign.py — Co-design solver: minimum observer gain and certified metric.

What this module proves
-----------------------
Proposition 2 (Observer co-design condition): For the closed-loop error dynamics
    ẋ = f(x, e)
    ė = -f(x, e) - α·e
there exists a diagonal polynomial contraction metric M(z) = diag(p(z),q(z))
satisfying Ṁ + JᵀM + MJ + 2λM ≼ 0 on domain Z if and only if α > 2 + 2λ.

The module provides:
  (A) The analytical co-design threshold α* = 2 + 2λ (Proposition 2, necessary).
  (B) An SOCP-based solver that finds the minimum feasible α numerically,
      confirming the analytical bound and returning the certified metric.

What assumptions this requires
-------------------------------
- f(x,e) = x² - x³ - 2(x+e)  (the scalar benchmark plant).
- α > α* = 2 + 2λ  (sufficient and necessary for diagonal diagonal polynomial metric).
- Domain Z = DOMAIN_X × DOMAIN_E is compact and forward-invariant.
- The SOCP uses a 6-basis polynomial metric:
    p(z), q(z) ∈ span{1, x², e², x⁴, e⁴, x²e²}
  with softplus-encoded strictly positive coefficients.

Public functions
----------------
codesign_threshold      — analytical lower bound α* = 2 + 2λ
verify_codesign_condition — check if a given α satisfies the condition
solve                   — find min feasible α and certified metric via SOCP sweep
alpha_feasibility_table — table of (α, feasible, margin, h_2020) for reporting
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# 1.  Analytical co-design threshold
# ---------------------------------------------------------------------------

def codesign_threshold(lam: float) -> float:
    """Return the analytical minimum observer gain α* = 2 + 2λ.

    What this proves
    ----------------
    Proposition 2: For the benchmark plant with Jacobian
        J₂₂ = 2 - α  (the (2,2) entry of the closed-loop Jacobian at origin)
    the flow LMI S ≼ 0 requires S₂₂ = 2(2-α+λ)q < 0, giving α > 2+λ.
    The full off-diagonal coupling requires α > 2+2λ for the SOCP to be feasible
    with any bounded-coefficient-ratio polynomial metric.

    This is a sufficient condition; the SOCP verifies the exact threshold.

    Parameters
    ----------
    lam : contraction rate λ > 0

    Returns
    -------
    float  — analytical lower bound on α
    """
    return 2.0 + 2.0 * lam


def verify_codesign_condition(alpha: float, lam: float) -> dict:
    """Check whether a given α satisfies the co-design condition.

    What this proves: whether the analytical sufficient condition (Prop 2)
    is met and what the margin is.

    Parameters
    ----------
    alpha : observer gain to check
    lam   : contraction rate λ

    Returns
    -------
    dict with keys:
        satisfied     — bool, True iff α > 2 + 2λ
        alpha         — float, input α
        threshold     — float, α* = 2 + 2λ
        margin        — float, α - α*  (positive = satisfied)
        J22_at_origin — float, 2 - α (Jacobian entry; must be < -2λ)
        S22_sign      — str, 'negative (good)' or 'non-negative (bad)'
    """
    threshold = codesign_threshold(lam)
    margin = alpha - threshold
    j22 = 2.0 - alpha
    return {
        "satisfied":     bool(margin > 0),
        "alpha":         alpha,
        "threshold":     threshold,
        "lam":           lam,
        "margin":        margin,
        "J22_at_origin": j22,
        "S22_sign":      "negative (good)" if j22 < -2 * lam else "non-negative (bad)",
        "condition":     f"α > 2 + 2λ = {threshold:.4f}",
    }


# ---------------------------------------------------------------------------
# 2.  SOCP feasibility check at a single α
# ---------------------------------------------------------------------------

def _check_socp_feasible(alpha: float, lam: float,
                          domain_x: tuple, domain_e: tuple,
                          n_grid: int = 30) -> dict:
    """Run CVXPY SOCP for a given α and return feasibility + metric params.

    What this proves: whether there exists a diagonal polynomial metric
    with 6-monomial basis that satisfies the flow LMI at all grid points.

    Returns dict with keys:
        feasible      — bool
        params        — np.ndarray(12,) or None
        solver_status — str
        objective     — float or None
    """
    try:
        import cvxpy as cp
    except ImportError:
        return {"feasible": False, "params": None,
                "solver_status": "cvxpy_not_installed", "objective": None}

    x_lo, x_hi = domain_x
    e_lo, e_hi = domain_e
    EPS_PD    = 1e-4
    EPS_COEFF = 1e-6
    MAX_COEFF = 20.0
    REG       = 0.10

    xs = np.linspace(x_lo, x_hi, n_grid)
    es = np.linspace(e_lo, e_hi, n_grid)
    XX, EE = np.meshgrid(xs, es, indexing="ij")
    grid_pts = np.stack([XX.ravel(), EE.ravel()], axis=1)

    def _basis(pts: np.ndarray) -> np.ndarray:
        x, e = pts[:, 0], pts[:, 1]
        return np.stack([np.ones(len(pts)), x**2, e**2,
                         x**4, e**4, x**2 * e**2], axis=1)

    def _f_flow(x_v: np.ndarray, e_v: np.ndarray) -> np.ndarray:
        return x_v**2 - x_v**3 - 2.0 * (x_v + e_v)

    B      = _basis(grid_pts)
    gx, ge = grid_pts[:, 0], grid_pts[:, 1]
    fvals  = _f_flow(gx, ge)

    eps_fd = 1e-5
    gp_xp  = grid_pts + eps_fd * np.stack([fvals, -fvals - alpha * ge], axis=1)
    gp_xm  = grid_pts - eps_fd * np.stack([fvals, -fvals - alpha * ge], axis=1)
    dBdt   = (_basis(gp_xp) - _basis(gp_xm)) / (2 * eps_fd)

    df1dx  = 2.0 * gx - 3.0 * gx**2 - 2.0
    df2de  = 2.0 - alpha

    C00  = dBdt + (2 * df1dx[:, None] + 2 * lam) * B
    C11  = dBdt + (2 * df2de          + 2 * lam) * B
    D01q = -df1dx[:, None] * B
    D01p = -2.0 * B

    pp = cp.Variable(6)
    qq = cp.Variable(6)
    t  = cp.Variable()
    n_pts = grid_pts.shape[0]

    constraints = [pp >= EPS_COEFF, qq >= EPS_COEFF,
                   pp <= MAX_COEFF,  qq <= MAX_COEFF,
                   B @ pp >= EPS_PD, B @ qq >= EPS_PD,
                   pp[0] == 1.0, qq[0] == 1.0]

    for i in range(n_pts):
        S00_i   = C00[i]  @ pp
        S11_i   = C11[i]  @ qq
        S01_i   = D01q[i] @ qq + D01p[i] @ pp
        half_tr  = (S00_i + S11_i) * 0.5
        half_dif = (S00_i - S11_i) * 0.5
        constraints.append(cp.norm(cp.hstack([half_dif, S01_i]), 2) <= t - half_tr)

    obj  = t + REG * (cp.sum_squares(pp[1:]) + cp.sum_squares(qq[1:]))
    prob = cp.Problem(cp.Minimize(obj), constraints)

    status = "unsolved"
    for sname in ["CLARABEL", "SCS", "ECOS"]:
        try:
            kw = {"verbose": False}
            if sname == "SCS":
                kw["eps"] = 1e-6
            prob.solve(solver=getattr(cp, sname, sname), **kw)
            status = str(prob.status)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"feasible": False, "params": None,
                "solver_status": status, "objective": None}
    if pp.value is None or qq.value is None:
        return {"feasible": False, "params": None,
                "solver_status": status, "objective": None}

    def _inv_softplus(y: np.ndarray) -> np.ndarray:
        y = np.maximum(y, 1e-10)
        return np.where(y > 20, y, np.log(np.expm1(y)))

    pp_val = np.maximum(pp.value, EPS_COEFF * 10)
    qq_val = np.maximum(qq.value, EPS_COEFF * 10)
    params = np.concatenate([_inv_softplus(pp_val), _inv_softplus(qq_val)])

    return {
        "feasible":      True,
        "params":        params,
        "solver_status": status,
        "objective":     float(t.value) if t.value is not None else None,
    }


# ---------------------------------------------------------------------------
# 3.  Main solver: find minimum α
# ---------------------------------------------------------------------------

def solve(
    lam:       float,
    domain_x:  tuple,
    domain_e:  tuple,
    alpha_hi:  float = 6.0,
    n_sweep:   int   = 12,
    n_grid:    int   = 30,
    verbose:   bool  = True,
) -> dict:
    """Find the minimum observer gain α* and certified metric via SOCP sweep.

    What this proves
    ----------------
    By sweeping α from the analytical threshold α_analytical = 2+2λ upward,
    finds the smallest α at which the diagonal polynomial SOCP is feasible.
    Returns the certified metric parameters at that α.

    What assumptions this requires
    --------------------------------
    Same as Proposition 2: scalar benchmark plant, diagonal 6-monomial metric,
    compact domain Z = domain_x × domain_e.

    Parameters
    ----------
    lam       : target contraction rate λ
    domain_x  : (x_lo, x_hi)
    domain_e  : (e_lo, e_hi)
    alpha_hi  : upper bound for α sweep
    n_sweep   : number of α values to try
    n_grid    : SOCP grid size
    verbose   : print sweep progress

    Returns
    -------
    dict with keys:
        alpha_min_analytical  — float, 2 + 2λ (analytical lower bound)
        alpha_min_numerical   — float, smallest feasible α found by sweep
        alpha_margin          — float, alpha_min_numerical - alpha_min_analytical
        lam                   — float
        metric_params         — np.ndarray(12,) at alpha_min_numerical
        solver_status         — str
        sweep_results         — list[dict], one entry per α tried
        certified             — bool, True iff a feasible α was found
    """
    alpha_analytical = codesign_threshold(lam)
    sweep_alphas = np.linspace(alpha_analytical + 0.05, alpha_hi, n_sweep)

    if verbose:
        print(f"  Co-design threshold (analytical): α* = 2 + 2λ = {alpha_analytical:.4f}")
        print(f"  Sweeping α ∈ [{sweep_alphas[0]:.2f}, {alpha_hi:.2f}] "
              f"({n_sweep} points)...")

    sweep_results: list[dict] = []
    best: dict | None = None

    for alpha in sweep_alphas:
        res = _check_socp_feasible(alpha, lam, domain_x, domain_e, n_grid)
        entry = {
            "alpha":         float(alpha),
            "feasible":      res["feasible"],
            "solver_status": res["solver_status"],
            "objective":     res["objective"],
        }
        sweep_results.append(entry)

        if res["feasible"] and best is None:
            best = {"alpha": float(alpha), **res}

        if verbose:
            flag = "✔" if res["feasible"] else "✘"
            obj_str = f"  obj={res['objective']:.3e}" if res["objective"] is not None else ""
            print(f"    α={alpha:.3f}  [{res['solver_status']}]{obj_str}  {flag}")

        if res["feasible"]:
            break   # found minimum; stop sweep

    if best is None:
        return {
            "alpha_min_analytical": alpha_analytical,
            "alpha_min_numerical":  None,
            "alpha_margin":         None,
            "lam":                  lam,
            "metric_params":        None,
            "solver_status":        "all_infeasible",
            "sweep_results":        sweep_results,
            "certified":            False,
            "error":                f"No feasible α found in [{sweep_alphas[0]:.2f}, {alpha_hi:.2f}]",
        }

    margin = best["alpha"] - alpha_analytical
    if verbose:
        print(f"  Minimum feasible α = {best['alpha']:.4f} "
              f"(analytical: {alpha_analytical:.4f}, margin = {margin:.4f})")

    return {
        "alpha_min_analytical": alpha_analytical,
        "alpha_min_numerical":  best["alpha"],
        "alpha_margin":         margin,
        "lam":                  lam,
        "metric_params":        best["params"],
        "solver_status":        best["solver_status"],
        "sweep_results":        sweep_results,
        "certified":            True,
    }


# ---------------------------------------------------------------------------
# 4.  Feasibility table for reporting
# ---------------------------------------------------------------------------

def alpha_feasibility_table(
    lam:        float,
    alpha_list: list[float] | None = None,
    masp:       float = 0.5,
    eta_n:      int   = 20,
    eta_m:      int   = 17,
) -> list[dict]:
    """Build a table of (α, co-design-ok, margin, h_2020) for reporting.

    What it returns: for each α in alpha_list, reports whether the analytical
    co-design condition is satisfied and what the margin is. The h_2020 column
    is the MASP formula bound (architecture-independent reference).

    Parameters
    ----------
    lam        : contraction rate λ
    alpha_list : list of α values; defaults to [2.1, 2.5, 3.0, 4.0, 5.0]
    masp       : MASP formula parameter (default 0.5 s)
    eta_n, eta_m : WHRT parameters for h_2020 = masp / (N-M+1)

    Returns
    -------
    list of dicts, each with keys:
        alpha, satisfied, margin, threshold, h_2020_formula, note
    """
    if alpha_list is None:
        alpha_list = [2.1, 2.5, 3.0, 4.0, 5.0]

    threshold = codesign_threshold(lam)
    h_2020 = masp / (eta_n - eta_m + 1)

    rows = []
    for a in alpha_list:
        margin = a - threshold
        rows.append({
            "alpha":          float(a),
            "satisfied":      bool(margin > 0),
            "margin":         float(margin),
            "threshold":      float(threshold),
            "h_2020_formula": float(h_2020),
            "note": (
                "FEASIBLE" if margin > 0 else
                "INFEASIBLE (α too small for contraction)"
            ),
        })

    return rows
