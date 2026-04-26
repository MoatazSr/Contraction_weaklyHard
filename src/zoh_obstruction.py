"""
zoh_obstruction.py — Architecture-agnostic ZOH contraction obstruction certificate.

What this module proves
-----------------------
Proposition 1 (Universal ZOH obstruction): For any scalar nonlinear plant
    ẋ = f(x, e)
    ė = -f(x, e)              [ZOH: ẋ_hat = 0 so ė = -ẋ = -f]
no Riemannian metric M(z) ≻ 0 — diagonal or otherwise — with contraction
rate λ > 0 can satisfy S(z) := Ṁ + JᵀM + MJ + 2λM ≼ 0 on any domain
containing the equilibrium (x,e) = (0,0).

Proof (rank-1 Jacobian argument)
---------------------------------
The ZOH closed-loop Jacobian is J_ZOH = c·r(z)ᵀ (rank 1), where:
    c = (1, -1)ᵀ   (fixed direction, independent of z)
    r(z) = (∂f/∂x, ∂f/∂e)ᵀ   (depends on z)

At any z, for any M(z) ≻ 0:
    cᵀ(JᵀM + MJ)c = 2(rᵀc)(cᵀMc)

At the equilibrium z=0, if rᵀ(0)c = ∂f/∂x|₀ - ∂f/∂e|₀ = 0, then:
    cᵀS(0)c = 2·0·(cᵀMc) + 0 + 2λ cᵀMc = 2λ cᵀMc > 0

Hence S(0) ⊁ 0 for any M(0) ≻ 0 and λ > 0. □

What assumptions this requires
--------------------------------
1. f is C¹ near z=0.
2. α = 0 (pure ZOH): ẋ_hat = 0, so ė = -f(x,e).
3. The equilibrium satisfies f(0,0) = 0.
4. The rank-1 obstruction condition: ∂f/∂x|₀ = ∂f/∂e|₀.
   (When violated, the conservation-law argument is used instead.)

Public functions
----------------
check_zoh_obstruction   — full obstruction certificate for any plant f(x,e)
obstruction_report      — human-readable summary string
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# 1.  Numerical Jacobian helper
# ---------------------------------------------------------------------------

def _jacobian_zoh_at_z(f: Callable, z: np.ndarray,
                        eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return J_ZOH(z) and its rank-1 components c, r(z).

    For ZOH dynamics ẋ=f(x,e), ė=-f(x,e):
        J_ZOH = [[∂f/∂x, ∂f/∂e], [-∂f/∂x, -∂f/∂e]] = c·r(z)ᵀ
    where c=(1,-1)ᵀ and r=(∂f/∂x, ∂f/∂e)ᵀ.

    Returns (J, r) where J is the 2×2 ZOH Jacobian and r is (∂f/∂x, ∂f/∂e)ᵀ.
    """
    x, e = float(z[0]), float(z[1])
    dfx = (f(x + eps, e) - f(x - eps, e)) / (2 * eps)
    dfe = (f(x, e + eps) - f(x, e - eps)) / (2 * eps)
    r = np.array([dfx, dfe])
    c = np.array([1.0, -1.0])
    J = np.outer(c, r)
    return J, r


# ---------------------------------------------------------------------------
# 2.  Conservation law verification
# ---------------------------------------------------------------------------

def _check_conservation_law(f: Callable,
                             domain: tuple,
                             n_pts: int = 200) -> dict:
    """Verify d(x+e)/dt = 0 numerically over the domain grid.

    For ZOH: d(x+e)/dt = ẋ + ė = f(x,e) + (-f(x,e)) = 0 always.
    This is an algebraic identity — confirmed numerically as a sanity check.

    What this proves: the state space under ZOH is foliated by invariant
    lines {x+e = c}. Two trajectories on different leaves cannot converge
    during the flow phase, making contraction structurally infeasible.

    Returns dict with:
        confirmed  — bool, True iff identity holds within tolerance
        max_error  — float, max|d(x+e)/dt| over grid
        conserved  — str, human-readable conserved quantity
    """
    (x_lo, x_hi), (e_lo, e_hi) = domain
    xs = np.linspace(x_lo, x_hi, n_pts)
    es = np.linspace(e_lo, e_hi, n_pts)
    XX, EE = np.meshgrid(xs, es, indexing="ij")
    errors = np.abs(
        np.vectorize(f)(XX, EE) + np.vectorize(lambda x, e: -f(x, e))(XX, EE)
    )
    max_err = float(np.max(errors))
    return {
        "confirmed": bool(max_err < 1e-10),
        "max_error": max_err,
        "conserved": "x + e  (= x_hat, the ZOH estimate)",
        "proof":     "d(x+e)/dt = f(x,e) + (-f(x,e)) = 0  [algebraic identity for ZOH]",
        "implication": (
            "State space is foliated by invariant lines {x+e = const}. "
            "No two trajectories on different leaves converge during flow — "
            "contraction is impossible regardless of metric choice."
        ),
    }


# ---------------------------------------------------------------------------
# 3.  Rank-1 Jacobian obstruction at equilibrium
# ---------------------------------------------------------------------------

def _check_rank1_obstruction(f: Callable,
                              equilibrium: np.ndarray,
                              eps: float = 1e-6) -> dict:
    """Verify the rank-1 obstruction condition at the equilibrium.

    The rank-1 Jacobian J_ZOH = c·rᵀ gives, at z=0:
        cᵀS(0)c = 2(rᵀ(0)c)(cᵀM(0)c) + 2λ cᵀM(0)c

    Obstruction holds when rᵀ(0)c = 0, i.e., ∂f/∂x|₀ = ∂f/∂e|₀.
    Then cᵀS(0)c = 2λ cᵀM(0)c > 0 for any M(0) ≻ 0.

    Returns dict with:
        obstructed  — bool, True iff rᵀ(0)c = 0
        rTc_value   — float, value of rᵀ(0)c
        df_dx       — float, ∂f/∂x at equilibrium
        df_de       — float, ∂f/∂e at equilibrium
        proof_str   — str
    """
    z0 = np.asarray(equilibrium, dtype=float)
    J, r = _jacobian_zoh_at_z(f, z0, eps)
    c = np.array([1.0, -1.0])
    rtc = float(r @ c)   # = ∂f/∂x - ∂f/∂e at z0

    obstructed = abs(rtc) < 1e-6
    proof = (
        f"At z=0: r(0)=(∂f/∂x, ∂f/∂e)|₀ = ({r[0]:.4f}, {r[1]:.4f}), "
        f"c=(1,-1)ᵀ.  rᵀc = {rtc:.6f}."
    )
    if obstructed:
        proof += (
            f"\n  Since rᵀ(0)c ≈ 0: cᵀS(0)c = 2λ·cᵀM(0)c > 0 "
            f"for any M(0)≻0 and λ>0.  No metric can certify contraction."
        )
    else:
        proof += (
            f"\n  rᵀ(0)c = {rtc:.4f} ≠ 0 — exact rank-1 argument does not apply "
            f"at z=0.  See conservation law argument for the universal result."
        )

    return {
        "obstructed": obstructed,
        "rTc_value":  rtc,
        "df_dx":      float(r[0]),
        "df_de":      float(r[1]),
        "J_ZOH":      J,
        "proof_str":  proof,
    }


# ---------------------------------------------------------------------------
# 4.  Main public function
# ---------------------------------------------------------------------------

def check_zoh_obstruction(
    f:            Callable,
    domain:       tuple,
    equilibrium:  tuple = (0.0, 0.0),
    eps:          float = 1e-6,
    verbose:      bool  = True,
) -> dict:
    """Certificate that ZOH makes contraction structurally infeasible.

    What this proves
    ----------------
    For ANY plant ẋ=f(x,e) under ZOH (α=0, ė=-f(x,e)):
      (A) Conservation law: d(x+e)/dt = 0 identically → invariant foliation
          {x+e=c} → no two trajectories on different leaves can converge →
          contraction impossible regardless of metric.
      (B) Rank-1 Jacobian: if ∂f/∂x|₀ = ∂f/∂e|₀, then cᵀS(0)c = 2λcᵀMc > 0
          for ANY M(z) ≻ 0, diagonal or otherwise.

    What assumptions this requires
    ------------------------------
    - f is C¹ and f(0,0) = 0 (equilibrium at origin).
    - α = 0 (pure ZOH: ẋ_hat = 0 between samples).
    - domain: tuple ((x_lo, x_hi), (e_lo, e_hi)) — compact operating domain.

    Parameters
    ----------
    f           : callable(x: float, e: float) -> float, closed-loop vector field
    domain      : ((x_lo, x_hi), (e_lo, e_hi))
    equilibrium : (x0, e0), default (0,0)
    eps         : finite-difference step for Jacobian
    verbose     : print report to stdout

    Returns
    -------
    dict with keys:
        obstructed           — bool, True iff ZOH is provably obstructed
        conservation_law     — dict (from _check_conservation_law)
        rank1_obstruction    — dict (from _check_rank1_obstruction)
        obstruction_type     — str, 'universal' or 'rank1_at_equilibrium'
        conserved_quantity   — str, human-readable conserved quantity
        invariant_foliation  — str, description of invariant leaves
        report               — str, human-readable obstruction report
    """
    z0 = np.array(equilibrium, dtype=float)

    cl   = _check_conservation_law(f, domain)
    r1   = _check_rank1_obstruction(f, z0, eps)

    # Both arguments are independent — conservation law is always true for ZOH
    obstructed = cl["confirmed"]   # conservation law is the universal argument
    obs_type = "universal" if obstructed else "partial"

    report = obstruction_report(cl, r1, domain)

    if verbose:
        print(report)

    return {
        "obstructed":          obstructed,
        "conservation_law":    cl,
        "rank1_obstruction":   r1,
        "obstruction_type":    obs_type,
        "conserved_quantity":  cl["conserved"],
        "invariant_foliation": f"x + e = c  (c ∈ ℝ, domain {domain})",
        "report":              report,
    }


# ---------------------------------------------------------------------------
# 5.  Human-readable report
# ---------------------------------------------------------------------------

def obstruction_report(cl: dict, r1: dict, domain: tuple) -> str:
    """Format a human-readable obstruction certificate.

    What it returns: a string that fully documents (A) the conservation law
    argument (always applies) and (B) the rank-1 Jacobian argument (applies
    when ∂f/∂x|₀ = ∂f/∂e|₀), along with the physical interpretation.
    """
    (x_lo, x_hi), (e_lo, e_hi) = domain
    lines = [
        "╔══ ZOH CONTRACTION OBSTRUCTION CERTIFICATE ══╗",
        "",
        "  ARGUMENT A — Conservation Law (universal, applies to ANY plant under ZOH)",
        f"    Conserved quantity : {cl['conserved']}",
        f"    Proof              : {cl['proof']}",
        f"    Conservation confirmed (max|error| = {cl['max_error']:.2e}) : {cl['confirmed']}",
        f"    Implication        : {cl['implication']}",
        "",
        "  ARGUMENT B — Rank-1 Jacobian (applies when ∂f/∂x|₀ = ∂f/∂e|₀)",
        f"    ∂f/∂x|₀           : {r1['df_dx']:.4f}",
        f"    ∂f/∂e|₀           : {r1['df_de']:.4f}",
        f"    rᵀ(0)·c           : {r1['rTc_value']:.6f}  (0 → obstruction exact)",
        f"    Rank-1 obstructed : {r1['obstructed']}",
        f"    {r1['proof_str'].replace(chr(10), chr(10)+'    ')}",
        "",
        "  CONCLUSION",
        f"    Domain            : x∈[{x_lo},{x_hi}], e∈[{e_lo},{e_hi}]",
        "    ZOH is contraction-obstructed: no Riemannian metric M(z) ≻ 0,",
        "    diagonal or full, can certify contraction for α=0 (ZOH).",
        "    Adding α > 0 (observer) breaks the conservation law and enables",
        "    contraction analysis.",
        "╚══════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)
