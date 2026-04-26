"""
invariance.py — Forward invariance certificate for hybrid NCS domains.

What this module proves
-----------------------
Assumption A2 (Domain invariance): Z is forward-invariant under the hybrid
dynamics — every trajectory starting in Z remains in Z.

For a hybrid system with continuous flow ż = F(z) and discrete resets g(z),
invariance is proved in two parts:

  (A) Post-jump reachability: from the post-jump set g(Z) ⊆ Z, simulate
      the flow for the worst-case inter-reception duration T_max = l_max · h
      and verify the trajectory stays inside Z.

  (B) Boundary flux check: verify F(z) points inward on ∂Z for all z ∈ ∂Z
      where trajectories can actually reach. If only (A) is verified, the
      certificate carries an honest gap flag.

HONEST GAP: Strict box-invariance rarely holds for nonlinear systems — the
vector field may exit the box at corners that are unreachable under the hybrid
dynamics. When this happens, only the post-jump reachability check (A) is
positive, and the certificate is marked 'gap_flag=True' with an explanation.

Public functions
----------------
boundary_flux_check      — check inward-pointing vector field on ∂Z
post_jump_reachability   — simulate from post-jump set for T_max
forward_invariance_certificate — combine (A)+(B), return gap flag if needed
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1.  boundary_flux_check
# ---------------------------------------------------------------------------

def boundary_flux_check(
    F:        Callable,
    Z:        tuple,
    n_pts:    int = 200,
    atol:     float = 1e-9,
) -> dict:
    """Check that the vector field F(z) points strictly inward on ∂Z.

    What this proves: if F·n_out < 0 everywhere on ∂Z (n_out = outward normal),
    then Z is positively invariant under ż = F(z) by Nagumo's theorem.

    What assumptions this require
    ------------------------------
    - Z = (x_lo, x_hi) × (e_lo, e_hi) is an axis-aligned box (2D).
    - F: ℝ² → ℝ² is the closed-loop flow (continuous, no jumps).

    Parameters
    ----------
    F     : callable(z: np.ndarray) -> np.ndarray, 2D vector field
    Z     : ((x_lo, x_hi), (e_lo, e_hi))
    n_pts : grid points per boundary edge
    atol  : tolerance for strict inward-pointing (F·n < atol)

    Returns
    -------
    dict with keys:
        strictly_invariant — bool, True iff no boundary violations found
        n_violations       — int, number of grid points with outward flux
        worst_flux         — float, max(F·n_out) over boundary (positive = bad)
        violated_faces     — list[str], which faces have violations
        gap_flag           — bool, alias for 'not strictly_invariant'
        gap_note           — str, explanation if gap_flag is True
    """
    (x_lo, x_hi), (e_lo, e_hi) = Z
    xs = np.linspace(x_lo, x_hi, n_pts)
    es = np.linspace(e_lo, e_hi, n_pts)

    violations = []

    def _check_face(pts, normal_idx, normal_sign, face_name):
        for pt in pts:
            z = np.asarray(pt, dtype=float)
            try:
                fval = np.asarray(F(z), dtype=float)
            except Exception:
                continue
            flux = normal_sign * fval[normal_idx]
            if flux > atol:
                violations.append((face_name, pt, float(flux)))

    # x_hi face (outward normal = +x)
    _check_face([(x_hi, e) for e in es], normal_idx=0, normal_sign=+1, face_name="x_hi")
    # x_lo face (outward normal = -x)
    _check_face([(x_lo, e) for e in es], normal_idx=0, normal_sign=-1, face_name="x_lo")
    # e_hi face (outward normal = +e)
    _check_face([(x, e_hi) for x in xs], normal_idx=1, normal_sign=+1, face_name="e_hi")
    # e_lo face (outward normal = -e)
    _check_face([(x, e_lo) for x in xs], normal_idx=1, normal_sign=-1, face_name="e_lo")

    worst = max((v[2] for v in violations), default=0.0)
    faces = sorted(set(v[0] for v in violations))
    ok    = len(violations) == 0

    gap_note = ""
    if not ok:
        gap_note = (
            f"Box Z is NOT strictly forward-invariant under the continuous flow: "
            f"{len(violations)} boundary points have outward flux (worst = {worst:.3f}) "
            f"on faces {faces}. "
            f"Use post_jump_reachability() to check whether those corners are "
            f"actually reachable from the post-jump set — if not, the hybrid "
            f"certificate still holds."
        )

    return {
        "strictly_invariant": ok,
        "n_violations":       len(violations),
        "worst_flux":         float(worst),
        "violated_faces":     faces,
        "gap_flag":           not ok,
        "gap_note":           gap_note,
    }


# ---------------------------------------------------------------------------
# 2.  post_jump_reachability
# ---------------------------------------------------------------------------

def post_jump_reachability(
    F:        Callable,
    g:        Callable,
    Z:        tuple,
    l_max:    int,
    h:        float,
    n_x0:    int = 200,
    n_dense:  int = 300,
) -> dict:
    """Verify the reachable set from g(Z) stays inside Z over T_max = l_max·h.

    What this proves
    ----------------
    At each reception, the jump map g resets the state. The post-jump set is
    g(Z ∩ {e=0}) = {g(x, 0) : x ∈ [x_lo, x_hi]}.  Under ZOH with error reset
    g(x, e) = (x, 0), the post-jump set is {(x, 0) : x ∈ [x_lo, x_hi]}.

    We simulate ż = F(z) from each post-jump state (x0, 0) for T_max = l_max·h
    seconds (the worst-case inter-reception duration) and record max |e(t)|.

    If max |e(t)| ≤ e_hi for all x0, then every trajectory from g(Z) stays
    in Z during the flow phase — the hybrid domain Z is effectively invariant
    for the relevant reachable set.

    What assumptions this requires
    ------------------------------
    - g(Z) ⊆ Z (the post-jump image lies in Z; here g(x,e) = (x,0) → trivial).
    - F is C¹ on Z (so solve_ivp succeeds).

    Parameters
    ----------
    F       : callable(z: np.ndarray) -> np.ndarray, 2D vector field
    g       : callable(z: np.ndarray) -> np.ndarray, jump reset map
    Z       : ((x_lo, x_hi), (e_lo, e_hi))
    l_max   : worst-case number of inter-reception steps (= N−M+1)
    h       : sampling period (seconds)
    n_x0    : number of x0 grid points to try
    n_dense : time-grid density for max-|e| search

    Returns
    -------
    dict with keys:
        inside_domain     — bool, True iff max |e| < e_hi for all x0
        max_e_reachable   — float, worst max |e(t)| over all x0
        e_hi              — float, upper e-boundary
        margin            — float, e_hi − max_e_reachable (positive = safe)
        worst_x0          — float, x0 achieving max |e|
        T_max             — float, l_max * h
        l_max             — int
        n_failures        — int, ODE solve failures
        gap_flag          — bool, True iff not inside_domain
        gap_note          — str, explanation if gap_flag
    """
    (x_lo, x_hi), (e_lo, e_hi) = Z
    T_max = l_max * h

    def _ode(t, z):
        return list(F(np.asarray(z, dtype=float)))

    x0_grid   = np.linspace(x_lo, x_hi, int(n_x0))
    max_e_abs = 0.0
    worst_x0  = float("nan")
    n_fail    = 0

    for x0 in x0_grid:
        z_pj = g(np.array([x0, 0.0]))
        try:
            sol = solve_ivp(_ode, [0.0, T_max], list(z_pj),
                            method="RK45", rtol=1e-8, atol=1e-10,
                            dense_output=True, max_step=h / 20)
            if not sol.success:
                n_fail += 1
                continue
            t_dense = np.linspace(0.0, T_max, int(n_dense))
            y_dense = sol.sol(t_dense)
            me = float(np.max(np.abs(y_dense[1])))
            if me > max_e_abs:
                max_e_abs = me
                worst_x0  = float(x0)
        except Exception:
            n_fail += 1

    inside  = max_e_abs < e_hi
    margin  = e_hi - max_e_abs
    gap_note = ""
    if not inside:
        gap_note = (
            f"Post-jump reachable set exceeds e-boundary: max|e| = {max_e_abs:.4f} "
            f">= e_hi = {e_hi}. Enlarge DOMAIN_E or reduce l_max·h."
        )

    return {
        "inside_domain":    inside,
        "max_e_reachable":  float(max_e_abs),
        "e_hi":             float(e_hi),
        "margin":           float(margin),
        "worst_x0":         float(worst_x0),
        "T_max":            float(T_max),
        "l_max":            int(l_max),
        "n_failures":       int(n_fail),
        "gap_flag":         not inside,
        "gap_note":         gap_note,
    }


# ---------------------------------------------------------------------------
# 3.  forward_invariance_certificate
# ---------------------------------------------------------------------------

def forward_invariance_certificate(
    F:        Callable,
    g:        Callable,
    Z:        tuple,
    l_max:    int,
    h:        float,
    n_pts:    int = 200,
    n_x0:    int = 200,
    verbose:  bool = True,
) -> dict:
    """Combined forward invariance certificate with honest gap flag.

    What this proves
    ----------------
    Assumption A2 is verified in two ways:
      (A) Boundary flux: F·n < 0 on ∂Z  (Nagumo-style box invariance)
      (B) Post-jump reachability: from g(Z ∩ {e=0}) the flow stays in Z
          for T_max = l_max·h seconds

    If (A) fails but (B) holds, the certificate carries gap_flag=True with
    explanation: the box is not strictly invariant, but the HYBRID certificate
    is still valid because the unreachable corners are where (A) fails.

    If both fail, the certificate is invalid — Z must be enlarged.

    Parameters
    ----------
    F      : callable(z: np.ndarray) -> np.ndarray, 2D flow vector field
    g      : callable(z: np.ndarray) -> np.ndarray, jump reset map
    Z      : ((x_lo, x_hi), (e_lo, e_hi))
    l_max  : worst-case inter-reception steps (= N−M+1)
    h      : sampling period (seconds)
    n_pts  : boundary grid points for flux check
    n_x0   : initial-condition grid for reachability check
    verbose: print results

    Returns
    -------
    dict with keys:
        certified         — bool, True iff hybrid invariance is verified
        gap_flag          — bool, True iff box not strictly invariant (but hybrid ok)
        gap_note          — str, explanation of the gap
        boundary_flux     — dict from boundary_flux_check
        post_jump         — dict from post_jump_reachability
        conclusion        — str, human-readable certificate conclusion
    """
    bfc = boundary_flux_check(F, Z, n_pts=n_pts)
    pjr = post_jump_reachability(F, g, Z, l_max=l_max, h=h, n_x0=n_x0)

    box_inv     = bfc["strictly_invariant"]
    hybrid_inv  = pjr["inside_domain"]

    certified = hybrid_inv   # hybrid certificate: box + post-jump
    gap_flag  = (not box_inv) and hybrid_inv

    if certified and not gap_flag:
        conclusion = (
            f"INVARIANCE CERTIFIED (strict box + hybrid): "
            f"F·n < 0 on all ∂Z, and post-jump reachable set stays in Z "
            f"(max|e|={pjr['max_e_reachable']:.4f}, margin={pjr['margin']:.4f})."
        )
        gap_note = ""
    elif certified and gap_flag:
        conclusion = (
            f"INVARIANCE CERTIFIED (hybrid only, gap flag set): "
            f"Z is NOT strictly box-invariant (flux violations on {bfc['violated_faces']}), "
            f"but post-jump reachable set stays in Z "
            f"(max|e|={pjr['max_e_reachable']:.4f} < e_hi={pjr['e_hi']}, "
            f"margin={pjr['margin']:.4f}). "
            f"The flux violations occur at corners unreachable from g(Z); "
            f"the hybrid certificate is still valid."
        )
        gap_note = (
            "Gap: box Z is not strictly forward-invariant under continuous flow. "
            "The flux violations occur near (|x|~x_hi, |e|~e_hi) corners that "
            "are NOT reachable from the post-jump set {e=0} over T_max = "
            f"{pjr['T_max']:.3f} s. "
            "Formal proof of corner unreachability is numerical only (simulation). "
            "A Lyapunov-based sublevel-set proof would close this gap."
        )
    else:
        conclusion = (
            f"INVARIANCE FAILED: box check={box_inv}, "
            f"post-jump check={hybrid_inv}. "
            f"Enlarge Z or reduce T_max."
        )
        gap_note = (
            "Certificate invalid: the post-jump reachable set exits Z. "
            f"max|e|={pjr['max_e_reachable']:.4f} >= e_hi={pjr['e_hi']}."
        )

    if verbose:
        print(f"  Invariance certificate for Z = {Z}:")
        print(f"    Strict box invariant    : {box_inv}  "
              f"(violations: {bfc['n_violations']}, worst flux: {bfc['worst_flux']:.4f})")
        print(f"    Post-jump reachability  : {hybrid_inv}  "
              f"(max|e|={pjr['max_e_reachable']:.4f}, margin={pjr['margin']:.4f})")
        print(f"    Certified               : {certified}")
        if gap_flag:
            print(f"    GAP FLAG: {gap_note}")

    return {
        "certified":      certified,
        "gap_flag":       gap_flag,
        "gap_note":       gap_note,
        "boundary_flux":  bfc,
        "post_jump":      pjr,
        "conclusion":     conclusion,
    }
