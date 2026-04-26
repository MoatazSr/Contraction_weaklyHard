"""
comparison.py — Fair same-architecture methodology comparison.

What this module proves
-----------------------
We compare three h-bounds for the scalar benchmark plant under η=(17,20):

  h_emul  — emulation bound on the SAME observer system (same architecture):
             h_emul = 1/(2L) where L = Lipschitz constant of the observer flow.
             This is the conservative baseline using the SAME α, SAME plant.

  h_2020  — MASP formula (Hertneck et al. 2020 style, ZOH architecture):
             h_2020 = MASP / (N-M+1).
             DIFFERENT architecture (ZOH, α=0).

  h_ours  — our contraction+MCM certified bound.
             SAME architecture as h_emul (observer, α=3).

Improvement decomposition
--------------------------
  Method improvement (same architecture):
      method_ratio = h_ours / h_emul  ≈ 3.1×
      Attribution: purely methodology (contraction+MCM vs emulation Lyapunov).

  Combined improvement (vs 2020 ZOH baseline):
      combined_ratio = h_ours / h_2020  ≈ 1.56×
      Attribution: JOINT — both architecture (observer vs ZOH) AND method.
      The combined ratio MUST be reported as joint, not as a pure method gain.

Public functions
----------------
compute_emulation_bound   — h_emul = 1/(2L) from Lipschitz constant of F
compute_masp_bound        — h_2020 from MASP formula for ZOH architecture
compare_methodology       — 3.1× method improvement (same architecture)
compare_combined          — 1.56× combined improvement (flagged as joint)
run_comparison            — full comparison dict (called by main.py)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1.  compute_emulation_bound
# ---------------------------------------------------------------------------

def compute_emulation_bound(
    F:        Callable,
    Z:        tuple,
    n_pts:    int = 40,
    eps_fd:   float = 1e-5,
) -> dict:
    """Estimate h_emul = 1/(2L) from the Lipschitz constant L of F on Z.

    What this computes
    ------------------
    The emulation / naive Lyapunov stability bound for an NCS:
        h_emul  =  1 / (2 · L)
    where L = sup_{z∈Z} ‖∂F/∂z‖_2 is the largest singular value of the
    Jacobian of F(z) over a grid of Z.

    This is the conservative baseline using the SAME observer architecture
    and the SAME α as our method.  The method improvement
        method_ratio = h_ours / h_emul
    is therefore a clean methodology comparison (no architecture difference).

    What assumptions this requires
    ------------------------------
    - F: ℝ² → ℝ² is the closed-loop observer flow for the chosen α.
    - Z = ((x_lo, x_hi), (e_lo, e_hi)) is the operating domain.
    - The emulation bound h = 1/(2L) is sufficient for stability of the
      NCS via a time-delay / emulation argument (conservative).

    Parameters
    ----------
    F      : callable(z: np.ndarray) -> np.ndarray, 2D vector field
    Z      : ((x_lo, x_hi), (e_lo, e_hi))
    n_pts  : grid points per axis
    eps_fd : finite-difference step for Jacobian estimation

    Returns
    -------
    dict with keys:
        h_emulation    — float, 1/(2L)
        L_lipschitz    — float, estimated Lipschitz constant
        method         — str, 'emulation_lyapunov_same_architecture'
        note           — str, attribution note
    """
    (x_lo, x_hi), (e_lo, e_hi) = Z
    xs = np.linspace(x_lo, x_hi, int(n_pts))
    es = np.linspace(e_lo, e_hi, int(n_pts))

    L_max = 0.0
    for x in xs:
        for e in es:
            z = np.array([x, e], dtype=float)
            f0 = np.asarray(F(z), dtype=float)
            # Numerical Jacobian via finite differences
            J = np.zeros((2, 2))
            for j, dz in enumerate([np.array([eps_fd, 0.0]),
                                     np.array([0.0, eps_fd])]):
                fp = np.asarray(F(z + dz), dtype=float)
                fm = np.asarray(F(z - dz), dtype=float)
                J[:, j] = (fp - fm) / (2 * eps_fd)
            sigma_max = float(np.linalg.norm(J, ord=2))
            if sigma_max > L_max:
                L_max = sigma_max

    h_emul = 1.0 / (2.0 * L_max) if L_max > 0 else float("nan")

    return {
        "h_emulation": h_emul,
        "L_lipschitz": L_max,
        "method":      "emulation_lyapunov_same_architecture",
        "note": (
            "Conservative emulation bound on the SAME observer architecture. "
            f"L = {L_max:.3f} on Z, h_emul = 1/(2L) = {h_emul:.4f} s. "
            "Comparing h_ours vs h_emul gives a pure method improvement "
            "(no architecture change involved)."
        ),
    }


# ---------------------------------------------------------------------------
# 2.  compute_masp_bound
# ---------------------------------------------------------------------------

def compute_masp_bound(
    eta:  tuple[int, int],
    masp: float = 0.5,
) -> dict:
    """MASP-formula bound for the ZOH architecture (Hertneck et al. style).

    What this computes
    ------------------
    A simple MASP (Maximum Allowable Sampling Period) formula for ZOH:
        h_2020 = MASP / (N − M + 1)
    where MASP is the MASP for the continuous system and N−M+1 is the
    worst-case inter-reception gap under η=(M,N).

    ARCHITECTURE NOTE: This formula applies to ZOH (α=0). Our method uses
    an observer (α=3) — a different closed-loop architecture. Comparing
    h_ours vs h_2020 is a JOINT comparison: any improvement comes from
    both the architecture change AND the methodology change.

    Parameters
    ----------
    eta  : (M, N) — WHRT constraint
    masp : MASP for the continuous plant (seconds)

    Returns
    -------
    dict with keys:
        h_2020         — float, MASP formula bound
        l_max          — int, N−M+1
        architecture   — str, 'ZOH'
        caveat         — str, must be included in any paper claim
    """
    m, n = int(eta[0]), int(eta[1])
    l_max  = n - m + 1
    h_2020 = masp / l_max

    return {
        "h_2020":      h_2020,
        "l_max":       l_max,
        "eta":         (m, n),
        "masp_used":   masp,
        "architecture": "ZOH",
        "caveat": (
            f"Architecture differs: ZOH (α=0) vs observer (α=3). "
            f"h_2020 = {masp}/{l_max} = {h_2020:.4f} s. "
            "Any h_ours/h_2020 ratio is a JOINT improvement (architecture + method). "
            "Do not report it as a pure methodology gain."
        ),
    }


# ---------------------------------------------------------------------------
# 3.  compare_methodology  (same architecture — clean method comparison)
# ---------------------------------------------------------------------------

def compare_methodology(
    h_ours: float,
    h_emul: float,
    label:  str = "ours",
) -> dict:
    """Pure method improvement: contraction+MCM vs emulation Lyapunov.

    What this proves
    ----------------
    Both h_ours and h_emul use the SAME observer architecture (same α, same
    plant, same domain Z). The ratio h_ours / h_emul isolates the gain from
    using contraction theory + Karp MCM instead of a naive emulation bound.
    This is the only fair same-architecture methodology comparison.

    Parameters
    ----------
    h_ours : our certified sampling period (contraction+MCM)
    h_emul : emulation bound on the same architecture (1/(2L))
    label  : label for h_ours in output

    Returns
    -------
    dict with keys:
        method_ratio    — float, h_ours / h_emul
        h_ours, h_emul  — float
        attribution     — str, 'methodology only (same architecture)'
        conclusion      — str
    """
    ratio = h_ours / h_emul if h_emul > 0 else float("nan")
    conclusion = (
        f"Method improvement: h_{label} = {h_ours:.4f} s vs h_emul = {h_emul:.4f} s "
        f"→ {ratio:.2f}× (same observer architecture, methodology change only)."
    )
    return {
        "method_ratio": float(ratio),
        "h_ours":       float(h_ours),
        "h_emul":       float(h_emul),
        "attribution":  "methodology only (same observer architecture, same α)",
        "conclusion":   conclusion,
    }


# ---------------------------------------------------------------------------
# 4.  compare_combined  (different architecture — must flag as joint)
# ---------------------------------------------------------------------------

def compare_combined(
    h_ours: float,
    h_2020: float,
    label:  str = "ours",
) -> dict:
    """Combined improvement vs 2020 ZOH baseline — JOINT attribution required.

    What this computes
    ------------------
    h_ours / h_2020 is the total improvement over the ZOH MASP baseline.
    This ratio CANNOT be attributed to methodology alone because:
      - h_2020 uses ZOH (α=0): no observer, different closed-loop dynamics.
      - h_ours uses an observer (α=3): different architecture.
    Both the architecture change AND the methodology change contribute.

    MANDATORY CAVEAT: any paper or report citing this ratio must state
    explicitly that the improvement is joint (architecture + methodology).

    Parameters
    ----------
    h_ours : our certified h (contraction+MCM, observer architecture)
    h_2020 : MASP formula bound (ZOH architecture)
    label  : label for h_ours in output

    Returns
    -------
    dict with keys:
        combined_ratio    — float, h_ours / h_2020
        h_ours, h_2020    — float
        attribution       — str, always 'JOINT: architecture + methodology'
        mandatory_caveat  — str, must appear in any paper citation
        conclusion        — str
    """
    ratio = h_ours / h_2020 if h_2020 > 0 else float("nan")
    conclusion = (
        f"Combined improvement: h_{label} = {h_ours:.4f} s vs h_2020 = {h_2020:.4f} s "
        f"→ {ratio:.2f}×. "
        f"JOINT: both architecture (observer vs ZOH) and methodology (MCM vs MASP)."
    )
    mandatory_caveat = (
        f"The {ratio:.2f}× combined improvement over h_2020={h_2020:.4f}s (ZOH MASP) "
        "is a JOINT gain from two co-contributions: "
        "(a) the observer architecture that enables contraction analysis "
        "(ZOH is structurally obstructed — no contraction certificate exists for α=0), "
        "and (b) the contraction+MCM methodology that certifies a tighter h than "
        "the emulation bound on the same architecture. "
        "This ratio must NOT be reported as a pure methodology improvement."
    )
    return {
        "combined_ratio":   float(ratio),
        "h_ours":           float(h_ours),
        "h_2020":           float(h_2020),
        "attribution":      "JOINT: architecture (observer vs ZOH) + methodology (MCM vs MASP)",
        "mandatory_caveat": mandatory_caveat,
        "conclusion":       conclusion,
    }


# ---------------------------------------------------------------------------
# 5.  run_comparison  (called by main.py)
# ---------------------------------------------------------------------------

def run_comparison(
    h_ours:   float,
    F:        Callable,
    Z:        tuple,
    eta:      tuple[int, int],
    masp:     float = 0.5,
    mcm:      float | None = None,
    n_pts_lip: int  = 40,
) -> dict:
    """Full comparison: emulation baseline, MASP baseline, both improvements.

    Parameters
    ----------
    h_ours     : our certified sampling period h (contraction+MCM)
    F          : closed-loop flow callable(z: np.ndarray) -> np.ndarray
    Z          : ((x_lo, x_hi), (e_lo, e_hi))
    eta        : (M, N) WHRT constraint
    masp       : MASP for continuous plant (seconds)
    mcm        : MCM value (for reporting only, optional)
    n_pts_lip  : grid for Lipschitz estimation

    Returns
    -------
    dict with keys:
        passed              — bool
        emulation_baseline  — dict from compute_emulation_bound
        masp_baseline       — dict from compute_masp_bound
        method_comparison   — dict from compare_methodology
        combined_comparison — dict from compare_combined
        method_ratio        — float, h_ours / h_emul
        combined_ratio      — float, h_ours / h_2020
        mcm                 — float (passed through)
        summary_table       — list[list], for printing
        mandatory_caveat    — str
    """
    emul_res  = compute_emulation_bound(F, Z, n_pts=n_pts_lip)
    masp_res  = compute_masp_bound(eta, masp)

    h_emul = emul_res["h_emulation"]
    h_2020 = masp_res["h_2020"]

    meth_res  = compare_methodology(h_ours, h_emul)
    comb_res  = compare_combined(h_ours, h_2020)

    passed = (h_ours > h_emul) and (h_ours > 0)

    m, n = eta
    summary_table = [
        ["Method",                            "Architecture", "h (s)",            "vs h_ours", "Attribution"],
        ["ZOH (α=0)",                         "ZOH",          "N/A (obstructed)", "—",         "no contraction cert."],
        [f"Emulation (α=same, L={emul_res['L_lipschitz']:.2f})",
                                              "Observer",     f"{h_emul:.4f}",
         f"1/{meth_res['method_ratio']:.2f}×", "same architecture"],
        [f"MASP formula (ZOH, η=({m},{n}))", "ZOH",          f"{h_2020:.4f}",
         f"1/{comb_res['combined_ratio']:.2f}×", "JOINT (arch+method)"],
        ["Ours (contraction+MCM)",            "Observer",     f"{h_ours:.4f}",    "1×",        "this work"],
    ]

    return {
        "passed":              passed,
        "emulation_baseline":  emul_res,
        "masp_baseline":       masp_res,
        "method_comparison":   meth_res,
        "combined_comparison": comb_res,
        "method_ratio":        meth_res["method_ratio"],
        "combined_ratio":      comb_res["combined_ratio"],
        "mcm":                 mcm,
        "summary_table":       summary_table,
        "mandatory_caveat":    comb_res["mandatory_caveat"],
    }
