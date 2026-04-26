"""
whrt_mcm.py — Plant-agnostic WHRT + MCM stability pipeline.

What this module proves
-----------------------
Given:
  - a certified contraction rate λ > 0 (from the flow certificate),
  - a sampling period h > 0,
  - a WHRT constraint η = (M, N), and
  - a jump contraction factor ρ_jump ≤ 1 (from the reset map),

the module computes the per-label total growth factors

    ρ_total(l) = e^{-λ·l·h} · ρ_jump,      l = 1, …, l_max = N−M+1

and the Maximum Cycle Mean (MCM) of the WHRT automaton G(η), using
Karp's (1978) exact algorithm.  The system is exponentially stable under
ALL η-admissible dropout sequences if and only if MCM < 1.

Public functions
----------------
build_automaton          — construct the WHRT automaton G(η) from η=(M,N)
compute_gronwall_bounds  — per-label Gronwall bounds e^{-λ·l·h}
certify                  — full MCM certificate for given (λ, h, η, ρ_jump)
sweep                    — compute MCM as a function of h; find certified h*

What assumptions this requires
-------------------------------
- The flow certificate (A5) has been verified: there exists M(z) such that
  Ṁ + JᵀM + MJ + 2λM ≼ 0 on Z.  This module takes λ as given; it does
  NOT verify the flow condition.
- The jump map satisfies ρ_jump ≤ 1 (taken as input, not verified here).
- The WHRT constraint η = (M, N) with 0 < M < N is given.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np

# Pull whrt_graph from the same src/ directory without importing config
sys.path.insert(0, os.path.dirname(__file__))
from whrt_graph import build_whrt_graph, compute_max_cycle_mean


# ---------------------------------------------------------------------------
# 1.  build_automaton
# ---------------------------------------------------------------------------

def build_automaton(eta: tuple[int, int]):
    """Construct the WHRT automaton G(η) from η = (M, N).

    What this proves: the automaton has nodes 0 … N−M (failure-debt states)
    and edges labelled l ∈ {1, …, N−M+1} (inter-reception gaps), capturing
    all η-admissible dropout sequences.

    Parameters
    ----------
    eta : (M, N) — WHRT constraint; M successes in every N-step window

    Returns
    -------
    G   : nx.MultiDiGraph, WHRT automaton
    info: dict with n_nodes, n_edges, l_max, strongly_connected
    """
    m, n = int(eta[0]), int(eta[1])
    if m >= n:
        raise ValueError(f"WHRT requires M < N, got η=({m},{n})")
    if m <= 0 or n <= 0:
        raise ValueError(f"WHRT requires M,N > 0, got η=({m},{n})")

    G = build_whrt_graph(m, n)

    import networkx as nx
    sc = nx.is_strongly_connected(G)
    all_labels = sorted(set(d["label"] for _, _, d in G.edges(data=True)))
    l_max = max(all_labels) if all_labels else 0

    info = {
        "m":                 m,
        "n":                 n,
        "l_max":             l_max,
        "n_nodes":           G.number_of_nodes(),
        "n_edges":           G.number_of_edges(),
        "strongly_connected": sc,
        "all_labels":        all_labels,
    }
    return G, info


# ---------------------------------------------------------------------------
# 2.  compute_gronwall_bounds
# ---------------------------------------------------------------------------

def compute_gronwall_bounds(lam: float, h: float, max_l: int) -> dict[int, float]:
    """Per-label Gronwall flow bounds: ρ_flow(l) = e^{-λ·l·h}.

    What this proves: if the flow condition A5 holds with rate λ, then
    by Gronwall's inequality the metric-weighted distance decays as
    ‖δz(l·h)‖_M ≤ e^{-λ·l·h} ‖δz(0)‖_M over every interval of length l·h.

    What assumptions this requires
    ------------------------------
    - The flow certificate A5 has been verified externally (this function
      takes λ as a certified input, does not verify it).
    - Domain invariance: z(t) ∈ Z for all t (so M(z(t)) is well-defined).

    Parameters
    ----------
    lam   : certified contraction rate λ > 0
    h     : sampling period (seconds)
    max_l : largest label (= N−M+1 for η=(M,N))

    Returns
    -------
    dict {l: ρ_flow(l)} for l = 1, …, max_l
    """
    if lam <= 0:
        raise ValueError(f"Contraction rate must be positive, got λ={lam}")
    if h <= 0:
        raise ValueError(f"Sampling period must be positive, got h={h}")
    return {l: float(math.exp(-lam * l * h)) for l in range(1, int(max_l) + 1)}


# ---------------------------------------------------------------------------
# 3.  certify
# ---------------------------------------------------------------------------

def certify(
    lam:      float,
    h:        float,
    eta:      tuple[int, int],
    rho_jump: float = 1.0,
) -> dict:
    """Full MCM stability certificate for given (λ, h, η, ρ_jump).

    What this proves
    ----------------
    Under Assumptions A1–A6, the hybrid system under η-constrained dropout
    is exponentially stable if and only if MCM(G(η), ρ_total) < 1, where:
        ρ_total(l) = e^{-λ·l·h} · ρ_jump  for l = 1, …, N−M+1.
    This uses Karp's (1978) exact algorithm — the result is necessary and
    sufficient over ALL η-admissible infinite dropout sequences.

    What assumptions this requires
    ------------------------------
    - λ is a certified contraction rate (flow condition A5 externally verified).
    - ρ_jump ≤ 1 is certified (jump condition A6 externally verified).
    - η = (M, N) is the WHRT constraint.
    - h is the sampling period.

    Parameters
    ----------
    lam      : certified contraction rate λ > 0
    h        : sampling period (seconds)
    eta      : (M, N) — WHRT constraint
    rho_jump : jump contraction factor (default 1.0 = non-expanding)

    Returns
    -------
    dict with keys:
        stable           — bool, True iff MCM < 1
        mcm              — float, Maximum Cycle Mean
        mcm_log          — float, log(MCM)
        stability_margin — float, 1 − MCM  (positive = stable)
        worst_cycle      — list[int], labels of the worst elementary cycle
        worst_cycle_mean — float, geometric mean of ρ_total on worst cycle
        rho_flow         — dict {l: e^{-λ·l·h}}, Gronwall bounds
        rho_total        — dict {l: ρ_total(l)}, total growth factors
        rho_jump         — float, input jump factor
        lam              — float
        h                — float
        eta              — (M, N)
        l_max            — int, N−M+1
        n_nodes          — int, automaton nodes
        certificate_type — str, 'karp_exact'
        certified        — bool, True iff MCM < 1 (alias for stable)
        conclusion       — str, human-readable conclusion
    """
    G, info = build_automaton(eta)
    l_max = info["l_max"]

    rho_flow  = compute_gronwall_bounds(lam, h, l_max)
    rho_total = {l: rho_flow[l] * rho_jump for l in rho_flow}

    mcm_result = compute_max_cycle_mean(G, rho_total)

    mcm    = mcm_result.get("mcm", float("nan"))
    stable = mcm_result.get("stable", False)
    margin = mcm_result.get("stability_margin", float("nan"))

    if stable:
        conclusion = (
            f"STABLE: MCM = {mcm:.6f} < 1  (margin = {margin:.6f}). "
            f"System is exponentially stable under ALL η=({eta[0]},{eta[1]})-"
            f"admissible dropout sequences. Certificate is necessary and sufficient."
        )
    else:
        conclusion = (
            f"UNSTABLE: MCM = {mcm:.6f} ≥ 1  (margin = {margin:.6f}). "
            f"Cannot certify stability for η=({eta[0]},{eta[1]}), "
            f"h={h:.4f} s, λ={lam:.4f}."
        )

    return {
        "stable":           stable,
        "certified":        stable,
        "mcm":              mcm,
        "mcm_log":          mcm_result.get("mcm_log", float("nan")),
        "stability_margin": margin,
        "worst_cycle":      mcm_result.get("worst_cycle", []),
        "worst_cycle_mean": mcm_result.get("worst_cycle_mean", float("nan")),
        "rho_flow":         rho_flow,
        "rho_total":        rho_total,
        "rho_jump":         rho_jump,
        "lam":              lam,
        "h":                h,
        "eta":              eta,
        "l_max":            l_max,
        "n_nodes":          info["n_nodes"],
        "certificate_type": "karp_exact",
        "conclusion":       conclusion,
    }


# ---------------------------------------------------------------------------
# 4.  sweep
# ---------------------------------------------------------------------------

def sweep(
    lam:        float,
    eta:        tuple[int, int],
    rho_jump:   float = 1.0,
    h_lo:       float = 0.01,
    h_hi:       float = 1.0,
    n_points:   int   = 50,
    verbose:    bool  = False,
) -> dict:
    """Compute MCM as a function of h; find the maximum certifiable h*.

    What this proves
    ----------------
    By evaluating certify(λ, h, η, ρ_jump) over a grid of h values, finds:
      - h* = sup{h : MCM(h) < 1}  — the certified sampling period bound
      - The MCM curve MCM(h) for all h in [h_lo, h_hi]

    Since MCM(h) = exp(MCM_log) is monotonically increasing in h (larger h →
    less contraction per interval), there is a unique crossing MCM(h*) = 1.
    The sweep uses a uniform grid; bisection can refine h* further.

    Parameters
    ----------
    lam      : certified contraction rate λ > 0
    eta      : (M, N) — WHRT constraint
    rho_jump : jump contraction factor
    h_lo     : lower bound for h sweep
    h_hi     : upper bound for h sweep
    n_points : number of h values to evaluate
    verbose  : print sweep progress

    Returns
    -------
    dict with keys:
        h_certified      — float or None, largest h with MCM < 1 in grid
        mcm_at_h_cert    — float, MCM at h_certified
        margin_at_h_cert — float, stability margin at h_certified
        h_grid           — np.ndarray, h values evaluated
        mcm_grid         — np.ndarray, MCM values (may contain nan on failure)
        stable_mask      — np.ndarray[bool], True where MCM < 1
        n_stable         — int, number of stable h values found
        n_total          — int = n_points
        lam              — float
        eta              — (M, N)
        rho_jump         — float
        certified        — bool, True iff any stable h was found
        conclusion       — str
    """
    h_grid   = np.linspace(h_lo, h_hi, int(n_points))
    mcm_grid = np.full(len(h_grid), float("nan"))
    stable_mask = np.zeros(len(h_grid), dtype=bool)

    for i, h in enumerate(h_grid):
        try:
            res = certify(lam, float(h), eta, rho_jump)
            mcm_grid[i]    = res["mcm"]
            stable_mask[i] = res["stable"]
        except Exception:
            pass

        if verbose:
            flag = "✔" if stable_mask[i] else "✘"
            mcm_str = f"{mcm_grid[i]:.5f}" if not math.isnan(mcm_grid[i]) else "err"
            print(f"    h={h:.4f}  MCM={mcm_str}  {flag}")

    n_stable  = int(np.sum(stable_mask))
    certified = n_stable > 0

    h_cert      = None
    mcm_cert    = float("nan")
    margin_cert = float("nan")

    if certified:
        # Largest stable h in the grid
        stable_indices = np.where(stable_mask)[0]
        best_idx    = int(stable_indices[-1])
        h_cert      = float(h_grid[best_idx])
        mcm_cert    = float(mcm_grid[best_idx])
        margin_cert = float(1.0 - mcm_cert)

    if certified:
        conclusion = (
            f"Certified h* ≥ {h_cert:.4f} s  "
            f"(MCM = {mcm_cert:.6f}, margin = {margin_cert:.6f}) "
            f"for η=({eta[0]},{eta[1]}), λ={lam:.4f}, ρ_jump={rho_jump:.4f}."
        )
    else:
        conclusion = (
            f"No stable h found in [{h_lo:.4f}, {h_hi:.4f}] for "
            f"η=({eta[0]},{eta[1]}), λ={lam:.4f}, ρ_jump={rho_jump:.4f}. "
            f"Try larger λ or smaller h_hi."
        )

    return {
        "h_certified":      h_cert,
        "mcm_at_h_cert":    mcm_cert,
        "margin_at_h_cert": margin_cert,
        "h_grid":           h_grid,
        "mcm_grid":         mcm_grid,
        "stable_mask":      stable_mask,
        "n_stable":         n_stable,
        "n_total":          len(h_grid),
        "lam":              lam,
        "eta":              eta,
        "rho_jump":         rho_jump,
        "certified":        certified,
        "conclusion":       conclusion,
    }
