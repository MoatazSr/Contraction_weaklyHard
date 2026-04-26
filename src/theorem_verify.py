"""
theorem_verify.py — Stability certificate via Maximum Cycle Mean.

Certificate strategy
--------------------
PRIMARY:   compute_max_cycle_mean() from whrt_graph.py  [EXACT, N&S condition]
SECONDARY: verify_theorem_walks()                        [validation over sampled walks]

The MCM gives a NECESSARY AND SUFFICIENT condition for stability under ALL
WHRT-admissible infinite dropout sequences (including cyclic patterns).
The walk-product check is retained as a sanity cross-check only.

Public functions
----------------
verify_mcm_certificate  — primary certificate (wraps whrt_graph.compute_max_cycle_mean)
verify_theorem_walks    — secondary validation (walk-product check)
save_report             — write text report to disk
"""

from __future__ import annotations

import datetime
import numpy as np
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# 1.  verify_mcm_certificate  ← PRIMARY CERTIFICATE
# ---------------------------------------------------------------------------

def verify_mcm_certificate(G, rho_total: dict,
                            flow_cert: dict,
                            rho_jump_info: dict) -> dict:
    """Compute the exact MCM certificate for the WHRT system.

    Calls whrt_graph.compute_max_cycle_mean() and enriches the result with
    the upstream certificate chain status.

    Parameters
    ----------
    G             : WHRT automaton (from build_whrt_graph)
    rho_total     : dict {label → float}, per-label total growth factors
    flow_cert     : dict from lipschitz_certified_flow_viol
    rho_jump_info : dict from compute_rho_jump

    Returns
    -------
    dict with all keys from compute_max_cycle_mean, plus:
        flow_certified       — bool (from flow_cert)
        rho_jump_certified   — bool (from rho_jump_info)
        full_chain_certified — bool, True iff entire certificate chain is valid
        conclusion           — str, formal conclusion
        certificate_summary  — str, human-readable summary
    """
    # Import here to avoid circular import
    from whrt_graph import compute_max_cycle_mean

    mcm_result = compute_max_cycle_mean(G, rho_total)

    flow_certified     = bool(flow_cert.get("certified", False))
    rho_jump_certified = bool(rho_jump_info.get("certified", False))
    full_chain         = (flow_certified and rho_jump_certified
                          and mcm_result.get("stable", False))

    mcm  = mcm_result.get("mcm", float("nan"))
    stab = mcm_result.get("stable", False)
    margin = mcm_result.get("stability_margin", float("nan"))

    if full_chain:
        conclusion = (
            f"COMPLETE STABILITY CERTIFICATE: MCM = {mcm:.6f} < 1.0 "
            f"(margin = {margin:.6f}). The system is exponentially "
            f"incrementally stable for ALL WHRT-admissible dropout sequences. "
            f"Certificate is NECESSARY AND SUFFICIENT."
        )
        cert_type = "COMPLETE_NSC"
    elif stab and not (flow_certified and rho_jump_certified):
        conclusion = (
            f"PARTIAL CERTIFICATE: MCM = {mcm:.6f} < 1.0 but upstream "
            f"certificate chain is incomplete (flow_certified={flow_certified}, "
            f"rho_jump_certified={rho_jump_certified}). "
            f"Result is numerically suggestive but not formally rigorous."
        )
        cert_type = "PARTIAL"
    else:
        conclusion = (
            f"CERTIFICATE FAILED: MCM = {mcm:.6f} "
            f"({'< 1 (stable)' if stab else '>= 1 (unstable)'})."
        )
        cert_type = "FAILED"

    mcm_result.update({
        "flow_certified":       flow_certified,
        "rho_jump_certified":   rho_jump_certified,
        "full_chain_certified": full_chain,
        "conclusion":           conclusion,
        "certificate_type":     cert_type,
        "passed":               full_chain,
    })

    return mcm_result


# ---------------------------------------------------------------------------
# 2.  verify_theorem_walks  (secondary validation)
# ---------------------------------------------------------------------------

def verify_theorem_walks(walks: list, rho_total: dict,
                          walk_set_complete: bool = False) -> dict:
    """Walk-product validation check (NOT the primary certificate).

    Computes P(w) = ∏ rho_total(l) for each walk w and checks P(w) < 1.

    NOTE: This function is provided for VALIDATION only.  It cannot certify
    stability over infinite dropout sequences because the walk set is finite
    (truncated DFS).  Use verify_mcm_certificate() for the formal proof.

    Parameters
    ----------
    walks             : list of label-sequences
    rho_total         : dict {label → float}
    walk_set_complete : bool (True only if DFS fully exhausted the cost budget)

    Returns
    -------
    dict with keys:
        passed, max_product, worst_walk, stability_margin,
        product_distribution, n_walks_checked, n_walks_exceeding_1,
        walk_set_complete, conclusion, note
    """
    if not walks:
        return {
            "passed":               False,
            "max_product":          float("nan"),
            "worst_walk":           [],
            "stability_margin":     float("nan"),
            "product_distribution": np.array([], dtype=float),
            "n_walks_checked":      0,
            "n_walks_exceeding_1":  0,
            "walk_set_complete":    walk_set_complete,
            "conclusion":           "FAILED (no walks provided)",
            "note":                 "Validation only — use MCM for formal certificate",
        }

    products: list[float] = []
    max_product = -np.inf
    worst_walk: list = []

    for w in walks:
        p = 1.0
        missing = None
        for l in w:
            if l not in rho_total:
                missing = l
                break
            p *= float(rho_total[l])
        if missing is not None:
            return {
                "passed": False, "max_product": float("nan"),
                "worst_walk": list(w), "stability_margin": float("nan"),
                "product_distribution": np.array([], dtype=float),
                "n_walks_checked": len(walks), "n_walks_exceeding_1": 0,
                "walk_set_complete": walk_set_complete,
                "conclusion": f"FAILED (missing rho_total for l={missing})",
                "note": "Validation only",
            }
        products.append(p)
        if p > max_product:
            max_product = p
            worst_walk  = list(w)

    arr     = np.array(products, dtype=float)
    n_exceed = int(np.sum(arr > 1.0))
    margin  = float(1.0 - max_product)
    passed  = bool(max_product < 1.0)

    note = (
        "VALIDATION CHECK ONLY — this walk-product check covers "
        f"{len(walks)} finite walks from a truncated DFS. "
        "It cannot certify stability over infinite dropout sequences. "
        "The formal certificate is compute_max_cycle_mean() (MCM, exact)."
    )

    if passed:
        conclusion = (
            f"Validation passed: {len(walks)} walks checked, "
            f"max product = {max_product:.6f} < 1.0."
        )
    else:
        conclusion = (
            f"Validation FAILED: {n_exceed}/{len(walks)} walks exceed 1.0."
        )

    return {
        "passed":               passed,
        "max_product":          float(max_product),
        "worst_walk":           worst_walk,
        "stability_margin":     margin,
        "product_distribution": arr,
        "n_walks_checked":      len(walks),
        "n_walks_exceeding_1":  n_exceed,
        "walk_set_complete":    walk_set_complete,
        "conclusion":           conclusion,
        "note":                 note,
    }


# Backward-compatible alias
def verify_theorem(walks, growth_factors, walk_set_complete=False):
    """Alias for verify_theorem_walks (backward compatibility)."""
    rt = {l: v["rho_total"] for l, v in growth_factors.items()
          if "rho_total" in v}
    return verify_theorem_walks(walks, rt, walk_set_complete)


# ---------------------------------------------------------------------------
# 3.  save_report
# ---------------------------------------------------------------------------

def save_report(mcm_result: dict, walk_result: dict,
                params_info: dict, path: str) -> bool:
    """Write a human-readable certificate report to *path*.

    Returns True on success, False on any failure.
    """
    try:
        sep  = "=" * 70
        dash = "-" * 70
        lines: list[str] = []

        lines += [sep,
                  "  WHRT STABILITY CERTIFICATE REPORT  (publication version)",
                  f"  Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
                  sep, ""]

        # ── Parameters ───────────────────────────────────────────────────
        lines += ["PARAMETERS", dash]
        for k, v in params_info.items():
            lines.append(f"  {k:<30s}: {v}")
        lines.append("")

        # ── Config ───────────────────────────────────────────────────────
        lines += ["CONFIG", dash]
        for attr in ("M", "N", "H", "LAM", "RHO", "ALPHA",
                     "N_GRID_FINE", "N_LIP_GRID", "N_GRID_RJUMP", "SEED"):
            lines.append(f"  {attr:<30s}: {getattr(config, attr, 'N/A')}")
        lines.append("")

        # ── Primary certificate: MCM ──────────────────────────────────────
        lines += ["PRIMARY CERTIFICATE — Maximum Cycle Mean (Karp 1978)", dash]
        mcm = mcm_result.get("mcm", float("nan"))
        lines.append(f"  MCM                            : {mcm:.8f}")
        lines.append(f"  MCM < 1?                       : {mcm_result.get('stable','?')}")
        lines.append(f"  Stability margin (1-MCM)       : "
                     f"{mcm_result.get('stability_margin', float('nan')):.8f}")
        lines.append(f"  Certificate type               : "
                     f"{mcm_result.get('certificate_type','?')}")
        lines.append(f"  Flow condition certified       : "
                     f"{mcm_result.get('flow_certified','?')}")
        lines.append(f"  ρ_jump from reset map          : "
                     f"{mcm_result.get('rho_jump_certified','?')}")
        lines.append(f"  Full chain certified           : "
                     f"{mcm_result.get('full_chain_certified','?')}")
        lines.append(f"  Worst cycle labels             : "
                     f"{mcm_result.get('worst_cycle', [])}")
        lines.append(f"  Worst cycle geometric mean     : "
                     f"{mcm_result.get('worst_cycle_mean', float('nan')):.8f}")
        lines.append("")
        lines.append(f"  Conclusion: {mcm_result.get('conclusion','')}")
        lines.append("")

        # ── Secondary validation: walk products ───────────────────────────
        lines += ["SECONDARY VALIDATION — Walk Product Check (NOT certificate)", dash]
        lines.append(f"  NOTE: {walk_result.get('note','')}")
        lines.append(f"  Walks checked                  : "
                     f"{walk_result.get('n_walks_checked','?')}")
        lines.append(f"  Max walk product               : "
                     f"{walk_result.get('max_product', float('nan')):.8f}")
        lines.append(f"  Walks exceeding 1.0            : "
                     f"{walk_result.get('n_walks_exceeding_1','?')}")
        lines.append(f"  Walk validation passed         : "
                     f"{walk_result.get('passed','?')}")
        lines.append("")

        # ── Footer ───────────────────────────────────────────────────────
        lines += [sep, "END OF REPORT", sep]
        text = "\n".join(lines) + "\n"
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False
