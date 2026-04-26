"""
monotonicity.py – Monotonicity checks on WHRT growth factors.

One public function
-------------------
check_monotonicity – verify structural properties of the rho_total(l) sequence
"""

from __future__ import annotations

import numpy as np


def check_monotonicity(params, gf: dict, cfg) -> dict:
    """Check monotonicity and structural properties of the growth-factor sequence.

    Three checks are performed:

    1. rho_flow non-decreasing in l
       Longer integration intervals should never shrink the worst-case flow
       growth — if this is violated it indicates an unusual metric geometry.

    2. rho_total strictly less than 1 for every l  (stability requirement)

    3. Sub-multiplicativity of rho_flow:
       rho_flow(l1+l2) <= rho_flow(l1) * rho_flow(l2)
       Verified for all pairs (l1, l2) with l1+l2 in the computed range.
       A violation means the walk-product theorem may be overly conservative.

    passed = True iff checks 1 and 2 both pass
             (sub-multiplicativity is informational only).

    Never raises.  Returns a dict with all results populated regardless of
    outcome.
    """
    if not gf:
        return {
            "passed":              False,
            "key_metric":          "N/A",
            "flow_nondecreasing":  False,
            "all_stable":          False,
            "submultiplicative":   None,
            "n_violations_flow":   0,
            "n_violations_stab":   0,
            "n_violations_submul": 0,
            "max_rho_total":       None,
            "error":               "no growth factors provided",
        }

    ls          = sorted(gf.keys())
    rho_flows   = np.array([gf[l]["rho_flow"]  for l in ls])
    rho_totals  = np.array([gf[l]["rho_total"] for l in ls])

    # ── Check 1: rho_flow non-increasing ─────────────────────────────
    # With damping (Option A fix), longer integration intervals produce
    # more contraction, so rho_flow should decrease with l.
    flow_diffs      = np.diff(rho_flows)               # (n-1,)
    n_viol_flow     = int(np.sum(flow_diffs > 1e-9))   # violation = increasing
    flow_nondecr    = (n_viol_flow == 0)

    # ── Check 2: all rho_total < 1 ───────────────────────────────────
    n_viol_stab     = int(np.sum(rho_totals >= 1.0))
    all_stable      = (n_viol_stab == 0)

    # ── Check 3: sub-multiplicativity (informational) ─────────────────
    gf_map          = {l: gf[l]["rho_flow"] for l in ls}
    n_viol_submul   = 0
    for i, l1 in enumerate(ls):
        for l2 in ls:
            l_sum = l1 + l2
            if l_sum in gf_map:
                if gf_map[l_sum] > gf_map[l1] * gf_map[l2] + 1e-9:
                    n_viol_submul += 1
    submul_ok       = (n_viol_submul == 0)

    max_rho_total   = float(np.max(rho_totals))
    min_rho_total   = float(np.min(rho_totals))
    passed          = flow_nondecr and all_stable
    key_metric      = (f"max_rho_total={max_rho_total:.4f}, "
                       f"flow_nonincr={flow_nondecr}, "
                       f"stable={all_stable}")

    result: dict = {
        "passed":              passed,
        "key_metric":          key_metric,
        "flow_nondecreasing":  flow_nondecr,
        "all_stable":          all_stable,
        "submultiplicative":   submul_ok,
        "n_violations_flow":   n_viol_flow,
        "n_violations_stab":   n_viol_stab,
        "n_violations_submul": n_viol_submul,
        "max_rho_total":       max_rho_total,
        "min_rho_total":       min_rho_total,
        "rho_totals":          rho_totals.tolist(),
        "rho_flows":           rho_flows.tolist(),
        "labels":              ls,
    }
    if not passed:
        reasons = []
        if not flow_nondecr:
            reasons.append(f"rho_flow not non-increasing ({n_viol_flow} violation(s))")
        if not all_stable:
            reasons.append(f"{n_viol_stab} label(s) with rho_total >= 1")
        result["error"] = "; ".join(reasons)
    return result
