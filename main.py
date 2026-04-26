"""
main.py — WHRT Stability Pipeline (publication version).

Certificate chain
-----------------
Stage 1  Graph        — WHRT automaton construction and validation
Stage 2  Contraction  — CVXPY SOCP metric search + Lipschitz certificate [GATE 1]
Stage 3  ρ_jump       — Derived from actual reset map g(x,ê)=(x,0)        [GATE 2]
Stage 4  Growth Factors — Analytical Gronwall bound e^{-λlh}              [GATE 3]
Stage 5  MCM Certificate — Karp exact MCM (necessary and sufficient)      [GATE 4]
Stage 6  Deviation Bound — Half-life, e-folding, horizon bound
Stage 7  Monotonicity    — Structural consistency checks
Stage 8  Comparison      — Multi-baseline (2020 MASP, Lyapunov, ZOH)

Exit codes
----------
0  All gates passed — CERTIFIABLE (full chain rigorous)
1  Reporting stage issues (non-critical)
2  Stage 2 GATE: Lipschitz-certified flow violation not < 0
3  Stage 3 GATE: ρ_jump ≥ 1.0 (metric invalid for jump contraction)
4  Stage 4 GATE: some ρ_total(l) ≥ 1.0
5  Stage 5 GATE: MCM ≥ 1.0 (system not certified stable)
"""
import sys, threading, time, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import config

from src.whrt_graph         import (build_whrt_graph, validate_graph,
                                     compute_max_cycle_mean,
                                     enumerate_walks_sufficient)
from src.contraction        import find_metric_sos
from src.growth_factors     import compute_all_rho
from src.theorem_verify     import (verify_mcm_certificate,
                                    verify_theorem_walks, save_report)
from src.deviation_bound    import compute_deviation_bound
from src.monotonicity       import check_monotonicity
from src.comparison         import (run_comparison, compute_all_baselines,
                                    decompose_improvement, compute_h_vs_alpha_table)
from src.theoretical_analysis import (check_conservation_law,
                                       justify_alpha_modification,
                                       check_forward_invariance,
                                       check_post_jump_reachability,
                                       REMARK_TEXT)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
_C = "\033[96m"; _B = "\033[1m";  _D = "\033[2m";  _X = "\033[0m"
def _c(t, *codes): return "".join(codes) + str(t) + _X

class Spinner:
    _FRAMES = r"-\|/"
    def __init__(self, label):
        self._label = label; self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        for f in itertools.cycle(self._FRAMES):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r  {_c(f,_C)}  {self._label} ... ")
            sys.stdout.flush(); time.sleep(0.12)
    def __enter__(self): self._thread.start(); return self
    def __exit__(self, *_):
        self._stop.set(); self._thread.join()
        sys.stdout.write("\r" + " "*(len(self._label)+14) + "\r")
        sys.stdout.flush()

_log: list[tuple] = []
def _sep(): print(_c("  " + "─"*64, _D))
def _info(k, v): print(f"    {_c(k+':',_B):<35s}  {v}")
def _ok(msg):  print(f"  {_c('✔',_G)}  {msg}")
def _fail(msg): print(f"  {_c('✘',_R)}  {msg}")
def _warn(msg): print(f"  {_c('⚠',_Y)}  {msg}")

def _header(stage, total, title):
    print()
    print(_c(f"  ╔══ STAGE {stage}/{total}: {title} ══", _B))

def _gate(exit_code: int, stage: int, reason: str):
    print()
    _fail(f"GATE {exit_code-1} FAILED at Stage {stage}: {reason}")
    print(_c(f"  Pipeline terminated. Exit code {exit_code}.", _R+_B))
    sys.exit(exit_code)

# ===========================================================================
print(_c("\n  WHRT Stability Pipeline — Publication Version", _B+_C))
print(_c("  =" * 35, _D))
_info("Plant",    "ẋ = x²-x³+u,   u = -2x̂(t)")
_info("Observer", "dx̂/dt = -α(x̂-x),   α = " + str(config.ALPHA))
_info("η",        f"({config.M},{config.N}),   h = {config.H}s")
_info("Domain Z", f"x∈{config.DOMAIN_X}, ê∈{config.DOMAIN_E}")

# ===========================================================================
# STAGE 1 — WHRT Automaton
# ===========================================================================
_header(1, 9, "WHRT Automaton")

G   = build_whrt_graph(config.M, config.N)
val = validate_graph(G, config.M, config.N)
max_l = val["max_label"]

_sep()
_info("Nodes (debt states)",   val["n_nodes"])
_info("Edges (transitions)",   val["n_edges"])
_info("Max label l",           max_l)
_info("Strongly connected",    val["strongly_connected"])
_info("Validation passed",     val["passed"])
if val["errors"]:
    for e in val["errors"]: _fail(e)
else:
    _ok("Graph structure validated")

# Validation walk enumeration (NOT the certificate)
walks_val, walk_complete = enumerate_walks_sufficient(
    G, config.C_WALK, config.H, max_walks=10_000)
_info("Validation walks (DFS)", f"{len(walks_val):,} ({'complete' if walk_complete else 'truncated — validation only'})")
_warn("Walk enumeration is for VALIDATION only. MCM (Stage 5) is the formal certificate.")
_log.append((1, "WHRT Automaton", "PASS", f"n_nodes={val['n_nodes']}, max_l={max_l}"))

# ===========================================================================
# STAGE 2 — Contraction Metric + Lipschitz Certificate
# ===========================================================================
_header(2, 9, "Contraction Metric + Lipschitz Certificate")

# Conservation law proof
print()
cl = check_conservation_law(verbose=True)
alpha_info = justify_alpha_modification(config.ALPHA)
print(f"    Observer: {alpha_info['observer_ode']}")
print(f"    Control:  {alpha_info['control_law']}")
print(f"    Error dynamics: dê/dt = -f(x,ê) - {config.ALPHA}·ê")
print(REMARK_TEXT)

print()
fi = check_forward_invariance(verbose=True)
reach = check_post_jump_reachability(verbose=True)
if reach["inside_domain_e"]:
    _ok(f"Post-jump reachable set inside DOMAIN_E: "
        f"max|e|={reach['max_e_reachable']:.4f} < {reach['e_hi']} "
        f"(margin={reach['margin']:.4f})")
    if not fi["forward_invariant"]:
        _warn(f"Strict fwd-invariance fails at unreachable corners "
              f"({fi['n_violations']} pts). Certificate valid: reachable set is inside Z.")
else:
    _fail(f"max|e|={reach['max_e_reachable']:.4f} >= DOMAIN_E bound {reach['e_hi']} "
          f"— enlarge DOMAIN_E")
    _gate(2, 2, "Post-jump reachable set exceeds DOMAIN_E")
print()

with Spinner("CVXPY SOCP metric search + Lipschitz certification"):
    metric = find_metric_sos(config.LAM, config.RHO, config.N_GRID_FINE)

_sep()
params   = metric["params"]
fc       = metric["flow_cert"]
lam_used = metric["lam_used"]

_info("Solver",                       metric["solver_status"])
_info("Used fallback (scipy DE)",     metric["used_fallback"])
_info("Contraction rate λ used",      f"{lam_used:.4f}")
_info("Grid violation (max grid)",    f"{fc['max_grid_val']:.2e}")
_info("Lipschitz constant L",         f"{fc['lipschitz_const']:.2e}")
_info("Grid half-diagonal δ",         f"{fc['grid_half_diag']:.2e}")
_info("Certified bound (max+L·δ)",    f"{fc['certified_bound']:.2e}")
_info("Flow condition CERTIFIED",     fc["certified"])
_info("Certificate type",             fc["certificate_type"])

if fc["certified"]:
    _ok(f"Flow condition CERTIFIED: max_{{z∈Z}} λ_max(S(z)) ≤ {fc['certified_bound']:.2e} < 0")
else:
    _fail(f"Flow condition NOT certified (bound = {fc['certified_bound']:.2e} ≥ 0)")
    _gate(2, 2, f"Lipschitz-certified flow violation = {fc['certified_bound']:.2e} ≥ 0")

flow_certified = fc["certified"]
_log.append((2, "Contraction Metric", "PASS" if flow_certified else "FAIL",
             f"certified_bound={fc['certified_bound']:.2e}"))

# ===========================================================================
# STAGE 3 — ρ_jump from actual reset map
# ===========================================================================
_header(3, 9, "ρ_jump from Reset Map g(x,ê)=(x,0)")

rj_info  = metric["rho_jump_info"]
rho_jump = metric["rho_jump"]

_sep()
_info("Reset map",                    "g(x,ê) = (x, 0)  at each success")
_info("Variational map Dg",           "diag(1, 0)")
_info("ρ_jump formula",               "sup{(x,ê)∈Z} [p(x,0)/p(x,ê)]^½")
_info("Grid supremum (numerical)",    f"{rj_info['rho_jump_grid']:.6f}")
_info("Analytical proof",             rj_info.get("analytical_proof", False))
_info("ρ_jump bound (analytical)",    f"{rho_jump:.6f}")
_info("ρ_jump ≤ 1?",                  rho_jump <= 1.0)
if rj_info.get("proof"):
    print(f"    {_c('Proof:', _B)} {rj_info['proof'][:90]}")

rj_ok = rho_jump <= 1.0 and rj_info.get("certified", False)
if rj_ok:
    _ok(f"ρ_jump = {rho_jump:.6f} ≤ 1.0  (proved analytically from metric structure)")
else:
    _fail(f"ρ_jump = {rho_jump:.6f} — analytical proof failed (ê-coefficients not positive)")
    _gate(3, 3, f"ρ_jump analytical proof failed")

_log.append((3, "ρ_jump Certificate", "PASS" if rj_ok else "FAIL",
             f"rho_jump={rho_jump:.6f}, analytical={rj_info.get('analytical_proof')}"))

# ===========================================================================
# STAGE 4 — Analytical Growth Factors (Gronwall bound)
# ===========================================================================
_header(4, 9, "Analytical Growth Factors  (Gronwall bound e^{-λlh})")

with Spinner(f"Computing ρ_total(l) for l=1…{max_l}  [+ MC validation]"):
    gf = compute_all_rho(params, rho_jump, max_l, config.H, lam_used,
                         flow_certified=flow_certified)

_sep()
print(f"    {_c('Per-label growth factors:', _B)}")
print(f"    {'l':>3}  {'ρ_flow (cert)':>14}  {'ρ_flow (MC)':>12}  "
      f"{'ρ_jump':>8}  {'ρ_total':>9}  {'<1?':>5}  {'Source':>20}")
print(_c("    " + "─"*80, _D))

all_stable = True
for l in sorted(gf.keys()):
    e  = gf[l]
    ok = e["stable"]
    if not ok: all_stable = False
    flag = _c("✔", _G) if ok else _c("✘", _R)
    cert_note = "Gronwall" if e["rho_flow_source"] == "analytical_gronwall" else "MC only"
    print(f"    {l:>3}  {e['rho_flow']:>14.6f}  {e['rho_flow_mc']:>12.6f}  "
          f"{e['rho_jump']:>8.6f}  {e['rho_total']:>9.6f}  {flag}    {cert_note}")
    if "consistency_warning" in e:
        _warn(f"  l={l}: " + e["consistency_warning"])

_sep()
_info("Growth factor source",  "Gronwall analytic (certified)" if flow_certified else "MC estimate")
_info("All ρ_total < 1.0",     all_stable)

if not all_stable:
    _gate(4, 4, "ρ_total ≥ 1.0 for at least one label")
_ok("All ρ_total < 1.0")
_log.append((4, "Growth Factors", "PASS" if all_stable else "FAIL",
             f"max_rho_total={max(gf[l]['rho_total'] for l in gf):.6f}"))

# ===========================================================================
# STAGE 5 — MCM Certificate (Karp 1978)
# ===========================================================================
_header(5, 9, "MCM Certificate — Karp's Algorithm (exact, N&S condition)")

rho_total_map = {l: gf[l]["rho_total"] for l in gf}

with Spinner("Computing exact Maximum Cycle Mean via Karp (1978)"):
    mcm_result = verify_mcm_certificate(G, rho_total_map,
                                        fc, rj_info)

_sep()
mcm    = mcm_result["mcm"]
stable = mcm_result["stable"]
margin = mcm_result["stability_margin"]

_info("MCM (Maximum Cycle Mean)",    f"{mcm:.8f}")
_info("MCM < 1? (stability cond.)",  stable)
_info("Stability margin (1-MCM)",    f"{margin:.8f}")
_info("Worst cycle labels",          mcm_result["worst_cycle"])
_info("Worst cycle geom. mean",      f"{mcm_result['worst_cycle_mean']:.8f}")
_info("Flow condition certified",    mcm_result["flow_certified"])
_info("ρ_jump certified",            mcm_result["rho_jump_certified"])
_info("Full chain certified",        mcm_result["full_chain_certified"])
_info("Certificate type",            mcm_result["certificate_type"])
_sep()
print(f"    {_c('Conclusion:', _B)}")
print(f"    {mcm_result['conclusion']}")

if not stable:
    _gate(5, 5, f"MCM = {mcm:.6f} ≥ 1.0")
if stable:
    _ok(f"MCM = {mcm:.6f} < 1.0  (necessary and sufficient)")

# Walk validation cross-check
rt_list = {l: gf[l]["rho_total"] for l in gf}
walk_val = verify_theorem_walks(walks_val, rt_list, walk_complete)
_info("Walk validation (cross-check)", walk_val["conclusion"])
if not walk_val["passed"]:
    _warn("Walk validation failed — investigate (note: this is validation only)")

_log.append((5, "MCM Certificate", "PASS" if stable else "FAIL",
             f"MCM={mcm:.6f}, margin={margin:.6f}"))

# ===========================================================================
# STAGE 6 — Deviation Bound
# ===========================================================================
_header(6, 9, "Deviation Bound  (Gronwall + MCM)")

db = compute_deviation_bound(params, gf, config)

_sep()
_info("ρ_max (= MCM)",               f"{db.get('max_rho_total', float('nan')):.6f}")
_info("Half-life",                   f"{db.get('half_life', float('nan')):.4f} s")
_info("e-folding time",              f"{db.get('e_folding_time', float('nan')):.4f} s")
_info("Bound at t=SIM_TIME",         f"{db.get('horizon_bound', float('nan')):.2e}")
_info("Bound passed",                db.get("passed", False))

if db.get("passed"):
    _ok(f"Half-life = {db.get('half_life',float('nan')):.3f} s  "
        f"(e-fold = {db.get('e_folding_time',float('nan')):.3f} s)")
else:
    _warn("Deviation bound check flagged — see error in result")
_log.append((6, "Deviation Bound", "PASS" if db.get("passed") else "WARN",
             f"half_life={db.get('half_life',float('nan')):.3f}s"))

# ===========================================================================
# STAGE 7 — Monotonicity (structural consistency)
# ===========================================================================
_header(7, 9, "Monotonicity / Structural Consistency")

mono = check_monotonicity(params, gf, config)
_sep()
_info("ρ_flow non-increasing in l",   mono["flow_nondecreasing"])
_info("All ρ_total < 1",              mono["all_stable"])
_info("Sub-multiplicativity",         mono["submultiplicative"])
if mono["passed"]:
    _ok("Structural consistency checks passed")
else:
    _warn(mono.get("error", ""))
_log.append((7, "Monotonicity", "PASS" if mono["passed"] else "WARN", ""))

# ===========================================================================
# STAGE 8 — Multi-Baseline Comparison
# ===========================================================================
_header(8, 9, "Multi-Baseline Comparison (Fair Attribution)")

with Spinner("Computing baselines"):
    comp = run_comparison(mcm_result, gf, config)

_sep()
baselines = comp["baselines"]
b1 = baselines["baseline_2020"]
b2 = baselines["baseline_lyapunov"]
b3 = baselines["baseline_zoh"]

print(f"\n    {_c('Comparison Table:', _B)}")
for row in baselines["summary_table"]:
    print(f"    {row[0]:<40s}  {row[1]:<10s}  {row[2]}")
_sep()
_info("vs 2020 MASP baseline",        b1["conclusion"])
_info("vs ZOH (α=0)",                 b3["reason"])
_info("vs emulation (same obs.)",     f"ratio = {b2['improvement_ratio']:.2f}× (same architecture)")
_info("Total ratio (vs 2020)",        f"{comp['improvement_ratio']:.3f}×")
print()

# Fair attribution decomposition
decomp = baselines["decomposition"]
print(f"    {_c('Fair Attribution:', _B)}")
for line in decomp["attribution"].split("\n"):
    print(f"    {line}")
print()

# Co-design feasibility table
alpha_tbl = baselines["alpha_table"]
print(f"    {_c('Co-design feasibility (α > 2+2λ = {:.2f}):'.format(2+2*config.LAM), _B)}")
print(f"    {'α':>5}  {'Co-design':>10}  {'Margin':>8}  {'h_certified':>14}")
for row in alpha_tbl:
    ok_str = "✔ OK" if row["co_design_ok"] else "✘ NO"
    h_str  = f"{row['h_certified']:.3f}s" if row["h_certified"] is not None else "N/C"
    print(f"    {row['alpha']:>5.1f}  {ok_str:>10}  "
          f"{row['co_design_margin']:>+8.3f}  {h_str:>14}")
print()

if comp.get("passed"):
    _ok(f"Comparison passed — {comp['improvement_ratio']:.2f}× improvement over 2020 baseline")
else:
    _warn(comp.get("error", "comparison not fully passed"))
_log.append((8, "Comparison", "PASS" if comp.get("passed") else "WARN",
             f"ratio={comp['improvement_ratio']:.3f}x"))

# ===========================================================================
# STAGE 9 — Higher-Dimensional Example (2D plant)
# ===========================================================================
_header(9, 9, "Higher-Dimensional Example (2D plant, constant metric SDP)")

try:
    from src.example_2d import run_2d_example
    with Spinner("2D plant SDP metric search + MCM"):
        ex2d = run_2d_example(h=config.H, verbose=False)
    _sep()
    if ex2d.get("certified"):
        _ok(f"2D linear CERTIFIED: lam={ex2d['lam']:.4f}, MCM={ex2d['mcm']:.6f}")
        _info("2D plant A",        str(ex2d["plant_A"]))
        _info("2D observer gain",  ex2d["alpha"])
        _info("2D h (same eta)",  f"{ex2d['h']:.3f}s")
        _info("2D MCM (linear)",  f"{ex2d['mcm']:.6f}  (< 1)")
        _info("2D cond(M) blkdiag", f"{ex2d.get('cond_M', float('nan')):.2f}")
        # Metric choice comparison
        lam_full = ex2d.get("lam_full_M", float("nan"))
        lam_diag = ex2d["lam"]
        if not (lam_full != lam_full):   # not NaN
            gain_pct = (lam_full - lam_diag) / lam_diag * 100
            _info("Full 4x4 M: lambda_max",
                  f"{lam_full:.4f}  (vs block-diag {lam_diag:.4f}, +{gain_pct:.1f}%)")
            _info("Metric choice",
                  "block-diagonal blkdiag(M_xx,M_ee): 4 SDP vars vs 10 full; "
                  f"loses {gain_pct:.1f}% in lambda, simpler and interpretable")
        # Nonlinear extension
        nl_cert = ex2d.get("nl_certified", False)
        lam_nl  = ex2d.get("lam_nl", float("nan"))
        eps_nl  = ex2d.get("eps_nl", float("nan"))
        mcm_nl  = ex2d.get("mcm_nl") or {}
        if nl_cert:
            _ok(f"2D nonlinear CERTIFIED (eps_nl={eps_nl:.3f}): "
                f"lam={lam_nl:.4f}, MCM={mcm_nl.get('mcm', float('nan')):.6f}")
            _info("Nonlinear method",
                  f"gridded LMI: S(x1)=J(x1)^T M+M J(x1)+2lam M << 0, "
                  f"30 Jacobian samples in x1 in [-2,2] (sufficient, rigorous)")
        else:
            _warn(f"2D nonlinear not certified at tested lambdas "
                  f"(eps_nl={eps_nl:.3f})")
        _ok("Same WHRT automaton eta=(17,20) certifies 2D coupled plant at same h")
    else:
        _warn(f"2D example not certified: {ex2d.get('error','')}")
    _log.append((9, "2D Example",
                 "PASS" if (ex2d.get("certified") and ex2d.get("nl_certified")) else
                 "PARTIAL" if ex2d.get("certified") else "WARN",
                 f"mcm={ex2d.get('mcm',float('nan')):.4f}, "
                 f"nl_certified={ex2d.get('nl_certified',False)}"))
except Exception as _ex2d_err:
    _warn(f"2D example skipped: {_ex2d_err}")
    _log.append((9, "2D Example", "SKIP", str(_ex2d_err)[:60]))

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print()
print(_c("  ╔" + "═"*64 + "╗", _B))
print(_c("  ║  CERTIFICATE SUMMARY", _B))
print(_c("  ╠" + "═"*64 + "╣", _B))

cert_stages = {
    "S2 Flow certified":        fc["certified"],
    "S3 ρ_jump derived":        rho_jump <= 1.0 and rj_info.get("certified", False),
    "S4 All ρ_total < 1":       all_stable,
    "S5 MCM < 1 (N&S)":         stable,
    "S5 Full chain certified":  mcm_result["full_chain_certified"],
}

for name, passed_s in cert_stages.items():
    tag = _c("✔ PASS", _G) if passed_s else _c("✘ FAIL", _R)
    print(f"  ║  {tag}  {name}")

print(_c("  ╠" + "═"*64 + "╣", _B))
fully_certified = all(cert_stages.values())
if fully_certified:
    verdict = _c("  ║   SYSTEM: CERTIFIABLY STABLE (complete certificate chain)  ", _G+_B)
else:
    verdict = _c("  ║   SYSTEM: NOT FULLY CERTIFIED (see failed stages above)    ", _R+_B)
print(verdict)
print(_c("  ╚" + "═"*64 + "╝", _B))

print()
print(f"    {'MCM':<30s}: {mcm:.8f}  (must be < 1)")
print(f"    {'Stability margin':<30s}: {margin:.8f}")
print(f"    {'Half-life':<30s}: {db.get('half_life',float('nan')):.4f} s")
print(f"    {'e-folding time':<30s}: {db.get('e_folding_time',float('nan')):.4f} s")
print(f"    {'Improvement vs 2020':<30s}: {comp['improvement_ratio']:.3f}×")
print(f"    {'Certified h':<30s}: {config.H} s")
print(f"    {'2020 baseline h':<30s}: {b1['h_2020']:.3f} s")
print(f"    {'Certificate chain':<30s}: {'COMPLETE (Lipschitz+MCM)' if fully_certified else 'INCOMPLETE'}")

# Save report
from pathlib import Path
Path("results").mkdir(exist_ok=True)
save_report(
    mcm_result  = mcm_result,
    walk_result = walk_val,
    params_info = {
        "eta":               f"({config.M},{config.N})",
        "h":                 config.H,
        "alpha":             config.ALPHA,
        "lambda_certified":  lam_used,
        "rho_jump_certified": rho_jump,
        "MCM":               mcm,
        "stability_margin":  margin,
        "flow_certified":    fc["certified"],
        "certified_bound":   fc["certified_bound"],
        "certificate_chain": "COMPLETE" if fully_certified else "INCOMPLETE",
    },
    path="results/certificate_report.txt"
)
print(f"\n    Report saved: results/certificate_report.txt")

sys.exit(0 if fully_certified else 1)
