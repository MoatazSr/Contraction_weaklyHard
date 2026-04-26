"""
main.py — WHRT Stability Pipeline (publication version).

Certificate chain
-----------------
Stage 1  ZOH Obstruction  — proves ZOH (α=0) is structurally obstructed
Stage 2  Co-design        — observer gain threshold α* = 2+2λ
Stage 3  Contraction Metric — CVXPY SOCP search + Lipschitz certificate [GATE 1]
Stage 4  Invariance       — forward invariance of domain Z              [GATE 2]
Stage 5  ρ_jump           — derived from reset map g(x,ê)=(x,0)        [GATE 3]
Stage 6  Growth Factors   — analytical Gronwall bound e^{-λlh}
Stage 7  MCM Certificate  — Karp exact MCM (necessary and sufficient)   [GATE 4]
Stage 8  h Sweep          — maximum certifiable h* via whrt_mcm.sweep
Stage 9  Comparison       — methodology vs combined improvement

Exit codes
----------
0  All gates passed — CERTIFIABLE (full chain rigorous)
1  Reporting stage issues (non-critical)
2  Stage 3 GATE: Lipschitz-certified flow violation not < 0
3  Stage 5 GATE: ρ_jump ≥ 1.0
4  Stage 7 GATE: MCM ≥ 1.0 (system not certified stable)
"""
import sys, threading, time, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import config

from src.zoh_obstruction  import check_zoh_obstruction
from src.observer_codesign import (codesign_threshold,
                                    verify_codesign_condition,
                                    alpha_feasibility_table)
from src.invariance        import forward_invariance_certificate
from src.whrt_mcm          import build_automaton, certify as whrt_certify, sweep as whrt_sweep
from src.contraction       import find_metric_sos, f_flow
from src.growth_factors    import compute_all_rho
from src.theorem_verify    import (verify_mcm_certificate,
                                   verify_theorem_walks, save_report)
from src.deviation_bound   import compute_deviation_bound
from src.monotonicity      import check_monotonicity
from src.comparison        import run_comparison
from src.whrt_graph        import validate_graph, enumerate_walks_sufficient


# ── ANSI colours ───────────────────────────────────────────────────────────────
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
def _sep():  print(_c("  " + "─"*64, _D))
def _info(k, v): print(f"    {_c(k+':',_B):<35s}  {v}")
def _ok(msg):    print(f"  {_c('✔',_G)}  {msg}")
def _fail(msg):  print(f"  {_c('✘',_R)}  {msg}")
def _warn(msg):  print(f"  {_c('⚠',_Y)}  {msg}")

def _header(stage, total, title):
    print()
    print(_c(f"  ╔══ STAGE {stage}/{total}: {title} ══", _B))

def _gate(exit_code: int, stage: int, reason: str):
    print()
    _fail(f"GATE FAILED at Stage {stage}: {reason}")
    print(_c(f"  Pipeline terminated. Exit code {exit_code}.", _R+_B))
    sys.exit(exit_code)


# ===========================================================================
print(_c("\n  WHRT Stability Pipeline — Publication Version", _B+_C))
print(_c("  " + "="*35, _D))
_info("Plant",    "ẋ = x²-x³+u,   u = -2x̂(t)")
_info("Observer", f"dx̂/dt = -α(x̂-x),   α = {config.ALPHA}")
_info("η",        f"({config.M},{config.N}),   h = {config.H}s,   λ = {config.LAM}")
_info("Domain Z", f"x∈{config.DOMAIN_X}, ê∈{config.DOMAIN_E}")

# Shared plant definition (used across stages)
def _F(z: np.ndarray) -> np.ndarray:
    """Observer closed-loop flow at config.ALPHA."""
    return f_flow(z)

def _g(z: np.ndarray) -> np.ndarray:
    """Reset map: g(x, ê) = (x, 0) at each successful reception."""
    return np.array([float(z[0]), 0.0])

def _f_scalar(x: float, e: float) -> float:
    """Plant function f(x,e) = x²-x³-2(x+e)."""
    return x**2 - x**3 - 2.0*(x + e)

Z_DOMAIN = (config.DOMAIN_X, config.DOMAIN_E)
ETA      = (config.M, config.N)

# ===========================================================================
# STAGE 1 — ZOH Obstruction
# ===========================================================================
N_STAGES = 9
_header(1, N_STAGES, "ZOH Contraction Obstruction (Proposition 1)")

zoh_res = check_zoh_obstruction(
    f=_f_scalar,
    domain=Z_DOMAIN,
    equilibrium=(0.0, 0.0),
    verbose=True,
)

_sep()
_info("ZOH obstructed",      zoh_res["obstructed"])
_info("Obstruction type",    zoh_res["obstruction_type"])
_info("Conserved quantity",  zoh_res["conserved_quantity"])
cl   = zoh_res["conservation_law"]
r1   = zoh_res["rank1_obstruction"]
_info("Conservation law",    f"confirmed={cl['confirmed']}, max|error|={cl['max_error']:.2e}")
_info("Rank-1: rᵀ(0)·c",    f"{r1['rTc_value']:.6f}  (0 → obstruction exact)")
_info("Rank-1 obstructed",   r1["obstructed"])

if zoh_res["obstructed"]:
    _ok("ZOH is provably contraction-obstructed (conservation law + rank-1 Jacobian)")
else:
    _warn("ZOH obstruction check inconclusive — conservation law not confirmed")

_log.append((1, "ZOH Obstruction", "PASS" if zoh_res["obstructed"] else "WARN",
             f"obstructed={zoh_res['obstructed']}"))

# ===========================================================================
# STAGE 2 — Observer Co-design Threshold
# ===========================================================================
_header(2, N_STAGES, "Observer Co-design Threshold (Proposition 2)")

alpha_star  = codesign_threshold(config.LAM)
codesign    = verify_codesign_condition(config.ALPHA, config.LAM)
alpha_table = alpha_feasibility_table(
    lam=config.LAM,
    alpha_list=[2.1, 2.5, 3.0, 4.0, 5.0],
    masp=0.5,
    eta_n=config.N,
    eta_m=config.M,
)

_sep()
_info("Co-design threshold α*",  f"{alpha_star:.4f}  (= 2 + 2λ)")
_info("Configured α",            f"{config.ALPHA}")
_info("Condition satisfied",     codesign["satisfied"])
_info("Margin α − α*",          f"{codesign['margin']:.4f}")
_info("J₂₂ at origin",          f"{codesign['J22_at_origin']:.4f}  (= 2 − α)")
_info("S₂₂ sign",               codesign["S22_sign"])

print()
print(f"    {_c('Co-design feasibility table:', _B)}")
print(f"    {'α':>5}  {'α > α*':>8}  {'Margin':>8}  {'Note'}")
for row in alpha_table:
    ok_str = "✔ YES" if row["satisfied"] else "✘ NO"
    print(f"    {row['alpha']:>5.1f}  {ok_str:>8}  {row['margin']:>+8.3f}  {row['note']}")

if codesign["satisfied"]:
    _ok(f"α = {config.ALPHA} > α* = {alpha_star:.4f} — co-design condition satisfied")
else:
    _warn(f"α = {config.ALPHA} does not satisfy co-design condition α* = {alpha_star:.4f}")

_log.append((2, "Co-design", "PASS" if codesign["satisfied"] else "WARN",
             f"alpha={config.ALPHA}, alpha_star={alpha_star:.4f}"))

# ===========================================================================
# STAGE 3 — Contraction Metric + Lipschitz Certificate  [GATE 1]
# ===========================================================================
_header(3, N_STAGES, "Contraction Metric + Lipschitz Certificate")

with Spinner("CVXPY SOCP metric search + Lipschitz certification"):
    metric = find_metric_sos(config.LAM, config.RHO, config.N_GRID_FINE)

_sep()
params   = metric["params"]
fc       = metric["flow_cert"]
lam_used = metric["lam_used"]

_info("Solver",               metric["solver_status"])
_info("Used fallback",        metric["used_fallback"])
_info("Contraction rate λ",   f"{lam_used:.4f}")
_info("Grid violation",       f"{fc['max_grid_val']:.2e}")
_info("Lipschitz const L",    f"{fc['lipschitz_const']:.2e}")
_info("Grid half-diagonal δ", f"{fc['grid_half_diag']:.2e}")
_info("Certified bound",      f"{fc['certified_bound']:.2e}")
_info("Flow CERTIFIED",       fc["certified"])

if fc["certified"]:
    _ok(f"Flow condition CERTIFIED: max_{{z∈Z}} λ_max(S(z)) ≤ {fc['certified_bound']:.2e} < 0")
else:
    _fail(f"Flow condition NOT certified (bound = {fc['certified_bound']:.2e} ≥ 0)")
    _gate(2, 3, f"Lipschitz-certified flow violation = {fc['certified_bound']:.2e} ≥ 0")

flow_certified = fc["certified"]
_log.append((3, "Contraction Metric", "PASS" if flow_certified else "FAIL",
             f"certified_bound={fc['certified_bound']:.2e}"))

# ===========================================================================
# STAGE 4 — Forward Invariance  [GATE 2 — gap flag allowed]
# ===========================================================================
_header(4, N_STAGES, "Forward Invariance Certificate")

l_max_inv = config.N - config.M + 1

with Spinner("Boundary flux + post-jump reachability check"):
    inv_res = forward_invariance_certificate(
        F=_F, g=_g, Z=Z_DOMAIN,
        l_max=l_max_inv, h=config.H,
        n_pts=200, n_x0=200, verbose=False,
    )

_sep()
bfc = inv_res["boundary_flux"]
pjr = inv_res["post_jump"]

_info("Strict box invariant",    bfc["strictly_invariant"])
_info("Boundary violations",     f"{bfc['n_violations']}  (worst flux = {bfc['worst_flux']:.4f})")
_info("Violated faces",          bfc["violated_faces"] or "none")
_info("Post-jump: inside Z",     pjr["inside_domain"])
_info("Max |ê| reachable",       f"{pjr['max_e_reachable']:.4f}  (e_hi={pjr['e_hi']})")
_info("Post-jump margin",        f"{pjr['margin']:.4f}")
_info("Certified",               inv_res["certified"])
_info("Gap flag",                inv_res["gap_flag"])

if inv_res["certified"] and inv_res["gap_flag"]:
    _ok("Domain Z certified for HYBRID invariance (post-jump reachability)")
    _warn(f"Gap flag: {inv_res['gap_note']}")
elif inv_res["certified"]:
    _ok("Domain Z strictly forward-invariant")
else:
    _fail(inv_res["conclusion"])
    _gate(3, 4, "Domain Z not forward-invariant — enlarge Z or reduce T_max")

_log.append((4, "Invariance", "PASS" if inv_res["certified"] else "FAIL",
             f"gap_flag={inv_res['gap_flag']}, max_e={pjr['max_e_reachable']:.4f}"))

# ===========================================================================
# STAGE 5 — ρ_jump from reset map  [GATE 3]
# ===========================================================================
_header(5, N_STAGES, "ρ_jump from Reset Map g(x,ê)=(x,0)")

rj_info  = metric["rho_jump_info"]
rho_jump = metric["rho_jump"]

_sep()
_info("Reset map",           "g(x,ê) = (x, 0)  at each reception")
_info("Variational map Dg",  "diag(1, 0)")
_info("ρ_jump formula",      "sup{(x,ê)∈Z} [p(x,0)/p(x,ê)]^½")
_info("Grid supremum",       f"{rj_info['rho_jump_grid']:.6f}")
_info("Analytical proof",    rj_info.get("analytical_proof", False))
_info("ρ_jump bound",        f"{rho_jump:.6f}")
if rj_info.get("proof"):
    print(f"    {_c('Proof:',_B)} {rj_info['proof'][:90]}")

rj_ok = rho_jump <= 1.0 and rj_info.get("certified", False)
if rj_ok:
    _ok(f"ρ_jump = {rho_jump:.6f} ≤ 1.0  (proved analytically)")
else:
    _fail(f"ρ_jump = {rho_jump:.6f} — analytical proof failed")
    _gate(3, 5, f"ρ_jump analytical proof failed")

_log.append((5, "ρ_jump", "PASS" if rj_ok else "FAIL",
             f"rho_jump={rho_jump:.6f}"))

# ===========================================================================
# STAGE 6 — Growth Factors (Gronwall bound)
# ===========================================================================
_header(6, N_STAGES, "Analytical Growth Factors  (Gronwall e^{-λlh})")

G_graph, graph_info = build_automaton(ETA)
val = validate_graph(G_graph, config.M, config.N)
max_l = val["max_label"]

walks_val, walk_complete = enumerate_walks_sufficient(
    G_graph, config.C_WALK, config.H, max_walks=10_000)
_info("WHRT automaton nodes",  val["n_nodes"])
_info("WHRT automaton edges",  val["n_edges"])
_info("Max label l",           max_l)
_info("Validation walks (DFS)", f"{len(walks_val):,} ({'complete' if walk_complete else 'truncated'})")
_warn("Walk DFS is validation only — MCM (Stage 7) is the formal certificate.")

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
    src  = "Gronwall" if e["rho_flow_source"] == "analytical_gronwall" else "MC only"
    print(f"    {l:>3}  {e['rho_flow']:>14.6f}  {e['rho_flow_mc']:>12.6f}  "
          f"{e['rho_jump']:>8.6f}  {e['rho_total']:>9.6f}  {flag}    {src}")
    if "consistency_warning" in e:
        _warn(f"  l={l}: " + e["consistency_warning"])

_info("Source",           "Gronwall analytic (certified)" if flow_certified else "MC estimate")
_info("All ρ_total < 1",  all_stable)

if not all_stable:
    _gate(4, 6, "ρ_total ≥ 1.0 for at least one label")
_ok("All ρ_total < 1.0")
_log.append((6, "Growth Factors", "PASS" if all_stable else "FAIL",
             f"max={max(gf[l]['rho_total'] for l in gf):.6f}"))

# ===========================================================================
# STAGE 7 — MCM Certificate (Karp 1978)  [GATE 4]
# ===========================================================================
_header(7, N_STAGES, "MCM Certificate — Karp's Algorithm (exact, N&S)")

rho_total_map = {l: gf[l]["rho_total"] for l in gf}

with Spinner("Computing exact Maximum Cycle Mean via Karp (1978)"):
    mcm_result = verify_mcm_certificate(G_graph, rho_total_map, fc, rj_info)

_sep()
mcm    = mcm_result["mcm"]
stable = mcm_result["stable"]
margin = mcm_result["stability_margin"]

_info("MCM",                    f"{mcm:.8f}")
_info("MCM < 1? (stability)",   stable)
_info("Stability margin 1-MCM", f"{margin:.8f}")
_info("Worst cycle labels",     mcm_result["worst_cycle"])
_info("Worst cycle geom. mean", f"{mcm_result['worst_cycle_mean']:.8f}")
_info("Flow certified",         mcm_result["flow_certified"])
_info("ρ_jump certified",       mcm_result["rho_jump_certified"])
_info("Full chain certified",   mcm_result["full_chain_certified"])
_sep()
print(f"    {_c('Conclusion:', _B)}")
print(f"    {mcm_result['conclusion']}")

if not stable:
    _gate(4, 7, f"MCM = {mcm:.6f} ≥ 1.0")
_ok(f"MCM = {mcm:.6f} < 1.0  (necessary and sufficient)")

# Cross-check via walk products
rt_map   = {l: gf[l]["rho_total"] for l in gf}
walk_val = verify_theorem_walks(walks_val, rt_map, walk_complete)
_info("Walk validation (cross-check)", walk_val["conclusion"])
if not walk_val["passed"]:
    _warn("Walk validation failed — investigate (validation only, not certificate)")

_log.append((7, "MCM Certificate", "PASS" if stable else "FAIL",
             f"MCM={mcm:.6f}, margin={margin:.6f}"))

# ===========================================================================
# STAGE 8 — h Sweep (maximum certifiable h*)
# ===========================================================================
_header(8, N_STAGES, "Maximum Certifiable h* via WHRT+MCM Sweep")

print(f"  Sweeping h ∈ [0.05, 0.40] with λ={lam_used:.4f}, ρ_jump={rho_jump:.4f}, η={ETA}")
with Spinner("WHRT MCM sweep over h"):
    sweep_res = whrt_sweep(
        lam=lam_used,
        eta=ETA,
        rho_jump=rho_jump,
        h_lo=0.05, h_hi=0.40, n_points=36, verbose=False,
    )

_sep()
h_cert  = sweep_res["h_certified"]
mcm_cert = sweep_res["mcm_at_h_cert"]

_info("h certified (config)",   f"{config.H:.4f} s")
_info("h* (sweep max, Gronwall)", f"{h_cert:.4f} s  (MCM = {mcm_cert:.6f})" if h_cert else "N/A")
_info("n stable in sweep",      f"{sweep_res['n_stable']}/{sweep_res['n_total']}")

# Show stability boundary
h_grid   = sweep_res["h_grid"]
mcm_grid = sweep_res["mcm_grid"]
mask     = sweep_res["stable_mask"]
print()
print(f"    {_c('Stability boundary (transition from MCM<1 to ≥1):', _B)}")
for i, (h, m, ok) in enumerate(zip(h_grid, mcm_grid, mask)):
    if i > 0 and mask[i-1] != ok:
        prev_h = float(h_grid[i-1])
        print(f"    MCM crosses 1 between h={prev_h:.4f}s (MCM={float(mcm_grid[i-1]):.4f}) "
              f"and h={float(h):.4f}s (MCM={float(m):.4f})")
        break

if h_cert:
    _ok(sweep_res["conclusion"])
else:
    _warn(sweep_res["conclusion"])

_log.append((8, "h Sweep", "PASS" if sweep_res["certified"] else "WARN",
             f"h_cert={h_cert}"))

# ===========================================================================
# STAGE 9 — Deviation Bound + Monotonicity + Comparison
# ===========================================================================
_header(9, N_STAGES, "Deviation Bound, Monotonicity, Comparison")

db   = compute_deviation_bound(params, gf, config)
mono = check_monotonicity(params, gf, config)

_sep()
_info("Half-life",              f"{db.get('half_life', float('nan')):.4f} s")
_info("e-folding time",         f"{db.get('e_folding_time', float('nan')):.4f} s")
_info("Bound at t=SIM_TIME",    f"{db.get('horizon_bound', float('nan')):.2e}")
_info("ρ_flow non-increasing",  mono["flow_nondecreasing"])
_info("Sub-multiplicativity",   mono["submultiplicative"])

# Comparison (new signature: compute Lipschitz baseline from plant)
print()
with Spinner("Computing methodology comparison (Lipschitz baseline)"):
    comp = run_comparison(
        h_ours=config.H,
        F=_F,
        Z=Z_DOMAIN,
        eta=ETA,
        masp=0.5,
        mcm=mcm,
        n_pts_lip=25,
    )

_sep()
print(f"\n    {_c('Comparison Table:', _B)}")
for row in comp["summary_table"]:
    print(f"    {row[0]:<45s}  {row[1]:<12s}  {row[2]:<10s}  {row[3]:<12s}")

_sep()
meth_cmp = comp["method_comparison"]
comb_cmp = comp["combined_comparison"]
_info("Method improvement",     meth_cmp["conclusion"])
_info("Combined improvement",   comb_cmp["conclusion"])
print()
print(f"    {_c('Mandatory caveat:', _B)}")
for line in comp["mandatory_caveat"].split(". "):
    if line.strip():
        print(f"    {line.strip()}.")

if comp["passed"]:
    _ok(f"Comparison: method ratio = {comp['method_ratio']:.2f}× "
        f"(same arch), combined = {comp['combined_ratio']:.2f}× (joint)")
else:
    _warn("Comparison: h_ours does not exceed emulation bound — check λ or h")

# 2D extension
print()
try:
    from src.example_2d import run_2d_example
    with Spinner("2D plant SDP metric search + MCM"):
        ex2d = run_2d_example(h=config.H, verbose=False)
    if ex2d.get("certified"):
        _ok(f"2D linear CERTIFIED: lam={ex2d['lam']:.4f}, MCM={ex2d['mcm']:.6f}")
        nl_cert = ex2d.get("nl_certified", False)
        if nl_cert:
            _ok(f"2D nonlinear CERTIFIED (lam={ex2d.get('lam_nl', '?'):.4f})")
        else:
            _warn("2D nonlinear not certified at tested λ values")
    else:
        _warn(f"2D example not certified: {ex2d.get('error','')}")
    _log.append((9, "2D Example",
                 "PASS" if ex2d.get("certified") else "WARN",
                 f"mcm={ex2d.get('mcm', float('nan')):.4f}"))
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
    "S1 ZOH obstruction proved":     zoh_res["obstructed"],
    "S2 Co-design condition met":     codesign["satisfied"],
    "S3 Flow condition certified":    fc["certified"],
    "S4 Invariance (hybrid)":         inv_res["certified"],
    "S5 ρ_jump ≤ 1 (analytical)":    rj_ok,
    "S7 MCM < 1 (N&S certificate)":  stable,
    "S7 Full chain certified":        mcm_result["full_chain_certified"],
}

for name, passed_s in cert_stages.items():
    tag = _c("✔ PASS", _G) if passed_s else _c("✘ FAIL", _R)
    print(f"  ║  {tag}  {name}")

if inv_res.get("gap_flag"):
    print(f"  ║  {_c('⚠ GAP', _Y)}  S4 invariance: hybrid only (box not strictly invariant)")

print(_c("  ╠" + "═"*64 + "╣", _B))
fully_certified = all(cert_stages.values())
if fully_certified:
    verdict = _c("  ║   SYSTEM: CERTIFIABLY STABLE (complete certificate chain)  ", _G+_B)
else:
    verdict = _c("  ║   SYSTEM: NOT FULLY CERTIFIED (see failed stages above)    ", _R+_B)
print(verdict)
print(_c("  ╚" + "═"*64 + "╝", _B))

print()
print(f"    {'MCM':<35s}: {mcm:.8f}  (must be < 1)")
print(f"    {'Stability margin':<35s}: {margin:.8f}")
print(f"    {'Half-life':<35s}: {db.get('half_life',float('nan')):.4f} s")
print(f"    {'e-folding time':<35s}: {db.get('e_folding_time',float('nan')):.4f} s")
print(f"    {'Certified h (config)':<35s}: {config.H} s")
print(f"    {'h* (Gronwall sweep)':<35s}: {h_cert:.4f} s" if h_cert else
      f"    {'h* (sweep)':<35s}: N/A")
print(f"    {'Method improvement (same arch)':<35s}: {comp['method_ratio']:.2f}×")
print(f"    {'Combined improvement [JOINT]':<35s}: {comp['combined_ratio']:.2f}×  (arch+method)")
print(f"    {'Certificate chain':<35s}: {'COMPLETE' if fully_certified else 'INCOMPLETE'}")
if inv_res.get("gap_flag"):
    print(f"    {'Pending gap':<35s}: box invariance (hybrid cert valid)")

# Save report
from pathlib import Path
Path("results").mkdir(exist_ok=True)
save_report(
    mcm_result  = mcm_result,
    walk_result = walk_val,
    params_info = {
        "eta":                f"({config.M},{config.N})",
        "h":                  config.H,
        "alpha":              config.ALPHA,
        "alpha_star":         alpha_star,
        "lambda_certified":   lam_used,
        "rho_jump_certified": rho_jump,
        "MCM":                mcm,
        "stability_margin":   margin,
        "flow_certified":     fc["certified"],
        "certified_bound":    fc["certified_bound"],
        "h_star_sweep":       h_cert,
        "method_ratio":       f"{comp['method_ratio']:.2f}x (same arch)",
        "combined_ratio":     f"{comp['combined_ratio']:.2f}x [JOINT: arch+method]",
        "invariance_gap":     inv_res["gap_flag"],
        "certificate_chain":  "COMPLETE" if fully_certified else "INCOMPLETE",
    },
    path="results/certificate_report.txt"
)
print(f"\n    Report saved: results/certificate_report.txt")

sys.exit(0 if fully_certified else 1)
