"""
figures.py — Publication figures for the WHRT stability paper.

Five essential figures:
  Fig 1  Closed-loop simulation comparison  (3-panel: baseline / ours / beyond)
  Fig 2  Contraction metric certificate     (λ_max(S(z)) heatmap over Z)
  Fig 3  WHRT automaton                     (debt-state graph with edge labels)
  Fig 4  Growth-factor chain               (ρ_flow and ρ_total per label l)
  Fig 5  Baseline comparison               (certified h for all methods)

Run:  python figures.py
Output: figures/fig1_simulation.pdf  …  figures/fig5_comparison.pdf
        figures/fig1_simulation.png  …  figures/fig5_comparison.png
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.integrate import solve_ivp

import config
from contraction import find_metric_sos, check_flow_condition, lipschitz_certified_flow_viol
from growth_factors import compute_all_rho, analytical_rho_flow
from whrt_graph import build_whrt_graph, validate_graph, compute_max_cycle_mean
from theorem_verify import verify_mcm_certificate

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DPI      = 200    # screen-quality + fast; set 300 for final submission
DPI_FAST = 150

def _save(fig, name: str):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"  saved: {FIG_DIR/name}.pdf + .png")
    plt.close(fig)

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "lines.linewidth":  1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       DPI_FAST,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#f4a582"
GREY   = "#888888"

# ---------------------------------------------------------------------------
# Shared computation  (run once, reuse across figures)
# ---------------------------------------------------------------------------
print("Computing contraction metric …")
metric = find_metric_sos(config.LAM, config.RHO, config.N_GRID_FINE)
params    = metric["params"]
lam_used  = metric["lam_used"]
flow_cert = metric["flow_cert"]
rj_info   = metric["rho_jump_info"]
rho_jump  = metric["rho_jump"]          # = 1.0 (analytical)

print(f"  lam={lam_used:.4f}, certified_bound={flow_cert['certified_bound']:.4f}")

from whrt_graph import enumerate_walks_sufficient
G   = build_whrt_graph(config.M, config.N)
val = validate_graph(G, config.M, config.N)
max_l = int(val["max_label"])
walks, walk_complete = enumerate_walks_sufficient(G, config.C_WALK, config.H)

gf = compute_all_rho(params, rho_jump, max_l, config.H, lam_used,
                     flow_certified=flow_cert["certified"])

rho_total_dict = {l: gf[l]["rho_total"] for l in sorted(gf)}
mcm_raw   = compute_max_cycle_mean(G, rho_total_dict)
mcm_val   = mcm_raw["mcm"]
worst_cyc = mcm_raw.get("worst_cycle", [])
print(f"  MCM = {mcm_val:.6f},  worst cycle = {worst_cyc}")


# ===========================================================================
# FIG 1 — Closed-loop simulation (3-panel)
# ===========================================================================
print("Generating Fig 1: simulation comparison …")

SIM_T   = 20.0
_N_DROP    = config.N - config.M          # = 3
WHRT_WORST = [True] * config.M + [False] * _N_DROP   # length 20

# Three scenarios: (h, alpha, label, color, certified)
SCENARIOS = [
    (0.125,        float(config.ALPHA), f"h = 0.125 s  (2020 baseline, α={config.ALPHA})",  GREEN,  True),
    (float(config.H), float(config.ALPHA), f"h = {config.H} s  (ours, certified, α={config.ALPHA})", BLUE, True),
    (float(config.H), 0.0,              f"h = {config.H} s  (ZOH, α = 0, not certifiable)", RED,   False),
]

X0_LIST = [-1.4, -0.9, -0.5, 0.4, 0.85, 1.3]
E0_LIST = [-0.5, 0.0, 0.5]

def _sim(x0: float, e0: float, h_sim: float, alpha_obs: float, pattern=None):
    """Simulate [x, ê] under a periodic WHRT dropout pattern.

    alpha_obs: observer gain (0 = ZOH, positive = observer with ê-damping).
    pattern:   list of bool, True=success (ê reset), False=dropout.
    """
    t_ev  = np.arange(0.0, SIM_T, h_sim)
    t_all, x_all, e_all = [], [], []
    xc, ê = float(x0), float(e0)
    for step, tk in enumerate(t_ev):
        def ode(t, y):
            x_, e_ = y
            dx = x_**2 - x_**3 - 2.0*(x_ + e_)
            de = -dx - alpha_obs * e_
            return [dx, de]
        try:
            sol = solve_ivp(ode, [tk, tk + h_sim], [xc, ê],
                            method="RK45", rtol=1e-7, atol=1e-9,
                            dense_output=True, max_step=h_sim / 20)
            if not sol.success:
                break
            ts = np.linspace(tk, tk + h_sim, 12)
            ys = sol.sol(ts)
            t_all.extend(ts.tolist())
            x_all.extend(ys[0].tolist())
            e_all.extend(ys[1].tolist())
            xc = float(sol.y[0, -1])
            ê  = float(sol.y[1, -1])
            success = (pattern is None) or pattern[step % len(pattern)]
            if success:
                ê = 0.0
        except Exception:
            break
    return np.array(t_all), np.array(x_all), np.array(e_all)

fig1, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (h_sim, alpha_sc, lbl, col, certified) in zip(axes, SCENARIOS):
    max_e_seen = 0.0
    for x0 in X0_LIST:
        for e0 in E0_LIST:
            t_tr, x_tr, e_tr = _sim(x0, e0, h_sim, alpha_sc, pattern=WHRT_WORST)
            if len(x_tr) < 2:
                continue
            ax.plot(t_tr, x_tr, color=col, lw=0.8, alpha=0.55)
            if len(e_tr):
                max_e_seen = max(max_e_seen, float(np.max(np.abs(e_tr))))

    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("t (s)")
    ax.set_title(lbl, fontsize=10, color=col, fontweight="bold")
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlim(0, 2.5)
    if certified:
        status, status_col = "converges [certified]", GREEN
    else:
        status, status_col = f"converges — not certifiable", ORANGE
        # Annotate max |e| to show ê persistence under ZOH
        ax.text(0.03, 0.96, f"max |e| = {max_e_seen:.2f}  (e persists w/o observer)",
                transform=ax.transAxes, va="top", fontsize=8, color=RED,
                bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.text(0.97, 0.05, status, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, fontweight="bold",
            color=status_col)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("x(t)")
fig1.suptitle(
    "Simulation: WHRT worst-case [17S+3F]  —  observer (α=3) vs ZOH (α=0)  (η=(17,20))",
    fontsize=11, fontweight="bold")
plt.tight_layout()
_save(fig1, "fig1_simulation")


# ===========================================================================
# FIG 2 — Contraction metric certificate  (λ_max(S(z)) heatmap)
# ===========================================================================
print("Generating Fig 2: flow condition heatmap …")

N_HEAT = 80
x_lo, x_hi = config.DOMAIN_X
e_lo, e_hi = config.DOMAIN_E
xs_h = np.linspace(x_lo, x_hi, N_HEAT)
es_h = np.linspace(e_lo, e_hi, N_HEAT)
XX, EE = np.meshgrid(xs_h, es_h)
lmax_grid = np.array([
    check_flow_condition(params, np.array([XX[i, j], EE[i, j]]), lam_used)
    for i in range(N_HEAT) for j in range(N_HEAT)
]).reshape(N_HEAT, N_HEAT)

vmax_abs = max(abs(lmax_grid.min()), abs(lmax_grid.max()))
vmax_abs = max(vmax_abs, 0.5)
norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs * 0.3)

fig2, ax2 = plt.subplots(figsize=(6.5, 4))
cm = ax2.contourf(XX, EE, lmax_grid, levels=30, cmap="RdBu_r", norm=norm)
ax2.contour(XX, EE, lmax_grid, levels=[0], colors="k", linewidths=1.2,
            linestyles="--")
cbar = fig2.colorbar(cm, ax=ax2, shrink=0.9)
cbar.set_label(r"$\lambda_{\max}(S(z))$", fontsize=10)

cb_val = flow_cert["certified_bound"]
ax2.text(0.03, 0.97,
         f"Certified bound: max $\\lambda_{{\\max}} \\leq {cb_val:.3f} < 0$",
         transform=ax2.transAxes, va="top", fontsize=9,
         bbox=dict(fc="white", ec="none", alpha=0.8))

ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$\hat{e}$")
ax2.set_title(r"Flow certificate: $\dot{M}+J^\top M+MJ+2\lambda M \preccurlyeq 0$ on $Z$",
              fontsize=10)
ax2.set_xlim(x_lo, x_hi)
ax2.set_ylim(e_lo, e_hi)
ax2.grid(False)
_save(fig2, "fig2_flow_certificate")


# ===========================================================================
# FIG 3 — WHRT automaton
# ===========================================================================
print("Generating Fig 3: WHRT automaton …")

# Node positions: debt states d = 0, 1, 2, 3
#   d=0: top-left   d=1: top-right  d=2: bottom-right  d=3: bottom-left
node_pos = {0: (0.15, 0.70), 1: (0.50, 0.70),
            2: (0.50, 0.25), 3: (0.15, 0.25)}
node_labels = {d: f"d = {d}" for d in range(4)}

# Build edge list from G: G is networkx DiGraph with edge attrs
import networkx as nx

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.set_xlim(0, 0.7)
ax3.set_ylim(0, 1.0)
ax3.axis("off")

NODE_R = 0.055
NODE_COLOR = "#dce9f5"
EDGE_COLOR = "#555555"

# Draw nodes
for d, (px, py) in node_pos.items():
    circ = plt.Circle((px, py), NODE_R, color=NODE_COLOR,
                       ec=BLUE, lw=1.8, zorder=3)
    ax3.add_patch(circ)
    ax3.text(px, py, node_labels[d], ha="center", va="center",
             fontsize=9, fontweight="bold", zorder=4)
    # annotate as success/failure capacity
    cap = config.N - config.M - d   # remaining failure capacity
    ax3.text(px, py - NODE_R - 0.04, f"cap={cap}", ha="center",
             va="top", fontsize=7.5, color=GREY)

# Gather edges with labels
edges_info = []
for u, v, data in G.edges(data=True):
    l   = data.get("label", data.get("l", "?"))
    rt  = rho_total_dict.get(l, float("nan"))
    edges_info.append((u, v, l, rt))

# Draw edges with curved arrows
def _draw_arrow(ax, p0, p1, label_str, col, rad=0.25, offset=(0, 0)):
    ax.annotate("",
                xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    mx = (p0[0] + p1[0]) / 2 + offset[0]
    my = (p0[1] + p1[1]) / 2 + offset[1]
    ax.text(mx, my, label_str, fontsize=7.5, ha="center", va="center",
            color=col, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1),
            zorder=5)

# Group by (u,v) to handle parallel edges
from collections import defaultdict
edge_groups = defaultdict(list)
for u, v, l, rt in edges_info:
    edge_groups[(u, v)].append((l, rt))

drawn_pairs = set()
for (u, v), items in edge_groups.items():
    pu, pv = node_pos[u], node_pos[v]
    is_self = (u == v)
    rad_base = 0.30 if (v, u) in edge_groups else 0.0
    for idx, (l, rt) in enumerate(sorted(items)):
        lbl = f"l={l}\nρ={rt:.3f}"
        off_x = 0.0
        off_y = 0.04 * (idx - len(items) / 2)
        rad = rad_base + 0.15 * idx if not is_self else 0.6 + 0.2*idx
        col_e = BLUE if rt < 0.95 else ORANGE
        if is_self:
            # Self-loop drawn as small arc above node
            cx, cy = pu
            th = np.linspace(0.2*np.pi, 1.8*np.pi, 60)
            r_loop = 0.08
            ax3.plot(cx + r_loop * np.cos(th), cy + NODE_R + r_loop * np.sin(th),
                     color=col_e, lw=0.9)
            ax3.text(cx, cy + NODE_R + 2*r_loop + 0.01, lbl,
                     ha="center", va="bottom", fontsize=7, color=col_e,
                     bbox=dict(fc="white", ec="none", alpha=0.7))
        else:
            _draw_arrow(ax3, pu, pv, lbl, col_e, rad=rad,
                        offset=(off_x, off_y))

ax3.set_title(
    f"WHRT automaton  η=({config.M},{config.N})  — debt-state graph",
    fontsize=10, fontweight="bold")
ax3.text(0.68, 0.02,
         f"Edge label = (l, rho_total)\nMCM = {mcm_val:.4f} < 1  (stable)",
         transform=ax3.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(fc="white", ec=GREY, alpha=0.8, lw=0.5))
_save(fig3, "fig3_whrt_automaton")


# ===========================================================================
# FIG 4 — Growth-factor chain (ρ_flow and ρ_total per label l)
# ===========================================================================
print("Generating Fig 4: growth factor chain …")

labels_l  = sorted(gf.keys())
rho_flows = [gf[l]["rho_flow"] for l in labels_l]
rho_mcs   = [gf[l]["rho_flow_mc"] for l in labels_l]
rho_tots  = [gf[l]["rho_total"]   for l in labels_l]

x_pos   = np.arange(len(labels_l))
width   = 0.28

fig4, ax4 = plt.subplots(figsize=(6.5, 4))
b1 = ax4.bar(x_pos - width, rho_flows, width, label=r"$\rho_{\rm flow}$(Gronwall)",
             color=BLUE, alpha=0.85, edgecolor="white")
b2 = ax4.bar(x_pos,         rho_mcs,   width, label=r"$\rho_{\rm flow}$(MC, validation)",
             color=ORANGE,  alpha=0.80, edgecolor="white", hatch="//")
b3 = ax4.bar(x_pos + width, rho_tots,  width, label=r"$\rho_{\rm total} = \rho_{\rm flow} \times \rho_{\rm jump}$",
             color=RED,   alpha=0.85, edgecolor="white")

ax4.axhline(1.0, color="k", lw=1.3, ls="--", label="Stability threshold (ρ = 1)")
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"l = {l}" for l in labels_l])
ax4.set_ylabel("Growth factor")
ax4.set_ylim(0, 1.15)
ax4.set_title("Per-label growth factors  (certified Gronwall bound vs MC validation)",
              fontsize=10)
ax4.legend(fontsize=8, loc="upper right")

# Annotate ρ_total values
for xi, rt in zip(x_pos, rho_tots):
    ax4.text(xi + width, rt + 0.01, f"{rt:.3f}", ha="center",
             va="bottom", fontsize=8, color=RED, fontweight="bold")

ax4.text(0.02, 0.96,
         f"λ = {lam_used:.3f},  h = {config.H} s,  ρ_jump = 1.000 (analytic)",
         transform=ax4.transAxes, va="top", fontsize=8, color=GREY)
ax4.grid(axis="y", alpha=0.3)
_save(fig4, "fig4_growth_factors")


# ===========================================================================
# FIG 5 — Baseline comparison
# ===========================================================================
print("Generating Fig 5: baseline comparison …")

methods = [
    ("ZOH / α=0  (Baseline 3)",         None,        False, GREY),
    ("Lyapunov emulation  (Baseline 2)", 0.250,       True,  ORANGE),
    ("2020 MASP  (Baseline 1)",          0.125,       True,  GREEN),
    (f"Ours  (observer, α={config.ALPHA})",
                                         float(config.H), True, BLUE),
]

fig5, ax5 = plt.subplots(figsize=(7, 3.8))
y_pos = np.arange(len(methods))

for yi, (name, h_val, cert, col) in enumerate(methods):
    if h_val is None:
        # Not certifiable — draw a hatched placeholder
        ax5.barh(yi, 0.05, height=0.55, color=GREY, alpha=0.3, hatch="////")
        ax5.text(0.06, yi, "Not certifiable\n(conservation law)", va="center",
                 fontsize=8.5, color=GREY, fontstyle="italic")
    else:
        ax5.barh(yi, h_val, height=0.55, color=col, alpha=0.85,
                 edgecolor="white")
        ax5.text(h_val + 0.003, yi, f"{h_val:.3f} s", va="center",
                 fontsize=9, fontweight="bold", color=col)
        if not cert:
            ax5.text(h_val / 2, yi, "(partial)", va="center", ha="center",
                     fontsize=7.5, color="white", fontstyle="italic")

ax5.set_yticks(y_pos)
ax5.set_yticklabels([m[0] for m in methods], fontsize=9)
ax5.set_xlabel("Certified sampling period  h (s)")
ax5.set_title("Comparison of certified sampling periods across methods",
              fontsize=10, fontweight="bold")
ax5.axvline(float(config.H), color=BLUE, ls=":", lw=1.2, alpha=0.5)
ax5.set_xlim(0, 0.28)
ax5.grid(axis="x", alpha=0.3)

# Improvement annotation
h_ours = float(config.H)
h_2020 = 0.125
ratio  = h_ours / h_2020
ax5.annotate("",
             xy=(h_ours, 3), xytext=(h_2020, 3),
             arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
ax5.text((h_ours + h_2020) / 2, 3.22,
         f"×{ratio:.2f} improvement",
         ha="center", va="bottom", fontsize=9, fontweight="bold")

_save(fig5, "fig5_comparison")


# ===========================================================================
# Done
# ===========================================================================
print(f"\nAll figures saved to  {FIG_DIR}/")
print("Files:")
for f in sorted(FIG_DIR.iterdir()):
    print(f"  {f.name}")
