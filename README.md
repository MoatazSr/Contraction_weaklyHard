# Contraction-Based Stability under Weakly Hard Real-Time Dropout

**Moataz Seghyar** — LASTIMI Laboratory, EST Salé, Mohammed V University in Rabat  
PhD research companion code · `paper.tex` + `design_rationale.tex`

---

## 1. The Problem

A nonlinear control system executes over a network subject to packet dropout.
The dropout sequence is not arbitrary: it satisfies a **weakly hard real-time
(WHRT)** constraint `η = (M, N)`, meaning every window of `N` consecutive
transmission attempts contains at least `M` successes. This is a worst-case,
non-probabilistic model that captures real scheduling guarantees.

The question is: **given a WHRT dropout constraint and a nonlinear plant, can
we certify that the closed-loop system is exponentially stable — and if so, for
how large a sampling period h?**

This project answers that question via contraction theory.

---

## 2. System Model

### 2.1 Hybrid dynamics

The closed-loop system is a hybrid system alternating between continuous flow
and discrete resets:

```
Flow  (between receptions, duration l·h):
    ż(t)  =  F(z(t))

Jump  (at each successful reception t_k):
    z(t_k⁺)  =  g(z(t_k))

Dropout (no reset):
    z(t_k⁺)  =  z(t_k)
```

The state `z ∈ Z ⊂ ℝⁿ` lives in a compact operating domain. The gap
between two consecutive receptions is `l ∈ {1, …, l_max}` sampling intervals,
where `l_max = N − M + 1`.

### 2.2 WHRT dropout constraint

A sequence `(s_k) ∈ {0,1}^ω` satisfies `η = (M, N)` if

```
∑_{i=k}^{k+N−1} s_i  ≥  M     for all k ≥ 0.
```

This is encoded as a finite directed automaton `G = (V, E)` whose states track
the current failure debt within the sliding window. Each edge carries a label
`l` (inter-reception gap) and a weight `ρ(l)` (growth factor for that interval).

For `η = (17, 20)`: `|V| = 4` debt states, `l_max = 4`, at most 3 consecutive
dropouts.

---

## 3. Assumptions

**A1 (Smoothness).** `F : Z → ℝⁿ` and `g : Z → Z` are continuously
differentiable on the compact domain `Z`.

**A2 (Domain).** `Z` is forward-invariant under the hybrid dynamics: every
trajectory starting in `Z` remains in `Z`. *(In the scalar example, verified
via post-jump reachability analysis; see §6.)*

**A3 (WHRT constraint).** The reception sequence `(s_k)` satisfies
`η = (M, N)` with automaton `G` as defined in §2.2.

**A4 (Metric feasibility).** There exists a Riemannian metric `M(z) ≻ 0` on `Z`
satisfying the flow contraction condition (A5) and the jump non-expansion
condition (A6).

**A5 (Flow).** The metric `M(z)` satisfies

```
S(z)  :=  Ṁ(z) + ∂F(z)ᵀ M(z) + M(z) ∂F(z) + 2λ M(z)  ≼  0     on Z
```

for some rate `λ > 0`. This is the contraction condition: the metric-weighted
distance between any two trajectories decays at rate `λ` during the flow phase.

**A6 (Jump).** The variational map `Dg(z)` satisfies

```
ρ_jump  :=  sup_{z ∈ Z}  ‖Dg(z)‖_{M}  ≤  1.
```

Resets do not expand the metric. *(Sufficient condition: `g(z)ᵀ M(g(z)) g(z) ≤ zᵀ M(z) z`
for all `z ∈ Z`.)*

---

## 4. Main Results

### Theorem — Contraction under WHRT dropout

**Statement.** Under Assumptions A1–A6, define the per-label growth factor

```
ρ_total(l)  :=  ρ_flow(l) · ρ_jump,     ρ_flow(l) ≤ e^{−λlh}   (Gronwall)
```

and the maximum cycle mean of the WHRT automaton

```
MCM(G, ρ_total)  :=  max_{cycle C in G}  ( ∏_{e ∈ C} ρ_total(l(e)) )^{1/|C|}.
```

If `MCM(G, ρ_total) < 1`, then the hybrid system is **exponentially stable** on Z
for every admissible `η`-sequence: there exist `C, μ > 0` such that

```
‖z(t)‖_{M}  ≤  C · e^{−μt} · ‖z(0)‖_{M}     for all z(0) ∈ Z,  t ≥ 0.
```

**Proof structure.**
1. *Flow phase:* A5 + Gronwall → `‖δz(lh)‖_M ≤ e^{−λlh} ‖δz(0)‖_M`.
2. *Jump phase:* A6 → metric does not expand at each reception.
3. *Composition over one WHRT cycle:* the metric decreases by
   `∏_{e ∈ C} ρ_total(l(e))` for each cycle `C`.
4. *MCM < 1* (Karp 1978, necessary and sufficient) → every cycle in `G` has
   geometric mean less than 1 → metric decreases geometrically across cycles
   → exponential stability.

**Why MCM, not `max_l ρ_total(l) < 1`.**
The per-label condition is sufficient but not necessary. MCM captures
cycle-averaging: a cycle alternating a long gap (large `l`, small `ρ`) with
several short gaps (small `l`, large `ρ`) can be stable even if individual
large-`l` edges have `ρ` close to 1. MCM is the exact necessary-and-sufficient
condition; using it rather than the per-edge bound is what makes the certified
`h` tight.

---

### Proposition — Grid-to-domain flow certificate

**Statement.** Let `L` be a Lipschitz constant of `λ_max(S(·))` on Z, and
let `δ` be the half-diagonal of a uniform grid `G_N ⊂ Z` with `N` points per
axis. If

```
max_{z ∈ G_N}  λ_max(S(z))  +  L · δ  <  0,
```

then A5 holds on all of Z.

**Role.** The flow condition A5 is enforced over a finite grid by SOCP; this
proposition extends the certificate to the full compact domain with a computable
correction `L·δ`.

---

## 5. Certificate Chain

The four verification stages that instantiate the theorem:

| Stage | Verifies | Method | Passes if |
|---|---|---|---|
| 1 | WHRT automaton `G` is correctly constructed | Graph algorithm | All valid `(l, debt)` transitions encoded |
| 2 | Flow condition A5 holds on Z | SOCP on grid + Lipschitz bound | `grid_max + L·δ < 0` |
| 3 | Jump condition A6 holds | Analytical or numerical | `ρ_jump ≤ 1` |
| 4 | `MCM(G, ρ_total) < 1` | Karp's algorithm (exact) | `MCM < 1` |

All four gates must pass. The certificate is rigorous if stages 2–4 use
verified bounds; numerical if Monte Carlo is substituted.

---

## 6. Instantiation: Scalar Nonlinear Plant, `η = (17, 20)`

This codebase instantiates the framework on a specific benchmark plant.

**Plant and state.**
`ż = F(z)` where `z = (x, ê) ∈ ℝ²`, `x` is the plant state,
`ê` is the estimation error, and `F` is determined by the closed-loop dynamics
on `Z = [−2, 2] × [−1.1, 1.1]`.

**Metric.** Diagonal polynomial `M(z) = diag(p(z), q(z))` with softplus-encoded
positive coefficients, found by SOCP.

**Jump map.** `g(x, ê) = (x, 0)` — reception resets the estimation error to zero.
`ρ_jump = 1` analytically (softplus ensures `p(x, ê) ≥ p(x, 0)` for all `ê`).

**Domain invariance.** `Z` is not forward-invariant as a box. The correct
argument uses the hybrid structure: the reset `ê → 0` defines a post-jump set
from which reachable trajectories are confined to `|ê| ≤ 1.078` over the
worst-case inter-reception duration `l_max · h = 0.780 s`. The domain is
enlarged to `|ê| ≤ 1.1` to contain this with margin 0.092.

**Numerical result.**

```
λ = 0.15,   h = 0.195 s,   η = (17, 20),   α = 3.0

Flow certificate:   −1.450 + 0.017  =  −1.433  <  0     ✓
ρ_jump          :   1.000  ≤  1                          ✓
max_l ρ_total   :   0.9712  <  1                         ✓
MCM             :   0.97117  <  1   (margin 0.02883)     ✓

Certified stable.   Half-life 4.62 s.
```

---

## 7. 2D Extension (`src/example_2d.py`)

The same framework applied to a 2D coupled linear plant, demonstrating that
the certificate chain scales beyond the scalar case.

**Plant.** `ż = F(z)` where `z = (x₁, x₂, ê₁, ê₂) ∈ ℝ⁴`,
`A = [[-1, 0.5], [-0.5, -2]]`, same `η = (17, 20)`, same `h`.

**Metric.** Constant block-diagonal `M = blkdiag(M_xx, M_ee)` found by SDP.
For a linear plant, the Jacobian is constant and a constant metric is exact
(no Lipschitz correction needed).

**Nonlinear extension.** Adding `g(x) = [0.05 x₁³, 0]ᵀ` makes the Jacobian
state-dependent. The gridded LMI enforces the flow condition at 30 sampled
Jacobian values rather than using a Lipschitz correction (which fails when M
is ill-conditioned).

| Variant | λ | MCM |
|---|---|---|
| Linear, block-diagonal M (4 SDP vars) | 0.40 | 0.925 |
| Linear, full 4×4 M (10 SDP vars) | 0.45 | — |
| Nonlinear, gridded LMI, block-diagonal M | 0.25 | 0.952 |

---

## 8. Repository Structure

```
project_pub/
├── main.py                     # Full 9-stage certificate pipeline (run this)
├── config.py                   # Parameters: M=17, N=20, h=0.195, λ=0.15
├── paper.tex                   # Publication draft
├── design_rationale.tex        # Companion: reasoning behind every decision
├── figures.py                  # All figures
├── src/
│   ├── whrt_graph.py           # WHRT automaton + Karp MCM
│   ├── contraction.py          # SOCP metric search
│   ├── growth_factors.py       # Gronwall bounds + Monte Carlo validation
│   ├── theorem_verify.py       # Certificate chain verifier
│   ├── comparison.py           # Baseline comparison
│   ├── theoretical_analysis.py # Forward invariance, co-design analysis
│   └── example_2d.py           # 2D extension (SDP + gridded LMI)
└── results/
    └── certificate_report.txt
```

```bash
python main.py            # Full pipeline, ~60 s
python src/example_2d.py  # 2D example only
python figures.py         # Regenerate figures
```

**Dependencies:** `numpy`, `scipy`, `cvxpy` (SCS), `matplotlib`.

---

## 9. Open Problems

| Gap | Status |
|---|---|
| Forward invariance: formal sublevel-set proof | Numerical argument only |
| Proposition: full Schur complement proof of flow feasibility condition | Proof sketch in `paper.tex` |
| General plant class (beyond scalar benchmark) | Not yet stated |
| Incremental stability corollary | Follows from contraction; not yet written |
| Differentiation from Hertneck et al. 2020 (same ×1.56, different method and baseline) | Explicit comparison pending |
