# Contraction-Based Stability of a Dual-Sensor NCS under WHRT Dropout

**Moataz Seghyar** — LASTIMI Laboratory, EST Salé, Mohammed V University in Rabat  
PhD research companion code · `paper.tex` + `design_rationale.tex`

---

## What this is

A complete, executable certificate chain proving exponential stability of a nonlinear
networked control system (NCS) subject to the `η=(17,20)` weakly hard real-time (WHRT)
dropout constraint. The certified sampling period is **h = 0.195 s**, a 56 % improvement
over the 2020 MASP baseline (h = 0.125 s).

The code is the ground truth. Every number in the paper is produced by `main.py`.

---

## Core theoretical result

### Why ZOH cannot be certified — for *any* metric

The standard zero-order-hold (ZOH) formulation holds `x̂` constant between samples,
giving error dynamics `ê̇ = -f(x, ê)`. The augmented Jacobian is:

```
J_ZOH = [[ ∂f/∂x,   ∂f/∂ê ],   =  c · rᵀ   (rank 1)
          [-∂f/∂x,  -∂f/∂ê ]]
```

where `c = (1, -1)ᵀ` and `r = (∂f/∂x, ∂f/∂ê)ᵀ`. This rank-1 structure is a
**universal obstruction**: for *any* positive definite metric M (diagonal, full,
state-dependent), the contraction matrix satisfies

```
λ_max(JᵀM + MJ)  =  cᵀr + ‖M^{1/2}c‖ ‖M^{-1/2}r‖  ≥  0
```

by Cauchy-Schwarz (`ĉᵀr̂ = cᵀr` is M-invariant; `‖ĉ‖‖r̂‖ ≥ |cᵀr|`). Adding the
rate term `2λM` makes `S = JᵀM + MJ + 2λM` strictly indefinite for all `λ > 0`.
**No metric, however general, can certify contraction under ZOH.**

The obstruction is not a metric-design failure. It is the conservation law
`d(x+ê)/dt = 0`: the ZOH estimate `x̂ = x + ê` is constant between samples,
foliation the state space into invariant lines that no contraction can cross.

### The observer fix

Replacing ZOH with a continuous observer `ẋ̂ = -α(x̂ - x_local)` modifies the
error dynamics to `ê̇ = -αê - f(x, ê)`. The Jacobian becomes:

```
J = [[ ∂f/∂x,   ∂f/∂ê ],   —  full rank for α > 0
     [-∂f/∂x,   2 - α  ]]
```

The (2,2) entry `2 - α` is no longer `-∂f/∂ê`. The rank-1 structure is broken,
the conservation law is broken (`d(x+ê)/dt = -αê`), and contraction becomes
achievable. The explicit co-design condition for a diagonal polynomial metric with
rate λ is:

```
α  >  2 + 2λ
```

With λ = 0.15 the threshold is α > 2.30; we use α = 3.0 (margin 0.70).

---

## Certificate chain

| Stage | What it certifies | Method | Key result |
|---|---|---|---|
| 1 | WHRT automaton valid | Graph construction | 4 debt states, l_max = 4 |
| 2 | Flow contracts at rate λ | SOCP + Lipschitz bound | −1.435 < 0 on Z |
| 3 | Jump does not expand metric | Analytical (softplus) | ρ_jump = 1.000 |
| 4 | Per-label Gronwall bounds | exp(−λlh) | ρ_total(l) < 1 for l=1..4 |
| 5 | MCM < 1 (necessary & sufficient) | Karp 1978 | MCM = 0.9712 |

All five gates must pass. Exit code 0 = fully certified.

### Why MCM and not a simpler condition

Requiring `max_l ρ_total(l) < 1` is sufficient but not necessary. The MCM captures
cycle-averaging: a sequence alternating `l=4` (ρ=0.890) with three `l=1` (ρ=0.971)
steps has geometric mean 0.960 < 1, which is stable even though a naive worst-case
argument would flag the l=4 edge alone. MCM is the exact necessary-and-sufficient
stability condition for the switched system on the WHRT automaton.

---

## Metric design

### Why polynomial diagonal

The closed-loop Jacobian `J(x, ê)` varies with state (the `∂f/∂x = 2x−3x²−2` entry
ranges over [−18, −5/3] on `[-2,2]`). A constant metric cannot simultaneously certify
contraction at all points; a state-dependent polynomial metric `M(z) = diag(p(z), q(z))`
adapts to the local geometry.

Diagonal is sufficient (not just a simplification) because the Jacobian has
block-triangular cascade structure: the ê-subsystem with eigenvalue `2−α = −1`
is independently contracting, and the x-subsystem is independently stable; the
coupling is absorbed by scaling the two metric blocks. A full off-diagonal metric
would gain marginally in certified λ at the cost of breaking the analytical jump proof.

### Why softplus positivity

The metric components must be strictly positive on all of Z. Softplus encoding
`θ_i = log(1 + exp(a_i)) > 0` ensures positivity algebraically before the SOCP
runs, without adding semidefinite constraints. This also makes the jump proof
analytical: all ê-dependent monomials in `p(x, ê)` have strictly positive
coefficients, so `p(x, ê) ≥ p(x, 0)` everywhere, giving `ρ_jump ≤ 1` directly.

---

## Forward invariance

The domain `Z = [-2,2] × [-1,1]` has outward-pointing vector field near the
corners `(x=±2, ê=±1)`. The correct argument is post-jump reachability:

- After every reception, `ê` is reset to 0 (post-jump set `{(x,0): x ∈ [-2,2]}`)
- Worst-case inter-reception duration: `l_max · h = 4 × 0.195 = 0.780 s`
- Numerical simulation over 200 starting points confirms `max|ê| = 1.0078`
- Enlarged domain `Ẑ = [-2,2] × [-1.1, 1.1]` contains the full reachable set
  with margin 0.092

The flow certificate is re-verified on `Ẑ` (bound −1.435 < 0 remains valid).

---

## Fair comparison

The 56 % improvement decomposes into two co-contributions:

| Contribution | From | To | Factor |
|---|---|---|---|
| (A) Architecture | ZOH (contraction obstructed) | Observer α=3 (feasible) | undefined ratio — ZOH has no contraction bound |
| (B) Methodology | Emulation on observer system (h≈0.063 s) | Contraction+MCM (h=0.195 s) | ×3.1 |
| Combined vs 2020 MASP | h=0.125 s (ZOH+MASP) | h=0.195 s (observer+MCM) | ×1.56 |

Claiming the ×1.56 as a pure methodology improvement would be misleading.
Both contributions are stated explicitly in the paper.

---

## 2D higher-dimensional example (`src/example_2d.py`)

Demonstrates that the framework scales beyond the scalar benchmark plant.

**Plant:** `ẋ = (A−2I)x − 2ê`, `A = [[-1, 0.5], [-0.5, -2]]`, same `η=(17,20)`, same h.

Three sub-results:

| Variant | Metric | λ | MCM |
|---|---|---|---|
| Linear, block-diagonal M | Standard SDP (4 vars) | 0.40 | 0.925 |
| Linear, full 4×4 M | Full SDP (10 vars) | 0.45 | — |
| Nonlinear g=[0.05·x₁³, 0], block-diagonal M | Gridded LMI (30 Jacobian samples) | 0.25 | 0.952 |

**Why block-diagonal suffices for 2D:** The augmented 4D Jacobian has block structure
`[[A_bar, -K], [-A_bar, (2-α)I]]`. Both diagonal blocks are independently stable;
a block-diagonal Lyapunov function is sufficient for such cascade structures (Lin,
Sontag, Wang 1996). Full 4×4 M gains +12.5 % in λ but requires 10 SDP variables,
has 4× worse conditioning (cond 307 vs 74), and is harder to interpret.

**Gridded LMI for nonlinear extension:** Instead of a Lipschitz correction (which
fails when M is ill-conditioned), the SDP enforces `S(x₁) ≼ 0` at 30 sampled
x₁ values using the actual nonlinear Jacobian at each point. Verified on 150-point
fine grid. No approximation — a rigorous sufficient condition.

---

## Repository structure

```
project_pub/
├── main.py                     # Full certificate pipeline (run this)
├── config.py                   # All parameters (M=17, N=20, h=0.195, α=3.0, λ=0.15)
├── paper.tex                   # IEEE-style publication draft
├── design_rationale.tex        # Companion: reasoning behind every design decision
├── figures.py                  # Simulation and certificate figures
├── src/
│   ├── whrt_graph.py           # WHRT automaton + Karp MCM
│   ├── contraction.py          # SOCP metric search (softplus, polynomial diagonal)
│   ├── growth_factors.py       # Gronwall bounds + MC validation
│   ├── theorem_verify.py       # Full certificate chain verification
│   ├── comparison.py           # Baseline comparison + fair attribution decomposition
│   ├── theoretical_analysis.py # ZOH obstruction, co-design, forward invariance
│   └── example_2d.py           # 2D linear + nonlinear extension (SDP + gridded LMI)
└── results/
    └── certificate_report.txt  # Generated on each run
```

---

## Running

```bash
cd project_pub
python main.py          # Full pipeline, all 9 stages, ~60 s
python src/example_2d.py    # 2D example only
python figures.py           # Regenerate all figures
```

**Requirements:** `numpy`, `scipy`, `cvxpy` (with SCS solver), `matplotlib`.

**Expected output (all gates pass):**

```
CERTIFIABLY STABLE (complete certificate chain)
  MCM        : 0.97117   (must be < 1)
  Margin     : 0.02883
  Half-life  : 4.621 s
  Certified h: 0.195 s
  vs 2020    : 1.560×
```

---

## Publication status and known gaps

| Gap | Status |
|---|---|
| ZOH obstruction proof (diagonal metric) | In `paper.tex` Proposition 1 |
| ZOH obstruction proof (all metrics, rank-1 argument) | Proved; not yet in `paper.tex` — upgrade pending |
| Forward invariance (rigorous sublevel set) | Numerical in `check_post_jump_reachability()`; formal proof pending |
| Complete proof of co-design Proposition 2 | Proof sketch only in `paper.tex`; full Schur complement argument pending |
| General plant class (beyond cubic benchmark) | Not yet stated; needed for Q1 journal |
| Incremental stability corollary | Follows directly from contraction; not yet stated |
| Comparison vs Hertneck et al. 2020 [9] | Same 1.56× claimed; baseline differs (MASP vs max-consecutive-miss); differentiation needed |

The strengthened ZOH obstruction result (rank-1 Jacobian → no metric works, not
just no diagonal metric) is the most significant pending upgrade to `paper.tex`.
It changes the main proposition from a diagonal-metric limitation to a universal
structural theorem.

---

## Key parameters

| Symbol | Value | Meaning |
|---|---|---|
| M, N | 17, 20 | WHRT window: ≥17 successes in every 20 slots |
| h | 0.195 s | Certified sampling period |
| α | 3.0 | Observer gain (threshold: 2 + 2λ = 2.30) |
| λ | 0.15 | Contraction rate |
| l_max | 4 | Max intervals between receptions (N−M+1) |
| MCM | 0.9712 | Maximum cycle mean (< 1 required) |
| ρ_jump | 1.000 | Jump growth factor (analytical) |
| Domain Z | [-2,2]×[-1.1,1.1] | Compact operating domain (enlarged for invariance) |
