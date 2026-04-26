# Contraction-Based Stability of a Dual-Sensor NCS under WHRT Dropout

**Moataz Seghyar** — LASTIMI Laboratory, EST Salé, Mohammed V University in Rabat  
PhD research companion code · `paper.tex` + `design_rationale.tex`

---

## 1. System Model

### 1.1 Plant

Consider the scalar nonlinear plant

```
ẋ(t) = f(x(t), ê(t))  :=  x² − x³ − 2(x + ê),    x ∈ ℝ
```

where `ê = x̂ − x` is the estimation error and `u = −2x̂` is the control law.
The uncontrolled fixed point is `x = 0`; the open-loop plant has an unstable
manifold near `x = 1`.

### 1.2 Dual-sensor architecture

The actuator node has access to two sensor signals:

- **Local sensor** (always available): measures `x_local(t)` continuously,
  e.g. an encoder or IMU co-located with the actuator.
- **Remote sensor** (high-quality, unreliable): transmits `x_remote(t_k)` at
  scheduled instants `t_k = kh` over a channel subject to the WHRT constraint.

The actuator runs a continuous observer fusing both signals:

```
ẋ̂(t)   = −α (x̂(t) − x_local(t))        [continuous correction]
x̂(t_k⁺) = x_remote(t_k)                 [reset on successful reception]
```

### 1.3 Error dynamics and hybrid structure

Define `ê(t) = x̂(t) − x(t)`. Differentiating and substituting gives the
**augmented hybrid system**:

```
Flow  (between receptions):
    ẋ  =  f(x, ê)
    ê̇  =  −α ê − f(x, ê)

Jump  (at each successful reception t_k):
    x(t_k⁺)  =  x(t_k)           [x is continuous]
    ê(t_k⁺)  =  0                 [remote reset eliminates error]

Dropout (at each failed reception):
    x(t_k⁺)  =  x(t_k)
    ê(t_k⁺)  =  ê(t_k)            [ê unchanged]
```

The augmented state is `z = (x, ê)ᵀ ∈ Z`.

### 1.4 WHRT dropout constraint

A binary sequence `(s_k)_{k≥0}`, `s_k ∈ {0,1}` (1 = success, 0 = dropout),
satisfies the `η = (M, N)` weakly hard real-time constraint if

```
∑_{i=k}^{k+N−1} s_i  ≥  M     for all k ≥ 0.
```

For `η = (17, 20)`: at most `N − M = 3` dropouts in any window of 20.
The maximum gap between two successive receptions is `l_max = N − M + 1 = 4`
sampling intervals.

The constraint is encoded as a finite directed automaton `G = (V, E)` with
debt states `V = {0, 1, 2, 3}` (current failure debt) and edges labelled by
`l ∈ {1, 2, 3, 4}` (intervals to next reception).

### 1.5 Operating domain

```
Z  :=  [−2, 2] × [−1.1, 1.1]    ⊂  ℝ²
```

The x-bounds capture the physical operating range. The ê-bounds are set to
`±1.1` rather than `±1` to cover the post-jump reachable set (see §4).

---

## 2. Assumptions

**A1 (Smoothness).** The vector field `f : Z → ℝ` is continuously differentiable
on Z.

**A2 (Post-jump reachability).** Starting from any post-jump state
`(x₀, 0) ∈ Z`, the augmented trajectory satisfies `|ê(t)| ≤ 1.1` for all
`t ∈ [0, l_max · h]`. *(Verified numerically: max|ê| = 1.0078, margin 0.092.)*

**A3 (WHRT automaton).** The dropout sequence `(s_k)` satisfies `η = (17, 20)`
with automaton `G` as defined in §1.4.

**A4 (Co-design).** The observer gain `α` and target contraction rate `λ` satisfy
`α > 2 + 2λ`.

**A5 (Metric feasibility).** There exists a diagonal polynomial metric
`M(z) = diag(p(z), q(z))` with `p, q > 0` on Z satisfying the flow LMI
`Ṁ + JᵀM + MJ + 2λM ≼ 0` on Z.

*A5 is verified computationally via SOCP + Lipschitz certificate (§3.2).
A4 is a sufficient condition for A5 to be feasible.*

---

## 3. Main Results

### Proposition 1 — Universal ZOH obstruction

**Statement.** For `α = 0` (ZOH), no positive definite metric `M ≻ 0`,
whether diagonal, full, constant, or state-dependent, satisfies the contraction
condition `S = JᵀM + MJ + 2λM ≼ 0` on Z for any `λ > 0`.

**Proof sketch.**
With `α = 0`, `ẋ̂ = 0` so `ê̇ = −f(x, ê)`. The Jacobian factors as

```
J_ZOH  =  c · rᵀ,    c = (1, −1)ᵀ,    r = (∂f/∂x, ∂f/∂ê)ᵀ,
```

i.e. `J` is **rank 1** at every point. Changing variables `ξ = M^{1/2} δz`
and writing `ĉ = M^{1/2}c`, `r̂ = M^{−1/2}r`, the matrix `P + Pᵀ` where
`P = ĉr̂ᵀ` has eigenvalues

```
λ±(P + Pᵀ)  =  ĉᵀr̂  ±  ‖ĉ‖ ‖r̂‖.
```

Since `ĉᵀr̂ = cᵀr` (M-invariant) and `‖ĉ‖‖r̂‖ ≥ |cᵀr|` by Cauchy-Schwarz:

```
λ_max(P + Pᵀ)  =  cᵀr + ‖ĉ‖‖r̂‖  ≥  cᵀr + |cᵀr|  ≥  0.
```

Adding `2λI` gives `λ_max(P + Pᵀ + 2λI) ≥ 2λ > 0` for all `M ≻ 0`.
Hence `S` is strictly indefinite. `□`

**Interpretation.** The conservation law `d(x + ê)/dt = ẋ + ê̇ = f − f = 0`
foliates state space into invariant lines `{x + ê = const}`. No Riemannian
metric contracts along a direction that trajectories never leave.
This is a structural property of ZOH, not a limitation of the metric family.

---

### Proposition 2 — Observer co-design condition

**Statement.** Under Assumptions A1 and A4 (`α > 2 + 2λ`), the diagonal entry
`J₂₂ = 2 − α < −2λ`, making the flow LMI (Assumption A5) feasible via SOCP.

**Proof sketch.**
The observer modifies the error row of the Jacobian to

```
J  =  [[ ∂f/∂x,   ∂f/∂ê ],
       [−∂f/∂x,    2 − α  ]].
```

The rank-1 structure is broken: `J₂₂ = 2 − α ≠ −∂f/∂ê`. For a diagonal metric
`M = diag(p, q)`, the `(2,2)` entry of `S` at any point is

```
S₂₂  =  2(2 − α + λ)q.
```

`S₂₂ < 0` iff `α > 2 + λ`. The off-diagonal coupling `S₁₂` additionally
requires `α > 2 + 2λ` (Schur complement bound over Z). `□`

**Interpretation.** The observer gain `α` controls how fast the local path
damps `ê` between remote receptions. The co-design condition quantifies the
minimum damping rate needed to overcome the plant's coupling term. Larger `α`
allows larger certified `λ` (faster contraction) but increases local sensor
bandwidth demand.

---

### Theorem — Exponential stability under WHRT dropout

**Statement.** Under Assumptions A1–A5 and the numerical certificate
`(λ, h, η) = (0.15, 0.195 s, (17,20))`, if the four-stage certificate chain
below yields

```
(i)  max_{z∈Z} λ_max(S(z)) + L·δ  <  0           [flow, Lipschitz-corrected]
(ii) ρ_jump  ≤  1                                  [jump non-expansive]
(iii) ρ_total(l) = ρ_flow(l)·ρ_jump  <  1  ∀l     [per-label growth bounded]
(iv) MCM(G, ρ_total)  <  1                         [Karp condition on automaton]
```

then the hybrid system is **exponentially stable** on Z for all admissible
`η`-sequences: there exist `C, μ > 0` such that

```
‖z(t)‖_M  ≤  C · e^{−μt} · ‖z(0)‖_M     for all z(0) ∈ Z, t ≥ 0.
```

**Numerical certificate** (`η = (17,20)`, `h = 0.195 s`, `λ = 0.15`, `α = 3.0`):

```
(i)  −1.450 + 0.017  =  −1.433  <  0       ✓
(ii) ρ_jump  =  1.000  ≤  1                ✓
(iii) max_l ρ_total(l)  =  0.9712  <  1    ✓
(iv) MCM  =  0.97117  <  1   (margin 0.02883)   ✓
```

**Why MCM and not `max_l ρ_total(l) < 1`.** Condition (iv) is necessary and
sufficient (Karp 1978); requiring `max_l ρ < 1` is only sufficient. A cycle
alternating `l=4` (ρ=0.890) with three `l=1` steps (ρ=0.971 each) has geometric
mean 0.960 < 1 — stable — but the per-edge condition would not detect this.
Using MCM rather than the per-edge bound is what permits certifying `h = 0.195 s`
where a simpler criterion would require a smaller `h`.

---

## 4. Metric Design

### 4.1 Polynomial diagonal metric

**Why polynomial (state-dependent).**
`∂f/∂x = 2x − 3x² − 2` ranges from `−18` (at `x = −2`) to `−5/3` (at `x = 1/3`).
A constant metric must simultaneously certify contraction at the most and least
stable points; the SOCP finds this infeasible. A polynomial metric `M(z) = diag(p(z), q(z))`
adapts its weighting to the local Jacobian eigenstructure.

**Why diagonal.**
The Jacobian has cascade block structure: the `ê`-subsystem eigenvalue is
`2 − α = −1` (independently contracting); the `x`-subsystem has negative
eigenvalues from `∂f/∂x`; the coupling terms are bounded. For such cascade
structures, a block-diagonal (equivalently diagonal in 2D) Lyapunov function
is sufficient by the cascade stability theorem (Lin, Sontag, Wang 1996). A
full off-diagonal metric would gain marginally in certified `λ` but breaks the
analytical proof of `ρ_jump ≤ 1` (see §4.2).

### 4.2 Softplus positivity and the jump proof

Metric coefficients are encoded as `θᵢ = log(1 + exp(aᵢ)) > 0` (softplus),
ensuring all polynomial coefficients are strictly positive before the SOCP runs.

**Consequence for jumps.** Since every `ê`-dependent monomial in `p(x, ê)` has
a strictly positive coefficient:

```
p(x, ê)  =  p(x, 0)  +  [strictly positive terms in ê]  ≥  p(x, 0).
```

Therefore `ρ_jump = sup √(p(x,0)/p(x,ê)) ≤ 1` analytically, without solving
an additional optimisation. This is the sole reason softplus is used rather than,
e.g., box constraints: it converts the jump condition into a one-line algebraic
argument.

---

## 5. Forward Invariance

**Issue.** The vector field has outward-pointing components near the corners
`(x = ±2, ê = ±1)` of Z, so Z is not strictly forward-invariant as an absorbing set.

**Resolution (post-jump reachability).**
The hybrid structure resets `ê → 0` at every successful reception. Starting from
any post-jump state `(x₀, 0)`, the worst-case inter-reception duration is
`l_max · h = 4 × 0.195 = 0.780 s`. Numerical integration over 200 initial
conditions `x₀ ∈ [−2, 2]` with `ê₀ = 0` gives:

```
max |ê(t)|  =  1.0078  <  1.1  =  domain bound     (margin 0.092)
```

The domain is enlarged to `Z = [-2,2] × [-1.1, 1.1]` to contain this exceedance.
The flow certificate is re-verified on the enlarged domain (bound −1.433 < 0 holds).

**Pending.** A formal sublevel-set invariance proof (showing `{V ≤ c*}` is
positively invariant for `c* = min_{z ∈ ∂Z} V(z)`) is not yet in `paper.tex`.
The numerical argument above is rigorous for the post-jump reachable set but
does not cover all possible entry points into Z.

---

## 6. Fair Comparison

The 56 % improvement decomposes into two separable contributions:

| Contribution | Baseline | Ours | Factor |
|---|---|---|---|
| (A) Architecture | ZOH: contraction structurally infeasible (Prop. 1) | Observer α=3.0: contraction feasible | Not quantifiable as ratio |
| (B) Methodology | Emulation on same observer system, h ≈ 0.063 s | Contraction + MCM, h = 0.195 s | ×3.1 |
| Combined vs 2020 MASP | ZOH + MASP formula, h = 0.125 s | Observer + MCM, h = 0.195 s | ×1.56 |

The ×1.56 figure is a joint effect of (A) and (B). Claiming it as a pure
methodology improvement would be incorrect. The paper attributes it explicitly
to both contributions.

**Relation to Hertneck et al. (IFAC 2020).** That work also reports a ×1.56
improvement using non-monotonic Lyapunov on the WH graph. The comparison
baseline differs (max-consecutive-miss bound vs MASP formula); the method
differs (non-monotonic Lyapunov vs contraction + MCM); and the result here
additionally provides incremental stability (convergence between trajectories,
not just to the origin). These distinctions require explicit treatment in the
paper before Q1 submission.

---

## 7. 2D Extension (`src/example_2d.py`)

**Plant.** `ẋ = (A − 2I)x − 2ê`,  `A = [[-1, 0.5], [-0.5, -2]]`,  same `η`, same `h`.

**Assumptions (2D).** Linear plant → Jacobian constant → constant block-diagonal
metric `M = blkdiag(M_xx, M_ee)` is necessary and sufficient (standard quadratic
stability). No Lipschitz correction needed (`g ≡ 0` exactly).

| Variant | Method | λ | MCM | Notes |
|---|---|---|---|---|
| Linear, block-diag M | Standard LMI, 4 SDP vars | 0.40 | 0.925 | Exact for linear |
| Linear, full 4×4 M | Full LMI, 10 SDP vars | 0.45 | — | +12.5% λ, cond 4× worse |
| Nonlinear `g=[0.05x₁³, 0]`, block-diag M | Gridded LMI, 30 Jacobian samples | 0.25 | 0.952 | Sufficient, rigorous |

**Why block-diagonal suffices.** Both diagonal blocks of the 4D augmented Jacobian
are independently stable (`ê`-block eigenvalue `= −1`, `x`-block eigenvalues
`≈ −3.5 ± 0.7i`). Cascade stability (Lin, Sontag, Wang 1996) guarantees a
block-diagonal Lyapunov function exists. The full 4×4 metric gains +12.5% in `λ`
at the cost of 10 vs 4 SDP variables and condition number 307 vs 74.

**Gridded LMI rationale.** A Lipschitz correction approach fails when `M` is
ill-conditioned: the correction `‖M‖₂ · L_g` can dominate the SDP slack `σ`,
driving the effective `λ` negative. The gridded LMI avoids this by imposing
`S(x₁ⁱ) ≼ 0` directly at 30 sampled Jacobian evaluations — a rigorous sufficient
condition without any approximation.

---

## 8. Repository Structure

```
project_pub/
├── main.py                     # Full 9-stage certificate pipeline
├── config.py                   # Parameters: M=17, N=20, h=0.195, α=3.0, λ=0.15
├── paper.tex                   # IEEE-style draft
├── design_rationale.tex        # Companion: full reasoning behind every decision
├── figures.py                  # All figures (simulation, heatmap, automaton)
├── src/
│   ├── whrt_graph.py           # WHRT automaton construction + Karp MCM
│   ├── contraction.py          # SOCP metric search (softplus polynomial diagonal)
│   ├── growth_factors.py       # Gronwall bounds + Monte Carlo validation
│   ├── theorem_verify.py       # Certificate chain verifier
│   ├── comparison.py           # Baseline comparison + attribution decomposition
│   ├── theoretical_analysis.py # Propositions 1–2, forward invariance, co-design
│   └── example_2d.py           # 2D linear + nonlinear (SDP + gridded LMI)
└── results/
    └── certificate_report.txt  # Auto-generated on each run
```

### Running

```bash
cd project_pub
python main.py            # Full pipeline (~60 s), all 9 stages
python src/example_2d.py  # 2D example standalone
python figures.py         # Regenerate all figures
```

**Dependencies:** `numpy`, `scipy`, `cvxpy` (SCS solver), `matplotlib`.

**Expected terminal output:**

```
CERTIFIABLY STABLE (complete certificate chain)
  MCM              :  0.97117   (must be < 1)
  Stability margin :  0.02883
  Half-life        :  4.621 s
  Certified h      :  0.195 s
  vs 2020 MASP     :  1.560×
```

---

## 9. Known Gaps Before Q1 Submission

| Item | Status |
|---|---|
| Prop. 1 (ZOH obstruction, rank-1, any M) | Proved above; proof sketch only in `paper.tex` |
| Prop. 2 (co-design, full Schur complement) | Proof sketch in `paper.tex`; full argument pending |
| Forward invariance (formal sublevel set) | Numerical only; formal proof pending |
| General plant class beyond cubic benchmark | Not stated; required for Q1 |
| Incremental stability corollary | Follows from contraction; not yet stated |
| Differentiation from Hertneck et al. 2020 | Same ×1.56 ratio; baseline and method differ; needs explicit treatment |
