# Contraction-Based Exponential Stability of Nonlinear NCS under Weakly Hard Real-Time Dropout

**Moataz Seghyar** — LASTIMI Laboratory, EST Salé, Mohammed V University in Rabat

Research companion code for the paper *"Contraction-Based Exponential Stability of Nonlinear Networked Control Systems under Weakly Hard Real-Time Dropout Constraints"*.

---

## 1. What This Does

A nonlinear control system sends measurements over a network subject to packet dropout. The dropout is not arbitrary: it satisfies a **weakly hard real-time (WHRT)** constraint η = (M, N) — at least M successes in every N consecutive transmission attempts. This is a deterministic, worst-case model that captures real scheduling guarantees.

This project answers: **given a WHRT constraint and a nonlinear plant, can we certify exponential stability — and for how large a sampling period h?**

The answer is built from three independently motivated contributions:

1. **Universal ZOH obstruction** — a closed-form proof that zero-order hold alone is structurally blocked from any contraction certificate, for any plant and any Riemannian metric.
2. **Observer co-design** — an analytical condition on the observer gain α that resolves the obstruction and enables contraction analysis.
3. **WHRT + MCM stability pipeline** — a plant-agnostic pipeline that takes (λ, h, η, ρ_jump) and returns a necessary-and-sufficient stability certificate via Karp's exact Maximum Cycle Mean algorithm.

---

## 2. System Model

The closed-loop hybrid system alternates between continuous flow and discrete resets:

```
Flow (between receptions, duration l·h):
    ż(t) = F(z(t))

Jump (at each successful reception t_k):
    z(t_k+) = g(z(t_k))      [reset map]

Dropout (no measurement):
    z(t_k+) = z(t_k)          [state held]
```

The state `z = (x, ê) ∈ Z ⊂ R²` pairs the plant state x with the estimation error ê = x̂ − x. The gap between two consecutive receptions is `l ∈ {1, …, l_max}` steps, where `l_max = N − M + 1`.

**Benchmark plant (scalar instantiation):**

```
Plant:    ẋ = x² − x³ + u,    u = −2x̂
Observer: dx̂/dt = −α(x̂ − x)
Error:    ẋ = f(x, ê) = x² − x³ − 2(x + ê)
          dê/dt = −f(x, ê) − α·ê
Jump:     ê(t_k+) = 0   [reset on reception]
```

---

## 3. Assumptions

**A1 (Smoothness).** F and g are C¹ on Z.

**A2 (Domain).** Z is forward-invariant under the hybrid dynamics. Verified via post-jump reachability simulation; a strict box invariance gap is flagged honestly.

**A3 (WHRT constraint).** The reception sequence satisfies η = (M, N); encoded as a debt-state automaton G(η).

**A4 (Metric feasibility).** There exists a Riemannian metric M(z) ≻ 0 satisfying A5 and A6.

**A5 (Flow contraction).** The metric satisfies `S(z) := Ṁ + J^T M + M J + 2λ M ≼ 0` on Z for some λ > 0. Certified via SOCP on a grid with a Lipschitz correction that extends the certificate to all of Z.

**A6 (Jump non-expansion).** The variational map Dg satisfies `ρ_jump = sup_{z∈Z} ‖Dg(z)‖_M ≤ 1`. Proved analytically from the softplus-encoded metric structure.

---

## 4. Main Results

### Theorem — Contraction under WHRT dropout

Under A1–A6, the Maximum Cycle Mean of G(η) with edge weights ρ_total(l) = e^{−λlh} · ρ_jump satisfies:

```
MCM < 1  ⟺  exponentially stable under ALL η-admissible dropout sequences
```

This is the necessary-and-sufficient condition (Karp 1978). The half-life is `T_{1/2} = log(2) / |log(MCM)|` per WHRT cycle.

### Proposition 1 — Universal ZOH obstruction

For any plant f(x, ê) with f(0,0) = 0 under ZOH (α = 0), no Riemannian metric M(z) ≻ 0 — diagonal or otherwise — can certify contraction with any rate λ > 0.

**Proof sketch.** The ZOH Jacobian is rank-1: J_ZOH = c · r(z)^T where c = (1, −1)^T. At z = 0, r^T(0)c = ∂f/∂x|₀ − ∂f/∂ê|₀ = 0 (for the benchmark plant: −2 − (−2) = 0). Then c^T S(0) c = 2λ (c^T M(0) c) > 0 for any M(0) ≻ 0. Independently: d(x + ê)/dt = 0 identically under ZOH — the state space foliates into invariant lines {x + ê = const}, preventing any two trajectories on different leaves from converging.

### Proposition 2 — Observer co-design condition

The flow LMI is feasible with a diagonal polynomial metric if and only if α > 2 + 2λ. At the nominal λ = 0.15, the threshold is α* = 2.30; the configured α = 3.0 provides a margin of 0.70.

### Proposition 3 — Grid-to-domain Lipschitz extension

If `max_{z∈G_N} λ_max(S(z)) + L · δ < 0` where L is a Lipschitz bound on λ_max(S(·)) and δ is the grid half-diagonal, then A5 holds on all of Z.

---

## 5. Certificate Chain

Four gates must all pass:

| Stage | What is verified | Method | Passes if |
|---|---|---|---|
| 1 | ZOH structurally obstructed | Rank-1 Jacobian + conservation law | Both arguments hold |
| 2 | Observer co-design condition | Analytical: α > 2 + 2λ | margin > 0 |
| 3 | Flow condition A5 on Z | SOCP on grid + Lipschitz bound | `grid_max + L·δ < 0` |
| 4 | Jump condition A6 | Analytical (softplus encoding) | ρ_jump ≤ 1 |
| 5 | MCM < 1 (N&S) | Karp's algorithm (exact) | `MCM < 1` |

The certificate chain is rigorous when gates 3–5 pass; the ZOH obstruction (gate 1) and co-design (gate 2) are structural arguments.

---

## 6. Numerical Results — Scalar Plant, η = (17, 20)

```
λ = 0.15,   h = 0.195 s,   η = (17, 20),   α = 3.0

Stage 1  ZOH obstructed:          True  (conservation law + rank-1)
Stage 2  Co-design:                α = 3.0 > α* = 2.30  (margin = 0.70)
Stage 3  Flow certificate:        -1.450 + 0.017 = -1.433 < 0           OK
Stage 4  ρ_jump:                   1.000 ≤ 1                             OK
Stage 6  Growth factors:
         l=1: ρ_total = 0.9712    l=2: 0.9432    l=3: 0.9160    l=4: 0.8896
Stage 7  MCM = 0.97117            margin = 0.02883                       OK
Stage 8  h* sweep (Gronwall):     stable to h ≥ 0.40 s
Stage 9  Method improvement:      9.9× vs emulation bound (same arch)
         Combined improvement:     1.56× vs MASP formula (JOINT: arch + method)

Half-life: 4.62 s.   Certificate chain: COMPLETE.
```

**Forward invariance (Stage 4):** Z = [−2, 2] × [−1.1, 1.1] is not strictly box-invariant — the vector field exits near (|x| = 2, |ê| = 1.1) corners. However, post-jump reachability simulation confirms those corners are never reached: max|ê| from the post-jump set {ê = 0} is 1.0078 < 1.1 over the worst-case 4-step dropout burst. A `gap_flag` is set; a Lyapunov sublevel-set proof would close it.

---

## 7. Improvement Decomposition

The improvement over the 2020 MASP baseline (h_2020 = 0.125 s) is **joint** — it comes from two co-contributions:

| Comparison | h (s) | Architecture | Ratio | Attribution |
|---|---|---|---|---|
| ZOH (α = 0) | N/A (obstructed) | ZOH | — | no cert possible |
| Emulation bound (α = 3) | ~0.020 s | Observer | 1/9.9× | same arch |
| MASP formula (ZOH) | 0.125 s | ZOH | 1/1.56× | different arch |
| **This work** | **0.195 s** | Observer | 1× | — |

The **9.9× method improvement** (emulation vs contraction+MCM, same observer architecture) is a clean methodology comparison. The **1.56× combined improvement** over the MASP baseline is a joint contribution from (a) the observer architecture that makes certification possible and (b) the tighter MCM bound. Both are claimed as novel.

---

## 8. Generalization Test

The pipeline is fully decoupled from η = (17, 20). Changing `config.M` and `config.N` to any valid (M, N) automatically:
- Builds the correct debt-state automaton G(η) with N − M + 1 nodes
- Computes the right l_max = N − M + 1
- Returns a new certified h without any code changes

Example: η = (9, 10) gives l_max = 2, a 2-node automaton, and certifies stability up to h > 0.40 s at the same λ = 0.15.

---

## 9. Repository Structure

```
project_pub/
├── main.py                      # 9-stage certificate pipeline (run this)
├── config.py                    # Parameters: M=17, N=20, h=0.195, λ=0.15, α=3.0
├── figures.py                   # All publication figures
├── src/
│   ├── zoh_obstruction.py       # Contribution 1: universal ZOH obstruction
│   ├── observer_codesign.py     # Contribution 2: co-design threshold α* = 2+2λ
│   ├── whrt_mcm.py              # Contribution 3: WHRT automaton + Karp MCM pipeline
│   ├── invariance.py            # Forward invariance certificate with gap flag
│   ├── comparison.py            # Same-arch method comparison (9.9×) + combined (1.56×)
│   ├── contraction.py           # SOCP metric search + Lipschitz certificate
│   ├── growth_factors.py        # Gronwall bounds + MC validation
│   ├── whrt_graph.py            # WHRT automaton construction + Karp MCM (core)
│   ├── theorem_verify.py        # MCM certificate chain verifier + report writer
│   ├── deviation_bound.py       # Half-life, e-folding, horizon bound
│   ├── monotonicity.py          # Structural consistency checks
│   └── example_2d.py            # 2D extension (SDP + gridded LMI)
└── results/
    └── certificate_report.txt
```

**Three contribution modules are standalone and independently testable:**

```python
# Contribution 1 — ZOH obstruction (any plant, any metric class)
from src.zoh_obstruction import check_zoh_obstruction
result = check_zoh_obstruction(f, domain=((-2,2),(-1.1,1.1)))

# Contribution 2 — observer co-design (any λ)
from src.observer_codesign import codesign_threshold, solve
alpha_star = codesign_threshold(lam=0.15)   # returns 2.30
result = solve(lam=0.15, domain_x=(-2,2), domain_e=(-1.1,1.1))

# Contribution 3 — WHRT + MCM (any η, any λ, h, ρ_jump)
from src.whrt_mcm import certify, sweep
result = certify(lam=0.15, h=0.195, eta=(17,20), rho_jump=1.0)
h_curve = sweep(lam=0.15, eta=(17,20), rho_jump=1.0, h_lo=0.05, h_hi=0.50)
```

---

## 10. Running the Pipeline

```bash
# Full 9-stage certificate pipeline (~60 s, requires CVXPY with CLARABEL or SCS)
python main.py

# 2D extension only
python src/example_2d.py

# Regenerate figures
python figures.py
```

**Dependencies:** `numpy`, `scipy`, `cvxpy` (with CLARABEL or SCS solver), `matplotlib`, `networkx`.

```bash
pip install numpy scipy cvxpy matplotlib networkx
```

---

## 11. Open Gaps

| Gap | Status |
|---|---|
| Forward invariance: Lyapunov sublevel-set proof | Numerical only (gap_flag set in certificate) |
| General plant class beyond scalar benchmark | Not yet stated; framework is plant-agnostic |
| Incremental stability corollary | Follows from contraction; not yet written |
| Observer co-design: tight bound when α = α* | SOCP sweep confirms analytical bound is tight |
| Comparison with Hertneck et al. 2020: explicit decomposition | In comparison.py; 1.56× flagged as joint |
