# config.py  — publication version
#
# Dual-sensor NCS under WHRT dropout constraint
# ---------------------------------------------
# Architecture: the actuator node has TWO sensor paths to x(t):
#
#   (A) LOCAL sensor  — always available, co-located with actuator/plant
#       (e.g. encoder, strain gauge, IMU).  Feeds the continuous observer.
#
#   (B) REMOTE sensor — high-quality / calibrated (e.g. vision, GPS).
#       Transmits x(t_k) over an unreliable WHRT channel to the actuator.
#       Subject to the η=(M,N) dropout constraint.
#
# The actuator runs an observer that fuses both paths:
#
#     dx̂/dt = -ALPHA * (x̂ - x_local(t))     [continuous correction from (A)]
#     x̂(t_k+) = x_remote(t_k)               [reset on WHRT reception from (B)]
#     u(t)  = -2 * x̂(t)                     [control law]
#
# Defining ê = x̂ - x (estimation error against local x):
#
#     ẋ = f(x, ê)  := x² - x³ - 2(x + ê)
#     dê/dt = -f(x, ê) - ALPHA * ê           [ALPHA damps ê between receptions]
#
# At each successful WHRT reception (t_k):
#     x̂(t_k+) = x_remote(t_k) ≈ x(t_k)  →  ê(t_k+) = 0   [reset map]
#
# Without the observer (ALPHA=0, pure ZOH): dê/dt = -f(x,ê), so
# d(x+ê)/dt = 0.  The estimate x̂ is held constant, ê is driven only
# by plant dynamics, and no contraction metric can certify stability.
# ALPHA > 0 breaks this structure and enables the contraction analysis.
#
# Reference architecture: emulation-based NCS with co-located auxiliary
# sensing (cf. Nesic & Teel, IEEE TAC 2004, Section IV).

M  = 17       # WHRT: minimum successes required in window
N  = 20       # WHRT: window size
H  = 0.195    # sampling period (seconds)
C_WALK = 20   # walk cost bound (used only for validation walk enumeration)

LAM   = 0.15  # target contraction rate λ  (tuned; feasibility studied in paper)
RHO   = 0.90  # jump factor design bound (used only as initial CVXPY target)
ALPHA = 3.0   # observer gain α — co-design condition: α > 2+2λ = 2.3 required
               # for J₂₂ = 2-α < -2λ, ensuring ê-component is contracting.
               # α=3.0 chosen for comfortable margin; larger α → faster observer.

DOMAIN_X = (-2.0, 2.0)   # compact domain X (forward invariance checked)
DOMAIN_E = (-1.1, 1.1)   # compact domain E — enlarged from (-1,1) to cover the
                          # max|ê|≈1.008 excursion from the post-jump set {ê=0}
                          # under worst-case 3-dropout bursts at |x|=2 boundary

# Numerical parameters
N_GRID_COARSE  = 30    # CVXPY grid (coarse, for fast optimisation)
N_GRID_FINE    = 150   # certification grid (fine, for Lipschitz certificate)
N_GRID_RJUMP   = 300   # grid for ρ_jump computation (must be fine)
N_LIP_GRID     = 25    # sub-grid for Lipschitz constant estimation
N_SAMPLES_GF   = 500   # MC samples for growth-factor validation (not certificate)
N_MC_TRAJ      = 1000  # MC trajectories for simulation figures
SIM_TIME       = 50.0  # simulation horizon (seconds)
SEED           = 42
