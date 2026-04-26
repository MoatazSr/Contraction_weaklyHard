"""
whrt_graph.py — WHRT automaton construction and exact stability certificate.

Public functions
----------------
build_whrt_graph        — construct the WHRT state graph
validate_graph          — structural sanity checks (returns max_label, etc.)
compute_max_cycle_mean  — EXACT MCM via Karp (1978); primary stability certificate
enumerate_walks         — simple-path DFS (legacy; validation only)
enumerate_walks_full    — full DFS (validation only; NOT the primary certificate)
enumerate_walks_sufficient — capped DFS (validation only)

Certificate strategy
--------------------
The stability of the system under ALL admissible dropout sequences is
characterised exactly by the Maximum Cycle Mean (MCM) of the WHRT automaton
weighted by the per-label growth factors ρ_total(l):

    MCM = max_{cycles C ⊆ Gη} { (∏_{e∈C} ρ_total(label(e)))^{1/|C|} }

Equivalently, with log-weights w(l) = log(ρ_total(l)):

    MCM = exp( max_C { (1/|C|) Σ_{e∈C} w(label(e)) } )

The system is stable under ALL WHRT-admissible infinite sequences
if and only if  MCM < 1  (equivalently MCM_log < 0).

Karp's Algorithm (1978) computes the MCM exactly in O(|V|²|E|) time.
For the WHRT graph with |V|=4 nodes and |E|=16 edges this is trivial.
This replaces the previous truncated-DFS "sufficient certificate" approach,
which could not certify stability over infinite (cyclic) sequences.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import config


# ---------------------------------------------------------------------------
# 1.  build_whrt_graph
# ---------------------------------------------------------------------------

def build_whrt_graph(m: int, n: int) -> nx.MultiDiGraph:
    """Return the WHRT state automaton as a directed multigraph.

    Nodes  0 … (n-m)  represent failure-debt: how many additional failures
    have accumulated beyond the WHRT quota in the current sliding window.

    For every node i and every label l ∈ [1, n-m+1]:
        destination = min(n-m, max(0, i + l - 2))
    Edge attribute ``label`` is set to l.

    The graph is strongly connected for all valid η=(m,n) with m<n.
    """
    debt_max: int = n - m
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_nodes_from(range(debt_max + 1))
    for i in range(debt_max + 1):
        for l in range(1, debt_max + 2):
            dest = min(debt_max, max(0, i + l - 2))
            G.add_edge(i, dest, label=l)
    return G


# ---------------------------------------------------------------------------
# 2.  validate_graph
# ---------------------------------------------------------------------------

def validate_graph(G: nx.MultiDiGraph, m: int, n: int) -> dict:
    """Structural sanity checks on the WHRT automaton.

    Returns
    -------
    dict with keys:
        max_label    — int, largest edge label present
        n_nodes      — int
        n_edges      — int
        strongly_connected — bool
        passed       — bool
        errors       — list[str]
    """
    debt_max      = n - m
    expected_max  = debt_max + 1
    all_labels    = [d["label"] for _, _, d in G.edges(data=True)]
    max_label     = max(all_labels) if all_labels else 0
    n_nodes       = G.number_of_nodes()
    n_edges       = G.number_of_edges()
    sc            = nx.is_strongly_connected(G)

    errors: list[str] = []
    if max_label != expected_max:
        errors.append(f"max_label: expected {expected_max}, got {max_label}")
    if n_nodes != debt_max + 1:
        errors.append(f"n_nodes: expected {debt_max+1}, got {n_nodes}")
    if not sc:
        errors.append("graph is not strongly connected")

    return {
        "max_label":          max_label,
        "n_nodes":            n_nodes,
        "n_edges":            n_edges,
        "strongly_connected": sc,
        "passed":             len(errors) == 0,
        "errors":             errors,
    }


# ---------------------------------------------------------------------------
# 3.  compute_max_cycle_mean   ← PRIMARY STABILITY CERTIFICATE
# ---------------------------------------------------------------------------

def compute_max_cycle_mean(G: nx.MultiDiGraph,
                           rho_total: dict[int, float]) -> dict:
    """Exact Maximum Cycle Mean via Karp's algorithm (1978).

    The MCM of the WHRT automaton with edge weights ρ_total(l) equals:

        MCM = max over all cycles C { (∏_{e∈C} ρ_total(label(e)))^{1/|C|} }

    For log-weights  w(l) = log(ρ_total(l)):

        MCM = exp( max_C { (1/|C|) Σ w(l) } )

    Karp's algorithm
    ----------------
    Add a virtual source s = −1 with zero-weight edges to every node.
    D[k][v] = maximum total log-weight of any walk of exactly k edges
              from s to v  (−∞ if unreachable).

    D[0][v] = 0   for all v  (zero-length path from source to v)
    D[k][v] = max_{u: u→v ∈ E} ( D[k−1][u] + w(label(u→v)) )

    MCM_log = max_v min_{0≤k≤n−1} { (D[n][v] − D[k][v]) / (n − k) }
    MCM = exp(MCM_log)

    Complexity: O(|V|·|E|) — trivial for this graph.

    Stability condition
    -------------------
    The system is stable under ALL WHRT-admissible infinite dropout
    sequences if and only if  MCM < 1.

    This is both NECESSARY and SUFFICIENT — unlike the truncated DFS
    approach which was only a sufficient condition over finitely many walks.

    Parameters
    ----------
    G          : WHRT automaton (from build_whrt_graph)
    rho_total  : dict {label → ρ_total(label)}, must cover all edge labels

    Returns
    -------
    dict with keys:
        mcm              — float, the maximum cycle mean
        mcm_log          — float, log(mcm) = max mean log-weight per step
        stable           — bool, True iff mcm < 1
        stability_margin — float, 1 − mcm  (positive = stable)
        worst_cycle      — list[int], labels of the worst elementary cycle found
        worst_cycle_mean — float, geometric mean of ρ_total along worst cycle
        n_nodes          — int
        certificate_type — str, 'karp_exact'
        all_edge_labels  — list[int], labels present in graph
        error            — str, present only on failure
    """
    nodes      = sorted(G.nodes())
    n          = len(nodes)
    node_idx   = {v: i for i, v in enumerate(nodes)}
    NEG_INF    = -1e18

    # Build adjacency list with log-weights (multiple parallel edges per pair)
    # Each entry: (source_idx, dest_idx, log_weight, label)
    edges_lw: list[tuple[int, int, float, int]] = []
    for u, v, data in G.edges(data=True):
        l   = int(data["label"])
        if l not in rho_total:
            return {
                "mcm": float("nan"), "mcm_log": float("nan"),
                "stable": False, "stability_margin": float("nan"),
                "worst_cycle": [], "worst_cycle_mean": float("nan"),
                "n_nodes": n, "certificate_type": "karp_exact",
                "all_edge_labels": [],
                "error": f"missing rho_total for label l={l}",
            }
        rho = float(rho_total[l])
        if rho <= 0:
            return {
                "mcm": float("nan"), "mcm_log": float("nan"),
                "stable": False, "stability_margin": float("nan"),
                "worst_cycle": [], "worst_cycle_mean": float("nan"),
                "n_nodes": n, "certificate_type": "karp_exact",
                "all_edge_labels": [],
                "error": f"rho_total[{l}]={rho} ≤ 0 (log undefined)",
            }
        lw = math.log(rho)
        edges_lw.append((node_idx[u], node_idx[v], lw, l))

    # D[k][v] = max total log-weight of a walk of exactly k edges ending at v
    # Starting condition: D[0][v] = 0 for all v (virtual source with 0-weight edges)
    D = [[NEG_INF] * n for _ in range(n + 1)]
    for v in range(n):
        D[0][v] = 0.0

    for k in range(1, n + 1):
        for u_idx, v_idx, lw, _ in edges_lw:
            if D[k - 1][u_idx] > NEG_INF:
                val = D[k - 1][u_idx] + lw
                if val > D[k][v_idx]:
                    D[k][v_idx] = val

    # Karp's formula: MCM_log = max_v min_{0≤k≤n−1} (D[n][v] − D[k][v]) / (n−k)
    mcm_log = NEG_INF
    best_v  = 0
    for v in range(n):
        if D[n][v] <= NEG_INF:
            continue
        min_ratio = float("inf")
        for k in range(n):
            if D[k][v] > NEG_INF:
                ratio = (D[n][v] - D[k][v]) / float(n - k)
                if ratio < min_ratio:
                    min_ratio = ratio
        if min_ratio < float("inf") and min_ratio > mcm_log:
            mcm_log = min_ratio
            best_v  = v

    if mcm_log <= NEG_INF + 1:
        # Fallback: graph not strongly connected or all paths have NEG_INF
        mcm_log = float("nan")
        mcm     = float("nan")
        stable  = False
    else:
        mcm    = math.exp(mcm_log)
        stable = bool(mcm < 1.0)

    # Identify the worst elementary cycle by brute-force (graph is tiny)
    worst_cycle, worst_mean = _find_worst_elementary_cycle(G, rho_total)

    all_labels = sorted(set(d["label"] for _, _, d in G.edges(data=True)))

    return {
        "mcm":              mcm,
        "mcm_log":          mcm_log,
        "stable":           stable,
        "stability_margin": float(1.0 - mcm) if not math.isnan(mcm) else float("nan"),
        "worst_cycle":      worst_cycle,
        "worst_cycle_mean": worst_mean,
        "n_nodes":          n,
        "certificate_type": "karp_exact",
        "all_edge_labels":  all_labels,
    }


def _find_worst_elementary_cycle(G: nx.MultiDiGraph,
                                  rho_total: dict[int, float]) -> tuple:
    """Return (label_sequence, geometric_mean) of the worst elementary cycle.

    Uses simple DFS cycle enumeration.  Feasible for |V| ≤ 10.
    """
    best_mean  = 0.0
    best_cycle: list[int] = []

    def _dfs(start: int, node: int, path_labels: list[int],
             path_nodes: list[int], visited: set[int]) -> None:
        nonlocal best_mean, best_cycle
        for _, nbr, data in G.edges(node, data=True):
            lbl = int(data["label"])
            if nbr == start and path_labels:
                # Completed a cycle
                prod  = 1.0
                for l in path_labels:
                    prod *= rho_total.get(l, 1.0)
                mean  = prod ** (1.0 / len(path_labels))
                if mean > best_mean:
                    best_mean  = mean
                    best_cycle = list(path_labels) + [lbl]
            elif nbr not in visited:
                visited.add(nbr)
                path_labels.append(lbl)
                path_nodes.append(nbr)
                _dfs(start, nbr, path_labels, path_nodes, visited)
                path_labels.pop()
                path_nodes.pop()
                visited.discard(nbr)

    for start in sorted(G.nodes()):
        _dfs(start, start, [], [start], {start})

    return best_cycle, best_mean


# ---------------------------------------------------------------------------
# 4-6.  Walk enumeration (VALIDATION only — NOT the primary certificate)
# ---------------------------------------------------------------------------

def enumerate_walks(G: nx.DiGraph, c_walk: float, h: float) -> list[list[int]]:
    """Simple-path DFS walk enumeration.  For validation / figure generation only.

    WARNING: This produces only simple-path walks (no node revisits) and
    cannot certify stability over all infinite dropout sequences.
    Use compute_max_cycle_mean() for the formal stability certificate.
    """
    LIMIT: int = 50_000
    seen: set[tuple[int, ...]] = set()
    exceeded = [False]

    def _dfs(node: int, labels: list[int], cost: float,
             visited: set[int]) -> None:
        if exceeded[0]:
            return
        if labels:
            seen.add(tuple(labels))
            if len(seen) > LIMIT:
                exceeded[0] = True
                return
        for _u, nbr, data in G.edges(node, data=True):
            if nbr in visited:
                continue
            lbl = int(data["label"])
            nc  = cost + lbl * h
            if nc <= c_walk:
                labels.append(lbl)
                visited.add(nbr)
                _dfs(nbr, labels, nc, visited)
                labels.pop()
                visited.discard(nbr)
                if exceeded[0]:
                    return

    for start in G.nodes():
        if exceeded[0]:
            break
        _dfs(start, [], 0.0, {start})

    if exceeded[0]:
        raise ValueError(f"Walk list exceeded {LIMIT} entries.")
    return [list(s) for s in seen]


def enumerate_walks_full(G: nx.DiGraph, c_walk: float,
                         h: float) -> list[list[int]]:
    """Node-revisiting DFS.  For validation only."""
    LIMIT: int = 100_000
    walks: list[list[int]] = []
    exceeded = [False]

    def _dfs(node: int, labels: list[int], cost: float) -> None:
        if exceeded[0]:
            return
        if labels:
            walks.append(list(labels))
            if len(walks) > LIMIT:
                exceeded[0] = True
                return
        for _u, nbr, data in G.edges(node, data=True):
            lbl = int(data["label"])
            nc  = cost + lbl * h
            if nc <= c_walk:
                labels.append(lbl)
                _dfs(nbr, labels, nc)
                labels.pop()
                if exceeded[0]:
                    return

    for start in G.nodes():
        if exceeded[0]:
            break
        _dfs(start, [], 0.0)

    if exceeded[0]:
        raise ValueError(f"Walk explosion: {len(walks)} walks found.")
    return walks


def enumerate_walks_sufficient(G: nx.DiGraph, c_walk: float,
                                h: float, max_walks: int = 10_000,
                                ) -> tuple[list[list[int]], bool]:
    """Capped node-revisiting DFS.  For validation only.

    Returns (walks, is_complete).  is_complete=False when truncated.
    NOTE: This is NOT the primary stability certificate — use
    compute_max_cycle_mean() for a complete (necessary and sufficient) result.
    """
    walks: list[list[int]] = []
    truncated = [False]

    def _dfs(node: int, labels: list[int], cost: float) -> None:
        if truncated[0]:
            return
        if labels:
            walks.append(list(labels))
            if len(walks) >= max_walks:
                truncated[0] = True
                return
        for _u, nbr, data in G.edges(node, data=True):
            lbl = int(data["label"])
            nc  = cost + lbl * h
            if nc <= c_walk:
                labels.append(lbl)
                _dfs(nbr, labels, nc)
                labels.pop()
                if truncated[0]:
                    return

    for start in G.nodes():
        if truncated[0]:
            break
        _dfs(start, [], 0.0)

    return walks, not truncated[0]


# ---------------------------------------------------------------------------
# 7.  plot_graph
# ---------------------------------------------------------------------------

def plot_graph(G: nx.MultiDiGraph, save_path: str) -> None:
    """Save a figure of the WHRT automaton."""
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800,
                           node_color="#2166ac", alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={v: f"d={v}" for v in G.nodes()},
                            font_color="white", font_size=9)
    # Collect edge labels
    edge_labels: dict = {}
    for u, v, data in G.edges(data=True):
        key = (u, v)
        lbl = str(data["label"])
        edge_labels[key] = (edge_labels.get(key, "") +
                             ("," if key in edge_labels else "") + lbl)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="->", arrowsize=15, alpha=0.7,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=7)
    ax.set_title(f"WHRT Automaton  η=({config.M},{config.N})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
