"""Hodge invalid-loop detection & persistent avoidance (PLAN §12.4) — DESIGN MODULE, NOT YET RUN.

Detects action sequences that form REVERSIBLE LOOPS (return to a visited state with no net progress —
"s1→s2→s3→s4→s1 is invalid"), separates them from PROGRESS (gradient) and IRREVERSIBLE no-return EVENTS
(harmonic, the goals), and accumulates a PERSISTENT penalty so the policy avoids those loops in later rollouts.

The crux vs the plain `hodge.py` (which fills ALL triangles): here `B₂` is the **A∧A-oracle fill** —
a cycle is filled (→ its circulation becomes CURL = invalid) IFF it is reversible (every edge undoable, so the
loop can be walked back to its start). Irreversible cycles stay unfilled → their circulation is HARMONIC = a
no-return ratchet = the goal. Reversibility = contractibility.

Curvature mapping (user, PLAN §12.4) — the three components ARE the curvature primitives:
  grad = B₁ᵀφ   ≡  dA        the MEASURED forward change (progress); φ = value/dist-to-goal (wake_sleep SLEEP)
  curl = B₂ψ    ≡  A∧A        the MEASURED "across-and-back" reversible loops          (penalize, avoid)
  harm = x−g−c  ≡  no-return  the residual we CANNOT measure ("don't know how to reach back") — Hodge infers it
dA and A∧A pin down two components from direct measurement; Hodge reveals the third (the unreachable no-return).
A∧A is the SWITCH: A∧A < τ_rev fills the cell → curl; A∧A large/UNKNOWN leaves it unfilled → harmonic.

Pure linear algebra (cycle basis, B₂-from-cells, decomposition, invalid score, penalty) is implemented concretely
and is unit-testable offline. The per-edge reversibility flag it consumes comes from the multi-step A∧A already
built in `cycle_test.py` / `diagnose_lr.smae_pair` during WAKE. Nothing here calls the env or trains.

  (smoke tests live in PLAN §12.6; run only on user's go)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

# τ_rev: an edge transition s→s' is "reversible" iff its 1-step A∧A < this (curvature units). A loop is filled
# (→ curl/invalid) iff ALL its edges are reversible. Tune on the position-aware scale (nav A∧A ≈ 0.67, event ≈ 74).
TAU_REV = 1.0


# --------------------------------------------------------------------------- cycle basis (which loops exist)
def _adj(nodes, edges):
    nid = {n: i for i, n in enumerate(nodes)}
    g = defaultdict(list)
    for ei, (u, v) in enumerate(edges):
        g[nid[u]].append((nid[v], ei)); g[nid[v]].append((nid[u], ei))
    return nid, g


def cycle_basis(nodes, edges):
    """Fundamental cycle basis: a BFS spanning forest, then each NON-tree edge closes exactly one cycle
    (its endpoints' tree paths). Returns cycles as ordered node-index sequences [a, b, ..., a] (closed).
    dim = |E| − |V| + #components = the cycle-space dimension (curl ⊕ harmonic)."""
    nid, g = _adj(nodes, edges)
    M = len(nodes)
    parent = [-1] * M; pedge = [-1] * M; depth = [-1] * M; seen_tree = set()
    order = []
    for s in range(M):
        if depth[s] >= 0:
            continue
        depth[s] = 0; q = deque([s])
        while q:
            u = q.popleft(); order.append(u)
            for (w, ei) in g[u]:
                if depth[w] < 0:
                    depth[w] = depth[u] + 1; parent[w] = u; pedge[w] = ei; seen_tree.add(ei); q.append(w)

    def path_to_root(x):
        p = [x]
        while parent[p[-1]] >= 0:
            p.append(parent[p[-1]])
        return p

    cycles = []
    for ei, (u, v) in enumerate(edges):
        if ei in seen_tree:
            continue
        a, b = nid[u], nid[v]                       # non-tree edge a-b closes a loop
        pa, pb = path_to_root(a), path_to_root(b)
        sa, sb = set(pa), {}
        # find LCA: first node of pa that is on pb
        idxb = {n: k for k, n in enumerate(pb)}
        lca_k = next((k for k, n in enumerate(pa) if n in idxb), None)
        if lca_k is None:
            continue                                # different components (shouldn't happen for an internal edge)
        up = pa[:lca_k + 1]                          # a → ... → lca
        down = pb[:idxb[pa[lca_k]]]                  # b → ... → (just below lca)
        cyc = up + down[::-1] + [a]                  # a→…→lca→…→b→a   (closed)
        cycles.append(cyc)
    return cycles


# --------------------------------------------------------------------------- A∧A-oracle fill → B₂ cells
def reversible_fill(nodes, edges, reversible_edges, cycles=None, max_cells=4096):
    """Pick the filled 2-cells (PLAN §12.4-2): a cycle is filled IFF every edge in it is reversible
    (the loop can be walked back to its start). `reversible_edges` is a bool array over `edges` (edge e is
    reversible iff its 1-step A∧A < TAU_REV, computed during WAKE). Returns filled_cells: each a list of
    (edge_index, sign) giving the oriented boundary, ready for B₂.

    UNKNOWN reversibility (no observed return) must be encoded as reversible_edges[e] = False so the cycle is
    left UNFILLED (never mis-certified reversible — PLAN fix #3)."""
    eidx = {e: i for i, e in enumerate(edges)}
    cycles = cycle_basis(nodes, edges) if cycles is None else cycles
    rev = np.asarray(reversible_edges, bool)
    filled = []
    for cyc in cycles:
        cell = []
        ok = True
        for k in range(len(cyc) - 1):
            a, b = nodes[cyc[k]], nodes[cyc[k + 1]]
            e = (a, b) if a < b else (b, a)
            ei = eidx.get(e)
            if ei is None or not rev[ei]:           # missing edge or an irreversible/UNKNOWN edge → don't fill
                ok = False; break
            cell.append((ei, 1.0 if a < b else -1.0))
        if ok and cell:
            filled.append(cell)
            if len(filled) >= max_cells:
                break
    return filled


# --------------------------------------------------------------------------- decomposition with explicit B₂
def hodge_decompose_filled(nodes, edges, x, filled_cells):
    """Helmholtz–Hodge decomposition of oriented flow x with B₂ given by `filled_cells` (the A∧A fill), NOT by
    triangle-fill. Mirrors hodge.hodge_decompose's sparse solve. Returns grad/curl/harm + per-edge masses +
    energy fractions + the B₁B₂=0 check. (curl = circulation in the filled reversible cells = invalid loops.)"""
    nid = {n: i for i, n in enumerate(nodes)}
    M, E, F = len(nodes), len(edges), len(filled_cells)
    x = np.asarray(x, np.float64)

    # B₁ᵀ as `d0` (E×M): −1 at edge source, +1 at edge dest (hodge.py convention) → grad = colspace(d0)
    ei = np.repeat(np.arange(E), 2)
    ej = np.empty(2 * E, np.int64); ev = np.empty(2 * E, np.float64)
    for e, (i, j) in enumerate(edges):
        ej[2 * e] = nid[i]; ev[2 * e] = -1.0; ej[2 * e + 1] = nid[j]; ev[2 * e + 1] = 1.0
    d0 = sp.csr_matrix((ev, (ei, ej)), shape=(E, M))

    # B₂ᵀ as `d1` (F×E) from the filled cells: row f has the oriented boundary of cell f → curl = rowspace(d1)
    if F > 0:
        tr, tc, tv = [], [], []
        for f, cell in enumerate(filled_cells):
            for (e, s) in cell:
                tr.append(f); tc.append(e); tv.append(s)
        d1 = sp.csr_matrix((tv, (tr, tc)), shape=(F, E))
    else:
        d1 = None

    grad = d0 @ lsqr(d0, x)[0]
    curl = (d1.T @ lsqr(d1.T, x)[0]) if d1 is not None else np.zeros(E)
    harm = x - grad - curl

    b1b2_ok = (float(abs((d0.T @ d1.T)).max()) < 1e-8) if d1 is not None else True   # ‖B₁B₂‖≈0 (PLAN fix #3)
    tot = float(x @ x) + 1e-12
    return {
        "nodes": nodes, "edges": edges, "x": x, "grad": grad, "curl": curl, "harm": harm,
        "g_e": np.abs(grad), "c_e": np.abs(curl), "h_e": np.abs(harm),
        "e_grad": float(grad @ grad) / tot, "e_curl": float(curl @ curl) / tot, "e_harm": float(harm @ harm) / tot,
        "n_cells": F, "B1B2_ok": b1b2_ok,
    }


# --------------------------------------------------------------------------- invalid-action score (per edge)
def invalid_scores(dec, eps=1e-9):
    """invalid(e) = |curl(e)| / (|grad(e)|+|curl(e)|+|harm(e)|+ε) ∈ [0,1]. ≈1 on a reversible loop's edges
    (pure curl), ≈0 on a progress edge (gradient) or a no-return event edge (harmonic). PLAN §12.4-4."""
    g, c, h = dec["g_e"], dec["c_e"], dec["h_e"]
    return c / (g + c + h + eps)


# --------------------------------------------------------------------------- fast tabu: closed loops in a rollout
def detect_closed_loops(state_seq, curv, tol):
    """Fast tabu without a solve (PLAN §12.4-4): scan a rollout's state sequence for a revisit (s_t ≈ s_τ,
    τ<t under `curv` < tol) and return the minimal closed cycles [(τ, t), ...] as index spans. The caller
    checks reversibility of those edges and, if reversible, marks them invalid immediately. `curv(a,b)->float`
    is the position-aware curvature (diagnose_lr.smae_pair)."""
    loops = []
    for t in range(len(state_seq)):
        for tau in range(t - 2, -1, -1):                 # need ≥2 steps to be a loop
            if curv(state_seq[tau], state_seq[t]) < tol:
                loops.append((tau, t)); break            # nearest revisit = minimal cycle
    return loops


# --------------------------------------------------------------------------- persistent cross-rollout penalty
class LoopPenalty:
    """Accumulates P(cluster, action) over rollouts and shapes the next rollout's policy logits (PLAN §12.4-5):

        after a rollout:  P[c,a] += η · invalid(edge(c,a))
        next rollout:     logit(c,a) = base + α·alignGrad − β·P[c,a] + γ·alignHarm   ;  π = softmax_a(logit)

    `alignGrad` steers DOWN the progress potential φ toward the goal (the reverse curriculum), `−β·P` removes
    known reversible loops, `alignHarm` seeks no-return events. P persists, so a loop found curl once is
    suppressed in every later rollout — "avoid these actions again next time, and so on.\""""

    def __init__(self, eta=1.0, alpha=1.0, beta=2.0, gamma=1.0):
        self.P = defaultdict(float)
        self.eta, self.alpha, self.beta, self.gamma = eta, alpha, beta, gamma

    def update(self, edge_keys, inval):
        """edge_keys[e] = (cluster_id, action) for edge e; inval = invalid_scores(dec)."""
        for e, key in edge_keys.items():
            self.P[key] += self.eta * float(inval[e])

    def penalty(self, cluster_id, action):
        return self.P.get((cluster_id, action), 0.0)

    def shape_logits(self, cluster_id, base_logits, align_grad=None, align_harm=None):
        """base_logits: dict action→logit. align_grad/align_harm: dict action→[0,1] (optional). Returns
        softmax probabilities over the actions in base_logits (Σ_a π = 1)."""
        ag = align_grad or {}; ah = align_harm or {}
        z = {a: (base_logits[a] + self.alpha * ag.get(a, 0.0) + self.gamma * ah.get(a, 0.0)
                 - self.beta * self.penalty(cluster_id, a)) for a in base_logits}
        m = max(z.values()); ex = {a: np.exp(v - m) for a, v in z.items()}; s = sum(ex.values()) + 1e-12
        return {a: ex[a] / s for a in ex}


# --------------------------------------------------------------------------- end-to-end SLEEP triage (glue)
def sleep_triage(nodes, edges, x, reversible_edges):
    """One SLEEP pass: A∧A fill → decompose → (φ progress potential, invalid(e) per edge). The caller maps
    edges→(cluster,action) and feeds invalid into LoopPenalty.update; φ feeds the reverse-curriculum value.
    Returns dict with dec, invalid, and energy split for logging. (Pure; safe to unit-test offline.)"""
    filled = reversible_fill(nodes, edges, reversible_edges)
    dec = hodge_decompose_filled(nodes, edges, x, filled)
    return {"dec": dec, "invalid": invalid_scores(dec), "n_filled": len(filled),
            "energy": (dec["e_grad"], dec["e_curl"], dec["e_harm"]), "B1B2_ok": dec["B1B2_ok"]}
