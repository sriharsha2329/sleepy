"""Object-level Hodge/Helmholtz decomposition of the affordance flow.

Ported from archive/brain_agent/hodge_object.py (its validated numerics: d0 gradient op, triangle-based curl,
harmonic remainder) with the §5.2 fix: the edge 1-chain is the **oriented signed flow** (antisymmetric), not a
non-negative magnitude. For each affordance traversal of object-state u->v with weight w, the reference edge
{min,max} accumulates +w if u<v else -w. A balanced eddy -> ~0 net flow; a one-way affordance -> large signed.

Decomposition (orthogonal):  x = x_grad (+) x_curl (+) x_harm
  x_grad = d0 @ lstsq(d0, x)        conservative / potential flow (net transport = PROGRESS)
  x_curl = d1^T @ lstsq(d1^T, x)    circulation around filled triangles (reversible eddies)
  x_harm = x - x_grad - x_curl      circulation around holes (irreversible RATCHET); beta1 = #holes

CURVATURE MAPPING (PLAN §12.4 — user):  the three components ARE the curvature primitives:
  x_grad  ≡  dA        the MEASURED forward change (net transport / progress); computed directly.
  x_curl  ≡  A∧A       the MEASURED "across-and-back": go across, A∧A says you can return (A∧A≈0 → reversible eddy).
  x_harm  ≡  no-return the residual we CANNOT measure directly — "we don't know how to reach the last state from
                       the initial (or get back)"; Hodge INFERS it as x − x_grad − x_curl.  beta1 = # such holes.
So dA and A∧A pin down two components from direct measurement; Hodge reveals the third (the unreachable no-return).
A∧A is the SWITCH: A∧A < τ_rev fills the cell → CURL (reversible); A∧A large/UNKNOWN leaves it unfilled → HARMONIC.

Per the settled mapping (PLAN §5.0/§5.4): KEEP gradient (progress) + harmonic (ratchet); DEDUP curl (busyness).
B2 here is the (validated) triangle fill; the **A∧A-oracle fill** that realizes the mapping above lives in
`hodge_loops.py` (PLAN §12.4): fill a cycle as curl iff every edge is reversible, else leave it harmonic.

BOUNDARY ADDITIVITY (PLAN §12.4-2b, user): a larger loop need NOT be its own 2-cell — if its CHORDS exist as
edges it triangulates, and curl(k-loop) = Σ curl(triangles), because shared chords cancel (opposite orientation).
E.g. (s1→s2→s3→s4→s1) = (s1→s2→s3→s1)+(s1→s3→s4→s1), chord s1↔s3 cancels. So this triangle fill ALREADY resolves
any TRIANGULABLE loop as curl; a loop stays HARMONIC only when NOT triangulable (chords missing / a triangle is a
no-return). To decide a suspected reversible k-loop, explore its chords; hodge_loops.cycle_basis is the chordless fallback.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.hodge      # synthetic smoke
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

MAX_EDGES = 6000        # above this the affordance graph is pathologically dense -> clustering granularity is
MAX_TRIS = 300_000      # wrong (no bimodal valley). Fail LOUD rather than OOM (review: dense d1 hit 6.4 GB).
BETA1_MAX_E = 1500      # exact beta1 needs a dense E x E eigendecomp; only do it when small.


def build_complex(op, oc, weight=None):
    """Affordance graph from per-slot object-state ids. op/oc [T,N] (-1 dead). weight [T,N] (default = count).

    Returns: nodes (sorted active object-state ids), edges (list of (u,v), u<v), x (oriented signed flow [E]),
    and cnt (per-edge total traversals, for diagnostics)."""
    T, N = op.shape
    if weight is None:
        weight = np.ones_like(op, dtype=np.float32)
    flow = defaultdict(float)   # (u,v) u<v -> net signed flow (+ = net u->v)
    cnt = defaultdict(float)
    for t in range(T):
        for i in range(N):
            u, v = int(op[t, i]), int(oc[t, i])
            if u < 0 or v < 0 or u == v:
                continue
            w = float(weight[t, i])
            (a, b), s = ((u, v), 1.0) if u < v else ((v, u), -1.0)
            flow[(a, b)] += s * w
            cnt[(a, b)] += w
    edges = sorted(flow.keys())
    nodes = sorted({n for e in edges for n in e})
    x = np.array([flow[e] for e in edges], np.float64)
    return nodes, edges, x, dict(cnt)


def hodge_decompose(nodes, edges, x, beta1_max_e=BETA1_MAX_E, reversible_edges=None):
    """Hodge decomposition of oriented flow x over (nodes, edges) — SPARSE (review: dense d1 OOMs on a
    saturated affordance graph). Returns the three components, energy fractions, per-edge magnitudes, beta1
    (exact only when small, else None with n_cycles = dim of the curl+harmonic cycle space).

    A∧A SWITCH (curvature mapping, PLAN §12.4): if `reversible_edges` (bool array over `edges`) is given, a
    triangle is filled into B₂ — so its circulation becomes CURL ≡ A∧A-reversible — ONLY if all 3 of its edges
    are reversible (A∧A < τ_rev). A triangle touching an irreversible/UNKNOWN edge is LEFT UNFILLED, so its
    circulation stays in the HARMONIC remainder ≡ no-return. `reversible_edges=None` ⇒ fill all triangles
    (legacy). General arbitrary-cycle A∧A fill (e.g. a 4-loop) lives in `hodge_loops.py`."""
    rev = None if reversible_edges is None else np.asarray(reversible_edges, bool)
    nid = {n: i for i, n in enumerate(nodes)}
    M, E = len(nodes), len(edges)
    if E > MAX_EDGES:
        raise ValueError(f"affordance graph too dense: E={E} > MAX_EDGES={MAX_EDGES}. Clustering granularity is "
                         f"likely wrong (no bimodal valley -> every state distinct). Reduce M / fix correspondence.")
    eidx = {e: i for i, e in enumerate(edges)}

    # sparse d0 (E x M): gradient operator (node potential -> edge flow)
    ei = np.repeat(np.arange(E), 2)
    ej = np.empty(2 * E, np.int64); ev = np.empty(2 * E, np.float64)
    for e, (i, j) in enumerate(edges):
        ej[2 * e] = nid[i]; ev[2 * e] = -1.0; ej[2 * e + 1] = nid[j]; ev[2 * e + 1] = 1.0
    d0 = sp.csr_matrix((ev, (ei, ej)), shape=(E, M))

    # filled triangles -> curl (B2)
    nbr = defaultdict(set)
    for (i, j) in edges:
        nbr[i].add(j); nbr[j].add(i)
    tris = []
    for (i, j) in edges:
        for l in (nbr[i] & nbr[j]):
            if l > j:
                if rev is not None and not (rev[eidx[(i, j)]] and rev[eidx[(j, l)]] and rev[eidx[(i, l)]]):
                    continue                       # A∧A SWITCH: a cell touching an irreversible/UNKNOWN edge is
                tris.append((i, j, l))             # NOT filled -> its circulation stays HARMONIC (no-return),
                if len(tris) > MAX_TRIS:           # only A∧A-reversible cells become CURL.
                    raise ValueError(f"too many filled triangles (> {MAX_TRIS}); affordance graph too dense.")
    Tn = len(tris)
    if Tn > 0:
        tr, tc, tv = [], [], []
        for t, (i, j, l) in enumerate(tris):
            tr += [t, t, t]; tc += [eidx[(i, j)], eidx[(j, l)], eidx[(i, l)]]; tv += [1.0, 1.0, -1.0]
        d1 = sp.csr_matrix((tv, (tr, tc)), shape=(Tn, E))
    else:
        d1 = None

    grad = d0 @ lsqr(d0, x)[0]                               # projection onto gradient space (sparse least sq)
    curl = (d1.T @ lsqr(d1.T, x)[0]) if d1 is not None else np.zeros(E)
    harm = x - grad - curl

    # diagnostics: connected components -> cycle-space dim; exact beta1 only when small enough
    A = sp.coo_matrix((np.ones(E), ([nid[i] for i, j in edges], [nid[j] for i, j in edges])), shape=(M, M))
    ncomp = connected_components(A, directed=False)[0]
    n_cycles = E - M + ncomp                                 # dim(curl (+) harmonic)
    if E <= beta1_max_e:
        L1 = (d0 @ d0.T) + (d1.T @ d1 if d1 is not None else sp.csr_matrix((E, E)))
        beta1 = int((np.linalg.eigvalsh(L1.toarray()) < 1e-7).sum())
    else:
        beta1 = None
    b1b2_ok = (float(abs((d0.T @ d1.T)).max()) < 1e-8) if d1 is not None else True

    tot = float(x @ x) + 1e-12
    return {
        "nodes": nodes, "edges": edges, "x": x,
        "grad": grad, "curl": curl, "harm": harm,
        "e_grad": float(grad @ grad) / tot, "e_curl": float(curl @ curl) / tot, "e_harm": float(harm @ harm) / tot,
        "g_e": np.abs(grad), "c_e": np.abs(curl), "h_e": np.abs(harm),
        "beta1": beta1, "n_cycles": int(n_cycles), "n_tris": Tn, "B1B2_ok": b1b2_ok,
    }


def node_mass(component, edges, nodes):
    """Lift a per-edge magnitude to per-object-state mass (sum of |flow| over incident edges)."""
    nid = {n: i for i, n in enumerate(nodes)}
    nm = np.zeros(len(nodes))
    for e, (i, j) in enumerate(edges):
        nm[nid[i]] += abs(component[e]); nm[nid[j]] += abs(component[e])
    return nm


# ----------------------------------------------------------------------------- synthetic smoke
def _decomp_from_edges(edges, x):
    nodes = sorted({n for e in edges for n in e})
    return hodge_decompose(nodes, edges, np.array(x, np.float64))


def _smoke():
    P = []
    # (1) square cycle (no diagonal) with circulation -> pure HARMONIC, beta1=1, curl=0
    sq = _decomp_from_edges([(0, 1), (1, 2), (2, 3), (0, 3)], [1, 1, 1, -1])  # 0->1->2->3->0
    P.append(("square: harmonic-dominant", sq["e_harm"] > 0.95))
    P.append(("square: curl ~ 0 (no triangles)", sq["e_curl"] < 1e-6))
    P.append(("square: beta1 == 1 (one hole)", sq["beta1"] == 1))

    # (2) filled triangle with circulation -> pure CURL, beta1=0
    tri = _decomp_from_edges([(0, 1), (0, 2), (1, 2)], [1, -1, 1])            # 0->1->2->0
    P.append(("triangle: curl-dominant", tri["e_curl"] > 0.95))
    P.append(("triangle: harm ~ 0 (filled)", tri["e_harm"] < 1e-6))
    P.append(("triangle: beta1 == 0 (no hole)", tri["beta1"] == 0))
    P.append(("triangle: B1 B2 == 0", tri["B1B2_ok"]))

    # (3) bridge (tree edge) net transport -> pure GRADIENT
    br = _decomp_from_edges([(0, 1), (1, 2)], [1, 1])                          # 0->1->2 chain
    P.append(("chain: gradient-dominant", br["e_grad"] > 0.99))
    P.append(("chain: beta1 == 0", br["beta1"] == 0))

    # (4) build_complex makes an oriented (antisymmetric) flow: balanced eddy cancels
    op = np.array([[0, -1], [1, -1]]); oc = np.array([[1, -1], [0, -1]])      # 0->1 and 1->0 once each
    nodes, edges, x, cnt = build_complex(op, oc)
    P.append(("balanced two-way traversal -> net flow ~ 0", abs(x[0]) < 1e-9 and cnt[edges[0]] == 2.0))

    ok = all(v for _, v in P)
    print("HODGE SMOKE (oriented flow; grad/curl/harm):")
    for n, v in P:
        print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    _smoke()
