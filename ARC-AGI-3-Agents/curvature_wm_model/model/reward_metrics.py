"""reward_metrics.py — the canonical REWARD / INTENSITY / EDGE-WEIGHT computation used to TRAIN the value & Q
heads (and to draw the metrics graphs). Lives in the model folder so the training-time signal is self-contained.

State identity is the INPUT-GRAPH hash (perception node attrs + centroids + Mahalanobis, HUD dropped) — NOT the
latent hash. Everything here (faces, intensity, degree, reward) is keyed by that input hash via the `edges` dict
the caller passes in; the trunk latent is only the head's feature input, never the identity.

Per directed transition  s --a--> s'   (β=10):
    r = dA + β·A∧A − 10·(intensity(s') − intensity(s)) − 10·intensity(s') − deg(s')
  · dA   = smae_pair(s, s')          semantic + Mahalanobis-position change (HUD excluded) — computed by caller.
  · A∧A  = no-return holonomy        model-straight 0 unless an event (z(dA)>1); brute-forced by the caller.
  · intensity(h) = #reversible(green) faces containing h  +  0.5 if h is on a one-way (unidirectional) edge.
                   ONLY one-way arrows get the 0.5 — NOT every member/vertex of a charged face.
  · deg(s') = number of discovered UNDIRECTED edges attached to s' (the "edge weight" term, coefficient 1).
"""
from __future__ import annotations

import networkx as nx

BETA = 10.0
INTENSITY_W = 10.0          # weight on the Δintensity and the intensity(s') penalties
EDGE_W = 1.0                # weight on the edge-count (degree) penalty
LEVELUP_BONUS = 100.0       # flat bonus when the transition completes a level (level-up) — the goal signal


def strip_hud(g, regions, tol=0):
    """Node-based HUD removal (whole pixels, no cy/cx ratios). `regions` are HUD-bar pixel bboxes
    [ymin,xmin,ymax,xmax] from hud_node_bboxes. Drop every node whose pixel bbox lies INSIDE a bar's bbox — the
    bar itself plus the fill / progress-segments / counters within it — and edges touching them. Returns a
    HUD-free graph used uniformly for identity / dA / A∧A / encoding."""
    if not regions:
        return g
    hud, keep = set(), []
    for n in g["nodes"]:
        by0, bx0, by1, bx1 = (int(v) for v in n.get("bbox", (0, 0, 0, 0)))
        if any(B0 <= by0 and by1 <= B1 and C0 <= bx0 and bx1 <= C1 for (B0, C0, B1, C1) in regions):
            hud.add(n.get("pid"))
        else:
            keep.append(n)
    if not hud:
        return g
    edges = [e for e in g.get("edges", []) if e.get("src") not in hud and e.get("dst") not in hud]
    return {"nodes": keep, "edges": edges, "deltas": g.get("deltas", [])}


def _order_face(graph, C):
    """Cyclic node order of a minimal face C (for polygon drawing). Minimal cycles are chordless so the induced
    subgraph IS the cycle -> cycle_basis returns it ordered; fall back to the raw set on the rare chorded case."""
    for cc in nx.cycle_basis(graph.subgraph(C)):
        if len(cc) == len(C):
            return cc
    return list(C)


def faces_intensity(edges, nodes):
    """From the discovered directed edge dict {(u_hash, v_hash): ...}, classify minimal faces and score per-node
    intensity. Returns {green, charged, rev, oneway_nodes, intensity}.

      reversible (green) = BIDIRECTIONAL edge (both directions discovered).
      charged           = minimal face of the full graph containing a one-way edge (a directed cycle).
      intensity(h)      = #green faces containing h  +  (0.5 if h is on a one-way edge).   [corrected rule]
    """
    de = dict(edges)
    und = set((min(u, v), max(u, v)) for (u, v) in de)
    rev = set(k for k in und if (k[0], k[1]) in de and (k[1], k[0]) in de)            # reversible = bidirectional
    Hrev = nx.Graph(); Hrev.add_edges_from(rev)
    green = [_order_face(Hrev, c) for c in nx.minimum_cycle_basis(Hrev) if len(c) >= 3] if Hrev.number_of_edges() else []
    Hall = nx.Graph(); Hall.add_edges_from(und)
    charged = []
    for c in (nx.minimum_cycle_basis(Hall) if Hall.number_of_edges() else []):
        if len(c) < 3:
            continue
        cyc = _order_face(Hall, c)
        ed = [(min(cyc[i], cyc[(i + 1) % len(cyc)]), max(cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))]
        if any(e not in rev for e in ed):                                             # has a one-way edge -> charged
            charged.append(cyc)
    oneway_nodes = set()
    for (a, b) in (und - rev):                                                        # endpoints of one-way arrows
        oneway_nodes.add(a); oneway_nodes.add(b)
    intensity = {h: sum(1 for cyc in green if h in cyc) + (0.5 if h in oneway_nodes else 0.0) for h in nodes}
    return {"green": green, "charged": charged, "rev": rev, "oneway_nodes": oneway_nodes, "intensity": intensity}


def undirected_degree(edges, h):
    """Number of discovered UNDIRECTED edges attached to state h (the edge-weight reward term)."""
    und = set((min(u, v), max(u, v)) for (u, v) in edges)
    return sum(1 for e in und if h in e)


def transition_reward(dA, aa, di, ds, deg, up=False, beta=BETA):
    """r = dA + β·A∧A − 10·(intensity(s')−intensity(s)) − 10·intensity(s') − deg(s')  (+100 on level-up)."""
    r = dA + beta * aa - INTENSITY_W * (di - ds) - INTENSITY_W * di - EDGE_W * deg
    return r + LEVELUP_BONUS if up else r
