"""flow.py — the scene-state graph + oriented signed flow accumulated from rollout transitions (Hodge-Flow §3).

Nodes = scene states, deduped by the POSITION-AWARE curvature (`smae_pair` incl. Mahalanobis < TOL).
For each observed transition (u --a--> v) we store:
  - directed model[u][a] = (v, dA, aa)            (what the rollout uses to plan)
  - undirected oriented flow x_e = net signed dA   (+ if traversed lo->hi)   (what Hodge decomposes)
  - per-edge min A∧A = reversibility               (the triangle-fill / curl switch)

The oriented flow makes a balanced eddy (equal both ways) cancel to ~0 and a one-way event large — exactly the
input the Helmholtz–Hodge split needs. Pure data structure; the env exploration that fills it lives in rollout.py.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401  (wires config / data / transform_catalyst, read-only)

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

from collections import defaultdict

import numpy as np

from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair

TOL = 1.0       # two scene states are "the same" if position-aware curvature < TOL (nav ~6, event ~79 → safe)
TAU_REV = 1.0   # an edge is REVERSIBLE (curl-eligible) if its 1-step A∧A < TAU_REV (nav A∧A ~0.67, event ~74)


class FlowGraph:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reps = []                          # representative graph per node id
        self.snaps = []                         # a FastEnv snapshot to re-enter each node id
        self.frames = {}                        # node id -> raw frame array (for the rollout-graph viz)
        self.model = defaultdict(dict)          # u -> {a: (v, dA, aa)}
        self._flow = defaultdict(float)         # (lo,hi) -> net signed dA  (+ means net lo->hi)
        self._cnt = defaultdict(float)          # (lo,hi) -> traversal count
        self._aa = {}                           # (lo,hi) -> min A∧A seen on that edge (reversibility)

    # ---- nodes (deduped by position-aware curvature) ----
    def node_id(self, graph):
        for i, r in enumerate(self.reps):
            if smae_pair(graph, r, self.cfg) < TOL:
                return i
        return -1

    def get_or_add(self, graph, snap):
        i = self.node_id(graph)
        if i < 0:
            i = len(self.reps); self.reps.append(graph); self.snaps.append(snap)
        return i

    # ---- edges ----
    def add(self, u, a, v, dA, aa):
        self.model[u][a] = (v, dA, aa)
        if u == v:
            return
        (lo, hi), s = ((u, v), 1.0) if u < v else ((v, u), -1.0)
        self._flow[(lo, hi)] += s * dA
        self._cnt[(lo, hi)] += 1.0
        self._aa[(lo, hi)] = min(self._aa.get((lo, hi), float("inf")), aa)

    def edge_aa(self, u, v):
        lo, hi = (u, v) if u < v else (v, u)
        return self._aa.get((lo, hi), float("inf"))

    # ---- Hodge inputs ----
    def hodge_inputs(self):
        """(nodes, edges, x, reversible) for the Hodge solve. `reversible` gates the triangle fill (curl)."""
        edges = sorted(self._flow.keys())
        nodes = sorted({n for e in edges for n in e})
        x = np.array([self._flow[e] for e in edges], np.float64)
        reversible = np.array([self._aa.get(e, float("inf")) < TAU_REV for e in edges], bool)
        return nodes, edges, x, reversible

    def stats(self):
        return {"nodes": len(self.reps), "edges": len(self._flow),
                "reversible_edges": int(sum(1 for e in self._flow if self._aa.get(e, 1e9) < TAU_REV))}
