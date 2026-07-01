"""Perception layer — D4-canonical graph representation + WL type identity.

Pipeline:
  F_t → G_t → Ḡ_t → h_t(v) → persistent IDs → ā_t

Modules:
  * wl        — 1-WL graph-orbit refinement (structural type identity)
  * canon     — D4 graph canonicalization + g_t* storage
  * tracker   — hash+position temporal matching (persistent IDs)
"""
from .wl import assign_graph_orbits, compute_graph_orbits, node_attrs, type_hash
from .tracker import GaugeTracker, DeltaEvent

__all__ = [
    "assign_graph_orbits", "compute_graph_orbits", "node_attrs", "type_hash",
    "GaugeTracker", "DeltaEvent",
]
