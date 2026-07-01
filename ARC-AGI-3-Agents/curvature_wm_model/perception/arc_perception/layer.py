"""High-level perception API: frame -> symmetry-typed temporal graph features.

This is the single source of truth for the per-frame perception pipeline
(the same logic the CLI extractor uses). It turns a raw 64x64 colour grid into
the quotient graph (nodes grouped by symmetry type, typed relation edges) and the
temporal delta events (object/edge births, deaths, moves, mutations, transitions).

Notebook usage:
    from arc_perception import PerceptionLayer
    pl = PerceptionLayer()
    feats = pl.process_frame(frame)        # {'nodes': [...], 'edges': [...], 'deltas': [...]}
    pl.reset()                             # clear tracker state at an episode boundary

The PerceptionLayer is STATEFUL: it carries a GaugeTracker so deltas (births,
deaths, moves, resurrections, edge births/deaths, level transitions) are computed
relative to the previous frame. Call reset() when starting a new trajectory.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .core import GaugeTracker, assign_graph_orbits
from .core.wl import type_hash as compute_type_hash, area_bin
from .region_finder.image_ops import estimate_background_label, segment_grid
from .region_finder.regions import build_regions
from .region_finder.graph import build_region_graph

_BG_KINDS = ("background", "background_component", "noise")


def _as_grid(frame) -> np.ndarray:
    """Coerce a frame to a 2D int grid (drops leading singleton/stacked dims)."""
    arr = np.asarray(frame, dtype=np.int32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"frame must reduce to a 2D grid; got shape {np.asarray(frame).shape}")
    return arr


def perceive_regions(arr: np.ndarray, near_threshold: float = 3.0, min_area: int = 3):
    """Stateless front-end: grid -> (regions, edges, label_map, background_label).

    regions : list[Region]   objects with colour, area, centroid, stabilizer, features
    edges   : list[EdgeRecord] spatial-relation edges (source, target, relation)
    """
    bg = int(estimate_background_label(arr))
    raw, label_map = segment_grid(arr, background_label=bg, min_area=min_area)
    regions = build_regions(raw, label_map)
    G, edges = build_region_graph(regions, label_map, near_threshold=near_threshold)
    assign_graph_orbits(regions, G)
    # build_region_graph may omit isolates; ensure every region is a node
    for r in regions:
        if r.region_id not in G.nodes():
            G.add_node(r.region_id)
    return regions, edges, label_map, bg


def quotient_graph(regions, edges):
    """Group regions by symmetry type_hash -> (nodes, typed edges, rid->type map)."""
    qnodes: dict[str, dict] = {}
    rid_to_type: dict[str, str] = {}
    for r in regions:
        if r.kind in _BG_KINDS:
            continue
        th = compute_type_hash(r)
        rid_to_type[r.region_id] = th
        if th not in qnodes:
            qnodes[th] = {
                "type_hash": th,
                "colour": list(r.color) if r.color else [],
                "stab": int(r.features.get("object_stabilizer", 1)),
                "area_bin": area_bin(r.area),
                "count": 0,
                "kinds": set(),
            }
        qnodes[th]["count"] += 1
        qnodes[th]["kinds"].add(r.kind)
    for n in qnodes.values():
        n["kinds"] = sorted(n["kinds"])

    qedges: set[tuple[str, str, str]] = set()
    for e in edges:
        ta = rid_to_type.get(e.source)
        tb = rid_to_type.get(e.target)
        if ta and tb and ta != tb:
            qedges.add((min(ta, tb), max(ta, tb), e.relation))

    nodes = list(qnodes.values())
    edges_out = [{"source": a, "target": b, "relation": rel} for a, b, rel in qedges]
    return nodes, edges_out, rid_to_type


class PerceptionLayer:
    """Stateful frame -> graph-features perceiver.

    Parameters
    ----------
    near_threshold : float   distance threshold for 'near' relation edges (px)
    min_area       : int     minimum component area kept as an object
    graveyard_ttl  : int     frames a vanished object is remembered (for resurrection)
    history_limit  : int     GaugeTracker frame-history cap (-1 disables history;
                             0 = unlimited; >0 = ring buffer)
    """

    def __init__(self, near_threshold: float = 3.0, min_area: int = 3,
                 graveyard_ttl: int = 30, history_limit: int = -1):
        self.near_threshold = near_threshold
        self.min_area = min_area
        self._tracker_kwargs = dict(graveyard_ttl=graveyard_ttl, history_limit=history_limit)
        self.tracker = GaugeTracker(**self._tracker_kwargs)

    def reset(self) -> None:
        """Clear tracker state (call at a new trajectory/episode boundary)."""
        self.tracker.reset()

    def process_frame(self, frame, *, raw: bool = False) -> dict:
        """Perceive one frame and advance the temporal tracker.

        Returns a dict with:
          nodes  : list of quotient nodes {type_hash, colour, stab, area_bin, count, kinds}
          edges  : list of typed edges    {source, target, relation}
          deltas : list of events         {kind, pid, detail}   (kind in
                   birth|death|move|mutation|state_change|resurrect|
                   edge_birth|edge_death|transition)
        If raw=True, also returns the underlying `regions` (per-object detail:
        centroid, area, stabilizer, features), `assignments` (region_id->pid),
        `rid_to_type`, and `background` label — for building richer per-object features.
        """
        arr = _as_grid(frame)
        regions, edges, label_map, bg = perceive_regions(
            arr, self.near_threshold, self.min_area)
        assignments, deltas = self.tracker.update(regions, edges=edges)
        nodes, qedges, rid_to_type = quotient_graph(regions, edges)
        out = {
            "nodes": nodes,
            "edges": qedges,
            "deltas": [{"kind": d.kind, "pid": d.pid, "detail": d.detail} for d in deltas],
        }
        if raw:
            out["regions"] = regions
            out["assignments"] = assignments
            out["rid_to_type"] = rid_to_type
            out["background"] = bg
        return out

    def process_trajectory(self, frames: Iterable, *, raw: bool = False,
                           reset: bool = True) -> list[dict]:
        """Perceive a sequence of frames in order. Resets the tracker first by default."""
        if reset:
            self.reset()
        return [self.process_frame(f, raw=raw) for f in frames]

    @property
    def state(self) -> dict:
        """Current tracker summary: frame_count, active, graveyard, total_pids_issued."""
        return self.tracker.summary()
