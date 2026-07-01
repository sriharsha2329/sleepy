"""arc_perception — symmetry-typed temporal-graph perception for ARC-AGI-3 frames.

Quick start (e.g. in a Jupyter notebook):

    from arc_perception import PerceptionLayer, iter_recording_frames

    pl = PerceptionLayer()
    for fr in iter_recording_frames("…/human_runs/ar25/<uuid>.recording.jsonl", max_frames=300):
        feats = pl.process_frame(fr["frame"])     # {'nodes', 'edges', 'deltas'}
        ...

    # one-shot helpers
    from arc_perception import perceive_recording
    rows = perceive_recording("…/ar25/<uuid>.recording.jsonl", max_frames=300)

Public API:
    PerceptionLayer        stateful frame -> {nodes, edges, deltas} perceiver
    perceive_regions       stateless front-end: grid -> (regions, edges, label_map, bg)
    quotient_graph         regions/edges -> (nodes, typed edges, rid->type)
    perceive_recording     convenience: a recording path -> list of per-frame feature dicts
    iter_recording_frames  stream frames from a .recording.jsonl
    GaugeTracker, type_hash, assign_graph_orbits   low-level building blocks
"""
from __future__ import annotations

from .layer import PerceptionLayer, perceive_regions, quotient_graph
from .core import GaugeTracker, DeltaEvent, type_hash, assign_graph_orbits
from .io import (iter_recording_frames, recording_meta, list_recordings, action_id)

__all__ = [
    "PerceptionLayer", "perceive_regions", "quotient_graph", "perceive_recording",
    "iter_recording_frames", "recording_meta", "list_recordings", "action_id",
    "GaugeTracker", "DeltaEvent", "type_hash", "assign_graph_orbits",
]

__version__ = "0.1.0"


def perceive_recording(path, max_frames=None, *, raw: bool = False,
                       near_threshold: float = 3.0, min_area: int = 3) -> list[dict]:
    """Run perception over a whole recording -> list of per-frame feature dicts.

    Each row carries the frame's perception features plus light context:
      {t, frame_idx, action, click, level_idx, nodes, edges, deltas}
    (and regions/assignments when raw=True).
    """
    pl = PerceptionLayer(near_threshold=near_threshold, min_area=min_area)
    rows = []
    for t, fr in enumerate(iter_recording_frames(path, max_frames=max_frames)):
        feats = pl.process_frame(fr["frame"], raw=raw)
        row = {
            "t": t,
            "frame_idx": fr["frame_idx"],
            "action": fr["action"],
            "click": fr["click"],
            "level_idx": fr["levels"],
            **feats,
        }
        rows.append(row)
    return rows
