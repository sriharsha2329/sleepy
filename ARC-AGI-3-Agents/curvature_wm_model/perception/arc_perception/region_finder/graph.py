"""Region adjacency graph + graph-context signatures.

Graph G = (V, E) where:
    V = the regions from Phase 1A
    E has three relation kinds:
        touches   - 4-neighbour pixel adjacency (computed from label_map)
        near      - bbox-distance ≤ threshold (default 2 cells)
        contains  - A's bbox encloses B's bbox AND area(A) > area(B)
    `near` edges carry a `direction` annotation (left_of / right_of / above /
    below) derived from centroid delta.

For each region we compute a depth-1 context signature:
    ctx₁(R) = sha1(own_attrs(R) || sorted({(relation_set, own_attrs(N))
                                              : N ∈ neighbours(R)}))

where own_attrs is a coarse-binned tuple — area is log₂-bucketed, fill ratio
is split into 5 bins, etc. — so the signature is robust to small numerical
noise but still discriminates obviously different regions.

A depth-2 signature adds the sorted multiset of neighbours' depth-1
signatures.  This is what makes "lookalike object next to a button" different
from "lookalike object alone".
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from .regions import Region


# --------------------------------------------------------------------------- edges

@dataclass
class EdgeRecord:
    source: str
    target: str
    relation: str
    features: dict


# --------------------------------------------------------------------------- coarse binning

def _area_bin(area: int) -> int:
    return 0 if area <= 0 else int(math.log2(area))


def _fill_bin(x: float) -> int:
    return min(4, max(0, int(x * 5)))


def _aspect_bin(x: float) -> int:
    if x <= 0:
        return 0
    return int(max(-3, min(3, round(math.log2(x)))))


def _pos_bin(p: float) -> int:
    """Map a normalised coordinate (0..1) into {0,1,2} = left/center/right."""
    return min(2, max(0, int(p * 3)))


def _own_attrs(region: Region) -> tuple:
    f = region.features
    return (
        region.kind,
        tuple(region.color),
        _area_bin(region.area),
        _fill_bin(f["fill_ratio"]),
        _aspect_bin(f["aspect_ratio"]),
        _pos_bin(f["norm_centroid"][0]),
        _pos_bin(f["norm_centroid"][1]),
        bool(f["touches_border"]),
    )


def _own_attrs_pos_invariant(region: Region) -> tuple:
    """Same coarse attrs as `_own_attrs` MINUS the 2 position bins.

    Used to compute the evidence-layer context signature (fix C1): cursor
    moves within the same coarse area shouldn't reset the learned evidence.
    """
    f = region.features
    return (
        region.kind,
        tuple(region.color),
        _area_bin(region.area),
        _fill_bin(f["fill_ratio"]),
        _aspect_bin(f["aspect_ratio"]),
        bool(f["touches_border"]),
    )


# --------------------------------------------------------------------------- geometry helpers

def _bbox_distance(b1: list[int], b2: list[int]) -> float:
    """Axis-aligned bbox-to-bbox distance (0 if they overlap)."""
    dy = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    dx = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return math.hypot(dy, dx)


def _bbox_contains(outer: list[int], inner: list[int]) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def _direction_label(src: Region, tgt: Region) -> str:
    dy = tgt.centroid[0] - src.centroid[0]
    dx = tgt.centroid[1] - src.centroid[1]
    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"


# --------------------------------------------------------------------------- touch edges

def _scan_touch_pairs(label_map: np.ndarray) -> set[tuple[int, int]]:
    """Pixel-level 4-adjacency on the label map; returns sorted index pairs."""
    h, w = label_map.shape
    pairs: set[tuple[int, int]] = set()
    # right neighbour: shift columns
    a = label_map[:, :-1]
    b = label_map[:, 1:]
    mask = (a != -1) & (b != -1) & (a != b)
    if mask.any():
        for av, bv in zip(a[mask], b[mask]):
            pairs.add((min(int(av), int(bv)), max(int(av), int(bv))))
    # bottom neighbour
    a = label_map[:-1, :]
    b = label_map[1:, :]
    mask = (a != -1) & (b != -1) & (a != b)
    if mask.any():
        for av, bv in zip(a[mask], b[mask]):
            pairs.add((min(int(av), int(bv)), max(int(av), int(bv))))
    return pairs


# --------------------------------------------------------------------------- main builder

def build_region_graph(
    regions: list[Region],
    label_map: np.ndarray,
    near_threshold: float = 2.0,
) -> tuple[nx.DiGraph, list[EdgeRecord]]:
    """Build G and a flat list of EdgeRecord for serialization.

    The graph is a DiGraph: touches/near edges are added both ways with the
    same `relation`; `contains` is one-way (outer -> inner).
    """
    G: nx.DiGraph = nx.DiGraph()
    by_id: dict[str, Region] = {r.region_id: r for r in regions}

    for r in regions:
        G.add_node(
            r.region_id,
            kind=r.kind,
            area=r.area,
            color=tuple(r.color),
            bbox=tuple(r.bbox),
            centroid=tuple(r.centroid),
        )

    edges: list[EdgeRecord] = []

    def _add_edge(a: str, b: str, rel: str, feats: dict | None = None) -> None:
        feats = feats or {}
        if G.has_edge(a, b):
            rels = G[a][b].get("relations", set())
            rels.add(rel)
            G[a][b]["relations"] = rels
        else:
            G.add_edge(a, b, relations={rel})
        edges.append(EdgeRecord(a, b, rel, feats))

    # 1. touches (symmetric)
    for ai, bi in _scan_touch_pairs(label_map):
        ra = f"R{ai + 1:04d}"
        rb = f"R{bi + 1:04d}"
        _add_edge(ra, rb, "touches")
        _add_edge(rb, ra, "touches")

    # 2. near (symmetric, with directional annotation per direction)
    for i, r1 in enumerate(regions):
        for r2 in regions[i + 1:]:
            d = _bbox_distance(r1.bbox, r2.bbox)
            if 0 < d <= near_threshold:
                dir12 = _direction_label(r1, r2)
                dir21 = _direction_label(r2, r1)
                _add_edge(r1.region_id, r2.region_id, "near",
                          {"distance": round(d, 3), "direction": dir12})
                _add_edge(r2.region_id, r1.region_id, "near",
                          {"distance": round(d, 3), "direction": dir21})

    # 3. contains (asymmetric)
    for r1 in regions:
        for r2 in regions:
            if r1.region_id == r2.region_id:
                continue
            if _bbox_contains(r1.bbox, r2.bbox) and r1.area > r2.area:
                _add_edge(r1.region_id, r2.region_id, "contains")

    return G, edges


# --------------------------------------------------------------------------- context signature

def region_context_signature(
    region: Region,
    by_id: dict[str, Region],
    G: nx.DiGraph,
    depth: int = 1,
) -> str:
    """Depth-1 ctx hash: own coarse attrs + sorted multiset of neighbour attrs."""
    own = _own_attrs(region)
    neighbours = []
    for nbr_id in G.successors(region.region_id):
        nbr = by_id.get(nbr_id)
        if nbr is None:
            continue
        rels = tuple(sorted(G[region.region_id][nbr_id].get("relations", set())))
        neighbours.append((rels, _own_attrs(nbr)))
    neighbours.sort()
    key = repr((own, tuple(neighbours)))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def region_context_signature_evidence(
    region: Region,
    by_id: dict[str, Region],
    G: nx.DiGraph,
) -> str:
    """Same shape as `region_context_signature` but uses position-invariant
    coarse attrs everywhere.  Used to key click evidence so a cursor moving
    1 cell doesn't rehash and reset what the agent has learned."""
    own = _own_attrs_pos_invariant(region)
    neighbours = []
    for nbr_id in G.successors(region.region_id):
        nbr = by_id.get(nbr_id)
        if nbr is None:
            continue
        rels = tuple(sorted(G[region.region_id][nbr_id].get("relations", set())))
        neighbours.append((rels, _own_attrs_pos_invariant(nbr)))
    neighbours.sort()
    key = repr((own, tuple(neighbours)))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def assign_context_signatures(
    regions: list[Region],
    G: nx.DiGraph,
    depth: int = 2,
) -> None:
    """Store ctx_signature + ctx_signature_evidence on each region (mutates)."""
    by_id = {r.region_id: r for r in regions}

    # depth-1 first
    d1 = {}
    for r in regions:
        sig = region_context_signature(r, by_id, G, depth=1)
        d1[r.region_id] = sig
        r.features["context_signature"] = sig

    # depth-2: own d1 sig + sorted d1 sigs of neighbours
    if depth >= 2:
        for r in regions:
            nbr_sigs = sorted(
                d1.get(n, "") for n in G.successors(r.region_id)
            )
            payload = f"{d1[r.region_id]}|{','.join(nbr_sigs)}"
            r.features["context_signature_d2"] = (
                hashlib.sha1(payload.encode()).hexdigest()[:12]
            )

    # Position-invariant variant for evidence layer (fix C1).  Depth-1 only —
    # the evidence layer doesn't need the d2 boost because the goal is to be
    # *coarser* than ctx_signature, not finer.
    for r in regions:
        r.features["context_signature_evidence"] = (
            region_context_signature_evidence(r, by_id, G)
        )


def graph_stats(G: nx.DiGraph, edges: list[EdgeRecord]) -> dict:
    """Quick counts used by the UI sidebar."""
    rel_counts: dict[str, int] = {}
    for e in edges:
        rel_counts[e.relation] = rel_counts.get(e.relation, 0) + 1
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "by_relation": rel_counts,
    }


def region_neighbour_summary(region_id: str, G: nx.DiGraph) -> list[dict]:
    """Compact per-neighbour summary for the hover tooltip."""
    out = []
    for nbr_id in G.successors(region_id):
        ed = G[region_id][nbr_id]
        out.append({
            "neighbour": nbr_id,
            "relations": sorted(ed.get("relations", set())),
        })
    out.sort(key=lambda d: d["neighbour"])
    return out
