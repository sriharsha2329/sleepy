"""WL type identity — structural hashes (no cryptographic hashing).

All hashes are human-readable structured strings that preserve locality:
  same input → same string
  similar input → similar string (shared prefix/components)

1-WL (node colouring):
    Round 0: "c(12)_s8_a3" = colour 12, |Stab|=8, area_bin=3
    Round k+1: compose own key with sorted neighbour keys

2-WL (pair-space matching):
    Pair cost = type_match_penalty + position_distance
    Decomposed by round-0 type key → O(n·d²)
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import networkx as nx

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


# ---- area bucketing --------------------------------------------------------

def area_bin(area: int) -> int:
    if area <= 1:    return 0
    if area <= 3:    return 1
    if area <= 8:    return 2
    if area <= 20:   return 3
    if area <= 50:   return 4
    if area <= 150:  return 5
    if area <= 400:  return 6
    if area <= 1200: return 7
    return 8


# ---- round-0 node attributes -----------------------------------------------

def node_attrs(region) -> tuple:
    """(pixel_colour, |Stab|, area_bin)."""
    colour = tuple(region.color) if region.color else ()
    stab = int(region.features.get("object_stabilizer", 1))
    abin = area_bin(region.area)
    return (colour, stab, abin)


def type_hash(region) -> str:
    """Structured type key: 'c(12)_s8_a3'.  Human-readable, locality-preserving."""
    colour = tuple(region.color) if region.color else ()
    stab = int(region.features.get("object_stabilizer", 1))
    abin = area_bin(region.area)
    col_str = ".".join(str(c) for c in colour) if colour else "nil"
    return f"c({col_str})_s{stab}_a{abin}"


def _attrs_to_key(attrs: tuple) -> str:
    """Convert a node_attrs tuple to a structured string key."""
    colour, stab, abin = attrs
    col_str = ".".join(str(c) for c in colour) if colour else "nil"
    return f"c({col_str})_s{stab}_a{abin}"


# ---- 1-WL graph orbits (per-frame) ----------------------------------------

def _partition_sig(colors: dict[str, str]) -> tuple:
    groups: dict[str, list[str]] = {}
    for nid, col in colors.items():
        groups.setdefault(col, []).append(nid)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def compute_graph_orbits(
    G: nx.DiGraph,
    regions: Iterable,
    max_iterations: int = 10,
    attr_fn: Callable | None = None,
) -> dict[str, str]:
    """Return {region_id → orbit_key} by 1-WL to fixed point.

    Keys are structured strings, not cryptographic hashes.
    Round 0: type_hash string.
    Round k+1: "own_key|nbr1_key,nbr2_key,..."
    """
    if attr_fn is None:
        attr_fn = node_attrs

    # Round 0 — structured human-readable keys
    colors: dict[str, str] = {}
    for r in regions:
        colors[r.region_id] = _attrs_to_key(attr_fn(r))

    prev_sig = _partition_sig(colors)

    # Subsequent rounds — dense-rank to short IDs to avoid string-length
    # explosion.  Same WL semantics (identical structured key → identical ID).
    for round_idx in range(max_iterations):
        structured: dict[str, tuple] = {}
        for rid in G.nodes():
            own = colors.get(rid, "?")
            nbrs = tuple(sorted(colors.get(n, "?") for n in G.successors(rid)))
            structured[rid] = (own, nbrs)
        # Dense rank: identical tuples → identical ID
        unique_keys = sorted(set(structured.values()))
        key_to_id = {k: f"o{round_idx+1}_{i}" for i, k in enumerate(unique_keys)}
        new_colors = {rid: key_to_id[structured[rid]] for rid in structured}

        new_sig = _partition_sig(new_colors)
        if new_sig == prev_sig:
            break
        colors = new_colors
        prev_sig = new_sig

    return colors


def assign_graph_orbits(regions: Iterable, G: nx.DiGraph) -> int:
    orbits = compute_graph_orbits(G, regions)
    for r in regions:
        r.features["graph_orbit_id"] = orbits.get(r.region_id, "")
    return len(set(orbits.values()))


def assign_round1_orbits(regions: Iterable, G: nx.DiGraph,
                         attr_fn: Callable | None = None) -> int:
    """Round-1 WL only: own type + sorted multiset of immediate neighbour types.

    More discriminating than round-0 type_hash (incorporates 1-hop edges)
    but more stable than the fully propagated orbit (which can include
    long-range structure that flips when distant nodes move).

    Stores 'round1_orbit_id' on each region as a structured human-readable
    string: 'c(12)_s8_a4|c(0)_s4_a5,c(10)_s1_a7'.
    """
    if attr_fn is None:
        attr_fn = node_attrs
    round0 = {r.region_id: _attrs_to_key(attr_fn(r)) for r in regions}
    unique_round1: set[str] = set()
    for r in regions:
        own = round0.get(r.region_id, "?")
        nbrs = sorted(round0.get(n, "?") for n in G.successors(r.region_id))
        key = own + ("|" + ",".join(nbrs) if nbrs else "")
        r.features["round1_orbit_id"] = key
        unique_round1.add(key)
    return len(unique_round1)


# ---- 2-WL pair-space matching (temporal) -----------------------------------

def pair_match_decomposed(
    old_objects: list[dict],
    new_objects: list[dict],
    hash_mismatch_cost: float = 50.0,
    colour_mismatch_cost: float = 200.0,
    max_match_cost: float = 100.0,
) -> list[tuple[int | None, int | None, float]]:
    """Decomposed pair matching: group by type_hash, Hungarian within each."""

    old_by_hash: dict[str, list[int]] = {}
    for i, o in enumerate(old_objects):
        old_by_hash.setdefault(o["type_hash"], []).append(i)

    new_by_hash: dict[str, list[int]] = {}
    for j, n in enumerate(new_objects):
        new_by_hash.setdefault(n["type_hash"], []).append(j)

    matched_old: set[int] = set()
    matched_new: set[int] = set()
    results: list[tuple[int | None, int | None, float]] = []

    # Pass 1: exact type-hash match
    for h in set(old_by_hash.keys()) & set(new_by_hash.keys()):
        olds = old_by_hash[h]
        news = new_by_hash[h]
        d = max(len(olds), len(news))
        cost = np.full((d, d), max_match_cost + 1, dtype=np.float64)
        for ii, i in enumerate(olds):
            for jj, j in enumerate(news):
                dr = abs(old_objects[i]["centroid"][0] - new_objects[j]["centroid"][0])
                dc = abs(old_objects[i]["centroid"][1] - new_objects[j]["centroid"][1])
                cost[ii, jj] = dr + dc
        row_ind, col_ind = _solve(cost)
        for ii, jj in zip(row_ind, col_ind):
            if ii < len(olds) and jj < len(news) and cost[ii, jj] <= max_match_cost:
                matched_old.add(olds[ii])
                matched_new.add(news[jj])
                results.append((olds[ii], news[jj], float(cost[ii, jj])))

    # Pass 2: mutation match (same colour, different hash) on leftovers
    leftover_old = [i for i in range(len(old_objects)) if i not in matched_old]
    leftover_new = [j for j in range(len(new_objects)) if j not in matched_new]

    if leftover_old and leftover_new:
        old_by_col: dict[tuple, list[int]] = {}
        for i in leftover_old:
            old_by_col.setdefault(old_objects[i]["colour"], []).append(i)
        new_by_col: dict[tuple, list[int]] = {}
        for j in leftover_new:
            new_by_col.setdefault(new_objects[j]["colour"], []).append(j)

        for col in set(old_by_col.keys()) & set(new_by_col.keys()):
            olds = old_by_col[col]
            news = new_by_col[col]
            d = max(len(olds), len(news))
            cost = np.full((d, d), max_match_cost + 1, dtype=np.float64)
            for ii, i in enumerate(olds):
                for jj, j in enumerate(news):
                    dr = abs(old_objects[i]["centroid"][0] - new_objects[j]["centroid"][0])
                    dc = abs(old_objects[i]["centroid"][1] - new_objects[j]["centroid"][1])
                    cost[ii, jj] = hash_mismatch_cost + dr + dc
            row_ind, col_ind = _solve(cost)
            for ii, jj in zip(row_ind, col_ind):
                if ii < len(olds) and jj < len(news) and cost[ii, jj] <= max_match_cost:
                    matched_old.add(olds[ii])
                    matched_new.add(news[jj])
                    results.append((olds[ii], news[jj], float(cost[ii, jj])))

    # Births & deaths
    for i in range(len(old_objects)):
        if i not in matched_old:
            results.append((i, None, 0.0))
    for j in range(len(new_objects)):
        if j not in matched_new:
            results.append((None, j, 0.0))

    return results


def _solve(cost: np.ndarray):
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost)
    n, m = cost.shape
    rows, cols = [], []
    ur, uc = set(), set()
    for idx in np.argsort(cost.ravel()):
        i, j = divmod(int(idx), m)
        if i in ur or j in uc:
            continue
        rows.append(i); cols.append(j)
        ur.add(i); uc.add(j)
        if len(rows) >= min(n, m):
            break
    return rows, cols
