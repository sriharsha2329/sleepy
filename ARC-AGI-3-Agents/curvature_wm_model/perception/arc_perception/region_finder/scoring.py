"""Per-candidate scoring — weighted feature combination.

Score formula:

    S(c) =   2.0·O(c)              — object indicator
           + 1.5·U(c)              — uniqueness  (1 / class_size)
           + 1.2·C(c)              — centre indicator
           + 1.0·J(c)              — junction (distinct labels in patch)
           + 0.4·H_patch(c)·C(c)   — patch |Stab|, gated to interior kinds
           + 0.5·H_obj(c)·C(c)     — object |Stab|, gated to interior kinds
           + 0.5·B(c)              — boundary indicator
           − 1.5·BG(c)             — background penalty
           − 1.0·tiny(c)           — tiny-region penalty

Every raw term lives in [0, 1] before weighting, so the weights have a clean
"this matters how much" reading.  H_patch and H_obj are *multiplied* by C(c)
(0 for non-interior kinds, 0.7 for centroid, 1.0 for region_center) so a
boundary or corner candidate gets no stabilizer credit even if its 7×7 patch
happens to look symmetric — that symmetry would be an artefact of the patch
hitting a flat edge, not a meaningful anchor signal.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


DEFAULT_WEIGHTS: dict[str, float] = {
    "O": 2.0, "U": 1.5, "C": 1.2, "J": 1.0,
    "H_patch": 0.4, "H_obj": 0.5, "B": 0.5,
    "BG": -1.5, "tiny": -1.0,
}

TINY_AREA = 4

OBJECT_KINDS = {
    "region_center", "centroid", "bbox_corner",
    "bbox_edge_midpoint", "boundary",
}
BOUNDARY_KIND = "boundary"
BACKGROUND_KIND = "background"

# Per-kind centre score in [0, 1].  Zero for non-centre kinds so the
# stabilizer gating (multiplication by C) zeros them out automatically.
CENTER_SCORE: dict[str, float] = {
    "region_center": 1.0,
    "centroid":      0.7,
}


# --------------------------------------------------------------------------- helpers

def _junction_score(point, label_map: np.ndarray, radius: int = 3) -> float:
    """Distinct region labels in the candidate's local label-patch, normalised.

      1 label  → 0.0   (pure interior of one region)
      2 labels → 0.25
      3 labels → 0.5   (T-junction)
      4 labels → 0.75
      5+ labels → 1.0  (busy multi-region junction)
    """
    h, w = label_map.shape
    r0, c0 = int(point[0]), int(point[1])
    r1, r2 = max(0, r0 - radius), min(h, r0 + radius + 1)
    c1, c2 = max(0, c0 - radius), min(w, c0 + radius + 1)
    patch = label_map[r1:r2, c1:c2]
    if patch.size == 0:
        return 0.0
    distinct = {int(v) for v in patch.flat if v != -1}
    return min(1.0, max(0.0, (len(distinct) - 1) / 4.0))


# --------------------------------------------------------------------------- main API

def _uniqueness_key(candidate, region) -> tuple:
    """Match the C3 evidence-hash equivalence: candidates that share this
    triple share evidence, so they should also share uniqueness mass.

    Pre-L3 we used Phase 1D's `class_key` (visual class), which inflated U
    for things like 4 identical buttons in 4 corners (each got U = 1.0
    because their visual class hashes differed by position).  Keying on
    `graph_orbit_id` instead aligns the prior with the evidence layer."""
    orbit_id = ""
    if region is not None:
        orbit_id = region.features.get("graph_orbit_id", "")
    return (candidate.kind, candidate.local_signature, orbit_id)


def score_candidates(
    candidates: Iterable,
    regions: Iterable,
    label_map: np.ndarray,
    weights: dict[str, float] | None = None,
) -> None:
    """In-place: set c.score and c.features['score_terms'] / ['score_reasons']."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    by_region = {r.region_id: r for r in regions}
    # Orbit-based uniqueness — aligned with the C3 evidence hash.
    orbit_sizes = Counter(
        _uniqueness_key(c, by_region.get(c.region_id)) for c in candidates
    )

    for c in candidates:
        region = by_region.get(c.region_id)
        if region is None:
            c.score = 0.0
            c.features["score_terms"] = {}
            c.features["score_reasons"] = []
            continue

        # Raw normalised terms in [0, 1].
        O = 1.0 if c.kind in OBJECT_KINDS else 0.0
        u_key = _uniqueness_key(c, region)
        cls_size = orbit_sizes.get(u_key, 1)         # structural orbit size
        U = 1.0 / max(1, cls_size)
        C = CENTER_SCORE.get(c.kind, 0.0)
        J = _junction_score(c.point, label_map)
        H_patch_raw = c.features.get("stabilizer_count", 1) / 8.0
        H_obj_raw   = region.features.get("object_stabilizer", 1) / 8.0
        # Gating: stabilizer credit only for interior kinds, scaled by C(c).
        H_patch = H_patch_raw * C
        H_obj   = H_obj_raw   * C
        B  = 1.0 if c.kind == BOUNDARY_KIND   else 0.0
        BG = 1.0 if c.kind == BACKGROUND_KIND else 0.0
        tiny = 1.0 if region.area < TINY_AREA else 0.0

        # Weighted contributions.
        terms = {
            "O":       w["O"]       * O,
            "U":       w["U"]       * U,
            "C":       w["C"]       * C,
            "J":       w["J"]       * J,
            "H_patch": w["H_patch"] * H_patch,
            "H_obj":   w["H_obj"]   * H_obj,
            "B":       w["B"]       * B,
            "BG":      w["BG"]      * BG,
            "tiny":    w["tiny"]    * tiny,
        }
        total = sum(terms.values())

        # Human-readable reasons (non-zero terms only).
        reasons: list[str] = []
        if O > 0:       reasons.append(f"+{terms['O']:.2f} object kind")
        if U > 0:       reasons.append(f"+{terms['U']:.2f} structural unique (orbit size {cls_size})")
        if C > 0:       reasons.append(f"+{terms['C']:.2f} centre kind ({c.kind})")
        if J > 0:       reasons.append(f"+{terms['J']:.2f} junction ({int(J*4)+1}-way)")
        if H_patch > 0: reasons.append(
            f"+{terms['H_patch']:.2f} patch |Stab| "
            f"{c.features.get('stabilizer_count',1)}/8")
        if H_obj > 0:   reasons.append(
            f"+{terms['H_obj']:.2f} object |Stab| "
            f"{region.features.get('object_stabilizer',1)}/8 "
            f"({region.features.get('symmetry_type','')})")
        if B > 0:       reasons.append(f"+{terms['B']:.2f} boundary kind")
        if BG > 0:      reasons.append(f"{terms['BG']:+.2f} background penalty")
        if tiny > 0:    reasons.append(
            f"{terms['tiny']:+.2f} tiny region (area {region.area})")

        c.score = round(total, 3)
        c.features["score_terms"]   = {k: round(v, 3) for k, v in terms.items() if v != 0}
        c.features["score_reasons"] = reasons


def select_representatives(candidates: Iterable) -> list:
    """One candidate per class_key — the highest-scoring member of each class."""
    best: dict[str, object] = {}
    for c in candidates:
        if not c.class_key:
            continue
        cur = best.get(c.class_key)
        if cur is None or c.score > cur.score:
            best[c.class_key] = c
    return sorted(best.values(), key=lambda c: -c.score)


def top_k_representatives(candidates: Iterable, k: int = 10) -> list:
    return select_representatives(candidates)[:k]
