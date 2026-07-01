"""Sparse click-candidate generation per region.

For every non-background region we emit up to ~16 candidate click points spread
across five kinds:

  region_center        argmax of the Euclidean distance transform (deepest
                       interior pixel). Stable for non-convex shapes where the
                       arithmetic centroid would fall outside the mask.
  centroid             the arithmetic mean of the region pixels, snapped onto
                       the nearest mask pixel.
  bbox_corner          each of the 4 bbox corners, snapped to the nearest mask
                       pixel. Useful for handles / corner-marked buttons.
  bbox_edge_midpoint   each of the 4 bbox edge midpoints, snapped to the mask.
  boundary             up to 8 mask-boundary pixels chosen by 45°-spaced
                       angular bins around the centroid. Silhouette samples.

For background_component regions we emit only ONE candidate (the distance
centre) so the background doesn't dominate later orbit grouping.

The output of this module is the input to Phase 1C (D4 canonical signatures).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .regions import Region, boundary_mask, region_index_from_id, region_mask


@dataclass
class Candidate:
    candidate_id: str
    region_id: str
    kind: str
    point: list[int]                     # [row, col] in grid coords
    score: float = 0.0                   # filled by Phase 1E
    class_key: str = ""                  # filled by Phase 1D ("C̃####" sequential id)
    local_signature: str = ""            # filled by Phase 1C — D4-orbit canonical κ
    reason: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- helpers

def _snap_to_mask(
    point: tuple[float, float],
    mask: np.ndarray,
    bbox: list[int],
) -> tuple[int, int]:
    """Return the mask pixel (global coords) closest to `point`.

    Distances are computed in the cropped bbox neighbourhood for speed.
    `point` may be a float and may lie outside the mask.
    """
    r0, c0, r1, c1 = bbox
    local = mask[r0:r1 + 1, c0:c1 + 1]
    ys, xs = np.where(local)
    if len(ys) == 0:                                 # defensive
        return int(round(point[0])), int(round(point[1]))
    pr = point[0] - r0
    pc = point[1] - c0
    d2 = (ys - pr) ** 2 + (xs - pc) ** 2
    i = int(np.argmin(d2))
    return int(ys[i] + r0), int(xs[i] + c0)


def _angular_boundary_samples(
    mask: np.ndarray,
    bbox: list[int],
    centroid: tuple[float, float],
    n_buckets: int = 8,
) -> list[tuple[int, int]]:
    """Pick `n_buckets` boundary pixels at 45°-spaced angular directions.

    For each bucket centre angle ``θ_k = -π + (k + 0.5) · 2π / n``, choose the
    boundary pixel whose direction-from-centroid is closest to ``θ_k`` in
    circular distance:

        d(α, β) = |((α − β + π) mod 2π) − π|              ∈ [0, π]

    This samples the silhouette of the region in equally-spaced compass
    directions; for an axis-aligned square you get all 4 corners + 4 edge
    midpoints, for a circle you get 8 evenly-spaced rim pixels.
    """
    r0, c0, r1, c1 = bbox
    local = mask[r0:r1 + 1, c0:c1 + 1]
    bdy = boundary_mask(local)
    ys, xs = np.where(bdy)
    if len(ys) == 0:
        return []

    cy = centroid[0] - r0
    cx = centroid[1] - c0
    # Note: arctan2(dy, dx) with row=y, col=x — atan2 returns (-π, π].
    angles = np.arctan2(ys - cy, xs - cx)

    samples: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    two_pi = 2.0 * math.pi
    for k in range(n_buckets):
        bc = -math.pi + (k + 0.5) * two_pi / n_buckets
        diff = np.abs(((angles - bc + math.pi) % two_pi) - math.pi)
        i = int(np.argmin(diff))
        p = (int(ys[i] + r0), int(xs[i] + c0))
        if p in seen:
            continue
        seen.add(p)
        samples.append(p)
    return samples


def _bbox_corner_targets(bbox: list[int]) -> list[tuple[float, float, str]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0, c0, "tl"), (r0, c1, "tr"),
        (r1, c0, "bl"), (r1, c1, "br"),
    ]


def _bbox_edge_targets(bbox: list[int]) -> list[tuple[float, float, str]]:
    r0, c0, r1, c1 = bbox
    mr = 0.5 * (r0 + r1)
    mc = 0.5 * (c0 + c1)
    return [
        (r0, mc, "top"), (r1, mc, "bottom"),
        (mr, c0, "left"), (mr, c1, "right"),
    ]


# --------------------------------------------------------------------------- main

def generate_candidates(
    regions: list[Region],
    label_map: np.ndarray,
    max_per_region: int = 16,
) -> list[Candidate]:
    """One sparse Candidate set covering every (non-tiny) region.

    Ordering inside `max_per_region`: region_center, centroid, 4 corners,
    4 edge midpoints, up to 8 boundary samples.  Duplicates (same physical
    pixel chosen by multiple kinds, common for tiny regions) are dropped — the
    first kind that lands on a pixel keeps it.
    """
    candidates: list[Candidate] = []
    next_id = 1

    def _emit(region_id: str, kind: str, point: tuple[int, int],
              reason: list[str], features: Optional[dict] = None) -> None:
        nonlocal next_id
        candidates.append(Candidate(
            candidate_id=f"C{next_id:04d}",
            region_id=region_id,
            kind=kind,
            point=[int(point[0]), int(point[1])],
            reason=reason,
            features=features or {},
        ))
        next_id += 1

    for region in regions:
        idx = region_index_from_id(region.region_id)
        mask = region_mask(idx, label_map)
        if not mask.any():
            continue

        # --- background components: a single "background" candidate -----------
        if region.kind == "background_component":
            dc = region.features.get("distance_center")
            if dc is not None and mask[dc[0], dc[1]]:
                _emit(
                    region.region_id,
                    "background",
                    (dc[0], dc[1]),
                    [f"background centre; dt-radius {region.features['distance_radius']:.2f}"],
                    {"distance_radius": region.features["distance_radius"]},
                )
            continue

        # --- object regions: assemble the sparse candidate list ---------------
        proposed: list[tuple[str, tuple[int, int], list[str], dict]] = []

        # 1. region_center = argmax distance_transform_edt
        dc = region.features.get("distance_center")
        if dc is not None and mask[dc[0], dc[1]]:
            proposed.append((
                "region_center",
                (dc[0], dc[1]),
                [f"dt argmax; r={region.features['distance_radius']:.2f}"],
                {"distance_radius": region.features["distance_radius"]},
            ))

        # 2. centroid snapped onto the mask
        cy, cx = region.centroid
        snap = _snap_to_mask((cy, cx), mask, region.bbox)
        proposed.append((
            "centroid",
            snap,
            [f"centroid ({cy:.1f},{cx:.1f}) snapped"],
            {"raw_centroid": [cy, cx]},
        ))

        # 3. bbox corners
        for r, c, tag in _bbox_corner_targets(region.bbox):
            p = _snap_to_mask((r, c), mask, region.bbox)
            proposed.append((
                "bbox_corner",
                p,
                [f"bbox {tag} corner snapped"],
                {"corner": tag},
            ))

        # 4. bbox edge midpoints
        for r, c, tag in _bbox_edge_targets(region.bbox):
            p = _snap_to_mask((r, c), mask, region.bbox)
            proposed.append((
                "bbox_edge_midpoint",
                p,
                [f"bbox {tag}-edge midpoint snapped"],
                {"edge": tag},
            ))

        # 5. boundary representatives (≤ 8)
        for p in _angular_boundary_samples(mask, region.bbox, (cy, cx)):
            proposed.append((
                "boundary",
                p,
                ["45°-bucket boundary sample"],
                {},
            ))

        # dedupe by physical pixel and cap at max_per_region
        seen: set[tuple[int, int]] = set()
        for kind, point, reason, feats in proposed:
            if point in seen:
                continue
            seen.add(point)
            _emit(region.region_id, kind, point, reason, feats)
            if len([c for c in candidates if c.region_id == region.region_id]) \
                    >= max_per_region:
                break

    return candidates
