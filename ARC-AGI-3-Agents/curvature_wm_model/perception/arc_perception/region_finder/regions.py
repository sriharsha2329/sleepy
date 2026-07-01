"""Region dataclass + per-region geometric features.

Inputs come from image_ops.segment_components; this module enriches each raw
region dict with shape statistics (fill ratio, compactness, distance-transform
center, boundary pixels).  No group-theory work happens here yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .object_stabilizer import compute_object_stabilizer
from .symmetry import canonical_shape_hash


@dataclass
class Region:
    region_id: str
    kind: str                       # "object" | "background_component" | "noise"
    color: Optional[list[int]]
    area: int
    bbox: list[int]                 # [r_min, c_min, r_max, c_max]
    centroid: list[float]           # [row, col]
    pixels_count: int
    parent_color_label: Optional[str]
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ----- mask helpers -----

def region_mask(region_index: int, label_map: np.ndarray) -> np.ndarray:
    return label_map == region_index


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Single-pixel boundary: pixels in mask whose 4-neighbours leave the mask."""
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def distance_center(mask: np.ndarray) -> tuple[list[int], float]:
    """Pixel furthest from the mask boundary; more stable than centroid."""
    dt = distance_transform_edt(mask)
    idx = int(np.argmax(dt))
    r, c = np.unravel_index(idx, mask.shape)
    return [int(r), int(c)], float(dt[r, c])


# ----- region construction + feature pass -----

def _crop(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0:r1 + 1, c0:c1 + 1]


def build_regions(raw: list[dict], label_map: np.ndarray) -> list[Region]:
    """Turn segment_components' raw dicts into full Region objects with features."""
    h, w = label_map.shape
    regions: list[Region] = []

    for raw_region in raw:
        idx = raw_region["_index"]
        bbox = raw_region["bbox"]
        bbox_h = bbox[2] - bbox[0] + 1
        bbox_w = bbox[3] - bbox[1] + 1

        mask = region_mask(idx, label_map)
        local_mask = _crop(mask, bbox)
        local_boundary = boundary_mask(local_mask)
        boundary_count = int(local_boundary.sum())
        interior_count = int(raw_region["area"] - boundary_count)

        dc, dc_radius = distance_center(local_mask)
        # back to global coords
        dc_global = [dc[0] + bbox[0], dc[1] + bbox[1]]

        # Object-level D4 stabilizer (Level 2): |Stab| of the mask itself.
        obj_stab, sym_type = compute_object_stabilizer(local_mask)
        # Canonical D4-mask hash — the LOSSLESS shape fingerprint.
        # Two regions whose binary masks are equivalent under ANY D4
        # transform (rotation/reflection) share this hash.  Captures
        # all stabilizer subgroups (trivial, mirror-*, Klein-V/D,
        # C2/C4-rot, D4) — not just |Stab|=8 shapes.
        shape_hash = canonical_shape_hash(local_mask)

        bbox_area = bbox_h * bbox_w
        fill_ratio = float(raw_region["area"] / bbox_area) if bbox_area else 0.0
        # Compactness ≈ area / perimeter² (clamped against zero perimeter).
        compactness = (
            float(raw_region["area"] / (boundary_count * boundary_count))
            if boundary_count > 0 else 0.0
        )
        aspect = float(bbox_h / bbox_w) if bbox_w else 1.0

        cy, cx = raw_region["centroid"]
        features = {
            "bbox_height": bbox_h,
            "bbox_width": bbox_w,
            "aspect_ratio": aspect,
            "fill_ratio": fill_ratio,
            "compactness": compactness,
            "boundary_pixels": boundary_count,
            "interior_pixels": interior_count,
            "distance_center": dc_global,
            "distance_radius": dc_radius,
            "touches_border": raw_region["touches_border"],
            "relative_area": float(raw_region["area"] / (h * w)),
            "norm_centroid": [cy / h, cx / w],
            "object_stabilizer": obj_stab,
            "symmetry_type": sym_type,
            "canonical_shape_hash": shape_hash,
        }

        color = raw_region["color"]
        color_label = ".".join(str(c) for c in color)

        regions.append(Region(
            region_id=f"R{idx + 1:04d}",
            kind=raw_region["kind"],
            color=list(color),
            area=int(raw_region["area"]),
            bbox=list(bbox),
            centroid=[float(cy), float(cx)],
            pixels_count=int(raw_region["area"]),
            parent_color_label=color_label,
            features=features,
        ))

    return regions


def region_index_from_id(region_id: str) -> int:
    """'R0007' -> 6 (0-based index into the regions list)."""
    return int(region_id[1:]) - 1

