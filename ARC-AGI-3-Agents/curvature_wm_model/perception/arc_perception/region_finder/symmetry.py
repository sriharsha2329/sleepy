"""D4 group symmetry + canonical patch signatures.

The dihedral group D4 has eight elements — four rotations and four reflections
of a square — which is exactly the symmetry group of an axis-aligned square
patch.  Two candidate points whose local patches are equivalent under D4
should get the *same* canonical signature; this lets Phase 1D group them as
one orbit.

Canonical signature, as defined in the spec:

    κ(p) = min_{g ∈ D4} Encode(canonicalize(g · Patch(I, p)))

Where:
  - Patch(I, p) is a (2r+1) × (2r+1) crop centred at p, padded with a sentinel
    value (PAD_VALUE) when p is near the image border.
  - `canonicalize` relabels values by their row-major first occurrence so the
    *shape* of the pattern matters, not the literal colors.
  - Encode is sha1 truncated to 12 hex digits.

Orbit-stabilizer theorem: |D4| / |Stab(P)| = |Orbit(P)|, so the stabilizer
count tells us how symmetric a patch is (1 = fully asymmetric, 8 = D4-invariant).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

PAD_VALUE = -1
_DEFAULT_RADIUS = 3                            # 7x7 patches on 64x64 grids


# --------------------------------------------------------------------------- patch

def extract_patch(
    grid: np.ndarray,
    point: tuple[int, int] | list[int],
    radius: int = _DEFAULT_RADIUS,
    pad_value: int = PAD_VALUE,
) -> np.ndarray:
    """Square (2r+1)×(2r+1) crop centred at `point`, padded out-of-bounds.

    `grid` may be the colour grid (palette indices) or the region label_map
    (region indices, -1 = noise). Either way values are integer.
    """
    h, w = grid.shape
    r0, c0 = int(point[0]), int(point[1])
    diam = 2 * radius + 1
    out = np.full((diam, diam), pad_value, dtype=grid.dtype)

    r1, r2 = r0 - radius, r0 + radius + 1
    c1, c2 = c0 - radius, c0 + radius + 1
    src_r1, src_r2 = max(0, r1), min(h, r2)
    src_c1, src_c2 = max(0, c1), min(w, c2)
    if src_r1 >= src_r2 or src_c1 >= src_c2:
        return out
    dst_r1 = src_r1 - r1
    dst_c1 = src_c1 - c1
    out[dst_r1:dst_r1 + (src_r2 - src_r1),
        dst_c1:dst_c1 + (src_c2 - src_c1)] = grid[src_r1:src_r2, src_c1:src_c2]
    return out


def patch_radius_for(grid_shape: tuple[int, int]) -> int:
    h, w = grid_shape
    return int(max(3, min(15, round(min(h, w) * 0.03))))


# --------------------------------------------------------------------------- D4 group

def d4_transforms(patch: np.ndarray) -> list[np.ndarray]:
    """All 8 elements of D4 applied to `patch`.

    Order:  e, r, r², r³, s, sr, sr², sr³
    where r = 90° CCW rotation and s = horizontal flip (flipud).
    """
    rotations = [
        patch,
        np.rot90(patch, 1),
        np.rot90(patch, 2),
        np.rot90(patch, 3),
    ]
    flipped = np.flipud(patch)
    reflections = [
        flipped,
        np.rot90(flipped, 1),
        np.rot90(flipped, 2),
        np.rot90(flipped, 3),
    ]
    return rotations + reflections


# --------------------------------------------------------------------------- color canonicalization

def canonicalize_patch_colors(patch: np.ndarray) -> np.ndarray:
    """Relabel values by row-major first-occurrence order.

    Examples:
        [[5,5,3],[3,5,2]]  ->  [[0,0,1],[1,0,2]]
        [[2,2,8],[8,2,3]]  ->  [[0,0,1],[1,0,2]]

    The two patches above differ in literal colors but share the same
    *pattern*, so canonicalization maps them to the same integer grid.  This
    is what makes the signature color-invariant.
    """
    seen: dict[int, int] = {}
    out = np.empty_like(patch, dtype=np.int8)
    flat_in = patch.ravel()
    flat_out = out.ravel()
    next_id = 0
    for i, raw in enumerate(flat_in):
        v = int(raw)
        if v not in seen:
            seen[v] = next_id
            next_id += 1
        flat_out[i] = seen[v]
    return out


# --------------------------------------------------------------------------- hash

def encode_patch(patch: np.ndarray) -> str:
    """sha1 truncated to 12 hex chars (48 bits)."""
    return hashlib.sha1(patch.astype(np.int8).tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- signature

def canonical_d4_signature(
    patch: np.ndarray,
    color_sensitive: bool = False,
) -> tuple[str, int]:
    """Return (κ, |Stab(P)|) for one patch.

    κ = min over g ∈ D4 of Encode(canonicalize(g·P)), as in the spec.
    Stabilizer count tells us how many D4 elements fix the canonical encoding.
    """
    encodings: list[str] = []
    for variant in d4_transforms(patch):
        if not color_sensitive:
            variant = canonicalize_patch_colors(variant)
        encodings.append(encode_patch(variant))
    canonical = min(encodings)
    stab = sum(1 for e in encodings if e == canonical)
    return canonical, stab


def canonical_shape_hash(mask: np.ndarray) -> str:
    """D4-canonical hash of a binary region mask.

    Two regions whose masks are equal modulo any element of D4 (rotation /
    reflection) return the same hash.  Independent of position, colour, and
    rotation — purely a shape fingerprint.
    """
    if mask.size == 0:
        return "0" * 12
    bin_mask = mask.astype(np.int8)
    return min(encode_patch(g) for g in d4_transforms(bin_mask))


# --------------------------------------------------------------------------- candidate API

def candidate_local_signature(
    candidate_kind: str,
    color_grid: np.ndarray,
    label_grid: np.ndarray,
    point: tuple[int, int] | list[int],
    radius: int,
) -> dict:
    """Compute everything Phase 1D needs to know about p's local neighbourhood.

    Returns a dict with:
      color_signature   κ over the colour patch (color-invariant)
      label_signature   κ over the region-label patch (region-invariant)
      stabilizer_count  |Stab| of the colour patch (1..8)
      stabilizer_score  stab / 8.0  (handy for ranking later)
      local_signature   sha1(kind || color || label) — the combined key Phase
                        1D groups on, so kind segregates centers from corners.
    """
    color_patch = extract_patch(color_grid, point, radius)
    label_patch = extract_patch(label_grid, point, radius)

    color_sig, color_stab = canonical_d4_signature(color_patch)
    label_sig, _ = canonical_d4_signature(label_patch)

    combined = hashlib.sha1(
        f"{candidate_kind}|{color_sig}|{label_sig}".encode()
    ).hexdigest()[:12]

    return {
        "color_signature": color_sig,
        "label_signature": label_sig,
        "stabilizer_count": color_stab,
        "stabilizer_score": color_stab / 8.0,
        "local_signature": combined,
    }


def assign_signatures(
    candidates: Iterable,
    color_grid: np.ndarray,
    label_grid: np.ndarray,
    radius: int | None = None,
) -> None:
    """In-place: attach local_signature + stabilizer fields to each candidate."""
    if radius is None:
        radius = patch_radius_for(color_grid.shape)
    for cand in candidates:
        sig = candidate_local_signature(
            cand.kind, color_grid, label_grid, cand.point, radius
        )
        cand.local_signature = sig["local_signature"]
        cand.features.update({
            "color_signature": sig["color_signature"],
            "label_signature": sig["label_signature"],
            "stabilizer_count": sig["stabilizer_count"],
            "stabilizer_score": sig["stabilizer_score"],
            "patch_radius": radius,
        })
