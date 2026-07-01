"""Object stabilizer — Level 2 of the stabilizer hierarchy.

Given a region's binary mask M, compute the order of the subgroup of D4 that
fixes the mask (after padding to a square so non-square bboxes are comparable
under rotation).

D4 elements in our convention:
    0  e             identity
    1  r             90° CCW rotation
    2  r²            180° rotation
    3  r³            270° CCW rotation
    4  s             horizontal-axis mirror (flipud)
    5  sr            main-diagonal mirror (transpose)
    6  sr²           vertical-axis mirror (fliplr)
    7  sr³           anti-diagonal mirror

Subgroup lattice of D4 used for the symmetry-type label:

    order 8 : D4
    order 4 : C4 ({e,r,r²,r³}), Klein-V ({e,r²,s,sr²}), Klein-D ({e,r²,sr,sr³})
    order 2 : C2-rot ({e,r²}), mirror-H ({e,s}), mirror-V ({e,sr²}),
              mirror-D ({e,sr}), mirror-A ({e,sr³})
    order 1 : trivial ({e})

Orbit-stabilizer: |D4-orbit(M)| = 8 / |Stab(M)|, so |Stab| ∈ {1, 2, 4, 8} only.
Any other count is a bug.
"""
from __future__ import annotations

import numpy as np


def _pad_to_square(mask: np.ndarray) -> np.ndarray:
    """Centre-pad a possibly non-square boolean mask into a square frame.

    Centre-padding rather than corner-padding so the D4 rotations/reflections
    of the padded square coincide with the rotations/reflections of the
    original (un-padded) mask considered as a centred figure.
    """
    h, w = mask.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=bool)
    pad_h = (s - h) // 2
    pad_w = (s - w) // 2
    out[pad_h:pad_h + h, pad_w:pad_w + w] = mask
    return out


def _d4_fixers(mask_sq: np.ndarray) -> list[bool]:
    """Per-element membership in Stab(M); returned in our canonical D4 order."""
    rot1 = np.rot90(mask_sq, 1)
    rot2 = np.rot90(mask_sq, 2)
    rot3 = np.rot90(mask_sq, 3)
    flip = np.flipud(mask_sq)
    return [
        True,                                 # e (identity, by definition)
        np.array_equal(rot1, mask_sq),        # r
        np.array_equal(rot2, mask_sq),        # r²
        np.array_equal(rot3, mask_sq),        # r³
        np.array_equal(flip, mask_sq),        # s = flipud
        np.array_equal(np.rot90(flip, 1), mask_sq),  # sr = transpose
        np.array_equal(np.rot90(flip, 2), mask_sq),  # sr² = fliplr
        np.array_equal(np.rot90(flip, 3), mask_sq),  # sr³ = anti-transpose
    ]


def _classify_subgroup(fixers: list[bool]) -> str:
    """Map the boolean fixer-tuple to a human-readable subgroup label."""
    n = sum(fixers)
    if n == 8:
        return "D4"
    if n == 4:
        # Determine which order-4 subgroup we landed in.
        if fixers[1] and fixers[2] and fixers[3]:
            return "C4"
        if fixers[2] and fixers[4] and fixers[6]:
            return "Klein-V"      # horizontal + vertical mirrors + 180° rot
        if fixers[2] and fixers[5] and fixers[7]:
            return "Klein-D"      # both diagonals + 180° rot
        return "K4-misc"          # shouldn't happen for valid masks
    if n == 2:
        if fixers[2]:
            return "C2-rot"       # 180° rotation only
        if fixers[4]:
            return "mirror-H"     # horizontal axis
        if fixers[5]:
            return "mirror-D"     # main diagonal
        if fixers[6]:
            return "mirror-V"     # vertical axis
        if fixers[7]:
            return "mirror-A"     # anti-diagonal
        return "C2-misc"
    return "trivial"


def compute_object_stabilizer(mask: np.ndarray) -> tuple[int, str]:
    """Return (|Stab|, symmetry_type) for the mask under D4.

    Empty masks degenerate to (8, "D4") since the empty set is fixed by every
    transform — but callers should not be passing empty masks.
    """
    if mask.size == 0 or not mask.any():
        return 8, "D4"
    sq = _pad_to_square(mask.astype(bool))
    fixers = _d4_fixers(sq)
    return sum(fixers), _classify_subgroup(fixers)
