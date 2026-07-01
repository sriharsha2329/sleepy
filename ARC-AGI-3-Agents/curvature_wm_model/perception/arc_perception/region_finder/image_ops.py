"""Image I/O, quantization, background estimation, color-connected components.

These are the deterministic preprocessing steps that turn a raw RGB frame into
a per-pixel region label map.  Heavy math lives in symmetry.py / graph.py; this
module is intentionally boring.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np
from PIL import Image
from skimage.measure import label as cc_label


# ----- 1. load -----

def load_image(path: str) -> np.ndarray:
    """Load any PIL-supported image as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# ----- 2. quantize -----

def unique_color_count(img: np.ndarray) -> int:
    flat = img.reshape(-1, img.shape[-1])
    return len({tuple(row) for row in flat})


def quantize_image(img: np.ndarray, max_colors: int = 32) -> np.ndarray:
    """Uniform-bin quantize only if image has more colors than max_colors.

    Connected-component labelling depends on stable color labels, so we want
    "close enough" colors to collapse without distorting the palette of crisp
    ARC-style frames (those already have < 32 unique colors -> no-op).
    """
    if unique_color_count(img) <= max_colors:
        return img
    bin_size = max(1, 256 // max_colors)
    return ((img // bin_size) * bin_size).astype(np.uint8)


# ----- 3. background -----

def _border_pixels(img: np.ndarray) -> Iterator[tuple[int, int, int]]:
    h, w, _ = img.shape
    for x in range(w):
        yield tuple(int(c) for c in img[0, x])
        yield tuple(int(c) for c in img[h - 1, x])
    for y in range(1, h - 1):
        yield tuple(int(c) for c in img[y, 0])
        yield tuple(int(c) for c in img[y, w - 1])


def estimate_background_color(img: np.ndarray) -> tuple[int, int, int]:
    """Most frequent color on the 1-pixel border; whole-image mode on tie."""
    border = Counter(_border_pixels(img))
    if not border:
        return (0, 0, 0)
    top, top_count = border.most_common(1)[0]
    ties = [c for c, n in border.items() if n == top_count]
    if len(ties) == 1:
        return top
    whole = Counter(tuple(p) for p in img.reshape(-1, 3))
    return max(ties, key=lambda c: whole[c])


# ----- 4a. grid-mode background + segmentation (for palette-index grids) -----

def estimate_background_label(grid: np.ndarray) -> int:
    """Most frequent label on the 1-pixel border of an integer grid."""
    border = np.concatenate([
        grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]
    ])
    vals, counts = np.unique(border, return_counts=True)
    top = counts.max()
    candidates = vals[counts == top]
    if len(candidates) == 1:
        return int(candidates[0])
    whole_vals, whole_counts = np.unique(grid, return_counts=True)
    mode_idx = max(candidates, key=lambda v: whole_counts[whole_vals == v][0])
    return int(mode_idx)


def segment_grid(
    grid: np.ndarray,
    background_label: int,
    connectivity: int = 1,
    min_area: int = 3,
) -> tuple[list[dict], np.ndarray]:
    """Connected-component label a 2-D integer grid directly.

    Faster path for ARC frames where pixel values are already small palette
    indices (0..15); skips the RGB-to-key trick of segment_components.
    """
    h, w = grid.shape
    label_map = np.full((h, w), -1, dtype=np.int32)
    regions: list[dict] = []
    next_id = 0
    for v in np.unique(grid):
        mask = grid == v
        cc = cc_label(mask, connectivity=connectivity)
        for cc_id in range(1, int(cc.max()) + 1):
            comp = cc == cc_id
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.where(comp)
            bbox = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
            centroid = [float(ys.mean()), float(xs.mean())]
            kind = "background_component" if int(v) == background_label else "object"
            touches_border = bool(
                bbox[0] == 0 or bbox[1] == 0
                or bbox[2] == h - 1 or bbox[3] == w - 1
            )
            regions.append({
                "_index": next_id,
                "color": [int(v)],
                "kind": kind,
                "area": area,
                "bbox": bbox,
                "centroid": centroid,
                "touches_border": touches_border,
            })
            label_map[comp] = next_id
            next_id += 1
    return regions, label_map


# ----- 4b. connected components per color (RGB image) -----

def segment_components(
    img: np.ndarray,
    background_color: tuple[int, int, int],
    connectivity: int = 1,
    min_area: int = 3,
) -> tuple[list[dict], np.ndarray]:
    """Per-color CC labeling.

    Each connected component is its own region; same-colored objects in
    different places are NOT merged.  Returns:
      regions   – list of raw dicts (region_id, color, kind, area, bbox, centroid,
                  touches_border, pixel mask key).  Full Region objects are built
                  in regions.py from these dicts.
      label_map – (H, W) int array where each pixel holds the region's index in
                  the regions list, or -1 for sub-min-area noise.
    """
    h, w, _ = img.shape
    # 1-D color key for fast equality without iterating channels in Python.
    color_key = (
        img[..., 0].astype(np.int32) * 1_000_000
        + img[..., 1].astype(np.int32) * 1_000
        + img[..., 2].astype(np.int32)
    )
    unique_keys = np.unique(color_key)
    bg_key = background_color[0] * 1_000_000 + background_color[1] * 1_000 + background_color[2]

    label_map = np.full((h, w), -1, dtype=np.int32)
    regions: list[dict] = []
    next_id = 0

    for key in unique_keys:
        mask = color_key == key
        cc = cc_label(mask, connectivity=connectivity)
        max_cc = int(cc.max())
        for cc_id in range(1, max_cc + 1):
            comp = cc == cc_id
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.where(comp)
            bbox = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
            centroid = [float(ys.mean()), float(xs.mean())]
            color = (int((key // 1_000_000) % 1000),
                     int((key // 1_000) % 1000),
                     int(key % 1_000))
            kind = "background_component" if int(key) == bg_key else "object"
            touches_border = bool(
                bbox[0] == 0 or bbox[1] == 0
                or bbox[2] == h - 1 or bbox[3] == w - 1
            )
            regions.append({
                "_index": next_id,
                "color": list(color),
                "kind": kind,
                "area": area,
                "bbox": bbox,
                "centroid": centroid,
                "touches_border": touches_border,
            })
            label_map[comp] = next_id
            next_id += 1

    return regions, label_map
