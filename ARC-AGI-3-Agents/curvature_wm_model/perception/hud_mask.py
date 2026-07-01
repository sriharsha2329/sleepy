"""hud_mask.py — per-game CONSTANT HUD region(s) + a node-dropper for training.

The HUD is a DYNAMIC solid-rectangle bar at an extreme edge (a counter/progress bar). We detect it from the
opening frames, then the removal region is the FIXED UNION bbox of that bar over those frames — a constant
mask, so masking the same region every frame never leaks progress (a dynamic, per-frame mask would).

  detect:  dynamic (area varies across frames) + solid rectangle (fill>=0.9, aspect>=4) + extreme edge + area>=32
  remove:  drop any graph node whose normalized centroid falls in a region (mask -> False), every frame.

Excludes sk48/sb26 (their near-top game-rule blocks are NOT extreme-edge bars — different approach needed).
Frame-based so it also catches m0r0's bar that is black (empty) at t=0 and fills with white over time.

NOTE: the principled fix is rung-3 causal (do(a) controllability); it's unstable for now, so this heuristic.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np
from scipy import ndimage

from curvature_wm_model import paths  # noqa: F401

HR = paths.HUMAN_RUNS                        # consolidated raw recordings (curvature_wm/data/human_runs)
EXCLUDE = set()                             # node-based detector applies to ALL games (incl. sk48/sb26)
M = 0.10                                     # extreme-edge margin (within 10% of a border)
AREA_MIN, AREA_MAX, FILL_MIN, ASPECT_MIN = 32, 192, 0.9, 4.0   # solid rectangle bar: min area 32, MAX area 192
DYN = 4                                      # area must vary > DYN px across frames -> dynamic (HUD, not static wall)
TOL = 0.02                                   # centroid-in-region tolerance (normalized)
CAP = 0.14                                    # max bar thickness (clip to a thin EDGE strip so we never eat playfield)
_CACHE: dict[str, list] = {}


def _opening_frames(game, k=400):                # wide enough that a slow-filling bar (e.g. m0r0) grows past AREA_MIN
    recs = sorted(glob.glob(str(HR / game / "*.recording.jsonl")))
    if not recs:
        return []
    out = []
    for line in open(recs[0]):
        d = json.loads(line).get("data", {})
        if isinstance(d, dict) and "frame" in d:
            out.append(np.clip(np.asarray(d["frame"])[0], 0, 15))   # defensive: clip stray indices to palette
        if len(out) >= k:
            break
    return out


def _edge_side(cyc, cxc):
    if cyc < M:
        return "top"
    if cyc > 1 - M:
        return "bottom"
    if cxc < M:
        return "left"
    if cxc > 1 - M:
        return "right"
    return None


def hud_regions_from_frames(frames):
    """Detect HUD bbox(es) from a list of opening FRAMES (a live env's first frames) — same algorithm as
    hud_regions, no file I/O. Used for ONLINE / Kaggle play where human recordings don't exist."""
    regions = []
    if frames:
        H, W = frames[0].shape
        vals, cnts = np.unique(frames[0], return_counts=True)
        bg = int(vals[cnts.argmax()])                               # background = most common color
        series = defaultdict(list)
        for fr in frames[::4]:
            for col in np.unique(fr):
                if int(col) == bg:
                    continue
                lab, n = ndimage.label(fr == col)
                for cid in range(1, n + 1):
                    ys, xs = np.where(lab == cid)
                    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; area = len(ys)
                    if area < AREA_MIN or area > AREA_MAX or area / (h * w) < FILL_MIN or max(w / h, h / w) < ASPECT_MIN:
                        continue
                    side = _edge_side(ys.mean() / H, xs.mean() / W)
                    if side is None:
                        continue
                    series[(int(col), side)].append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()), area))
        for (_col, side), obs in series.items():
            # STATIC detection from the OPENING frames (no dynamic-area requirement, no human runs): a solid edge
            # rectangle with area∈[32,192], fill≥0.9, aspect≥4 IS the HUD bar — caught right at game start.
            r0 = min(o[0] for o in obs) / H; r1 = max(o[1] for o in obs) / H
            c0 = min(o[2] for o in obs) / W; c1 = max(o[3] for o in obs) / W   # UNION = constant max extent
            if side == "top":       r1 = min(r1, CAP)
            elif side == "bottom":  r0 = max(r0, 1 - CAP)
            elif side == "left":    c1 = min(c1, CAP)
            elif side == "right":   c0 = max(c0, 1 - CAP)
            regions.append((r0, r1, c0, c1))
    return regions


def hud_node_bboxes(nodes):
    """HUD-bar whole-pixel bboxes detected from OUR PERCEPTION's nodes — runs at game start, for ANY game, with no
    human runs and no image re-analysis. A node is a HUD bar if its pixel bbox is a thin solid rectangle lying
    ALONG a frame extremity: area in [AREA_MIN, AREA_MAX], aspect >= ASPECT_MIN, stabilizer in {2,4}, and a
    horizontal bar hugs the top/bottom row (a vertical bar the left/right col). The whole bbox is then blanked in
    the frame and dropped from perception, so its interior is masked too. Square HUD (corner icons, rule-blocks)
    is intentionally OUT OF SCOPE — bars only (per user)."""
    if not nodes:
        return []
    H = max(int(n["bbox"][2]) for n in nodes) + 1
    W = max(int(n["bbox"][3]) for n in nodes) + 1
    M = max(1, round(0.10 * max(H, W)))                          # extremity margin, in PIXELS
    bars = []
    for n in nodes:
        by0, bx0, by1, bx1 = (int(v) for v in n["bbox"])        # whole-pixel box [ymin,xmin,ymax,xmax]
        h, w = by1 - by0 + 1, bx1 - bx0 + 1
        aspect = max(w / h, h / w) if h and w else 0.0
        # A HUD bar lies ALONG an extreme edge: a HORIZONTAL bar hugging the top/bottom row, or a VERTICAL bar
        # hugging the left/right column. Merely TOUCHING an edge with a tip is NOT enough -- a full-height center
        # divider (ar25) or a mid-field platform (vc33) touches an edge but is playfield, and must NOT be masked.
        if w >= h:
            at_edge = by1 <= M or by0 >= H - 1 - M               # horizontal bar -> near TOP or BOTTOM row
        else:
            at_edge = bx1 <= M or bx0 >= W - 1 - M               # vertical bar   -> near LEFT or RIGHT col
        if AREA_MIN <= int(n.get("area", 0)) <= AREA_MAX and aspect >= ASPECT_MIN \
                and int(n.get("stab", 1)) in (2, 4) and at_edge:    # HUD bars are 2- OR 4-fold symmetric (ls20=2, tn36=4)
            bars.append([by0, bx0, by1, bx1])
    return bars


def hud_regions(game):
    """List of CONSTANT normalized bboxes (cy0,cy1,cx0,cx1) covering the game's dynamic edge HUD bar(s)."""
    if game in EXCLUDE:
        return []
    if game in _CACHE:
        return _CACHE[game]
    frames = _opening_frames(game)
    regions = []
    if frames:
        H, W = frames[0].shape
        vals, cnts = np.unique(frames[0], return_counts=True)
        bg = int(vals[cnts.argmax()])                               # background = most common color
        series = defaultdict(list)                                  # (color, edge-side) -> [(r0,r1,c0,c1,area), ...]
        for fr in frames[::4]:
            for col in np.unique(fr):
                if int(col) == bg:
                    continue
                lab, n = ndimage.label(fr == col)
                for cid in range(1, n + 1):
                    ys, xs = np.where(lab == cid)
                    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; area = len(ys)
                    if area < AREA_MIN or area > AREA_MAX or area / (h * w) < FILL_MIN or max(w / h, h / w) < ASPECT_MIN:
                        continue
                    side = _edge_side(ys.mean() / H, xs.mean() / W)
                    if side is None:
                        continue
                    series[(int(col), side)].append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()), area))
        for (_col, side), obs in series.items():
            areas = [o[4] for o in obs]
            if len(obs) >= 2 and (max(areas) - min(areas)) > DYN:   # DYNAMIC -> HUD (static walls/borders skipped)
                r0 = min(o[0] for o in obs) / H; r1 = max(o[1] for o in obs) / H
                c0 = min(o[2] for o in obs) / W; c1 = max(o[3] for o in obs) / W   # UNION = constant max extent
                if side == "top":       r1 = min(r1, CAP)           # clip to a thin edge strip (never into playfield)
                elif side == "bottom":  r0 = max(r0, 1 - CAP)
                elif side == "left":    c1 = min(c1, CAP)
                elif side == "right":   c0 = max(c0, 1 - CAP)
                regions.append((r0, r1, c0, c1))
    _CACHE[game] = regions
    return regions


def drop_hud(ar, regions):
    """Set mask_{prev,cur}=False for any node whose normalized centroid (cy,cx) lies in a HUD region. In-place.
    If the click target node gets dropped, invalidate it (click_target=-1) so the click loss isn't supervised
    against a masked (-1e9-logit) slot."""
    if not regions:
        return 0
    dropped = 0
    for tag in ("prev", "cur"):
        pos = ar[f"pos_{tag}"]; mask = ar[f"mask_{tag}"]
        for i in range(len(mask)):
            if not mask[i]:
                continue
            cy, cx = float(pos[i][0]), float(pos[i][1])
            for (r0, r1, c0, c1) in regions:
                if r0 - TOL <= cy <= r1 + TOL and c0 - TOL <= cx <= c1 + TOL:
                    mask[i] = False; dropped += 1
                    break
    ct = int(ar.get("click_target", -1))                            # invalidate a click target we just removed
    if ct >= 0 and (not ar["mask_prev"][ct] or not ar["mask_cur"][ct]):
        ar["click_target"] = -1
    return dropped
