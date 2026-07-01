"""Featurization + cross-frame slot alignment (guide §1; reviewer §3).

A transition's prev/cur graphs are aligned into a fixed register of N_max slots by
**persistent id (pid)** — the bootstrap matcher. The learned SlotMatcher (matcher.py)
refines and is validated separately. Slots present in both frames share an index;
appear/disappear is a mask flip.
"""
from __future__ import annotations

import hashlib

import numpy as np

RELATIONS = {"touches": 1, "near": 2, "contains": 3}
DELTA_KINDS = [
    "transition", "state_change", "mutation", "edge_birth", "edge_death",
    "move", "birth", "death", "resurrect", "level_change", "level_start",
    "death_reset",
]
DK_IDX = {k: i for i, k in enumerate(DELTA_KINDS)}


def stable_hash(s: str, mod: int) -> int:
    """Deterministic string hash in [1, mod-1] (0 reserved for pad)."""
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)
    return 1 + (h % (mod - 1))


def delta_multihot(deltas: list, n: int = 12) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for d in deltas or []:
        kind = d.get("kind", "")
        sub = (d.get("detail", {}) or {}).get("type", "")
        for key in (kind, sub):
            if key in DK_IDX:
                v[DK_IDX[key]] = 1.0
    return v


def _color_multihot(color, n_colors=16) -> np.ndarray:
    v = np.zeros(n_colors, dtype=np.float32)
    for c in (color or []):
        if isinstance(c, int) and 0 <= c < n_colors:
            v[c] = 1.0
    return v


def align_slots(prev_nodes, cur_nodes, prev_edges, cur_edges, cfg) -> dict:
    """Align two frames' graphs into N_max slots keyed by pid. Returns numpy arrays."""
    N, C = cfg.n_max, cfg.n_colors
    pv = {n["pid"]: n for n in prev_nodes}
    cu = {n["pid"]: n for n in cur_nodes}

    def area(n):
        return float(n.get("area", 0)) if n else 0.0

    pids = set(pv) | set(cu)
    ranked = sorted(
        pids,
        key=lambda p: (p in pv and p in cu, max(area(pv.get(p)), area(cu.get(p)))),
        reverse=True,
    )[:N]
    slot = {p: i for i, p in enumerate(ranked)}

    out = {
        "pos_prev": np.zeros((N, 2), np.float32), "pos_cur": np.zeros((N, 2), np.float32),
        "color_prev": np.zeros((N, C), np.float32), "color_cur": np.zeros((N, C), np.float32),
        "stab_prev": np.zeros(N, np.int64), "stab_cur": np.zeros(N, np.int64),
        "areabin_prev": np.zeros(N, np.int64), "areabin_cur": np.zeros(N, np.int64),
        "wl_prev": np.zeros(N, np.int64), "wl_cur": np.zeros(N, np.int64),
        "logk_prev": np.zeros(N, np.float32), "logk_cur": np.zeros(N, np.float32),
        "mask_prev": np.zeros(N, bool), "mask_cur": np.zeros(N, bool),
        "edge_rel_prev": np.zeros((N, N), np.int64), "edge_rel_cur": np.zeros((N, N), np.int64),
        "edge_geo_prev": np.zeros((N, N, 3), np.float32), "edge_geo_cur": np.zeros((N, N, 3), np.float32),
        # per-object spatial FOOTPRINT (px,py,sx,sy in pixels) for the cell->node Mahalanobis i†
        "foot_prev": np.zeros((N, 4), np.float32), "foot_cur": np.zeros((N, 4), np.float32),
    }

    def fill_nodes(by_pid, tag):
        for p, i in slot.items():
            n = by_pid.get(p)
            if n is None:
                continue
            out[f"mask_{tag}"][i] = True
            out[f"pos_{tag}"][i] = (float(n.get("cy", 0.0)), float(n.get("cx", 0.0)))
            out[f"color_{tag}"][i] = _color_multihot(n.get("color"), C)
            out[f"stab_{tag}"][i] = int(np.clip(int(n.get("stab", 1)), 0, cfg.n_stab - 1))
            out[f"areabin_{tag}"][i] = int(np.clip(int(n.get("area_bin", 0)), 0, cfg.n_areabin - 1))
            out[f"wl_{tag}"][i] = stable_hash(str(n.get("type_hash", "")), cfg.th_vocab)
            out[f"logk_{tag}"][i] = float(np.log1p(float(n.get("k", 1))))
            out[f"foot_{tag}"][i] = (float(n.get("px", 0.0)), float(n.get("py", 0.0)),
                                     float(n.get("sx", 1.0)), float(n.get("sy", 1.0)))

    fill_nodes(pv, "prev")
    fill_nodes(cu, "cur")

    def fill_edges(edges, tag):
        for e in (edges or []):
            si, di = slot.get(e["src"]), slot.get(e["dst"])
            if si is None or di is None:
                continue
            out[f"edge_rel_{tag}"][si, di] = RELATIONS.get(e.get("relation", ""), 0)
            out[f"edge_geo_{tag}"][si, di] = (
                float(e.get("dx", 0.0)), float(e.get("dy", 0.0)), float(e.get("dist", 0.0)))

    fill_edges(prev_edges, "prev")
    fill_edges(cur_edges, "cur")
    return out


def click_identity(wl_row, stab_row, area_row, edge_rel_row, mask_row, t) -> int:
    """STANDARD-feature, game-agnostic identity of clicking node t: its type_hash/WL class +
    symmetry + area-bin, plus the multiset of relations on its incident edges. Stable across
    slot reassignment and across games (unlike a slot index), so a click verdict transfers."""
    t = int(t)
    rels = []
    for j in range(len(mask_row)):
        if not bool(mask_row[j]):
            continue
        ro, ri = int(edge_rel_row[t, j]), int(edge_rel_row[j, t])
        if ro:
            rels.append(ro)
        if ri:
            rels.append(ri)
    return hash((int(wl_row[t]), int(stab_row[t]), int(area_row[t]), tuple(sorted(rels))))


def action_key(a, click_target, wl_row, stab_row, area_row, edge_rel_row, mask_row):
    """Unified action identity. Moves -> ('move', id). Clicks -> ('click', standard click id),
    so 'where you clicked' (by node+edge features) is part of the action — never collapsed."""
    if int(a) == 6 and int(click_target) >= 0:
        return ("click", click_identity(wl_row, stab_row, area_row, edge_rel_row, mask_row,
                                        click_target))
    return ("move", int(a))


def click_target_slot(pos_prev, mask_prev, click_xy, grid: int, foot_prev=None):
    """Map a click to the live prev slot whose FOOTPRINT contains it, by Mahalanobis distance
    (Step-1 §1.3.1 i†) — each node is a REGION (centroid + per-axis spread), not a point:

        d_M((x,y), i) = sqrt( (x-px_i)²/σx_i² + (y-py_i)²/σy_i² )

    This is the rule that maps the r11l crossing cell (23,22) to the big region (area 3342,
    σ≈17) at d_M=0.76, instead of the nearest-CENTROID node 6.5σ away (the guarded-against bug).
    Far clicks are NOT rejected — they are confidence-ATTENUATED downstream via
    p_geo = exp(-d_M²/18) (far clicks still train, they just whisper).
    Falls back to nearest-centroid if footprints are absent. click_xy=[x(col),y(row)] pixels.
    Returns (slot_idx or -1, (row_norm, col_norm), d_M, delta_norm) where delta_norm is the
    Mahalanobis-normalized within-footprint offset δ̃ = ((x-px)/σx, (y-py)/σy) of the chosen slot."""
    if click_xy is None:
        return -1, (0.0, 0.0), 0.0, (0.0, 0.0)
    x, y = float(click_xy[0]), float(click_xy[1])
    row_n, col_n = y / grid, x / grid                  # pos stored as (cy=row, cx=col)
    best, bd = -1, 1e18
    best_dn = (0.0, 0.0)
    for i in range(len(mask_prev)):
        if not mask_prev[i]:
            continue
        if foot_prev is not None:
            px, py, sx, sy = foot_prev[i]
            # σ floor/cap [1, 20] px: thin regions need a near-exact hit; an inflated footprint may
            # not vacuum-claim the whole grid.
            sx = min(max(float(sx), 1.0), 20.0); sy = min(max(float(sy), 1.0), 20.0)
            dnx, dny = (x - float(px)) / sx, (y - float(py)) / sy
            d = dnx * dnx + dny * dny                                   # Mahalanobis²
        else:
            dr = pos_prev[i, 0] - row_n; dc = pos_prev[i, 1] - col_n
            d = dr * dr + dc * dc
            dnx = dny = 0.0
        if d < bd:
            bd, best, best_dn = d, i, (dnx, dny)
    d_m = float(bd) ** 0.5 if (best >= 0 and foot_prev is not None) else 0.0
    return best, (row_n, col_n), d_m, best_dn
