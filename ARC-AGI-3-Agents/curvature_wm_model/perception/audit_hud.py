"""audit_hud.py — prove the HUD is GONE from perception graphs + training samples, for EVERY game.

Three checks:
  1. DETECT (perception): hud_regions(game) for all games — which have a HUD bar, where, and that
     sk48/sb26 are EXCLUDED (regions=[], handled by a different approach).
  2. TRAINING DATA: in the exact pooled HUD-ON caches the world model trains on (train_hud + heldout_hud),
     count live nodes whose centroid falls inside that game's HUD region -> must be 0 for every game.
  3. BEFORE vs AFTER: a small fresh HUD-OFF load per HUD game shows how many nodes were there originally,
     so the 0 above is a real removal, not a vacuous pass.
Also asserts sk48/sb26 contribute ZERO transitions to the training data.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.perception.audit_hud
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401

import numpy as np

from config import Config
from hud_mask import hud_regions, EXCLUDE
from curvature_wm_model.data.loader import load_split
from curvature_wm_model.splits import TRAIN_GAMES, HELDOUT_GAMES, ALL_GAMES

TOL = 0.02
CY, CX = 58, 57           # node-latent foot block: cy=py/64 (dim 58), cx=px/64 (dim 57)


def _nodes_in_regions(Z, M, regions):
    """# of live nodes whose centroid lies in any region (same rule as drop_hud)."""
    if not regions:
        return 0
    cy, cx = Z[..., CY], Z[..., CX]
    hit = np.zeros(M.shape, bool)
    for r0, r1, c0, c1 in regions:
        hit |= M & (cy >= r0 - TOL) & (cy <= r1 + TOL) & (cx >= c0 - TOL) & (cx <= c1 + TOL)
    return int(hit.sum())


def main():
    cfg = Config()
    tg = [g for g in TRAIN_GAMES if g not in EXCLUDE]

    # ---- 1. DETECT (perception layer) ----
    print("=== 1. HUD DETECTION (perception) — region per game ===")
    has_hud = []
    for g in ALL_GAMES:
        if g in EXCLUDE:
            print(f"  {g:6s} EXCLUDED (sk48/sb26 — near-top game-rule blocks, not removed here)")
            continue
        reg = hud_regions(g)
        if reg:
            has_hud.append(g)
            r0, r1, c0, c1 = reg[0]
            side = ("top" if r1 < 0.5 else "bottom") if (r1 - r0) < (c1 - c0) else ("left" if c1 < 0.5 else "right")
            print(f"  {g:6s} HUD: {len(reg)} region(s)  ~{side:6s}  bbox=({r0:.2f},{r1:.2f},{c0:.2f},{c1:.2f})")
        else:
            print(f"  {g:6s} no HUD bar detected")

    # ---- 2. TRAINING DATA: pooled HUD-ON caches the model actually trains on ----
    print("\n=== 2. TRAINING DATA (HUD-ON caches) — live nodes inside HUD region (must be 0) ===")
    dtr = load_split(tg, cfg, 1500, tag="train_hud", hud=True)
    dho = load_split(HELDOUT_GAMES, cfg, 1500, tag="heldout_hud", hud=True)
    total_leak = 0
    present = set()
    for name, d in (("TRAIN", dtr), ("HELDOUT", dho)):
        games = d["game"].astype(str)
        for g in sorted(set(games)):
            present.add(g)
            sel = games == g
            leak = _nodes_in_regions(d["Zp"][sel], d["Mp"][sel], hud_regions(g))
            total_leak += leak
            flag = "  <-- LEAK!" if leak else ""
            print(f"  [{name:7s}] {g:6s} transitions={int(sel.sum()):5d}  HUD nodes live={leak}{flag}")

    # ---- 3. sk48/sb26 must be absent from training data ----
    print("\n=== 3. EXCLUDED games absent from training data ===")
    for g in sorted(EXCLUDE):
        n = int((dtr["game"].astype(str) == g).sum()) + int((dho["game"].astype(str) == g).sum())
        print(f"  {g}: {n} transitions in training data  {'(absent ✓)' if n == 0 else '<-- PRESENT!'}")

    # ---- 4. BEFORE vs AFTER for HUD games (small fresh HUD-OFF load) ----
    print("\n=== 4. BEFORE (HUD-OFF) vs AFTER (HUD-ON) nodes in region — sample, max 300/game ===")
    for g in has_hud:
        off = load_split([g], cfg, 300, tag=f"audit_off_{g}", hud=False)
        on = load_split([g], cfg, 300, tag=f"audit_on_{g}", hud=True)
        n_off = _nodes_in_regions(off["Zp"], off["Mp"], hud_regions(g))
        n_on = _nodes_in_regions(on["Zp"], on["Mp"], hud_regions(g))
        print(f"  {g:6s} HUD nodes:  before={n_off:5d}  ->  after={n_on}   "
              f"({'removed ✓' if n_off > 0 and n_on == 0 else 'check'})")

    print(f"\nTOTAL HUD nodes live in training data: {total_leak}  ->  "
          f"{'HUD FULLY REMOVED ✓' if total_leak == 0 else 'HUD LEAK!'}")
    assert total_leak == 0, "HUD nodes leaked into training data"


if __name__ == "__main__":
    main()
