"""convert_run.py — show the full HUMAN-RUN -> GRAPH + CLICKED-NODE + LATENT pipeline, end to end.

Demonstrates the consolidated stack on one real recording:
  1. PERCEIVE   raw recording frames -> per-object graph (graph_extract): nodes carry centroid (cy,cx)
                AND spatial footprint (px,py,sx,sy = mask centroid + per-axis std), edges carry
                relation + centroid offset (dx,dy,dist).
  2. CLICK->NODE  a raw click pixel (x,y) is mapped to the live node whose FOOTPRINT contains it by
                Mahalanobis distance d_M = sqrt((x-px)^2/sx^2 + (y-py)^2/sy^2) (featurize.click_target_slot,
                "i-dagger"): each node is a REGION, not a point. Far clicks aren't rejected, just
                confidence-attenuated p_geo = exp(-d_M^2/18).
  3. ENCODE     node_latents -> [N, obj_dim] latent; the footprint block [px/64, py/64, sx/16, sy/16]
                folds the centroid + Mahalanobis spread straight into the latent the world model reads.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.encoding.convert_run --game bp35
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from curvature_wm_model import paths  # noqa: F401  -- local substrate + offline wiring

import numpy as np

import graph_extract
from config import Config
from data import iter_transitions, featurize_transition
from featurize import click_target_slot
from transform_catalyst.data_adapter import node_latents, obj_dim

CLICK = 6


def _first_recording(game: str) -> Path:
    recs = sorted((paths.HUMAN_RUNS / game).glob("*.recording.jsonl"))
    if not recs:
        raise SystemExit(f"no recordings for {game} under {paths.HUMAN_RUNS}")
    return recs[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="bp35")
    ap.add_argument("--max_frames", type=int, default=400, help="cap frames perceived (speed)")
    args = ap.parse_args()
    cfg = Config()
    od = obj_dim(cfg)

    rec = _first_recording(args.game)
    print(f"=== 1. PERCEIVE  {args.game}  ({rec.name}) ===")
    mods, GaugeTracker = graph_extract._imports()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "graph.jsonl"
        # cap the recording to max_frames so the smoke test is fast
        capped = Path(td) / "capped.recording.jsonl"
        with rec.open() as f, capped.open("w") as w:
            for i, ln in enumerate(f):
                if i >= args.max_frames:
                    break
                w.write(ln)
        info = graph_extract.extract_trajectory(capped, out, mods, GaugeTracker)
        print(f"  perceived {info['n_frames']} frames -> per-object graphs")

        # find the first CLICK transition that maps to a live node (so we can show click->node)
        chosen = None
        n_trans = 0
        for tr in iter_transitions(out):
            n_trans += 1
            if int(tr["a"]) != CLICK or tr["click"] is None:
                continue
            arr = featurize_transition(tr, cfg)
            if int(arr.get("click_target", -1)) >= 0:
                chosen = (tr, arr)
                break
        if chosen is None:                                  # fall back to the first valid transition
            for tr in iter_transitions(out):
                arr = featurize_transition(tr, cfg)
                if arr["mask_prev"].sum() >= 2:
                    chosen = (tr, arr)
                    break
        tr, arr = chosen

    # ---- 2. the perceived GRAPH for this transition's prev frame ----
    print(f"\n=== 2. GRAPH (prev frame)  {len(tr['prev_nodes'])} objects, {len(tr['prev_edges'])} edges ===")
    print("  node            centroid(cy,cx)   footprint(px,py,sx,sy)        color  stab area")
    for n in tr["prev_nodes"][:8]:
        print(f"   {str(n['pid'])[:12]:<12}  ({n['cy']:.3f},{n['cx']:.3f})   "
              f"({n['px']:.1f},{n['py']:.1f},{n['sx']:.1f},{n['sy']:.1f})".ljust(30)
              + f"  {str(n.get('color'))[:8]:<8} {n['stab']:>2}  {n['area']}")
    if tr["prev_edges"]:
        e = tr["prev_edges"][0]
        print(f"  e.g. edge: {str(e['src'])[:8]} -{e['relation']}-> {str(e['dst'])[:8]}  "
              f"dx={e['dx']:.3f} dy={e['dy']:.3f} dist={e['dist']:.3f}")

    # ---- 3. CLICK -> NODE via Mahalanobis i-dagger ----
    a = int(tr["a"])
    print(f"\n=== 3. CLICK -> NODE (Mahalanobis i-dagger)   action={a} ({'CLICK' if a==CLICK else 'move'}) ===")
    if a == CLICK and tr["click"] is not None:
        ct, cxy, d_m, dn = click_target_slot(arr["pos_prev"], arr["mask_prev"], tr["click"],
                                             cfg.grid, foot_prev=arr["foot_prev"])
        x, y = tr["click"]
        px, py, sx, sy = arr["foot_prev"][ct]
        print(f"  raw click pixel (x,y) = ({x},{y})")
        print(f"  -> node slot {ct}: footprint (px,py,sx,sy)=({px:.1f},{py:.1f},{sx:.1f},{sy:.1f})")
        print(f"  Mahalanobis d_M = {d_m:.3f}   within-footprint offset d~=({dn[0]:.2f},{dn[1]:.2f})")
        print(f"  p_geo = exp(-d_M^2/18) = {float(np.exp(-(d_m**2)/18.0)):.3f}   "
              f"(stored click_target={int(arr['click_target'])}, click_pgeo={float(arr['click_pgeo']):.3f})")
        node_slot = ct
    else:
        node_slot = int(np.argmax(arr["mask_prev"]))
        print(f"  (not a click transition; showing first live node, slot {node_slot})")

    # ---- 4. ENCODE the node into the latent the world model reads ----
    Z = node_latents(arr, "prev", cfg)                      # [N, obj_dim]
    print(f"\n=== 4. ENCODE  node_latents -> Z[{Z.shape[0]},{Z.shape[1]}]  (obj_dim={od}) ===")
    foot_block = Z[node_slot, od - 5:od - 1]                # [px/64, py/64, sx/16, sy/16]
    px, py, sx, sy = arr["foot_prev"][node_slot]
    expect = np.array([px / cfg.grid, py / cfg.grid, sx / 16.0, sy / 16.0], np.float32)
    print(f"  latent footprint block Z[{node_slot}, {od-5}:{od-1}] = "
          f"[{foot_block[0]:.4f},{foot_block[1]:.4f},{foot_block[2]:.4f},{foot_block[3]:.4f}]")
    print(f"  expected [px/64,py/64,sx/16,sy/16]              = "
          f"[{expect[0]:.4f},{expect[1]:.4f},{expect[2]:.4f},{expect[3]:.4f}]")
    ok = np.allclose(foot_block, expect, atol=1e-4)
    print(f"  centroid + Mahalanobis spread folded into latent: {'MATCH ' if ok else 'MISMATCH'}")
    print(f"  full node latent (62-d) nonzero dims: {int((Z[node_slot] != 0).sum())} "
          f"(stab 1-hot + areabin 1-hot + color + WL 1-hot + foot4 + logk)")
    assert ok, "footprint encoding mismatch"
    print("\nOK: human run -> graph -> clicked node -> centroid/Mahalanobis latent verified.")


if __name__ == "__main__":
    main()
