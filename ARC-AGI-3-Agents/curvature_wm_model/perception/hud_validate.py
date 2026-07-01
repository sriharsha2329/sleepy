"""hud_validate.py — VALIDATION picture: human-run frames BEFORE vs AFTER HUD masking.

For a game, sample frames across a human run and show, per frame:
  row 1 (RAW)        : the frame + the detected CONSTANT HUD region (red box) + every perception node's
                       centroid (green = kept, red X = DROPPED because its centroid is in the HUD region).
  row 2 (HUD-MASKED) : the same frame with the HUD region blanked to background — i.e. exactly the pixels
                       whose perception NODES drop_hud() removes before the world model ever sees the graph.

This is the SAME region/rule the training used (hud_mask.hud_regions + drop_hud), so it validates that the
model trained on properly HUD-masked graphs.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.perception.hud_validate ls20
"""
from __future__ import annotations

import glob
import json
import sys

from curvature_wm_model import paths  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import graph_extract
from hud_mask import hud_regions, TOL

# canonical ARC-AGI-3 16-colour palette (white@0)
try:
    from arc_agi.rendering import COLOR_MAP as _CM
    _ARC = [_CM[i] for i in range(16)]
except Exception:
    _ARC = ["#FFFFFF", "#DDDDDD", "#AAAAAA", "#666666", "#333333", "#000000", "#F012BE", "#FF80C0",
            "#FF4136", "#0074D9", "#7FDBFF", "#FFDC00", "#FF851B", "#870C25", "#2ECC40", "#B10DC9"]
_CMAP = ListedColormap(_ARC)

_MODS, _GT = graph_extract._imports()


def _frames(game, k):
    recs = sorted(glob.glob(str(paths.HUMAN_RUNS / game / "*.recording.jsonl")))
    if not recs:
        raise SystemExit(f"no recordings for {game} under {paths.HUMAN_RUNS}")
    out = []
    for line in open(recs[0]):
        d = json.loads(line).get("data", {})
        if isinstance(d, dict) and "frame" in d:
            fr = np.asarray(d["frame"])
            while fr.ndim > 2:
                fr = fr[-1] if fr.shape[0] else fr[0]
            out.append(np.clip(fr.astype(np.int32), 0, 15))
    idx = np.linspace(8, len(out) - 1, k).astype(int)          # spread across the run (skip intro)
    return [out[i] for i in idx], idx, len(out)


def _perceive(fr):
    tr = _GT(graveyard_ttl=30, history_limit=-1); tr._frame_idx_hint = 0
    nodes, _, _ = graph_extract._process_frame(fr, tr, _MODS)
    return nodes


def _in_region(cy, cx, regions):
    for (r0, r1, c0, c1) in regions:
        if r0 - TOL <= cy <= r1 + TOL and c0 - TOL <= cx <= c1 + TOL:
            return True
    return False


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    regions = hud_regions(game)
    frames, idx, n = _frames(game, k)
    H, W = frames[0].shape

    fig, axes = plt.subplots(2, k, figsize=(3.0 * k, 6.6))
    tot_drop = 0
    for j, (fr, fi) in enumerate(zip(frames, idx)):
        bg = int(np.bincount(fr.ravel()).argmax())             # background = most common colour
        nodes = _perceive(fr)
        kept = dropped = 0
        masked = fr.copy()
        for (r0, r1, c0, c1) in regions:                       # blank the HUD region to background
            y0, y1 = int(r0 * H), int(np.ceil(r1 * H)); x0, x1 = int(c0 * W), int(np.ceil(c1 * W))
            masked[y0:min(y1 + 1, H), x0:min(x1 + 1, W)] = bg

        for row, img, ttl in ((0, fr, "RAW"), (1, masked, "HUD-MASKED")):
            ax = axes[row, j]
            ax.imshow(img, cmap=_CMAP, vmin=0, vmax=15, interpolation="nearest")
            for (r0, r1, c0, c1) in regions:                   # red box = the constant HUD region
                ax.add_patch(Rectangle((c0 * W - 0.5, r0 * H - 0.5), (c1 - c0) * W + 1, (r1 - r0) * H + 1,
                                       fill=False, edgecolor="red", lw=1.6))
            if row == 0:                                        # mark perception nodes: green=kept, red X=dropped
                for nd in nodes:
                    cy, cx = float(nd.get("cy", 0)), float(nd.get("cx", 0))
                    drop = _in_region(cy, cx, regions)
                    ax.plot(cx * W, cy * H, "x" if drop else "o", color=("#d62728" if drop else "#2ECC40"),
                            ms=6, mew=1.6, mfc="none")
                    kept += not drop; dropped += drop
                ax.set_title(f"frame {fi}\n{dropped} node(s) DROPPED, {kept} kept", fontsize=8)
                tot_drop += dropped
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(ttl, fontsize=11, fontweight="bold")
    reg_txt = ", ".join(f"({r0:.2f},{r1:.2f},{c0:.2f},{c1:.2f})" for (r0, r1, c0, c1) in regions) or "NONE"
    fig.suptitle(f"{game}: HUD-masking validation  —  red box = CONSTANT HUD region {reg_txt}  ·  "
                 f"row1 green o = kept node, red x = dropped node  ·  row2 = region blanked  "
                 f"(total dropped over {k} frames: {tot_drop})", fontsize=10)
    out = paths.HERE / "rollout_graphs_png" / f"hud_validate_{game}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out} | {len(regions)} HUD region(s) {reg_txt} | sampled {k}/{n} frames | total nodes dropped={tot_drop}")


if __name__ == "__main__":
    main()
