"""latents.py — the AUGMENTED latent: node_latents ⊕ un-squashed Mahalanobis position + centroid.

Base node_latents put the footprint in foot=(px,py,sx,sy)·[1/grid,1/grid,1/16,1/16] — centroid ~[0,1] and a
few-px player move is ~0.09, which smooth-L1 squashes to ~0 (position-blind). Here we APPEND, per node,
[ centroid_col, centroid_row, px/σx, py/σy ]: the last two are the σ-normalized (Mahalanobis) centroid,
magnitude ~grid/σ (large, NOT squashed) — analogous to the curvature's σ-normalized term. Feeding this to the
trunk (after the input-dim edit + retrain) makes the shared representation position-aware.

`augment_Z` does the same on already-loaded node_latents tensors (reads the foot block), so the trainer can
augment the loader output without re-featurizing.

  ENVIRONMENTS_DIR=... PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.data.latents   # smoke
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

import numpy as np

from transform_catalyst.data_adapter import node_latents, obj_dim

SIG_FLOOR, SIG_CAP = 1.0, 20.0
POS_DIM = 4                                                       # [centroid_col, centroid_row, maha_col, maha_row]
FOOT0 = 57                                                       # foot block start in node_latents: (px/grid,py/grid,σx/16,σy/16)


def aug_dim(cfg):
    return obj_dim(cfg) + POS_DIM


def pos_features(feat, tag, cfg):
    """[N,4] = centroid (normalized) + Mahalanobis centroid (px/σ, py/σ; un-squashed). Dead slots zeroed."""
    foot = feat[f"foot_{tag}"].astype(np.float32)                 # raw (px, py, sx, sy) in px
    pos = feat[f"pos_{tag}"].astype(np.float32)                   # (cy, cx) normalized
    sx = np.clip(foot[:, 2], SIG_FLOOR, SIG_CAP); sy = np.clip(foot[:, 3], SIG_FLOOR, SIG_CAP)
    cen = np.stack([pos[:, 1], pos[:, 0]], 1)                     # (cx, cy)
    maha = np.stack([foot[:, 0] / sx, foot[:, 1] / sy], 1)       # (px/σx, py/σy)  <- un-squashed
    return (np.concatenate([cen, maha], 1) * feat[f"mask_{tag}"][:, None]).astype(np.float32)


def aug_latents(feat, tag, cfg):
    """[N, obj_dim+4] — base node_latents with the un-squashed Mahalanobis position + centroid appended."""
    return np.concatenate([node_latents(feat, tag, cfg), pos_features(feat, tag, cfg)], 1)


def augment_Z(Z, cfg):
    """[..., obj_dim] -> [..., obj_dim+4], computing the pos block from Z's own foot block (no re-featurize).
    Works on numpy. foot = Z[..., 57:61] = (px/grid, py/grid, σx/16, σy/16)."""
    foot = Z[..., FOOT0:FOOT0 + 4]
    cen = foot[..., :2]                                           # (col/grid, row/grid) = centroid (normalized)
    px = foot[..., 0] * cfg.grid; py = foot[..., 1] * cfg.grid
    sx = np.clip(foot[..., 2] * 16, SIG_FLOOR, SIG_CAP); sy = np.clip(foot[..., 3] * 16, SIG_FLOOR, SIG_CAP)
    maha = np.stack([px / sx, py / sy], -1)
    return np.concatenate([Z, cen, maha], -1).astype(np.float32)


# ----------------------------------------------------------------------------- smoke: are BOTH Mahalanobis AND graph visible?
def _changes(feat, cfg):
    od = obj_dim(cfg)
    Zp_old, Zc_old = node_latents(feat, "prev", cfg), node_latents(feat, "cur", cfg)
    Zp_aug, Zc_aug = aug_latents(feat, "prev", cfg), aug_latents(feat, "cur", cfg)
    return {
        "both": feat["mask_prev"] & feat["mask_cur"],
        "foot": np.abs(Zc_old[:, 57:61] - Zp_old[:, 57:61]).sum(1),        # OLD position channel (squashed)
        "graph": np.abs(Zc_old[:, :57] - Zp_old[:, :57]).sum(1),           # GRAPH channels (stab/area/color/WL)
        "maha": np.abs(Zc_aug[:, od + 2:od + 4] - Zp_aug[:, od + 2:od + 4]).sum(1),   # NEW Mahalanobis dims
        "aug": (Zp_aug, Zc_aug),
    }


def smoke():
    from config import Config
    from curvature_wm_model.hodge_flow.envfast import FastEnv
    from curvature_wm_model.hodge_flow.diagnose_lr import _feat
    from online_v5 import Live
    cfg = Config()
    env = FastEnv(Live(cfg, "bp35")); s0 = env.graph(); snap0 = env.snap()

    env.step(4); s1 = env.graph()                                # scenario A: a pure MOVE (graph unchanged)
    A = _changes(_feat(s0, s1, cfg), cfg)
    mover = int(np.argmax(A["maha"] * A["both"]))

    env.restore(snap0)                                           # scenario B: a structural EVENT (graph changes)
    for _ in range(3):
        env.step(4)
    s_pre = env.graph(); env.step(4); s_post = env.graph()
    B = _changes(_feat(s_pre, s_post, cfg), cfg)
    graph_event = float((B["graph"] * B["both"]).sum())

    aug_match = np.allclose(augment_Z(node_latents(_feat(s0, s1, cfg), "prev", cfg), cfg),
                            aug_latents(_feat(s0, s1, cfg), "prev", cfg), atol=1e-2)   # float32 foot round-trip ~3e-4
    P = [("aug latent dim == obj_dim + 4", A["aug"][0].shape[1] == aug_dim(cfg)),
         ("augment_Z ≈ aug_latents (loader path matches, atol 1e-2)", aug_match),
         ("aug latents finite", np.isfinite(A["aug"][0]).all() and np.isfinite(B["aug"][1]).all()),
         (f"MOVE: Mahalanobis visible, LARGE (>1) [{A['maha'][mover]:.2f}]", A["maha"][mover] > 1.0),
         (f"MOVE: old foot squashed (<0.2)        [{A['foot'][mover]:.3f}]", A["foot"][mover] < 0.2),
         (f"MOVE: graph blind to a pure move (<0.2) [{A['graph'][mover]:.3f}]", A["graph"][mover] < 0.2),
         ("MOVE: Mahalanobis >> foot", A["maha"][mover] > 10 * max(A["foot"][mover], 1e-6)),
         (f"EVENT: graph visible from latent (>1) [{graph_event:.2f}]", graph_event > 1.0)]
    ok = all(v for _, v in P)
    print("LATENT SMOKE — BOTH Mahalanobis AND graph visible? (A=move, B=event)")
    for n, v in P:
        print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print("ALL PASS ✓" if ok else "SOME FAILED ✗")
    return ok


if __name__ == "__main__":
    smoke()
