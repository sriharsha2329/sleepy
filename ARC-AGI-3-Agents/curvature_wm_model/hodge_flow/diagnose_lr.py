"""Diagnosis: curvature = dA + A∧A (A∧A computed FOR REAL by branching return actions in the offline env),
and its Hodge gradient/harmonic split, for LEFT x3 vs RIGHT x3 on offline bp35.

dA(step)   = SMAE(z_cur - z_prev), PID-aligned (observed change).
A∧A(step)  = min over return actions b of SMAE(env(z_cur, b) - z_prev)   ['stay' candidate = dA, so A∧A <= dA].
             small A∧A => reversible (a return undoes it); A∧A ~ dA => irreversible (nothing returns).
curvature  = dA + A∧A.   Reversible(grad/curl) part ~ dA - A∧A ; irreversible(harmonic) part ~ A∧A.
Hodge      = object-state affordance graph over the LEFT+RIGHT+return transitions, decomposed grad/curl/harm.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.diagnose_lr
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from data import featurize_transition
import transform_catalyst.data_adapter as cda
from curvature_wm_model.hodge_flow.encode import build_gnn
from curvature_wm_model.hodge_flow import cluster as ccl
from curvature_wm_model.hodge_flow.hodge import build_complex, hodge_decompose, node_mass

LEFT, RIGHT, CLICK, UNDO = 3, 4, 6, 7
SMAE_BETA = 2.0
NAME = {0: "up", 1: "down", 2: "?2", 3: "LEFT", 4: "RIGHT", 5: "?5", 6: "CLICK", 7: "UNDO", "stay": "stay"}


def _feat(g_prev, g_cur, cfg, a=0):
    tr = {"prev_nodes": g_prev["nodes"], "cur_nodes": g_cur["nodes"], "prev_edges": g_prev["edges"],
          "cur_edges": g_cur["edges"], "a": a, "click": None, "deltas": g_cur.get("deltas", []),
          "level": 0, "r_ext": 0.0, "done": False}
    return featurize_transition(tr, cfg)


STABLE_DIM = 57   # [0:57]=stab+area+color+WL ; [57:62]=foot+logk
HUD_THRESH = 0.8  # nodes with normalized row cy >= this are HUD/step-counter (bottom rows ~51-63), NOT playfield


def _playfield(feat):
    """Union of alive nodes, EXCLUDING the HUD/step-counter region (bottom rows). The HUD counter ticks every
    frame ('the game ending') and otherwise inflates dA/A∧A even when the playfield is unchanged."""
    cy = np.where(feat["mask_cur"], feat["pos_cur"][:, 0], feat["pos_prev"][:, 0])
    return ((feat["mask_prev"] | feat["mask_cur"]) & (cy < HUD_THRESH)).astype(np.float32)


POS_W = 1.0            # weight of the Mahalanobis centroid displacement (σ units) in the curvature
SIG_FLOOR, SIG_CAP = 1.0, 20.0   # σ floor/cap in px (matches featurize click δ̃, §1.3.1)


def _maha_disp_from_feat(feat, cfg):
    """Σ over PID-aligned PLAYFIELD nodes of the MAHALANOBIS centroid displacement ‖Δpos·grid / σ‖.
    σ = clip(footprint spread sx,sy, [1,20]) px. This is the position signal the one-hot/graph channels miss:
    a few-px player move is ~0 in raw node_latents (foot·1/grid then smooth-L1) but several σ here."""
    both = feat["mask_prev"] & feat["mask_cur"]
    play = both & (feat["pos_cur"][:, 0] < HUD_THRESH)
    idx = np.where(play)[0]
    if idx.size == 0:
        return 0.0
    d = (feat["pos_cur"][idx] - feat["pos_prev"][idx]) * float(cfg.grid)     # (Δrow, Δcol) px
    sx = np.clip(feat["foot_cur"][idx, 2], SIG_FLOOR, SIG_CAP)               # σ_col (x)
    sy = np.clip(feat["foot_cur"][idx, 3], SIG_FLOOR, SIG_CAP)               # σ_row (y)
    return float(np.hypot(d[:, 0] / sy, d[:, 1] / sx).sum())


def smae_pair(g_prev, g_cur, cfg, stable=False, with_pos=True):
    """PID-aligned curvature between two frames, over PLAYFIELD nodes only (HUD/counter excluded):
      = semantic SMAE (stab/area/color/WL/foot) + POS_W · Mahalanobis centroid displacement.
    The Mahalanobis term (`with_pos`, default ON) makes a moving player register — the graph channels alone
    read ~0 for a several-px move. `stable` kept for callers but defaults off (uses full latent)."""
    feat = _feat(g_prev, g_cur, cfg)
    sl = slice(0, STABLE_DIM) if stable else slice(None)
    Zp = torch.from_numpy(cda.node_latents(feat, "prev", cfg)[:, sl]).float()
    Zc = torch.from_numpy(cda.node_latents(feat, "cur", cfg)[:, sl]).float()
    un = torch.from_numpy(_playfield(feat))
    sem = float((F.smooth_l1_loss(Zc, Zp, beta=SMAE_BETA, reduction="none").sum(-1) * un).sum())
    if with_pos:
        sem += POS_W * _maha_disp_from_feat(feat, cfg)
    return sem


def signed_gradient(g_prev, g_cur, cfg):
    """GRADIENT ≡ z_next − z_current, **SIGNED (no abs)**, over PID-aligned PLAYFIELD nodes (HUD excluded).
    PLAN §12.4 (user): "the gradient is z_nextframe − z_currentframe." Returns the signed flat vector
    [ semantic latent delta (Zc−Zp) | POS_W · signed Mahalanobis position delta ].  ‖·‖ of this == dA.
    NOTE: unlike smae_pair (which is a magnitude for clustering/identity), this KEEPS the sign for the
    oriented Hodge flow."""
    feat = _feat(g_prev, g_cur, cfg)
    un = _playfield(feat).astype(bool)
    Zp = cda.node_latents(feat, "prev", cfg); Zc = cda.node_latents(feat, "cur", cfg)
    sem = (Zc - Zp)[un].ravel()                                    # signed, NO abs
    both = (feat["mask_prev"] & feat["mask_cur"]) & un
    idx = np.where(both)[0]
    if idx.size:
        d = (feat["pos_cur"][idx] - feat["pos_prev"][idx]) * float(cfg.grid)     # signed (Δrow, Δcol) px
        sx = np.clip(feat["foot_cur"][idx, 2], SIG_FLOOR, SIG_CAP)
        sy = np.clip(feat["foot_cur"][idx, 3], SIG_FLOOR, SIG_CAP)
        pos = POS_W * np.stack([d[:, 0] / sy, d[:, 1] / sx], 1).ravel()          # signed Maha displacement
    else:
        pos = np.zeros(0, np.float32)
    return np.concatenate([sem.astype(np.float32), pos.astype(np.float32)])


def signed_curl(env, snap_post, g_prev, cfg):
    """CURL ≡ min_a ( f(z', a) − z ), **SIGNED (no abs)**, computed ACROSS EACH STEP. PLAN §12.4 (user).
    From the post-action snapshot `snap_post` (state z' = env.graph() there), branch every real action a
    (no undo/stay), pick a* = argmin‖f(z',a) − z‖, and return that action's **signed** residual
    `signed_gradient(g_prev, f(z',a*))` = f(z',a*) − z. ≈0 ⇒ reversible (fills → curl); ≈ gradient ⇒ no-return.
    ‖·‖ of this == A∧A. Requires a FastEnv-like `env` (restore/actions/step/graph)."""
    best = None; best_norm = float("inf"); best_a = None
    for a in env.actions():
        env.restore(snap_post)
        cxy = _node_xy(env.graph()) if a == CLICK else None
        env.step(a, cxy)
        resid = signed_gradient(g_prev, env.graph(), cfg)         # signed f(z',a) − z
        n = float(np.linalg.norm(resid))
        if n < best_norm:
            best_norm, best, best_a = n, resid, a
    return best, best_a                                           # signed best-return residual, and a*


def _node_xy(graph):
    ns = graph.get("nodes", [])
    if not ns:
        return None
    n = max(ns, key=lambda x: x.get("area", 0))
    x = n.get("cx", n.get("px")); y = n.get("cy", n.get("py"))
    return (int(x), int(y)) if x is not None else None


def replay(live, prefix):
    """Deterministically reach a state by replaying (action, click_xy) prefix from reset; returns its graph.
    Robust: stop if the game ends or the engine returns an empty frame at depth (returns last good graph)."""
    live.reset(); g = live.prev_graph
    for (a, cxy) in prefix:
        try:
            g2, lvl, done, _ = live.step(a, cxy)
        except Exception:
            break                                   # empty frame / engine edge at depth -> stop, keep last good
        live.prev_graph = g2; live.prev_level = lvl; g = g2
        if done:
            break
    return g


def diagnose_dir(live, cfg, direction, n=3):
    """Per-step dA, A∧A (real), curvature for `direction` x n from reset. Returns rows + the visited transitions
    (prev_g, cur_g, dir-label) and best-return transitions (cur_g, return_g, 'ret') for the Hodge graph."""
    prev_g = replay(live, [])                                       # reset state
    prefix = []; rows = []; trans = []
    for i in range(n):
        cur_g = replay(live, prefix + [(direction, None)])
        dA_f = smae_pair(prev_g, cur_g, cfg, False); dA_s = smae_pair(prev_g, cur_g, cfg, True)
        aa_f, aa_s, best_b, best_g = float("inf"), float("inf"), "none", cur_g   # NO stay, NO undo
        for b in live.available():
            if b == UNDO:
                continue
            cxy = _node_xy(cur_g) if b == CLICK else None
            sb = replay(live, prefix + [(direction, None), (b, cxy)])
            df = smae_pair(prev_g, sb, cfg, False); ds = smae_pair(prev_g, sb, cfg, True)
            if df < aa_f:
                aa_f, best_b, best_g = df, NAME[b], sb
            aa_s = min(aa_s, ds)
        rows.append({"dA_f": dA_f, "aa_f": aa_f, "dA_s": dA_s, "aa_s": aa_s, "ret": best_b})
        trans.append((prev_g, cur_g, NAME[direction]))
        trans.append((cur_g, best_g, "ret"))
        prefix = prefix + [(direction, None)]; prev_g = cur_g
    return rows, trans


@torch.no_grad()
def hodge_on(model, cfg, dev, all_trans, M=10):
    """Object-state Hodge over a set of (prev_g, cur_g, label) transitions (PID-aligned per transition).
    Returns energy split + per-LABEL harmonic/gradient edge mass."""
    Hp, Hc, Mp, Mc, lab = [], [], [], [], []
    for (gp, gc, label) in all_trans:
        feat = _feat(gp, gc, cfg)
        Zp = torch.from_numpy(cda.node_latents(feat, "prev", cfg)).float()[None].to(dev)
        Zc = torch.from_numpy(cda.node_latents(feat, "cur", cfg)).float()[None].to(dev)
        EFp = torch.from_numpy(cda.edge_feats(feat, "prev", cfg)).float()[None].to(dev)
        EFc = torch.from_numpy(cda.edge_feats(feat, "cur", cfg)).float()[None].to(dev)
        mp = torch.from_numpy(feat["mask_prev"]).bool()[None].to(dev)
        mc = torch.from_numpy(feat["mask_cur"]).bool()[None].to(dev)
        Hp.append(model.encode(Zp, mp, EFp)[0][0].cpu().numpy()); Mp.append(feat["mask_prev"].astype(bool))
        Hc.append(model.encode(Zc, mc, EFc)[0][0].cpu().numpy()); Mc.append(feat["mask_cur"].astype(bool))
        lab.append(label)
    Hp = np.stack(Hp); Hc = np.stack(Hc); Mp = np.stack(Mp); Mc = np.stack(Mc)
    C, op, oc = ccl.object_states(Hp, Hc, Mp, Mc, M=M)
    nodes, edges, x, _ = build_complex(op, oc)
    if len(edges) == 0:
        return None
    dec = hodge_decompose(nodes, edges, x)
    return dec


def main():
    cfg = Config(); dev = torch.device("cpu")
    model, _ = build_gnn(dev=dev)
    from online_v5 import Live
    live = Live(cfg, "bp35")
    print("DIAGNOSIS — curvature dA + A∧A for LEFT x3 vs RIGHT x3 on offline bp35")
    print("           (A∧A = min return by a REAL game action only — NO undo, NO stay)\n")

    all_trans = []
    for direction in (LEFT, RIGHT):
        rows, trans = diagnose_dir(live, cfg, direction, n=3)
        all_trans += trans
        print(f"=== {NAME[direction]} x3 ===")
        print("  step | FULL (with flicker): dA   A∧A  curv | STABLE (flicker removed): dA   A∧A  curv | return")
        for k, r in enumerate(rows, 1):
            print(f"   {k}   |  {r['dA_f']:6.2f} {r['aa_f']:6.2f} {r['dA_f']+r['aa_f']:6.2f} |"
                  f"   {r['dA_s']:6.2f} {r['aa_s']:6.2f} {r['dA_s']+r['aa_s']:6.2f} | {r['ret']}")
        cf = sum(r['dA_f'] + r['aa_f'] for r in rows); cs = sum(r['dA_s'] + r['aa_s'] for r in rows)
        print(f"  TOTAL curvature:  FULL(with flicker) = {cf:6.2f}   |   STABLE(no flicker) = {cs:6.2f}\n")

    # Hodge over the combined LEFT+RIGHT+return object-state graph
    dec = hodge_on(model, cfg, dev, all_trans, M=10)
    print("=== Hodge of the combined LEFT+RIGHT+return affordance graph ===")
    if dec is None:
        print("  (no object-state affordance edges — moves did not change object-state clusters)")
    else:
        print(f"  nodes={len(dec['nodes'])} edges={len(dec['edges'])} tris={dec['n_tris']} beta1={dec['beta1']}")
        print(f"  energy:  gradient {dec['e_grad']:.1%}   curl {dec['e_curl']:.1%}   harmonic {dec['e_harm']:.1%}")


if __name__ == "__main__":
    main()
