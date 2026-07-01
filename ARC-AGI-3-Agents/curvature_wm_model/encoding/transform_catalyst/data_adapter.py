"""Per-object latents + pid-matched pairs from the federation's own align_slots.

pid pairing is used as SOFT EVIDENCE only (to form Δz for the diagnostic + the δ target where the
matcher is confident). The MODEL never hard-tracks objects (correspondence-free, per the spec) — at
births/deaths the pairing is simply absent and we fall back to τ with no pairing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # curvature_wm/encoding (config.py, data.py for load_pairs)

N_WL_BUCKET = 16
N_RELS = 3
EDGE_DIM = N_RELS + 3


def obj_dim(cfg) -> int:
    return cfg.n_stab + cfg.n_areabin + cfg.n_colors + N_WL_BUCKET + 4 + 1


def node_latents(arr, tag, cfg) -> np.ndarray:
    """Deterministic per-object latent [N, obj_dim] for frame `tag`; dead slots zeroed."""
    m = arr[f"mask_{tag}"]; N = m.shape[0]
    v = np.zeros((N, obj_dim(cfg)), np.float32); o = 0; r = np.arange(N)
    v[r, o + np.clip(arr[f"stab_{tag}"], 0, cfg.n_stab - 1)] = 1.0; o += cfg.n_stab
    v[r, o + np.clip(arr[f"areabin_{tag}"], 0, cfg.n_areabin - 1)] = 1.0; o += cfg.n_areabin
    v[:, o:o + cfg.n_colors] = arr[f"color_{tag}"]; o += cfg.n_colors
    v[r, o + (arr[f"wl_{tag}"] % N_WL_BUCKET)] = 1.0; o += N_WL_BUCKET
    f = arr[f"foot_{tag}"].astype(np.float32)
    v[:, o:o + 4] = f * np.array([1.0 / cfg.grid, 1.0 / cfg.grid, 1.0 / 16, 1.0 / 16], np.float32); o += 4
    v[:, o] = np.clip(arr[f"logk_{tag}"], 0, 5) / 5.0; o += 1
    return v * m[:, None]


def edge_feats(arr, tag, cfg) -> np.ndarray:
    rel = arr[f"edge_rel_{tag}"]; geo = arr[f"edge_geo_{tag}"]; N = rel.shape[0]
    ef = np.zeros((N, N, EDGE_DIM), np.float32)
    for r in range(1, N_RELS + 1):
        ef[:, :, r - 1] = (rel == r)
    ef[:, :, N_RELS:N_RELS + 3] = geo / float(cfg.grid)
    return ef


def load_pairs(n_files: int = 40, cfg=None, cache: bool = True):
    """Pooled transitions: (Zp, Zc, Mp, Mc, EFp, ERc, acts) over the first n_files run files.
    Zp/Zc per-object latents [T,N,obj]; Mp/Mc masks [T,N]; EFp prev edge feats; ERc cur edge rels.
    CACHED to runs/pairs_*.npz (featurizing 21k transitions is ~18s; the cache makes reruns <1s)."""
    from config import Config
    from data import build_splits, iter_transitions, featurize_transition
    cfg = cfg or Config()
    cdir = Path(__file__).resolve().parent / "runs"; cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"pairs_n{n_files}_d{obj_dim(cfg)}_v4.npz"   # v4: + phi (dense trajectory-progress potential)
    if cache and cpath.exists():
        z = np.load(cpath)
        return {k: z[k] for k in z.files}, cfg
    files, _ = build_splits()
    n_dk = getattr(cfg, "n_delta_kinds", 12)
    H_bar = 69.0; gamma = 1.0 - 1.0 / H_bar                     # MC discount (federation-derived)
    Zp, Zc, Mp, Mc, EFp, ERc, acts = [], [], [], [], [], [], []
    GAM, PROG, MAG, GRET, CT, PHI = [], [], [], [], [], []     # federation targets + click target + Φ
    for fp in files[:n_files]:
        rows = []                                              # per-file transitions IN ORDER (for MC G)
        for tr in iter_transitions(fp):
            a = featurize_transition(tr, cfg)
            if a["mask_prev"].sum() >= 2:
                rows.append((a, float(tr["r_ext"]), bool(tr["done"])))
        g = 0.0; Gs = [0.0] * len(rows)                        # backward Monte-Carlo return per trajectory
        for t in range(len(rows) - 1, -1, -1):
            r, done = rows[t][1], rows[t][2]
            g = r + (0.0 if done else gamma * g); Gs[t] = g
        # Φ = DENSE progress potential: fractional position WITHIN each trajectory (0=start → 1=end/level-up).
        # Every state has it (unlike sparse reward), so the chain has a non-flat potential to ascend.
        pos = [0.0] * len(rows); seg = 0
        for t in range(len(rows)):
            if rows[t][2] or t == len(rows) - 1:               # trajectory boundary (done) or file end
                L = t - seg + 1
                for u in range(seg, t + 1):
                    pos[u] = (u - seg) / max(L - 1, 1)
                seg = t + 1
        for (a, r, _done), gt, ph in zip(rows, Gs, pos):
            Zp.append(node_latents(a, "prev", cfg)); Zc.append(node_latents(a, "cur", cfg))
            Mp.append(a["mask_prev"]); Mc.append(a["mask_cur"])
            EFp.append(edge_feats(a, "prev", cfg)); ERc.append((a["edge_rel_cur"] > 0))
            acts.append(int(a["a"]))
            dm = a["delta_multihot"].astype(np.float32)        # γ delta-kinds (what changes)
            GAM.append(dm); PROG.append(np.float32(r > 0.0)); MAG.append(np.float32(np.log1p(dm.sum())))
            GRET.append(np.float32(gt))
            CT.append(int(a.get("click_target", -1)))          # clicked slot (-1 if action != click)
            PHI.append(np.float32(ph))                          # dense progress potential
    out = {"Zp": np.stack(Zp), "Zc": np.stack(Zc),
           "Mp": np.stack(Mp).astype(bool), "Mc": np.stack(Mc).astype(bool),
           "EFp": np.stack(EFp), "ERc": np.stack(ERc).astype(np.float32),
           "a": np.array(acts, np.int64), "gamma_t": np.stack(GAM),
           "prog": np.array(PROG, np.float32), "mag": np.array(MAG, np.float32),
           "Gret": np.array(GRET, np.float32), "ct": np.array(CT, np.int64),
           "phi": np.array(PHI, np.float32)}
    if cache:
        np.savez(cpath, **out)
    return out, cfg
