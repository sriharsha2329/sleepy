"""Load human-run transitions for a chosen set of games (bounded per game, cached).

The load loop is vendored here (kept in the new folder) but reuses the proven featurize pipeline +
data_adapter from the archived tree READ-ONLY via paths.py. Adds EFc (NEXT-state edges) on top of the
original loader so the shared trunk can encode the next state too (needed by the inverse + click heads).

Returns a dict of stacked arrays:
  Zp,Zc [T,N,od]  Mp,Mc [T,N]  EFp,EFc [T,N,N,EDGE_DIM]  a [T]  ct [T] (click-target slot, -1 if none)
  pgeo [T]  game [T]
"""
from __future__ import annotations

import hashlib

from curvature_wm_model import paths  # noqa: F401  (sets sys.path for `data`, `transform_catalyst`)

import numpy as np
import transform_catalyst.data_adapter as cda
from data import iter_transitions, featurize_transition

from curvature_wm_model.splits import TRAIN_GAMES, HELDOUT_GAMES
from hud_mask import hud_regions, drop_hud           # curvature_wm/perception (on sys.path via paths)

CACHE = paths.HERE / "runs"
_COLS = ("Zp", "Zc", "Mp", "Mc", "EFp", "EFc", "a", "ct", "pgeo", "game")


def load_split(games, cfg, max_per_game: int = 1500, tag: str = "", hud: bool = False):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5((",".join(sorted(games)) + f"_{max_per_game}_{cda.obj_dim(cfg)}_cwm{'_hud' if hud else ''}").encode()).hexdigest()[:10]
    cpath = CACHE / f"games_{tag}_{key}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cols = {k: [] for k in _COLS}
    for g in games:
        gd = paths.GRAPH_DATA / g
        if not gd.exists():
            continue
        regions = hud_regions(g) if hud else []                     # per-game CONSTANT HUD region (computed once)
        cnt = 0
        for fp in sorted(gd.glob("*.jsonl")):
            for tr in iter_transitions(fp):
                if int(tr["a"]) < 0:
                    continue
                ar = featurize_transition(tr, cfg)
                if regions:
                    drop_hud(ar, regions)                           # mask out HUD nodes BEFORE the alive-count gate
                if ar["mask_prev"].sum() < 2 or ar["mask_cur"].sum() < 2:
                    continue
                cols["Zp"].append(cda.node_latents(ar, "prev", cfg)); cols["Zc"].append(cda.node_latents(ar, "cur", cfg))
                cols["Mp"].append(ar["mask_prev"]); cols["Mc"].append(ar["mask_cur"])
                cols["EFp"].append(cda.edge_feats(ar, "prev", cfg)); cols["EFc"].append(cda.edge_feats(ar, "cur", cfg))
                cols["a"].append(int(ar["a"]))
                cols["ct"].append(int(ar.get("click_target", -1))); cols["pgeo"].append(float(ar.get("click_pgeo", 1.0)))
                cols["game"].append(g); cnt += 1
                if cnt >= max_per_game:
                    break
            if cnt >= max_per_game:
                break
    out = {"Zp": np.stack(cols["Zp"]).astype(np.float32), "Zc": np.stack(cols["Zc"]).astype(np.float32),
           "Mp": np.stack(cols["Mp"]).astype(bool), "Mc": np.stack(cols["Mc"]).astype(bool),
           "EFp": np.stack(cols["EFp"]).astype(np.float32), "EFc": np.stack(cols["EFc"]).astype(np.float32),
           "a": np.array(cols["a"], np.int64), "ct": np.array(cols["ct"], np.int64),
           "pgeo": np.array(cols["pgeo"], np.float32), "game": np.array(cols["game"])}
    np.savez(cpath, **out)
    return out


def load_train(cfg, max_per_game: int = 1500):
    return load_split(TRAIN_GAMES, cfg, max_per_game, tag="train")


def load_heldout(cfg, max_per_game: int = 1500):
    return load_split(HELDOUT_GAMES, cfg, max_per_game, tag="heldout")
