"""live.py — minimal OFFLINE game env: arc_agi engine -> our perception graph -> featurized latents.

Self-contained replacement for the heavy archive ``online_v5.Live`` (which dragged in the causal /
value / dml RL heads). This one does ONLY what the perception + encoding pipeline needs: reset / step
a LOCAL offline game and turn each frame into our per-object graph (``graph_extract``) and, on
request, the ``node_latents`` (centroids + Mahalanobis footprint) the world model consumes.

Offline games live in ``curvature_wm/env/environment_files`` — wired by ``curvature_wm.paths`` via
``ENVIRONMENTS_DIR`` + ``OPERATION_MODE=offline``. No network / API.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.env.live --game bp35 --steps 8
"""
from __future__ import annotations

import argparse

from curvature_wm_model import paths  # noqa: F401  -- sets sys.path + OPERATION_MODE/ENVIRONMENTS_DIR (offline)

import numpy as np

import arc_agi
from arcengine import GameAction

import graph_extract                                   # curvature_wm/perception (on sys.path via paths)
from config import Config
from data import featurize_transition
from transform_catalyst.data_adapter import node_latents, edge_feats

CLICK = 6
_MODS, _GaugeTracker = graph_extract._imports()        # perception fns + GaugeTracker (arc_perception aliased)


def _aid(a) -> int:
    v = a.value
    return int(v[0]) if isinstance(v, tuple) else int(v)


_ACT = {_aid(a): a for a in GameAction}                 # action id -> GameAction


def list_games() -> list[str]:
    """Offline game ids available locally (folders under env/environment_files)."""
    d = paths.ENV_FILES
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


class Live:
    """Drive ONE offline game through the local perception stack. Frame -> {nodes, edges, deltas}."""

    def __init__(self, game: str = "bp35", cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.game = game
        self.arc = arc_agi.Arcade()
        self.env = None

    def _graph(self, obs) -> dict:
        arr = np.array(obs.frame[-1], dtype=np.int32)
        while arr.ndim > 2:
            arr = arr[0]
        self.last_frame = arr.copy()
        self.tracker._frame_idx_hint = self._fi
        self._fi += 1
        n, e, d = graph_extract._process_frame(arr, self.tracker, _MODS)
        return {"nodes": n, "edges": e, "deltas": d}

    def reset(self) -> dict:
        self.env = self.arc.make(self.game, render_mode=None)
        self.tracker = _GaugeTracker(graveyard_ttl=30, history_limit=-1)
        self._fi = 0
        obs = self.env.observation_space
        self.prev_graph = self._graph(obs)
        self.prev_level = int(obs.levels_completed)
        return self.prev_graph

    def available(self) -> list[int]:
        return [_aid(a) for a in self.env.action_space]

    def step(self, a: int, click_xy=None):
        """Apply action id `a` (6=click needs click_xy). Returns (cur_graph, level, done, info)."""
        a_obj = _ACT.get(int(a))
        if a_obj is None or a_obj not in self.env.action_space:
            return self.prev_graph, self.prev_level, False, {"level": self.prev_level, "win": False}
        data = {}
        if a_obj.is_complex() and click_xy is not None:
            data = {"x": int(click_xy[0]), "y": int(click_xy[1])}
        obs = self.env.step(a_obj, data=data)
        if obs is None:
            return self.prev_graph, self.prev_level, False, {"level": self.prev_level, "win": False}
        cur_g = self._graph(obs)
        lvl = int(obs.levels_completed)
        st = getattr(obs.state, "name", "")
        done = st in ("WIN", "GAME_OVER")
        info = {"level": lvl, "win": st == "WIN", "game_over": st == "GAME_OVER"}
        self.prev_graph, self.prev_level = cur_g, lvl
        return cur_g, lvl, done, info

    def feat(self, prev_g: dict, cur_g: dict, a: int, click=None) -> dict:
        """Featurize a (prev -> cur) transition exactly as the training loader does."""
        tr = {"prev_nodes": prev_g["nodes"], "cur_nodes": cur_g["nodes"],
              "prev_edges": prev_g["edges"], "cur_edges": cur_g["edges"],
              "a": int(a), "click": click, "deltas": cur_g["deltas"],
              "level": int(self.prev_level), "r_ext": 0.0, "done": False}
        return featurize_transition(tr, self.cfg)

    def latents(self, feat: dict, tag: str = "prev"):
        """(Z [N,obj_dim], M [N], EF [N,N,EDGE_DIM]) for the world model — centroids+Mahalanobis folded in."""
        Z = node_latents(feat, tag, self.cfg)
        M = feat[f"mask_{tag}"]
        EF = edge_feats(feat, tag, self.cfg)
        return Z, M, EF


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="bp35")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    print(f"offline games available: {list_games()}")
    live = Live(args.game)
    g0 = live.reset()
    print(f"[{args.game}] reset: {len(g0['nodes'])} objects, {len(g0['edges'])} edges | "
          f"frame {live.last_frame.shape} | actions {live.available()}")
    prev = g0
    rng = np.random.default_rng(0)
    for t in range(1, args.steps + 1):
        acts = [a for a in live.available() if a != CLICK]  # keep the smoke test click-free
        a = int(acts[rng.integers(len(acts))]) if acts else live.available()[0]
        cur, lvl, done, info = live.step(a)
        feat = live.feat(prev, cur, a)
        Z, M, EF = live.latents(feat, "prev")
        print(f"  step {t:2d}: a={a} -> {len(cur['nodes'])} obj | level={lvl} done={done} | "
              f"Z{tuple(Z.shape)} alive={int(M.sum())}")
        prev = cur
        if done:
            print(f"  episode ended: {info}")
            break


if __name__ == "__main__":
    main()
