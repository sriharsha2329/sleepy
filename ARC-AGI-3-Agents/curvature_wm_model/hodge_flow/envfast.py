"""Fast offline-env interaction via deepcopy SNAPSHOT/RESTORE instead of reset+replay.

The bottleneck was: to reach any state we reset() and replayed the whole action prefix (O(prefix) env.steps),
nested inside is_forced (branch all actions) and A∧A (branch all returns). copy.deepcopy(live.env) is ~5ms, so
we snapshot the engine, step one action, measure, and restore — O(1) branching. ~100x faster, and it is exactly
the exact-counterfactual "env fork" the plan needs.

UNDO excluded everywhere (only first-frame encoding uses it). A state is "forced/uncontrollable" if every real
action gives the same next state; we collapse those to the next controllable state.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.envfast      # fast LEFT vs RIGHT
"""
from __future__ import annotations

import copy
import os
import sys
import time

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

from config import Config
from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair, _node_xy

UNDO, CLICK = 7, 6
TOL = 0.5


class FastEnv:
    """Snapshot/restore wrapper over online_v5.Live (offline). No reset+replay."""
    def __init__(self, live):
        self.live = live
        self.live.reset()
        self.last_info = {}                                      # win / game_over from the last step

    def reset(self):
        self.live.reset(); self.last_info = {}

    def snap(self):
        return (copy.deepcopy(self.live.env), self.live.prev_graph, self.live.prev_level)

    def restore(self, s):
        self.live.env = copy.deepcopy(s[0]); self.live.prev_graph = s[1]; self.live.prev_level = s[2]

    def graph(self):
        return self.live.prev_graph

    def actions(self):
        return [a for a in self.live.available() if a != UNDO]

    def step(self, a, cxy=None):
        try:
            g, lvl, done, info = self.live.step(a, cxy)
        except Exception:
            self.last_info = {"win": False, "game_over": True}
            return self.live.prev_graph, True
        self.live.prev_graph = g; self.live.prev_level = lvl
        self.last_info = info if isinstance(info, dict) else {}    # {"win":..,"game_over":..} for end-of-level
        return g, done


def is_forced(env, cfg):
    """True if every real action gives the same next state (agent has no control -> env-forced)."""
    s = env.snap(); base = env.graph(); nx = []
    for b in env.actions():
        env.restore(s); cxy = _node_xy(base) if b == CLICK else None
        env.step(b, cxy); nx.append(env.graph())
    env.restore(s)
    return all(smae_pair(nx[0], n, cfg, True) < TOL for n in nx[1:])


def advance(env, cfg, cap=80):
    """Collapse forced/uncontrollable states (step forward in place) to the next controllable state."""
    sk = 0
    while sk < cap and is_forced(env, cfg):
        b = env.actions()[0]; cxy = _node_xy(env.graph()) if b == CLICK else None
        _, done = env.step(b, cxy); sk += 1
        if done:
            break
    return sk


def settle(env, cfg, a, cap=80):
    """Apply action a, then collapse forced states to the next controllable state (in place)."""
    cxy = _node_xy(env.graph()) if a == CLICK else None
    _, done = env.step(a, cxy)
    sk = 0 if done else advance(env, cfg, cap)
    return env.graph(), sk


def seq(env, cfg, direction, n=6):
    """Collapsed controllable sequence for `direction`: per decision dA + A∧A (env-cleared, no undo/stay)."""
    env.reset()
    advance(env, cfg)                                  # collapse intro -> first controllable
    s = env.graph(); rows = []
    for i in range(n):
        sd = env.snap()
        s2, sk = settle(env, cfg, direction)
        dA = smae_pair(s, s2, cfg, True)
        s2snap = env.snap()
        aa = float("inf")
        for b in env.actions():                        # A∧A: can any real action return s2 -> s?
            env.restore(s2snap)
            sb, _ = settle(env, cfg, b)
            aa = min(aa, smae_pair(s, sb, cfg, True))
        env.restore(s2snap)
        rows.append({"sk": sk, "dA": dA, "aa": aa, "curv": dA + aa})
        s = s2
    return rows


def main():
    cfg = Config()
    from online_v5 import Live
    env = FastEnv(Live(cfg, "bp35"))
    NAME = {3: "LEFT", 4: "RIGHT"}
    for d in (3, 4):
        t = time.time()
        rows = seq(env, cfg, d, n=6)
        print(f"=== {NAME[d]} (snapshot, env-cleared) | {time.time()-t:.1f}s ===")
        for i, r in enumerate(rows, 1):
            tag = "  <== BIG IRREVERSIBLE EVENT" if (r["dA"] > 5 and r["aa"] > 1) else ""
            print(f"  decision {i}: bounce_collapsed={r['sk']:>2}  dA={r['dA']:4.1f}  A^A={r['aa']:4.1f}  curv={r['curv']:4.1f}{tag}")
        print()


if __name__ == "__main__":
    main()
