"""online_env.py — a drop-in replacement for FastEnv that the EXISTING solver (online_rollout_undo.Roller)
can run over, but driven by the real arc_agi API (gateway ONLINE with the ARC key, or local offline — same API)
and with REAL action-undo instead of deepcopy.

It exposes exactly the interface the solver calls: snap() / restore() / step() / graph() / actions() / live /
last_info. The trick that makes the solver work online unchanged:
  * snap()    -> records the current ACTION PATH from start (+ a perception hash to verify)
  * restore() -> goes back to a snapshot by UNDOING the extra actions with the symmetric-opposite move
                 (left<->right, up<->down, space->space) or ACTION7 where the game offers a true undo; a no-op
                 move (hit a wall, stayed put) needs no undo. If the opposite-undo doesn't verify, it RESETs and
                 replays the snapshot's path (always reliable).
  * step()    -> a real arc_agi step (+ perception).
No online_v5 / cleanupv6 / trained Prior — perception is graph_extract; intensity/reward stay the solver's.
"""
import os

import numpy as np

from curvature_wm_model import paths  # noqa: F401
import graph_extract
from hud_mask import hud_node_bboxes
from curvature_wm_model.model.reward_metrics import strip_hud

try:
    from arcengine import GameAction
except Exception:
    GameAction = None

CLICK = 6
INV = {1: 2, 2: 1, 3: 4, 4: 3, 5: 5}             # symmetric-opposite undo
WALK_SIBLING = os.environ.get("WALK_SIBLING", "1") == "1"   # restore to a SIBLING by reverse-trail walk (vs reset+replay)

_GA = None
def _ga_map():
    global _GA
    if _GA is None and GameAction is not None:
        _GA = {0: GameAction.RESET, 1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
               4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6, 7: GameAction.ACTION7}
    return _GA


def _inhash(g):
    # WHOLE-PIXEL identity: integer pixel bbox [ymin,xmin,ymax,xmax] replaces normalized cy/cx/sx/sy (no sub-cell jitter)
    items = [(str(n.get("type_hash")), tuple(n.get("color", [])), int(n.get("stab", 0)), int(n.get("area_bin", 0)),
             tuple(int(v) for v in n.get("bbox", (0, 0, 0, 0)))) for n in g["nodes"]]
    items.sort()
    h = 1469598103934665603
    for b in repr(items).encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


class _LiveShim:
    """Mimics the .live attribute the solver reads (prev_level, last_frame, prev_graph)."""
    prev_level = 0
    last_frame = None
    prev_graph = None


class OnlineEnv:
    def __init__(self, game, api_key=None, online=True):
        # ARC-AGI API: ONLINE (live game via gateway/key) or OFFLINE (local env/environment_files) — same API.
        if online:
            os.environ["OPERATION_MODE"] = "online"         # arc_agi's env var wins over the constructor arg
            if api_key:
                os.environ["ARC_API_KEY"] = api_key
        else:
            os.environ["OPERATION_MODE"] = "offline"        # local game files (ENVIRONMENTS_DIR), no network
        import arc_agi
        from arc_agi.base import OperationMode
        self.game = game
        self.arc = arc_agi.Arcade(operation_mode=(OperationMode.ONLINE if online else OperationMode.OFFLINE))
        self.scorecard_id = self.arc.open_scorecard()
        self.env = self.arc.make(game, scorecard_id=self.scorecard_id)
        self.mods, GT = graph_extract._imports(); self._GT = GT
        self.live = _LiveShim()
        self.last_info = {}
        self.path = []                                   # [(aid, xy)] from start
        self.regions = []
        self.undo7 = False
        self.inv = dict(INV)
        self.win_levels = 1                              # levels needed to fully WIN the game (from the API frame)
        self.acount = 0                                  # total real API actions taken (diagnostics only)
        self._cur_hash = None
        self._reset_replay([])                           # initial reset + perceive

    # ---- perception ----
    def _grid(self, f):
        fr = getattr(f, "frame", None)
        if not fr:
            return self.live.last_frame if self.live.last_frame is not None else np.zeros((64, 64), dtype=np.int32)
        arr = np.array(fr[-1], dtype=np.int32)
        while arr.ndim > 2:
            arr = arr[0]
        for (B0, C0, B1, C1) in getattr(self, "regions", []):    # BLANK the saved HUD area + everything inside it to
            arr[B0:B1 + 1, C0:C1 + 1] = getattr(self, "_bg", 0)  # bg, EVERY frame -> HUD never enters frame OR perception
        return arr

    def _perceive(self, arr):
        self._tr._frame_idx_hint = self._fi; self._fi += 1
        n, e, d = graph_extract._process_frame(arr, self._tr, self.mods)
        g = {"nodes": n, "edges": e, "deltas": d}
        return strip_hud(g, self.regions) if self.regions else g

    def _update(self, f):
        self.live.last_frame = self._grid(f)
        self.live.prev_level = int(getattr(f, "levels_completed", self.live.prev_level))
        wl = getattr(f, "win_levels", None)
        if wl:
            self.win_levels = int(wl)
        self.live.prev_graph = self._perceive(self.live.last_frame)
        st = getattr(getattr(f, "state", None), "name", "")
        self.last_info = {"win": st == "WIN", "game_over": st == "GAME_OVER"}
        self._last_frame_obj = f
        self._cur_hash = _inhash(self.live.prev_graph)

    # ---- API helpers ----
    def _ga(self, aid, xy=None):
        a = _ga_map()[int(aid)]
        if int(aid) == CLICK and xy is not None:
            a.set_data({"x": int(xy[0]), "y": int(xy[1])})
        return a

    def _avail(self, f):
        return [int(getattr(a, "value", a)) for a in (getattr(f, "available_actions", []) or [])]

    def _api_step(self, aid, xy=None):
        a = _ga_map()[int(aid)]
        if int(aid) == CLICK and xy is not None:                          # click coords MUST go via the wrapper's data=
            return self.env.step(a, data={"x": int(xy[0]), "y": int(xy[1])})   # (set_data alone is ignored by the engine)
        return self.env.step(a)

    def _reset_replay(self, path):
        self._tr = self._GT(graveyard_ttl=30, history_limit=-1); self._fi = 0
        f = self.env.reset(); self.acount = getattr(self, "acount", 0) + 1
        for (aid, xy) in path:
            f = self._api_step(aid, xy); self.acount += 1
        self.path = list(path)
        self._update(f)

    # ---- FastEnv-compatible interface (what the solver calls) ----
    def graph(self):
        return self.live.prev_graph

    def actions(self):
        return [a for a in self._avail(self._last_frame_obj) if a not in (0, 7)]   # moves/clicks; undo(7) handled internally

    def step(self, aid, xy=None):
        f = self._api_step(aid, xy); self.acount += 1; self.n_fwd = getattr(self, "n_fwd", 0) + 1
        self.path.append((int(aid), tuple(xy) if xy else None))
        self._update(f)
        done = bool(self.last_info.get("win") or self.last_info.get("game_over"))
        return self.live.prev_graph, done

    def snap(self):
        return (list(self.path), _inhash(self.live.prev_graph))

    def restore(self, snap):
        path, target_hash = snap
        self.n_restore = getattr(self, "n_restore", 0) + 1
        if self._cur_hash == target_hash:
            self.n_restore_noop = getattr(self, "n_restore_noop", 0) + 1
            return                                                  # already there (e.g. probed a no-op)
        # WALK the REVERSE trail instead of reset+replay: undo back to the longest common ancestor of (current
        # path, target path), then replay forward to the target. Backtracking to an ANCESTOR is the special case
        # forward==[] (pure undo); a SIBLING branch shares a prefix, so undo to the fork then go forward — no reset.
        # Undo = ACTION7 (true undo, click games) / symmetric-opposite MOVE (left<->right, up<->down, space->space).
        # Reset+replay only if a step has no reverse (one-way) OR the walk would cost more than a reset.
        cp = 0
        n = min(len(self.path), len(path))
        while cp < n and self.path[cp] == path[cp]:                 # longest common prefix = the fork point
            cp += 1
        suffix = self.path[cp:]                                     # UNDO these (current -> fork)
        forward = path[cp:]                                         # then REPLAY these (fork -> target)
        eligible = (cp == len(path)) if not WALK_SIBLING else (len(suffix) + len(forward) <= 1 + len(path))
        if eligible:                                               # walk (prefix always; sibling when no costlier)
            ok = True
            avail = set(self.actions())                            # the game's OWN actions, read live (excl 0/7)
            for (aid, xy) in reversed(suffix):
                if self.undo7:                                     # ACTION7 reverts (click games: "undo for click")
                    u = 7
                elif aid == 5 and 5 in avail:                      # space -> space
                    u = 5
                elif aid == 1 and 2 in avail:                      # up   -> down
                    u = 2
                elif aid == 2 and 1 in avail:                      # down -> up
                    u = 1
                elif aid == 3 and 4 in avail:                      # left  -> right
                    u = 4
                elif aid == 4 and 3 in avail:                      # right -> left
                    u = 3
                else:
                    ok = False; break                             # this step is one-way -> no reverse -> reset
                self._api_step(u); self.acount += 1; self.path.pop()
            if ok:
                for (aid, xy) in forward:                          # fork -> target: replay the known-good actions
                    self._api_step(aid, xy); self.acount += 1
                    self.path.append((int(aid), tuple(xy) if xy else None))
                self.path = list(path); self._update(self.env.observation_space)
                if self._cur_hash == target_hash:
                    self.n_walk = getattr(self, "n_walk", 0) + 1
                    return                                          # verified via reverse-trail walk
        self.n_replay = getattr(self, "n_replay", 0) + 1            # walk failed / costlier -> O(path) fallback
        self.n_replay_steps = getattr(self, "n_replay_steps", 0) + 1 + len(path)
        self._reset_replay(path)                                    # reliable fallback: reset + replay the recorded path


def detect_undo(env):
    """Capture undo from the API: does ACTION7 revert a move? which move reverts which? (sets env.undo7/env.inv)."""
    base = env.snap()
    def seq(s):
        env.restore(base); env._reset_replay([])
        h0 = _inhash(env.live.prev_graph)
        for a in s:
            env.step(a)
        return _inhash(env.live.prev_graph), h0
    _, h0 = seq([])
    moves = [a for a in env.actions() if a != CLICK]
    inv, a7 = {}, False
    for a in moves:
        sa, _ = seq([a])
        if sa == h0:
            continue
        sa7, _ = seq([a, 7])
        if sa7 == h0:
            a7 = True
        for b in moves:
            sab, _ = seq([a, b])
            if sab == h0:
                inv[a] = b; break
    for a, b in list(inv.items()):
        inv.setdefault(b, a)
    env.undo7 = a7
    env.inv = inv or dict(INV)
    env._reset_replay([])


def detect_hud(env):
    """Detect the HUD ONCE at game start from OUR PERCEPTION's nodes (whole-pixel bboxes), SAVE its location, and
    BLANK that area + everything inside it to bg on EVERY frame forever (in _grid -> removes the HUD from the
    FRAME and therefore from PERCEPTION). Node-based, ANY game, no human runs, no image re-analysis."""
    try:
        env.regions = hud_node_bboxes(env.graph()["nodes"])   # regions empty here -> graph() is raw (un-masked)
        vals, cnts = np.unique(env.live.last_frame, return_counts=True)
        env._bg = int(vals[cnts.argmax()])                    # background color used to blank the saved HUD area
    except Exception:
        env.regions = []
    env._reset_replay([])                                      # re-perceive start WITH the area blanked (frame + perception)
