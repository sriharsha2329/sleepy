"""solving_graph_deep/online_rollout.py — OnlineRewardGuidedRollout.

An ONLINE reward-guided search that does NOT build the full graph first and does NOT run a global shortest-path
after. It grows an explored skeleton gradually by interacting with the env via do(action)/undo(action), ranks
actions by policy/Q/prior (lightly nudged by motif memory), and lets the CUSTOM REWARD decide which probed edges
are useful and which to commit. No-op actions (wall hits / useless clicks) are logged per (state, action) and
never repeated; click no-ops are also logged by clicked-node ORBIT so equivalent nodes are skipped. Local
radius-style triangles + intensity are maintained online as edges are observed; backtracking jumps to the best
OPEN FRONTIER edge (by reward/Q/prior), not the nearest parent.

Reward (the spec's variant — note max(0, Δintensity): moving high->low intensity is NOT punished):
    r = dA + 10·A∧A − 10·max(0, intensity_next − intensity_cur) − 10·intensity_next − degree_next
A∧A is model-straight 0 unless the transition is an EVENT (dA > running μ+σ) → brute-forced real no-return holonomy.
State identity = INPUT-graph hash (HUD dropped). Stops at the first level-up (goal). Policy/Q are pluggable
(default uniform / 0 → the real probed reward drives commits).

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.solving_graph_deep.online_rollout [game] [max_steps]
"""
import os
os.environ.setdefault("OPERATION_MODE", "online")     # default to the ARC-AGI API (online); OnlineEnv also forces it
import time
import heapq
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from curvature_wm_model import paths  # noqa: F401
sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

import numpy as np
import networkx as nx

from config import Config
from hud_mask import hud_regions, TOL
from curvature_wm_model.hodge_flow.agent import _name
from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair
from curvature_wm_model.hodge_flow.rollout import branch_actions, CLICK
from curvature_wm_model.solving_graph_deep.metrics_ls20 import InputHashSearch
from curvature_wm_model.solving_graph_deep.interactive_graph import _frame_b64
from curvature_wm_model.model.reward_metrics import faces_intensity, strip_hud

def _argvint(i, d):                                      # import-safe: don't crash when imported (argv is the host's)
    try:
        return int(sys.argv[i])
    except (IndexError, ValueError):
        return d
GAME = sys.argv[1] if (len(sys.argv) > 1 and not str(sys.argv[1]).startswith("-")) else "ls20"
MAX_STEPS = _argvint(2, 2000)
TARGET = _argvint(3, 1)                                   # number of levels to COMPLETE before stopping

SEEN_PRUNE = os.environ.get("SEEN_PRUNE", "1") == "1"   # depth-based seen-state pruning (diagnostic toggle)
SIGMA_LOCK_N = 5            # lock the "first sigma" of dA after this many committed samples
INTENSITY_MAX = float(os.environ.get("INTENSITY_MAX", "3"))   # if a state is DEEPER than this, UNDO back to <= this
DEATH_PENALTY = float(os.environ.get("DEATH_PENALTY", "100")) # game_over reward penalty (mirror of the +100 level-up)
BRUTE = os.environ.get("BRUTE", "0") == "1"                   # force brute-A∧A; DEFAULT OFF -> triangles use the MODEL
MODEL_RETURN_K = int(os.environ.get("MODEL_RETURN_K", "4"))   # triangles: env-VERIFY only the model's top-k returns
CLICK_BUDGET0 = int(os.environ.get("CLICK_BUDGET0", "2"))     # (kept for compat) top-2 clicks
ACTION_BUDGET0 = int(os.environ.get("ACTION_BUDGET0", "2"))   # start: top-2 OVERALL actions (moves+clicks ranked) per state
INT_RECOMPUTE = int(os.environ.get("INT_RECOMPUTE", "1"))     # recompute intensity (faces) every N new edges (speed)
GRID_CLICK = os.environ.get("GRID_CLICK", "0") == "1"        # fallback: probe a coarse grid of clicks at dead-ends
GRID_N = int(os.environ.get("GRID_N", "8"))                  # grid is GRID_N x GRID_N click positions over the frame
HUMAN_CLICKS = os.environ.get("HUMAN_CLICKS", "0") == "1"    # teacher prior: probe the click positions humans used
HCLICK_CAP = int(os.environ.get("HCLICK_CAP", "40"))        # cap on distinct human-click positions used
WALL_SECS = float(os.environ.get("WALL_SECS", "0"))         # self-stop after this many seconds (0=off) + emit the path
TEACHER_W = float(os.environ.get("TEACHER_W", "10"))        # sparse-space TEACHER: reward bonus per distinct-node-TYPE-count
                                                            # change, ONLY when dA<μ+σ (dA is ~noise there). TEACHER_W=0 off.
TRIANGLE_DA_TAU = 3.0       # triangle zero-gain thresholds (bootstrap; triangles are a secondary structure)
TRIANGLE_WEDGE_TAU = 1.0
CLICK_BUDGET_SCHEDULE = [2, 3, 4, 5, 6, 7, 8, 9, 10]    # top-k CLICKS: gradual 2->...->10, CAP at 10 (NEVER 'take all')
ACTION_BUDGET_SCHEDULE = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # top-k MOVES:  gradual 2->...->10, CAP at 10 (NEVER 'take all')


# ============================ action wrapper ============================
@dataclass(frozen=True)
class Action:
    type: str                 # "move" | "click"
    aid: int
    xy: Any = None            # (px,py) for clicks, else None
    node: Optional[str] = None    # clicked node pid
    orbit: Optional[str] = None   # clicked node orbit id

    @property
    def direction(self):
        return self.aid


def action_key(a: Action):
    if a.type == "move":
        return ("move", a.aid)
    return ("click", a.node if a.node is not None else a.xy)


def abstract_action_key(a: Action):
    if a.type == "move":
        return ("move", a.aid)
    return ("click_orbit", a.orbit) if a.orbit is not None else ("click", a.node)


# ============================ data structures ============================
@dataclass
class StateMemory:
    best_depth_seen: int = 10 ** 9
    tried_actions: set = field(default_factory=set)
    bad_actions: set = field(default_factory=set)
    click_budget: int = CLICK_BUDGET0
    action_budget: int = ACTION_BUDGET0
    closed: bool = False
    explored: bool = False            # have we been AT this state and probed its actions (vs merely discovered)?
    pending: list = field(default_factory=list)   # remaining UNEXPLORED escape candidates, escape-sorted (DFS)
    parent: Optional[str] = None
    parent_action: Any = None
    no_progress_count: int = 0
    saw_death: bool = False            # (deep) a probed action from this state led to game_over -> death-trap candidate


@dataclass
class EdgeRecord:
    src: str
    dst: str
    action: Any
    reward: float
    dA: float
    A_wedge_A: float
    intensity_src: float
    intensity_dst: float
    degree_dst: int
    object_event: bool
    level_event: bool
    depth: int


@dataclass(order=True)
class FrontierItem:
    src_hash: str = field(compare=False)
    action: Any = field(compare=False)
    dst_hash: str = field(compare=False)
    reward: float = field(compare=False)
    q_value: float = field(compare=False)
    prior_prob: float = field(compare=False)
    depth: int = field(compare=False)
    undo_distance: int = field(compare=False)
    motif_bonus: float = field(compare=False)
    intensity_dst: float = field(default=0.0, compare=False)   # escape ranking: lower = more "out"


@dataclass
class MotifActionStats:
    tries: int = 0
    successes: int = 0
    noops: int = 0
    reward_sum: float = 0.0
    dA_sum: float = 0.0

    @property
    def avg_reward(self):
        return self.reward_sum / max(1, self.tries)

    @property
    def success_rate(self):
        return self.successes / max(1, self.tries)

    @property
    def noop_rate(self):
        return self.noops / max(1, self.tries)


class OnlineGraph:
    def __init__(self):
        self.out_edges = defaultdict(list)   # src -> [EdgeRecord]
        self.in_edges = defaultdict(list)    # dst -> [EdgeRecord]
        self.edge_map = {}                   # (src, action_key, dst) -> EdgeRecord
        self.closed_triangles = set()
        self.open_triangles = set()

    def add_edge(self, edge: EdgeRecord):
        key = (edge.src, action_key(edge.action), edge.dst)
        if key in self.edge_map:
            return False
        self.edge_map[key] = edge
        self.out_edges[edge.src].append(edge)
        self.in_edges[edge.dst].append(edge)
        return True

    def undirected_edges(self):
        return set((min(e.src, e.dst), max(e.src, e.dst)) for e in self.edge_map.values() if e.src != e.dst)

    def degree(self, h):
        return sum(1 for (u, v) in self.undirected_edges() if h == u or h == v)


# ============================ online triangles (radius-2) ============================
def is_zero_gain_triangle(edges):
    total_reward = sum(e.reward for e in edges)
    total_abs_dA = sum(abs(e.dA) for e in edges)
    has_event = any(e.object_event or e.level_event for e in edges)
    max_wedge = max(abs(e.A_wedge_A) for e in edges)
    return (total_reward <= 0.0 and total_abs_dA < TRIANGLE_DA_TAU
            and max_wedge < TRIANGLE_WEDGE_TAU and not has_event)


def update_local_triangles(graph: OnlineGraph, new_edge: EdgeRecord):
    u, v = new_edge.src, new_edge.dst
    found = []
    for e1 in graph.out_edges[u]:                       # u -> x -> v  (+ u -> v)
        x = e1.dst
        for e2 in graph.out_edges[x]:
            if e2.dst == v and len({u, x, v}) == 3:
                found.append((tuple(sorted([u, x, v])), [e1, e2, new_edge]))
    for e1 in graph.out_edges[v]:                       # v -> x -> u  (+ new edge closes)
        x = e1.dst
        for e2 in graph.out_edges[x]:
            if e2.dst == u and len({u, x, v}) == 3:
                found.append((tuple(sorted([u, x, v])), [new_edge, e1, e2]))
    for tri, edges in found:
        if is_zero_gain_triangle(edges):
            graph.closed_triangles.add(tri); graph.open_triangles.discard(tri)
        else:
            graph.open_triangles.add(tri)
    return found


# ============================ motifs (weak teacher) ============================
def bucket(x):
    return int(min(4, max(0.0, x)))      # 0,1,2,3,4+


def _inhash(g):
    """INPUT-graph hash (FNV-1a), model-free. g is already HUD-stripped, so all its nodes are playfield.
    WHOLE-PIXEL identity: the integer pixel bbox [ymin,xmin,ymax,xmax] (1/64 resolution) replaces the normalized
    cy/cx/sx/sy -> sub-cell jitter no longer mints new states."""
    items = []
    for n in g["nodes"]:
        items.append((str(n.get("type_hash")), tuple(n.get("color", [])), int(n.get("stab", 0)),
                      int(n.get("area_bin", 0)), tuple(int(v) for v in n.get("bbox", (0, 0, 0, 0)))))
    items.sort()
    h = 1469598103934665603
    for b in repr(items).encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


class Roller:
    def __init__(self, game, policy=None, q_fn=None, env=None):
        if env is not None:                                  # injected ONLINE env (OnlineEnv: ARC API + action-undo)
            self.env = env
            self.cfg = Config()
            self.regions = getattr(env, "regions", None) or []   # live HUD: detect_hud(env) set env.regions
        else:
            self.s = InputHashSearch(Config(), game, k=3)
            self.env = self.s.env
            self.cfg = self.s.cfg
            self.regions = hud_regions(game)            # STANDARD per-game HUD regions (the hud_mask algorithm)
        self._hclicks = self._load_human_clicks(game) if HUMAN_CLICKS else []
        self.policy = policy
        self.q_fn = q_fn
        self.graph = OnlineGraph()
        self.memory = defaultdict(StateMemory)
        self.motif_memory = defaultdict(MotifActionStats)
        self.frontier = []                  # min-heap of (priority_tuple, FrontierItem)
        self.path = []                      # [(state_hash, action)]
        self.reg = {}                       # hash -> {snap, g, lvl, id}
        self.intensity = {}                 # hash -> intensity (recomputed online from real edges)
        self.da_vals = []                   # running dA for the event gate
        self.sigma0 = None                  # the FIRST sigma of dA -> all significance thresholds derive from it
        self.start_lvl = 0
        self.timeline = []
        self.n_level_probes = 0             # how many probes ever observed a level-up (diagnostic)

    # ---- significance thresholds, all derived from the first sigma of dA (no hand-picked constants) ----
    @property
    def sigma(self):
        return self.sigma0 if self.sigma0 else 1.0   # bootstrap = 1.0 until the first sigma is locked
    tau_a = property(lambda self: self.sigma)        # |dA| significance
    tau_r = property(lambda self: self.sigma)        # |reward| significance
    tau_wedge = property(lambda self: self.sigma)    # |A∧A| significance

    def _lock_sigma(self):
        if self.sigma0 is None and len(self.da_vals) >= SIGMA_LOCK_N:
            self.sigma0 = float(np.std(self.da_vals)) or 1.0

    # ---- env integration ----
    def _hash(self, g):
        return _inhash(g)                            # g is HUD-stripped by observe()

    def _register(self, g, snap, lvl, frame=None):
        h = self._hash(g)
        if h not in self.reg:
            self.reg[h] = {"snap": snap, "g": g, "lvl": lvl, "id": len(self.reg), "frame": frame}
        return h

    def _frame(self):
        return getattr(self.env.live, "last_frame", None)

    def observe(self):
        return strip_hud(self.env.graph(), self.regions)    # STANDARD HUD removal on every observation

    # ---- temp graph store: a fresh per-run graph the solver can refer back to (replaces the previous play) ----
    def _init_graph_tmp(self):
        import shutil
        self._graph_tmp = paths.HERE / "tmp_graph"
        shutil.rmtree(self._graph_tmp, ignore_errors=True)  # replace the previous play's graph
        self._graph_tmp.mkdir(parents=True, exist_ok=True)
        self._persist_graph()

    def _persist_graph(self):
        """Dump the ACTUAL state-transition graph the solver built: every registered state (node) with its
        per-object count, level, intensity; every observed edge (from OnlineGraph.edge_map: action, reward, dA,
        committed?); the triangles; and the committed path. File named per game: graph_<game>.json."""
        d = getattr(self, "_graph_tmp", None)
        if d is None:
            return
        idof = {h: self.reg[h]["id"] for h in self.reg}
        committed = set((e["src"], e["dst"]) for e in self.timeline)
        nodes = [{"id": self.reg[h]["id"], "hash": h, "lvl": int(self.reg[h]["lvl"]),
                  "inten": round(float(self.intensity.get(h, 0.0)), 3),
                  "objects": len(self.reg[h]["g"]["nodes"])} for h in self.reg]
        edges = []
        for e in self.graph.edge_map.values():
            edges.append({"src": idof.get(e.src), "dst": idof.get(e.dst), "action": _name(e.action.aid),
                          "xy": (list(e.action.xy) if getattr(e.action, "xy", None) else None),
                          "r": round(float(e.reward), 2), "dA": round(float(e.dA), 2),
                          "committed": (e.src, e.dst) in committed})
        path = [{"aid": int(a.aid), "action": _name(a.aid), "xy": (list(a.xy) if a.xy else None)}
                for (_h, a) in self.path]
        game = getattr(self.env, "game", None) or "game"
        (d / f"graph_{game}.json").write_text(json.dumps({
            "game": game, "states": len(self.reg), "edges": len(self.graph.edge_map),
            "open_triangles": len(self.graph.open_triangles), "closed_triangles": len(self.graph.closed_triangles),
            "nodes": nodes, "graph_edges": edges, "committed_path": path,
        }, default=str))

    # ---- reward terms ----
    def _thr(self):
        return (float(np.mean(self.da_vals)) + float(np.std(self.da_vals))) if self.da_vals else float("inf")

    def _brute_return(self, snap_next, g_orig):
        best = float("inf")
        for (_bk, baid, bxy) in branch_actions(self.env, self.cfg, self.env.graph()):   # enumerate on full graph
            self.env.restore(snap_next); self.env.step(baid, bxy)
            best = min(best, smae_pair(g_orig, strip_hud(self.env.graph(), self.regions), self.cfg))  # compare HUD-free
        self.env.restore(snap_next)
        return 0.0 if best == float("inf") else float(best)

    def _model_return(self, snap_next, g_orig, g_next):
        """A∧A with TRIANGLES FROM THE MODEL: the inverse/click heads rank which return action maps the
        post-action state g_next back toward the origin g_orig; the env then VERIFIES only the top-k of those
        (forward stays env/brute). Same smae_pair scale as _brute_return, but a handful of steps instead of
        every action — the win on click games. Falls back to brute if the model is absent or errors."""
        if self.policy is None or not hasattr(self.policy, "rank_returns"):
            return self._brute_return(snap_next, g_orig)
        self.env.restore(snap_next)
        try:
            cands = list(branch_actions(self.env, self.cfg, self.env.graph()))
            ranked = self.policy.rank_returns(g_next, g_orig, cands)
        except Exception:
            return self._brute_return(snap_next, g_orig)
        best = float("inf")
        for (_bk, baid, bxy) in ranked[:MODEL_RETURN_K]:                 # env VERIFIES the model's top-k returns
            self.env.restore(snap_next); self.env.step(baid, bxy)
            best = min(best, smae_pair(g_orig, strip_hud(self.env.graph(), self.regions), self.cfg))
        self.env.restore(snap_next)
        return 0.0 if best == float("inf") else float(best)

    def _grid_clicks(self):
        """coarse GRID_N x GRID_N grid of click positions over the frame (click-to-start / click-anywhere fallback)."""
        G = int(getattr(self.cfg, "grid", 64) or 64)
        stp = max(2, G // GRID_N)
        return [Action("click", CLICK, xy=(px, py))
                for py in range(stp // 2, G, stp) for px in range(stp // 2, G, stp)]

    def _load_human_clicks(self, game, cap=None):
        """teacher prior: the most-used distinct click (x,y) positions from this game's human runs."""
        from collections import Counter
        import json as _json
        cap = cap or HCLICK_CAP
        cnt = Counter()
        gd = paths.GRAPH_DATA / game
        for fp in sorted(gd.glob("*.jsonl"))[:6] if gd.exists() else []:
            try:
                for ln in open(fp):
                    o = _json.loads(ln)
                    if not o.get("meta") and o.get("action") == 6 and o.get("click"):
                        cnt[tuple(o["click"])] += 1
            except Exception:
                continue
        return [xy for xy, _ in cnt.most_common(cap)]

    def _try_action(self, cur_h, snap_cur, g_cur, a, mem, depth):
        """probe one action; register/edge/intensity + classify. Returns (status, payload):
        skip | noop | ('death',(a,p)) | useless | ('escape',(i_nxt,-r,a,p)) | ('levelup',(a,p))."""
        k = action_key(a)
        if k in mem.tried_actions or k in mem.bad_actions:
            return ("skip", None)
        mem.tried_actions.add(k)
        p = self.probe(cur_h, snap_cur, g_cur, a)
        self.update_motif(cur_h, a, p)
        if self.is_noop(p):
            mem.bad_actions.add(k)
            if a.type == "click" and a.orbit is not None:
                mem.bad_actions.add(("click_orbit", a.orbit))
            self._on_probe(cur_h, a, p, False)
            return ("noop", None)
        if p["game_over"]:                                      # DEATH (deep): penalize (in r) + PRUNE — never commit
            mem.bad_actions.add(k)                              # /retry this action, and never register/explore death
            if a.type == "click" and a.orbit is not None:
                mem.bad_actions.add(("click_orbit", a.orbit))
            self._on_probe(cur_h, a, p, False)
            return ("death", (a, p))
        self.n_level_probes += int(p["lvl_event"])
        self._register(p["g2"], p["snap2"], p["lvl2"], p["frame"])
        edge = EdgeRecord(cur_h, p["h2"], a, p["r"], p["dA"], p["A"], p["i_cur"], p["i_nxt"],
                          p["deg_nxt"], p["obj"], p["lvl_event"], depth + 1)
        if self.graph.add_edge(edge):
            update_local_triangles(self.graph, edge)
            self.da_vals.append(p["dA"])
            self._int_ctr = getattr(self, "_int_ctr", 0) + 1
            if self._int_ctr >= INT_RECOMPUTE:
                self._recompute_intensity(); self._int_ctr = 0
        self._on_probe(cur_h, a, p, True)
        if p["lvl_event"]:
            return ("levelup", (a, p))
        if self.is_useful(p) and not self.memory[p["h2"]].explored:
            return ("escape", (p["i_nxt"], -p["r"], a, p))
        if self.memory[p["h2"]].explored and not p["lvl_event"]:    # TRIANGULAR: action returns to an already-visited
            mem.bad_actions.add(k)                                   #   state -> cache it so we never re-probe this loop
        return ("useless", None)

    def _recompute_intensity(self):
        edges = {(e.src, e.dst): 1 for e in self.graph.edge_map.values()}
        nodes = set(self.reg) | set(self.intensity)
        self.intensity = faces_intensity(edges, nodes)["intensity"] if edges else {}

    # ---- enumerate / rank / filter ----
    def enumerate_actions(self, g):
        acts = []
        # px,py -> (pid, orbit) for resolving a click target to its node
        pxy = {}
        for n in g["nodes"]:
            pxy[(round(float(n.get("px", 0)), 2), round(float(n.get("py", 0)), 2))] = (n.get("pid"), n.get("orbit"))
        for (key, aid, xy) in branch_actions(self.env, self.cfg, g):
            if aid == CLICK and xy is not None:
                pid, orbit = pxy.get((round(float(xy[0]), 2), round(float(xy[1]), 2)), (None, None))
                acts.append(Action("click", CLICK, xy=tuple(xy), node=pid, orbit=orbit))
            else:
                acts.append(Action("move", int(aid)))
        return acts

    def motif_signatures(self, h, action=None):
        m = [("degree_bucket", bucket(self.graph.degree(h))), ("intensity_bucket", bucket(self.intensity.get(h, 0.0)))]
        if action is not None and action.type == "click":
            m.append(("click_orbit", action.orbit)); m.append(("click_node", action.node))
        if action is not None and action.type == "move":
            m.append(("move_dir", action.direction))
        return m

    def motif_bonus(self, h, action):
        bonus = 0.0
        for motif in self.motif_signatures(h, action):
            st = self.motif_memory.get((motif, abstract_action_key(action)))
            if st is None or st.tries < 2:
                continue
            conf = min(1.0, math.sqrt(st.tries) / 5.0)
            bonus += conf * (st.avg_reward + st.success_rate - st.noop_rate)
        return bonus

    def _prior(self, h, a):
        if self.policy is None:
            return 1.0
        if hasattr(self.policy, "state_dist"):              # PolicyPrior: the TRAINED model recommends from the state
            pd = self.__dict__.setdefault("_pdist", {})     # cache (a_p, c_p, pid->graph-index) per state hash
            if h not in pd:
                g = self.reg[h]["g"]
                a_p, c_p = self.policy.state_dist(g)
                pid2idx = {n.get("pid"): i for i, n in enumerate(g["nodes"])}   # click node id -> graph index (= c_p index)
                pd[h] = (a_p, c_p, pid2idx)
            a_p, c_p, pid2idx = pd[h]
            if a.type == "move":
                return float(a_p[a.aid]) if int(a.aid) < len(a_p) else 1e-6
            idx = pid2idx.get(a.node, -1)                   # P(click node i) = P(CLICK)*P(node i); i = clicked node's index
            return float(a_p[CLICK]) * (float(c_p[idx]) if 0 <= idx < len(c_p) else 1e-6)
        return self.policy(h, a)                             # legacy callable prior

    def _q(self, h, a):
        return self.q_fn(h, a) if self.q_fn else 0.0

    def rank_actions(self, h, actions):
        ranked = []
        for a in actions:
            pr = max(self._prior(h, a), 1e-8)
            qv = self._q(h, a)
            mb = self.motif_bonus(h, a)
            ranked.append((qv + 0.1 * math.log(pr) + 0.2 * mb, qv, pr, mb, a))
        ranked.sort(reverse=True, key=lambda x: x[0])
        return ranked

    def allowed_actions(self, h, ranked, mem):
        avail = []
        for _, _, _, _, a in ranked:                         # `ranked` is ordered by the model prior (best first)
            k = action_key(a)
            if k in mem.bad_actions or k in mem.tried_actions:
                continue
            if a.type == "click" and ("click_orbit", a.orbit) in mem.bad_actions:
                continue
            avail.append(a)
        moves = [a for a in avail if a.type == "move"]        # `avail` is ranked by the model prior (best first)
        clicks = [a for a in avail if a.type == "click"]
        if self.policy is not None:                           # MODEL-SELECTED: top-k OVERALL (moves+clicks ranked together)
            return avail[:mem.action_budget]                  # _widen escalates this k at stuck states (early-commit gated below)
        return moves + clicks[:mem.click_budget]             # (no model) all moves + top-budget clicks

    # ---- probe one action with do -> observe -> measure -> undo ----
    def probe(self, h, snap_cur, g_cur, action):
        self.env.restore(snap_cur)
        self.env.step(action.aid, action.xy)
        g2 = self.observe(); snap2 = self.env.snap(); fr2 = self._frame()
        h2 = self._hash(g2); lvl2 = self.env.live.prev_level; info = self.env.last_info
        lvl_cur = self.reg[h]["lvl"]
        level_event = (lvl2 > lvl_cur) or bool(info.get("win"))
        game_over = bool(info.get("game_over"))                # DEATH signal from the game (GAME_OVER state)
        same = (h2 == h)
        dA = smae_pair(g_cur, g2, self.cfg)
        aa = 0.0
        if dA > self._thr() and not level_event:                    # EVENT (dA > μ+σ) -> close the no-return A∧A triangle:
            aa = self._brute_return(snap2, g_cur) if BRUTE else self._model_return(snap2, g_cur, g2)   # MODEL ranks the
            #   return action(s); env VERIFIES only top-k (MODEL_RETURN_K) instead of brute-replaying every action
        i_cur = self.intensity.get(h, 0.0); i_nxt = self.intensity.get(h2, 0.0)
        deg_nxt = self.graph.degree(h2)
        obj_event = (not same) and _type_multiset(g_cur) != _type_multiset(g2)
        r = (dA + 10.0 * aa - 10.0 * max(0.0, i_nxt - i_cur) - 10.0 * i_nxt - deg_nxt)
        # TEACHER = pure SELECTION HEURISTIC. NOT added to reward r (that would corrupt A∧A / triangles). When dA is
        # sub-(μ+σ) (dA ~ noise) and the DISTINCT node-TYPE count changes (e.g. 6->5), flag it so selection can prefer it.
        nt_cur = len({str(n.get("type_hash")) for n in g_cur["nodes"]})
        nt2 = len({str(n.get("type_hash")) for n in g2["nodes"]})
        teacher = (abs(dA) < self._thr()) and (nt_cur != nt2)
        d_types = abs(nt_cur - nt2) if teacher else 0
        if game_over:
            r -= DEATH_PENALTY                                 # penalize death (mirror of +100 level-up)
        actionable = len(self.env.actions()) > 0                # does the NEXT frame offer a real move/click? (else not a state)
        self.env.restore(snap_cur)
        return {"g2": g2, "snap2": snap2, "frame": fr2, "h2": h2, "lvl2": lvl2, "dA": dA, "A": aa,
                "i_cur": i_cur, "i_nxt": i_nxt, "deg_nxt": deg_nxt, "obj": obj_event, "teacher": teacher,
                "lvl_event": level_event, "game_over": game_over, "r": r, "same": same, "actionable": actionable}

    def is_noop(self, p):
        # a frame that offers NO action is NOT a state for us (skip it), unless it's a level-up
        if not p.get("actionable", True) and not p["lvl_event"]:
            return True
        # SAME state (hashes match) + sub-sigma dA + no event => wall hit / useless click -> cache in bad_actions.
        # Keyed on `same` (matching states), NOT dA≈0 alone — the exploration space is sparse, don't over-prune.
        return p["same"] and abs(p["dA"]) < self.tau_a and not p["obj"] and not p["lvl_event"]

    def is_useful(self, p):
        return (p["r"] > 0.0 or abs(p["dA"]) >= self.tau_a or abs(p["r"]) >= self.tau_r
                or abs(p["A"]) >= self.tau_wedge or p["obj"] or p["lvl_event"] or not p["same"])

    def update_motif(self, h, action, p):
        success = self.is_useful(p) and p["r"] > 0.0
        noop = self.is_noop(p)
        for motif in self.motif_signatures(h, action):
            st = self.motif_memory[(motif, abstract_action_key(action))]
            st.tries += 1; st.reward_sum += p["r"]; st.dA_sum += p["dA"]
            st.successes += int(success); st.noops += int(noop)

    # ---- main loop (DEPTH-FIRST with restore-undo; backs OUT of intensity > INTENSITY_MAX) ----
    def run(self):
        g0 = self.observe(); snap0 = self.env.snap(); self.start_lvl = self.env.live.prev_level
        h0 = self._register(g0, snap0, self.start_lvl, self._frame())
        self.memory[h0].best_depth_seen = 0
        cur_h = h0
        self._t0 = time.time()
        self._init_graph_tmp()                  # fresh temp graph for THIS run (replaces the previous play's graph)

        for step in range(MAX_STEPS):
            if WALL_SECS and (time.time() - self._t0) > WALL_SECS:    # self-stop -> _result -> emit the AS-PLAYED path
                print(f"  wall-clock stop at {WALL_SECS:g}s, step {step}: committed {len(self.path)}, "
                      f"states {len(self.reg)}", flush=True)
                return self._result(False, step)
            self._lock_sigma()

            # ---- THE POWER OF UNDO: if we've drifted DEEPER than INTENSITY_MAX (into the belt core), UNDO
            #      (restore) one committed step at a time back to the CLOSEST ancestor lying on intensity <= MAX,
            #      then explore a different way OUT. (Env has no action-7 undo; snapshot-restore IS the undo.) ----
            undone = 0
            while self.intensity.get(cur_h, 0.0) > INTENSITY_MAX and self.path:
                src, _a = self.path.pop()
                self.env.restore(self.reg[src]["snap"]); cur_h = src; undone += 1
            if undone:
                print(f"  step {step} | UNDO x{undone} (too deep) -> state {self.reg[cur_h]['id']} "
                      f"intensity={self.intensity.get(cur_h,0.0):g} <= {INTENSITY_MAX:g}", flush=True)
                self.timeline.append({"src": self.reg[cur_h]["id"], "dst": self.reg[cur_h]["id"],
                                      "a": f"UNDOx{undone}", "r": 0.0, "dA": 0.0, "up": False})

            mem = self.memory[cur_h]
            snap_cur = self.reg[cur_h]["snap"]; g_cur = self.reg[cur_h]["g"]

            # ---- explore this state ONCE: probe all allowed actions -> escape-sorted pending (to UNEXPLORED) ----
            if not mem.explored:
                mem.explored = True
                self.env.restore(snap_cur)
                depth = len(self.path)
                escapes = []; jumped = False; saw_death = False
                acts = self.allowed_actions(cur_h, self.rank_actions(cur_h, self.enumerate_actions(g_cur)), mem)
                if GRID_CLICK:                                          # click-to-start / click-anywhere fallback
                    acts = acts + self._grid_clicks()
                if HUMAN_CLICKS:                                        # teacher prior: the click spots humans used
                    acts = acts + [Action("click", CLICK, xy=c) for c in self._hclicks]
                for a in acts:
                    st, pay = self._try_action(cur_h, snap_cur, g_cur, a, mem, depth)
                    if st == "levelup":
                        if self._do_levelup(cur_h, pay[0], pay[1], step) == "SOLVED":
                            return self._result(True, step)
                        cur_h = pay[1]["h2"]; jumped = True; break      # jump into the next level
                    if st == "escape":
                        escapes.append(pay)
                        p_e = pay[3]                                        # EARLY-COMMIT only at the BASE budget; after a
                        if (mem.action_budget <= ACTION_BUDGET0             #  _widen we probe the full widened set (broaden)
                                and (p_e["dA"] > self._thr() or p_e["i_nxt"] < 2.0)
                                and not self.memory[p_e["h2"]].explored):
                            break    # big EVENT (dA>μ+σ) OR next-intensity<2, reaches new ground -> take it, skip the rest
                    if st == "death":
                        saw_death = True
                if jumped:
                    continue
                # DELTA (teacher tie-break via SIGMA BUCKETS): bucket each escape's (dA+A∧A) into μ+kσ bands -- NOT the
                # raw score; same band == a tie. Within the best band prefer LOWER next-intensity, then FEWER distinct
                # node types (perception after HUD mask). Holds for events OR when all are sub-(μ+σ).
                _mu = float(np.mean(self.da_vals)) if self.da_vals else 0.0
                _sg = (float(np.std(self.da_vals)) or 1.0) if self.da_vals else 1.0
                def _band(p): return int((p["dA"] + p["A"] - _mu) // _sg)      # which σ-band above μ (higher = better)
                # within the band: LOWEST next-intensity wins outright (any intensity DROP is explored); the teacher
                # (fewest distinct node types) is consulted ONLY when intensities are EXACTLY equal -- it is not a score.
                escapes.sort(key=lambda x: (-_band(x[3]), x[3]["i_nxt"], _n_types(x[3]["g2"])))
                mem.pending = escapes
                mem.saw_death = saw_death

            # ---- commit the next pending escape that still leads to UNEXPLORED ground (depth-first) ----
            nxt = None
            while mem.pending:
                cand = mem.pending.pop(0)
                if not self.memory[cand[3]["h2"]].explored:
                    nxt = cand; break
            if nxt is not None:
                _i, _nr, a, p = nxt
                self.env.restore(snap_cur); self.env.step(a.aid, a.xy)
                self._commit(cur_h, a, p, step)
                cur_h = p["h2"]
                continue

            # ---- DEATH-UNDO (deep rule): this state's only remaining outcomes are death -> UNDO (restore) up the
            #      committed path until an ancestor that still has a non-death way forward ("undo until it does not
            #      make death occur"), then take that different route. ----
            if self.memory[cur_h].saw_death and self.path:
                undone = 0
                while self.path:
                    src, _a = self.path.pop()
                    self.env.restore(self.reg[src]["snap"]); cur_h = src; undone += 1
                    if any(not self.memory[c[3]["h2"]].explored for c in self.memory[src].pending):
                        break                                          # ancestor still has a surviving non-death escape
                print(f"  step {step} | DEATH-UNDO x{undone} -> state {self.reg[cur_h]['id']} "
                      f"(backed out of a death trap)", flush=True)
                self.timeline.append({"src": self.reg[cur_h]["id"], "dst": self.reg[cur_h]["id"],
                                      "a": f"DEATH-UNDOx{undone}", "r": 0.0, "dA": 0.0, "up": False})
                continue

            # ---- WIDEN before giving up: no unexplored escape at the current top-k budget -> grow it
            #      (2->3->4->...) and RE-EXPLORE this state (probes the next-ranked moves/clicks, tried ones
            #      filtered out). This is the "try two, move to three if no new states" escalation. ----
            if self._widen(mem):
                mem.explored = False
                continue

            # ---- dead-end (no escape left, budget exhausted): UNDO (restore) one step to the parent ----
            if self.path:
                src, _a = self.path.pop()
                self.env.restore(self.reg[src]["snap"]); cur_h = src
                continue
            return self._result(False, step)                           # all escapes exhausted
        return self._result(False, MAX_STEPS)

    def _commit(self, cur_h, a, p, step):
        self.path.append((cur_h, a))
        self._recompute_intensity()              # recompute face-intensity after EVERY committed action (each is valuable)
        self.timeline.append({"src": self.reg[cur_h]["id"], "dst": self.reg[p["h2"]]["id"], "a": _name(a.aid),
                              "r": round(p["r"], 1), "dA": round(p["dA"], 1), "up": p["lvl_event"]})
        self._persist_graph()                   # refresh the temp graph after each commit (referable mid-run)
        m2 = self.memory[p["h2"]]
        m2.best_depth_seen = min(m2.best_depth_seen, len(self.path))
        m2.parent = cur_h; m2.parent_action = a
        if len(self.path) % 25 == 0:
            print(f"  step {step} | committed {len(self.path)} | states {len(self.reg)} | "
                  f"edges {len(self.graph.edge_map)} | sigma={self.sigma:.2f} | "
                  f"int(dst)={self.intensity.get(p['h2'],0.0):g} | last r={p['r']:.1f} dA={p['dA']:.1f}", flush=True)
        self._after_commit(cur_h, a, p)             # hook: model rollouts collect (st,at,st+1) + train every N steps

    def _after_commit(self, cur_h, a, p):
        pass                                        # default no-op; ModelRoller (online_rollout_model.py) overrides

    def _on_probe(self, cur_h, a, p, registered):
        pass                                        # hook: live_viewer streams every probe (the search); default no-op

    def _do_levelup(self, cur_h, a, p, step):
        self.env.restore(self.reg[cur_h]["snap"]); self.env.step(a.aid, a.xy)
        self._commit(cur_h, a, p, step)
        lvls_done = p["lvl2"] - self.start_lvl
        print(f"  LEVEL {lvls_done} COMPLETE at step {step}: committed {len(self.path)}, "
              f"states {len(self.reg)} (action {_name(a.aid)} r={p['r']:.1f} dA={p['dA']:.1f})", flush=True)
        if lvls_done >= TARGET:
            return "SOLVED"
        self.path = []                  # fresh undo baseline for the NEW level (cannot undo across the boundary)
        return "CONTINUE"

    def _fpri(self, item: FrontierItem):
        # ESCAPE-directed: lowest next-intensity first (most "out", toward intensity-0), then reward, Q, prior, cost
        return (item.intensity_dst, -item.reward, -item.q_value,
                -math.log(max(item.prior_prob, 1e-8)), item.undo_distance)

    def _widen(self, mem: StateMemory):
        grew = False
        for x in ACTION_BUDGET_SCHEDULE:                      # grow top-k MOVES (2->3->4->...) when no new states
            if x > mem.action_budget:
                mem.action_budget = x; grew = True; break
        for x in CLICK_BUDGET_SCHEDULE:                       # grow top-k CLICKS (2->3->4->...)
            if x > mem.click_budget:
                mem.click_budget = x; grew = True; break
        return grew

    def backtrack(self):
        """Jump to the best OPEN frontier edge (by reward/Q/prior). Env supports restore -> teleport to src snap."""
        while self.frontier:
            _, item = heapq.heappop(self.frontier)
            sm = self.memory[item.src_hash]
            if sm.closed or action_key(item.action) in sm.bad_actions:
                continue
            if self.memory[item.dst_hash].explored:                 # dst already explored via another path -> no new ground
                continue
            self.env.restore(self.reg[item.src_hash]["snap"])       # teleport back to the frontier source
            self.env.step(item.action.aid, item.action.xy)
            g2 = self.observe(); h2 = self._hash(g2); lvl2 = self.env.live.prev_level
            self._register(g2, self.env.snap(), lvl2, self._frame())
            self.path.append((item.src_hash, item.action))
            m2 = self.memory[h2]
            m2.best_depth_seen = min(m2.best_depth_seen, len(self.path))
            m2.parent = item.src_hash; m2.parent_action = item.action
            return h2
        return None

    def _result(self, solved, steps):
        self._persist_graph()                   # final temp graph for this run
        return {"solved": solved, "steps": steps, "states": len(self.reg),
                "edges": len(self.graph.edge_map), "path_len": len(self.path),
                "open_tri": len(self.graph.open_triangles), "closed_tri": len(self.graph.closed_triangles),
                "level_probes": self.n_level_probes, "timeline": self.timeline}


def _type_multiset(g):
    return tuple(sorted(str(n.get("type_hash")) for n in g["nodes"]))


def _n_types(g):
    return len(set(str(n.get("type_hash")) for n in g["nodes"]))   # distinct node-TYPE count (HUD already masked) — teacher


HTML_ONLINE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>__GAME__ — online reward-guided rollout</title>
<style>
 html,body{margin:0;height:100%;font-family:-apple-system,Helvetica,sans-serif;background:#fbfbfb;overflow:hidden;}
 svg{position:fixed;top:0;left:0;width:100vw;height:100vh;}
 #bar{position:fixed;top:0;left:0;right:300px;z-index:10;background:rgba(255,255,255,.95);
      padding:7px 12px;border-bottom:1px solid #ccc;font-size:12px;}
 #bar b{font-size:13px}
 .sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:3px}
 #side{position:fixed;top:0;right:0;width:300px;height:100vh;overflow-y:auto;background:#fff;
       border-left:1px solid #ccc;font-size:11px;padding:6px 8px;box-sizing:border-box;}
 .row{padding:2px 4px;border-bottom:1px solid #f0f0f0;white-space:nowrap;}
 .row.up{color:#d62728;font-weight:700;background:#fff3f3;}
</style>
<script src="https://d3js.org/d3.v7.min.js"></script></head><body>
<div id="bar"><b>__GAME__ online rollout</b> &nbsp;<span id="sum"></span> &nbsp;·&nbsp;
  <span class="sw" style="background:#1f77b4"></span>committed path
  <span class="sw" style="background:#d62728"></span>level-up
  <span class="sw" style="background:#9ecae1"></span>reversible
  <span class="sw" style="background:#d9d9d9"></span>explored &nbsp;·&nbsp; ring = intensity
</div>
<svg></svg><div id="side"></div>
<script>
const D=__DATA__;
d3.select("#sum").text(`solved=${D.solved} · target L${D.target} · committed ${D.path_len} · states ${D.states} · edges ${D.edge_n}`);
const W=window.innerWidth-300,H=window.innerHeight;
const svg=d3.select("svg"),root=svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1,8]).on("zoom",e=>root.attr("transform",e.transform)));
const X=d3.scaleLinear().domain([0,1]).range([60,W-40]),Y=d3.scaleLinear().domain([0,1]).range([60,H-30]);
D.nodes.forEach(n=>{n.px=X(n.x);n.py=Y(n.y);});
const byId=new Map(D.nodes.map(n=>[n.id,n]));
D.edges.forEach(e=>{e.S=byId.get(e.s);e.T=byId.get(e.t);});
const col={committed:"#1f77b4",levelup:"#d62728",rev:"#9ecae1",explored:"#d9d9d9"};
const maxI=Math.max(1,d3.max(D.nodes,n=>n.inten)||1),greens=d3.scaleSequential(d3.interpolateGreens).domain([0,maxI]);
const eg=root.append("g"),cg=root.append("g"),ng=root.append("g");
eg.selectAll("line").data(D.edges).join("line").attr("stroke",d=>col[d.kind])
  .attr("stroke-width",d=>d.kind==="committed"||d.kind==="levelup"?2.2:1).attr("stroke-opacity",d=>d.kind==="explored"?.4:.85)
  .attr("x1",d=>d.S.px).attr("y1",d=>d.S.py).attr("x2",d=>d.T.px).attr("y2",d=>d.T.py);
cg.selectAll("path").data(D.edges.filter(d=>d.kind==="committed"||d.kind==="levelup")).join("path")
  .attr("d","M-4,-4 L4,0 L-4,4").attr("fill","none").attr("stroke",d=>col[d.kind]).attr("stroke-width",2)
  .attr("transform",d=>{const a=Math.atan2(d.T.py-d.S.py,d.T.px-d.S.px)*180/Math.PI;
     return `translate(${(d.S.px+d.T.px)/2},${(d.S.py+d.T.py)/2}) rotate(${a})`;});
const TS=26;
const node=ng.selectAll("g").data(D.nodes).join("g").attr("transform",d=>`translate(${d.px},${d.py})`)
  .call(d3.drag().on("drag",function(e,d){d.px=e.x;d.py=e.y;d3.select(this).attr("transform",`translate(${d.px},${d.py})`);
     eg.selectAll("line").attr("x1",l=>l.S.px).attr("y1",l=>l.S.py).attr("x2",l=>l.T.px).attr("y2",l=>l.T.py);
     cg.selectAll("path").attr("transform",l=>{const a=Math.atan2(l.T.py-l.S.py,l.T.px-l.S.px)*180/Math.PI;
        return `translate(${(l.S.px+l.T.px)/2},${(l.S.py+l.T.py)/2}) rotate(${a})`;});}));
node.append("rect").attr("x",-TS/2-2).attr("y",-TS/2-2).attr("width",TS+4).attr("height",TS+4).attr("rx",3)
  .attr("fill","none").attr("stroke",d=>d.inten>0?greens(d.inten):"#bbb").attr("stroke-width",d=>0.8+2.4*d.inten/maxI);
node.append("image").attr("href",d=>d.img).attr("x",-TS/2).attr("y",-TS/2).attr("width",TS).attr("height",TS);
node.append("text").attr("y",TS/2+8).attr("text-anchor","middle").attr("font-size",7).attr("fill","#333").text(d=>d.id);
node.append("title").text(d=>`state ${d.id} · level ${d.lvl} · intensity ${d.inten}`);
const side=d3.select("#side");
side.append("div").style("font-weight","700").style("padding","3px").text("committed path");
D.timeline.forEach((e,i)=>side.append("div").attr("class","row"+(e.up?" up":""))
  .text(`#${i+1}  ${e.src} —${e.a}→ ${e.t}  r=${e.r} dA=${e.dA}`+(e.up?"  ·LEVEL-UP!":"")));
</script></body></html>"""


def emit_html(R, res, suffix=""):
    und = R.graph.undirected_edges()
    G = nx.Graph(); G.add_edges_from(und)
    for h in R.reg:
        G.add_node(h)
    pos = nx.kamada_kawai_layout(G) if G.number_of_edges() else {h: (0.5, 0.5) for h in R.reg}
    xs = [p[0] for p in pos.values()] or [0]; ys = [p[1] for p in pos.values()] or [0]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    P = {h: (0.04 + 0.92 * (pos[h][0] - x0) / (x1 - x0 + 1e-9),
             0.04 + 0.92 * (pos[h][1] - y0) / (y1 - y0 + 1e-9)) for h in R.reg}
    idof = {h: R.reg[h]["id"] for h in R.reg}
    committed = set(); levelups = set()
    for e in res["timeline"]:
        committed.add((e["src"], e["dst"]))
        if e["up"]:
            levelups.add((e["src"], e["dst"]))
    dirset = set((e.src, e.dst) for e in R.graph.edge_map.values())
    revset = set((min(u, v), max(u, v)) for (u, v) in und if (u, v) in dirset and (v, u) in dirset)

    def b64(fr):
        try:
            return _frame_b64(fr) if fr is not None else ""
        except Exception:
            return ""
    nodes = [{"id": idof[h], "x": P[h][0], "y": P[h][1], "inten": float(R.intensity.get(h, 0.0)),
              "lvl": int(R.reg[h]["lvl"]), "img": b64(R.reg[h].get("frame"))} for h in R.reg]
    edata = []
    for e in R.graph.edge_map.values():
        s, t = idof[e.src], idof[e.dst]
        kind = ("levelup" if (s, t) in levelups else "committed" if (s, t) in committed
                else "rev" if (min(e.src, e.dst), max(e.src, e.dst)) in revset else "explored")
        edata.append({"s": s, "t": t, "kind": kind, "r": round(e.reward, 1), "dA": round(e.dA, 1), "a": _name(e.action.aid)})
    data = {"game": GAME, "nodes": nodes, "edges": edata, "timeline": res["timeline"], "solved": res["solved"],
            "target": TARGET, "path_len": res["path_len"], "states": res["states"], "edge_n": res["edges"]}
    out = paths.HERE / "solving_graph_deep" / f"online_undo_{GAME}{suffix}.html"
    out.write_text(HTML_ONLINE.replace("__GAME__", GAME).replace("__DATA__", json.dumps(data)))
    return out


def main():
    R = Roller(GAME)
    res = R.run()
    print(f"\n{GAME}: solved={res['solved']}  steps={res['steps']}  committed-path={res['path_len']}  "
          f"states={res['states']}  edges={res['edges']}  level-up-probes={res['level_probes']}", flush=True)
    out = emit_html(R, res, suffix=f"_lvl{TARGET}")
    print(f"saved {out}\nopen with:  open {out}", flush=True)


if __name__ == "__main__":
    main()
