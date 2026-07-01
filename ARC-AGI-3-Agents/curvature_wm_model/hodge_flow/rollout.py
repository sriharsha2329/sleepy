"""rollout.py — the Hodge-Flow wake→sleep→act rollout (Hodge-Flow §4). Offline bp35, NO Rung-3 causal, NO PPO.

WAKE  : explore from reset (snapshot/fork); record every transition with position-aware dA and 1-step A∧A.
SLEEP : decompose (A∧A-gated triangle fill) → gradient/curl/harmonic; pick no-return GOALS (A∧A); build φ
        (progress potential = dist-to-goal); mark reversible-loop (curl) edges.
ACT   : roll out from reset; at each state pick the action that  FOLLOWS the gradient (descend φ)  +  SEEKS the
        harmonic goal  −  AVOIDS curl / reversible loops (persistent penalty P). P carries across rollouts so a
        loop found once is suppressed next time.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.rollout
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

from collections import defaultdict

from config import Config

from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair, _node_xy
from curvature_wm_model.hodge_flow.envfast import FastEnv
from curvature_wm_model.hodge_flow.flow import FlowGraph, TAU_REV
from curvature_wm_model.hodge_flow import curl as curlmod
from curvature_wm_model.hodge_flow import gradient as gradmod
from curvature_wm_model.hodge_flow import harmonic as harmmod

CLICK = 6
NAME = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT5", 6: "CLICK", 7: "UNDO"}


def click_targets(graph, cfg, max_k=6):
    """One CLICK action per clickable playfield OBJECT (node). Target pixel = (px, py) of the object — NOT
    (cx,cy) which are normalized [0,1] (the old _node_xy clicked (0,0)). HUD is already removed upstream by the
    STANDARD hud_mask (strip_hud, area 32..192), so every node here is a real playfield object.
    Returns [(key, action_id, click_xy)] with key=(6, x, y) so the action is identified by which object is clicked."""
    cand = []
    for n in graph.get("nodes", []):
        px, py = n.get("px"), n.get("py")
        if px is None or py is None:
            continue
        cand.append((float(n.get("area", 0)), int(round(float(px))), int(round(float(py)))))
    cand.sort(key=lambda t: -t[0])                               # biggest objects first
    out, seen = [], set()
    for _area, x, y in cand[:max_k]:
        k = (6, x, y)
        if k not in seen:
            seen.add(k); out.append((k, 6, (x, y)))
    return out


def branch_actions(env, cfg, graph, max_clicks=6):
    """The full branch set at a state: each non-click action once, and CLICK expanded to one edge per object."""
    acts = []
    for a in env.actions():
        if a == CLICK:
            acts.extend(click_targets(graph, cfg, max_clicks))
        else:
            acts.append((a, a, None))
    return acts


def _decode(key):
    """Action key -> (action_id, click_xy). Moves: int key, xy=None. Clicks: key=(6,x,y) -> (6,(x,y))."""
    if isinstance(key, tuple):
        return key[0], (key[1], key[2])
    return key, None


class Prior:
    """Human action+click prior π(a|st) on the FROZEN aug trunk (st only; st+1 masked). `propose` returns the
    top-k candidates per state, ranking moves and click-on-object in ONE hierarchy:
      score(move a)      = P(a | st)
      score(click obj i) = P(CLICK | st) · P(obj i | st)        (action prob × click-on-node prob)
    so a click on the most-likely object competes directly with the moves."""

    def __init__(self, cfg, dev=None):
        raise RuntimeError(
            "rollout.Prior (the old aug-model prior) is obsolete — train_policy.py and the aug checkpoints were "
            "removed in the rebuild. Wire the rebuilt policy head from model/train.py --stage policy instead.")

    def propose_scored(self, graph, available, k=None):
        """[(score, key, action_id, click_xy)] sorted desc. score(move)=P(a|st); score(click@i)=P(CLICK)·P(i)."""
        import numpy as np
        import torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat, HUD_THRESH
        from curvature_wm_model.data.latents import aug_latents
        feat = _feat(graph, graph, self.cfg)
        Z = aug_latents(feat, "cur", self.cfg); M = feat["mask_cur"].astype(bool)
        with torch.no_grad():
            Zt = torch.from_numpy(Z).float()[None].to(self.dev)
            Mt = torch.from_numpy(M)[None].to(self.dev)
            Et = torch.from_numpy(cda.edge_feats(feat, "cur", self.cfg)).float()[None].to(self.dev)
            H, _ = self.model.encode(Zt, Mt, Et)
            a_logits, c_logits = self.head(H, Zt[:, :, self.od:self.od + 4], Mt)
            a_p = torch.softmax(a_logits[0], -1).cpu().numpy()
            c_p = torch.softmax(c_logits[0], -1).cpu().numpy()
        cands = []
        for a in available:
            if a == CLICK:
                for i in np.where(M)[0]:
                    if feat["pos_cur"][i, 0] >= HUD_THRESH:             # skip HUD/score region
                        continue
                    px = int(round(float(feat["foot_cur"][i, 0]))); py = int(round(float(feat["foot_cur"][i, 1])))
                    cands.append((float(a_p[CLICK]) * float(c_p[i]), (6, px, py), 6, (px, py)))
            else:
                cands.append((float(a_p[a]), a, a, None))
        cands.sort(key=lambda c: -c[0])
        return cands[:k] if k else cands

    def propose(self, graph, available, k=5):
        return [(key, aid, xy) for (_s, key, aid, xy) in self.propose_scored(graph, available, k)]


# ----------------------------------------------------------------------------- WAKE
def wake(env, cfg, fg, max_nodes=24, max_depth=10, max_clicks=6, prior=None, topk=5):
    """Explore from reset (snapshot/fork); fill `fg` with transitions keyed by action (int move / (6,x,y) click),
    each with position-aware dA and 1-step A∧A. If `prior` is given, branch ONLY the prior's top-`topk` candidates
    (moves + click-on-object ranked together) — this prunes the blow-up; else branch everything."""
    def branches(graph):
        return prior.propose(graph, env.actions(), topk) if prior else branch_actions(env, cfg, graph, max_clicks)

    env.reset()
    start = fg.get_or_add(env.graph(), env.snap())
    fg.frames.setdefault(start, getattr(env.live, "last_frame", None))    # capture the reset frame (viz)
    frontier = [(start, 0)]; explored = set()
    while frontier and len(fg.reps) < max_nodes:
        u, d = frontier.pop(0)
        if u in explored or d >= max_depth:
            continue
        explored.add(u)
        u_snap, u_graph = fg.snaps[u], fg.reps[u]
        env.restore(u_snap)                                       # env at u -> env.actions() valid
        fwd = branches(u_graph)
        for (key, a_id, xy) in fwd:
            env.restore(u_snap)
            env.step(a_id, xy); v_graph = env.graph(); v_snap = env.snap()   # env now at v
            v_frame = getattr(env.live, "last_frame", None)
            v_frame = None if v_frame is None else v_frame.copy()
            dA = smae_pair(u_graph, v_graph, cfg)
            ret = branch_actions(env, cfg, v_graph, max_clicks)   # FULL return set -> accurate reversibility (the
            aa = float("inf")                                     # inverse move may rank low in the prior, so the
            #                                                      A∧A must test all real actions, not the top-k
            for (_rk, r_id, r_xy) in ret:
                env.restore(v_snap)
                env.step(r_id, r_xy)
                aa = min(aa, smae_pair(u_graph, env.graph(), cfg))
            nbefore = len(fg.reps)
            v = fg.get_or_add(v_graph, v_snap)
            if v == nbefore:                                      # newly created node → store its frame
                fg.frames[v] = v_frame
            fg.add(u, key, v, dA, aa)
            if v not in explored:
                frontier.append((v, d + 1))
    return fg


# ----------------------------------------------------------------------------- SLEEP
def sleep(fg, tau_event=20.0):
    """Decompose and assign control signals: goals (no-return), φ (progress potential), curl loops (avoid)."""
    dec = curlmod.decompose(fg)
    goals = harmmod.no_return_goals(fg, tau_event)
    phi = gradmod.progress_potential(fg, goals)
    curl_e = curlmod.curl_edges(fg, dec)
    return {"dec": dec, "goals": goals, "phi": phi, "curl_edges": curl_e,
            "energy": curlmod.energy(dec), "beta1": harmmod.beta1(dec)}


# ----------------------------------------------------------------------------- ACT (shared policy)
# LOOP PENALTY POLICY (user refinement): penalize the LOOP, not the actions inside it. The only penalized move is
# one that CLOSES a loop — returns to an ALREADY-VISITED node via a reversible edge. So every state's FIRST visit
# is free (you can enter a loop region and take the 'third'/exit action there); only going AROUND again is
# penalized. No per-action `P[(s,a)]` blacklist (that would block an edge needed for a different path) and no
# per-edge curl penalty (curl edges ARE the loop's inner edges, which must stay usable).
def _pick_action(fg, cur, plan, visits, alpha, beta, gamma, actions):
    """FOLLOW gradient (descend φ) + SEEK harmonic goal − penalize LOOP CLOSURE (reversible revisit)."""
    phi, goals = plan["phi"], plan["goals"]
    best_a, best_v, best = None, None, -1e18
    for a in actions:
        mv = fg.model.get(cur, {}).get(a)
        if mv is None:
            continue
        v, _dA, _aa = mv
        reversible = fg.edge_aa(cur, v) < TAU_REV
        loop_closure = beta * visits.get(v, 0) if reversible else 0.0      # only a reversible RETURN is penalized
        score = -alpha * phi.get(v, 1e6) + (gamma if v in goals else 0.0) - loop_closure
        if score > best:
            best, best_a, best_v = score, a, v
    return best_a, best_v


def _record_loop(mem, path, v):
    """When the move closes a loop (returns to v already on the path), record the cycle's state-set (the LOOP),
    not its individual edges. `mem` is a dict {frozenset(cycle states): times-encountered}."""
    if v in path:
        cyc = frozenset(path[path.index(v):])
        mem[cyc] = mem.get(cyc, 0) + 1
        return True
    return False


def act(env, cfg, fg, plan, mem=None, alpha=1.0, beta=4.0, gamma=8.0, max_steps=20):
    """Roll out from reset, PLAYING in the real env (node tracked via the deterministic model). Penalizes loop
    CLOSURES (contextually, per rollout); records detected loops in `mem` (loop-level, not per action)."""
    mem = {} if mem is None else mem
    env.reset()
    cur = fg.node_id(env.graph())
    traj = []; visits = defaultdict(int); path = []; loops = 0
    for _ in range(max_steps):
        if cur in plan["goals"]:
            break
        visits[cur] += 1; path.append(cur)
        key, v = _pick_action(fg, cur, plan, visits, alpha, beta, gamma, list(fg.model.get(cur, {}).keys()))
        if key is None:
            break
        a_id, cxy = _decode(key)                                  # move (int) or click (6,x,y)
        env.step(a_id, cxy)                                       # actually play it
        traj.append((cur, key, v))
        if fg.edge_aa(cur, v) < TAU_REV and _record_loop(mem, path, v):
            loops += 1
        cur = v
    return {"reached": cur in plan["goals"], "steps": len(traj), "traj": traj, "loops": loops}


def simulate(fg, plan, start, mem=None, alpha=1.0, beta=4.0, gamma=8.0, max_steps=20):
    """Model-only rollout (no env) with the SAME policy — for planning and for testing without the env."""
    mem = {} if mem is None else mem
    cur = start; traj = []; visits = defaultdict(int); path = []; loops = 0
    for _ in range(max_steps):
        if cur in plan["goals"]:
            break
        visits[cur] += 1; path.append(cur)
        a, v = _pick_action(fg, cur, plan, visits, alpha, beta, gamma, list(fg.model.get(cur, {}).keys()))
        if a is None:
            break
        traj.append((cur, a, v))
        if fg.edge_aa(cur, v) < TAU_REV and _record_loop(mem, path, v):
            loops += 1
        cur = v
    return {"reached": cur in plan["goals"], "steps": len(traj), "traj": traj, "loops": loops}


# ----------------------------------------------------------------------------- agent (persistent across rollouts)
class FlowRolloutAgent:
    def __init__(self, cfg, game="bp35", tau_event=20.0):
        self.cfg = cfg; self.env = FastEnv(__import__("online_v5").Live(cfg, game))
        self.fg = FlowGraph(cfg); self.mem = {}; self.tau_event = tau_event; self.plan = None

    def wake(self, **kw):
        wake(self.env, self.cfg, self.fg, **kw); return self.fg.stats()

    def sleep(self):
        self.plan = sleep(self.fg, self.tau_event); return self.plan

    def act(self, **kw):
        return act(self.env, self.cfg, self.fg, self.plan, self.mem, **kw)


# ----------------------------------------------------------------------------- demo
def run_game(game="bp35", max_nodes=28, max_depth=12, tau_event=20.0, rollouts=3):
    cfg = Config()
    agent = FlowRolloutAgent(cfg, game, tau_event=tau_event)

    print(f"HODGE-FLOW ROLLOUT on offline {game}  (gradient=dA · curl=A∧A-triangle · harmonic=no-return)\n")
    print("WAKE: exploring ...")
    st = agent.wake(max_nodes=max_nodes, max_depth=max_depth)
    print(f"  graph: {st['nodes']} nodes, {st['edges']} edges, {st['reversible_edges']} reversible\n")

    plan = agent.sleep()
    eg = plan["energy"]
    print("SLEEP: Hodge decomposition + assignment")
    if eg:
        print(f"  energy  gradient {eg[0]:.0%}  curl {eg[1]:.0%}  harmonic {eg[2]:.0%}   beta1={plan['beta1']}")
    print(f"  no-return GOALS (nodes): {sorted(plan['goals'])}")
    print(f"  curl (reversible-loop) edges: {sorted(plan['curl_edges'])}")
    reach = {n: d for n, d in sorted(plan["phi"].items(), key=lambda kv: kv[1])}
    print(f"  φ (dist-to-goal) per node: {reach}\n")

    print("ACT: rollouts (loop CLOSURES penalized contextually; loops recorded, not the actions)")
    for r in range(rollouts):
        out = agent.act()
        path = " -> ".join(f"{NAME.get(a,a)}" for (_u, a, _v) in out["traj"])
        print(f"  rollout {r+1}: reached_goal={out['reached']}  steps={out['steps']}  loop_closures_avoided={out['loops']}  | {path}")
    print(f"\n  loops detected (state-sets, not actions): { {tuple(sorted(c)): n for c, n in agent.mem.items()} }")
    return agent


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "bp35"
    run_game(game)


if __name__ == "__main__":
    main()
