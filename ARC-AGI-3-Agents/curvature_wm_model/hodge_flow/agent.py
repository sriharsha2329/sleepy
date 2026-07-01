"""agent.py — topological search to the first HARMONIC / end-of-level. Three modes (--rollout):

  harmonic_dfs       prior top-3 + triangle loop-pruning, DFS on max dA+A∧A (the first version; checks each child
                     vs ALL known states -> O(states^2), slower).
  harmonic_frontier  BETTER: QUOTIENT the cheaply-reversible local basin (union-find over SIBLINGS only -> O(k^2)),
                     drop intra-quotient (reversible/triangle) edges, and run BEST-FIRST (priority queue) ranked by
                     the HARMONIC RESIDUAL (A∧A-dominant) — because a large dA can still be reversible curl/noise.
  mc                 (superseded) the old Monte-Carlo rollout.

State identity = GNN FRAME HASH: 64-bit FNV-1a (16 hex) over the quantized pooled trunk encoding — a real frame
hash, locality-sensitive (same frame -> same hash), non-cryptographic. (Centroid/position sigs may use any hash.)

  ENVIRONMENTS_DIR=... PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.agent \
      --game ls20 --rollout harmonic_frontier --top-k 3 --min-action-threshold 1.5 \
      --harmonic-threshold 20 --curvature-threshold 25 --max-depth 200
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
from collections import defaultdict

os.environ.setdefault("OPERATION_MODE", "offline")

from curvature_wm_model import paths  # noqa: F401

sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

from config import Config
from curvature_wm_model.hodge_flow.envfast import FastEnv
from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair, _node_xy, HUD_THRESH
from curvature_wm_model.hodge_flow.rollout import Prior, NAME, CLICK, branch_actions

TOL = 1.0


def _name(key):
    return f"CLICK@{key[1:]}" if isinstance(key, tuple) else NAME.get(key, key)


class Search:
    def __init__(self, cfg, game, k=3, dev=None):
        from online_v5 import Live
        self.cfg = cfg; self.k = k
        self.env = FastEnv(Live(cfg, game)); self.prior = Prior(cfg, dev)
        self.reps = []; self.snaps = []; self.lvl = []; self.label = []; self.depth = []
        self.key2id = {}; self.children = {}; self.expanded = set(); self.parent = {}     # union-find parent
        self.loops = set(); self.H = []; self.M = []                                      # cached per-state encoding (inverse head)
        self.frames = {}; self.par_state = {}; self.par_act = {}                          # replay: grid + discovery parent/action
        # A∧A bands: dA<da_mod -> cheap inverse-head return; dA>=da_mod -> BRUTE-FORCE (triangles + harmonic);
        #            dA>=da_large -> "large" -> harmonic-stop eligible (always brute-forced, so trustworthy).
        self.da_mod = 4.0; self.da_large = 12.0
        self._add(self.env.graph(), self.env.snap(), self.env.live.prev_level, 0)         # first frame
        self.frames[0] = getattr(self.env.live, "last_frame", None)

    def path_to(self, target):                                                            # discovery path start..target (state ids)
        path = []; cur = target; seen = set()
        while cur is not None and cur not in seen:                                         # `seen` guards against a parent cycle
            seen.add(cur); path.append(cur); cur = self.par_state.get(cur)
        return path[::-1]

    # ---- GNN frame hash (64-bit FNV-1a -> 16 hex) ----
    def _frame(self, g):
        import numpy as np
        import torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat
        from curvature_wm_model.data.latents import aug_latents
        feat = _feat(g, g, self.cfg)
        M0 = feat["mask_cur"].astype(bool)
        play = M0 & (feat["pos_cur"][:, 0] < HUD_THRESH)             # playfield only — drop the bottom HUD/step-counter bar
        idx = np.where(play if play.any() else M0)[0]               # SLICE it out (masking leaks via edge-attention)
        Z = aug_latents(feat, "cur", self.cfg)[idx]
        E = cda.edge_feats(feat, "cur", self.cfg)[np.ix_(idx, idx)]
        with torch.no_grad():
            Zt = torch.from_numpy(Z).float()[None].to(self.prior.dev)
            Mt = torch.ones(1, len(idx), dtype=torch.bool, device=self.prior.dev)
            Et = torch.from_numpy(E).float()[None].to(self.prior.dev)
            H, pooled = self.prior.model.encode(Zt, Mt, Et)
        # the pooled encoding is a MEAN over nodes -> the moving player's position washes out. So fold the
        # per-node centroid + Mahalanobis block (aug latent's last 4 dims) into the hash EXPLICITLY, sorted
        # so it is node-order-invariant. Now distinct player cells -> distinct hash (not collapsed).
        posblk = np.round(Z[:, -4:], 1).astype(np.float32)           # [centroid_col, centroid_row, maha_col, maha_row]
        posblk = posblk[np.lexsort(posblk.T)]                        # order-invariant
        key = np.round(pooled[0].cpu().numpy(), 1).tobytes() + posblk.tobytes()
        h = 1469598103934665603                                       # FNV-1a 64-bit (non-cryptographic)
        for b in key:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return key, format(h, "016x"), H, Mt

    def _add(self, g, snap, level, depth):
        key, label, H, M = self._frame(g)
        if key in self.key2id:
            return self.key2id[key]
        i = len(self.reps)
        self.reps.append(g); self.snaps.append(snap); self.lvl.append(level); self.label.append(label)
        self.depth.append(depth); self.key2id[key] = i; self.parent[i] = i
        self.H.append(H); self.M.append(M)
        return i

    def _inv_action(self, c_id, s_id, available):
        """Inverse head: action that maps state c_id -> s_id, RENORMALISED over the available actions only
        (a predicted CLICK on a click-less game is masked out and the best legal move is taken instead)."""
        import torch
        with torch.no_grad():
            union = (self.M[c_id].bool() | self.M[s_id].bool()).float()
            logits = self.prior.model.inverse_head(self.H[c_id], self.H[s_id], union)[0]
        masked = torch.full_like(logits, float("-inf"))
        for a in available:
            if 0 <= a < logits.shape[0]:
                masked[a] = logits[a]                                # keep only available -> argmax == renormalised pick
        return int(masked.argmax().item()) if torch.isfinite(masked).any() else available[0]

    # ---- union-find quotient ----
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def _branch(self, graph, available):
        """ALL move actions (never score-pruned — every direction is explored) + the prior's top-k CLICK targets
        (clicks can be many, so those stay pruned)."""
        out = [(a, a, None) for a in available if a != CLICK]
        if CLICK in available:
            out += [(k, aid, xy) for (_s, k, aid, xy) in self.prior.propose_scored(graph, [CLICK], self.k)]
        return out

    # ---- expand: branch all moves (+top-k clicks) via snapshot/undo, each with dA, A∧A, level-up ----
    def expand(self, i):
        if i in self.expanded:
            return self.children[i]
        self.expanded.add(i)
        env = self.env; env.restore(self.snaps[i]); l0 = env.live.prev_level; d0 = self.depth[i]
        ch = []
        for (key, aid, xy) in self._branch(self.reps[i], env.actions()):
            env.restore(self.snaps[i]); env.step(aid, xy); g = env.graph(); snp = env.snap()
            l1 = env.live.prev_level; up = l1 > l0
            win = bool(env.last_info.get("win")); over = bool(env.last_info.get("game_over"))
            cid = self._add(g, snp, l1, d0 + 1)                      # add FIRST so H[cid] is cached for the inverse head
            if cid not in self.par_state and cid != i:               # remember how this state was first reached (for replay)
                self.par_state[cid] = i; self.par_act[cid] = key
                self.frames[cid] = (lambda fr: None if fr is None else fr.copy())(getattr(env.live, "last_frame", None))
            dA = smae_pair(self.reps[i], g, self.cfg); aa = 0.0; ret = None
            if not (up or win or over) and dA >= self.da_mod:        # moderate+ dA -> brute-force the ACTUAL A∧A
                cs = env.snap(); aa = float("inf")                   # full branch: moves + CLICK-on-each-node (bp35)
                for (bkey, baid, bxy) in branch_actions(env, self.cfg, self.reps[cid]):
                    env.restore(cs); env.step(baid, bxy); d = smae_pair(self.reps[i], env.graph(), self.cfg)
                    if d < aa:
                        aa, ret = d, bkey                            # best-return key (move int, or (6,x,y) for a click)
                env.restore(cs); aa = 0.0 if aa == float("inf") else aa
            ch.append({"key": key, "cid": cid, "dA": dA, "aa": aa, "curv": dA + aa, "ret": ret,
                       "large": dA >= self.da_mod, "up": up, "win": win, "over": over})
        self.children[i] = ch
        return ch

    def reaches(self, i, j):
        best = None
        for c in self.expand(i):
            if c["cid"] == j and (best is None or c["dA"] < best[1]):
                best = (c["key"], c["dA"])
        return best

    def _reseed(self, expanded_q, blocked):
        """Frontier empty -> from the DEEPEST expanded state, walk UP the discovery ancestry; return the first
        top-k child whose quotient is still unexplored (an action that can still be 'added'). Walk one step
        above whenever a state has nothing addable; None only when the whole tree is exhausted."""
        for leaf in sorted(self.expanded, key=lambda i: -self.depth[i]):     # deepest leaves first
            cur, seen = leaf, set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                for c in self.children.get(cur, []):
                    cc = c["cid"]
                    if cc != cur and cc != blocked and self.find(cc) not in expanded_q \
                            and not c["up"] and not c["win"]:
                        return cc                                            # an unexplored branch off this ancestor
                cur = self.par_state.get(cur)                                # nothing here -> one step above
        return None

    # =============================== harmonic_dfs (first version) ===============================
    def run_dfs(self, max_expand=120, loop_thresh=15.0, event_thresh=25.0, verbose=True):
        log = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)
        loops = set(); path = [0]; visited = set(); steps = 0
        while path and steps < max_expand:
            cur = path[-1]
            if cur in visited:
                path.pop(); continue
            visited.add(cur); steps += 1; L = self.label
            ch = self.expand(cur)
            log(f"[{steps:3d}] exploring {L[cur]} -> {[L[c['cid']] for c in ch]}  (dA+A∧A={[round(c['curv'],1) for c in ch]})")
            for c in ch:
                if c["up"] or c["win"] or c["curv"] >= event_thresh:
                    log(f"     >>> HARMONIC / END OF LEVEL: {L[cur]} --{_name(c['key'])}--> {L[c['cid']]}  dA+A∧A={c['curv']:.1f}")
                    return {"completed": True, "path": [L[i] for i in path] + [L[c["cid"]]], "curv": c["curv"],
                            "level_up": c["up"], "win": c["win"], "steps": steps, "states": len(self.reps)}
            for c in ch:                                              # triangle pruning vs ALL states (slow)
                ci = c["cid"]
                for j in range(len(self.reps)):
                    if j == ci or frozenset({ci, j}) in loops:
                        continue
                    rij = self.reaches(ci, j); rji = self.reaches(j, ci)
                    if rij and rji and rij[1] < loop_thresh and rji[1] < loop_thresh:
                        loops.add(frozenset({ci, j}))
            on_path = set(path)
            cand = [c for c in ch if c["cid"] not in on_path and c["cid"] not in visited and c["cid"] != cur]
            if not cand:
                path.pop(); continue
            best = max(cand, key=lambda c: c["curv"])
            log(f"     {L[cur]} -> {L[best['cid']]} MAX dA+A∧A={best['curv']:.1f} via {_name(best['key'])}")
            path.append(best["cid"])
        return {"completed": False, "steps": steps, "states": len(self.reps), "reason": "budget/exhausted"}

    # =============================== harmonic_frontier (better) ===============================
    def run_frontier(self, max_expand=200, max_harmonics=5, min_action=1.5, harm_thresh=20.0, curv_thresh=25.0,
                     loop_thresh=2.0, wH=1.5, wF=1.0, wP=1.0, wV=0.25, wL=3.0, chain=True, verbose=True):
        log = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)
        L = self.label
        frontier = [(-0.0, 0)]                                        # max-heap (negated score) of state ids
        qvisits = defaultdict(int); expanded_q = set(); steps = 0; subgoals = []
        harm_srcs = set(); blocked = None        # blocked = the just-passed harmonic frame: forbidden until the NEXT harmonic is found
        while steps < max_expand:
            if not frontier:                                         # BACKTRACK: frontier empty -> walk up from the deepest leaf,
                rs = self._reseed(expanded_q, blocked)               # re-seed at the first ancestor with an unexplored top-k action
                if rs is None:
                    break                                            # nothing addable anywhere -> truly exhausted
                log(f"     [backtrack] frontier empty -> re-seed at {L[rs]} (depth {self.depth[rs]})")
                frontier = [(-0.0, rs)]
            negs, s = heapq.heappop(frontier)
            if s == blocked:                                         # cannot revisit the previous harmonic until a new one is found
                continue
            q = self.find(s)
            if q in expanded_q:
                continue
            expanded_q.add(q); steps += 1
            ch = self.expand(s)
            log(f"[{steps:3d}] exploring {L[s]} -> {[L[c['cid']] for c in ch]}  "
                f"(dA+A∧A={[round(c['curv'],1) for c in ch]}, A∧A={[round(c['aa'],1) for c in ch]})")
            for c in ch:                                              # reversible-loop (triangle) detection -> cache (PERSISTS across harmonics)
                if c["cid"] != s and c["ret"] is not None and c["aa"] < loop_thresh \
                        and frozenset({s, c["cid"]}) not in self.loops:
                    self.loops.add(frozenset({s, c["cid"]}))
                    log(f"     loop: {L[s]} --{_name(c['key'])}(dA={c['dA']:.1f})--> {L[c['cid']]} & "
                        f"{L[c['cid']]} --{_name(c['ret'])}--> {L[s]} (A∧A={c['aa']:.1f} reversible) -> cached (avoid triangular path)")
            for c in ch:                                              # end of level (the final harmonic)?
                if c["up"] or c["win"]:
                    subgoals.append((L[s], _name(c["key"]), L[c["cid"]], c["curv"], self.lvl[c["cid"]], c["cid"]))
                    log(f"     >>> END OF LEVEL: {L[s]} --{_name(c['key'])}--> {L[c['cid']]}  (env level={self.lvl[c['cid']]})")
                    return {"completed": True, "level_up": c["up"], "win": c["win"], "subgoals": subgoals,
                            "curv": c["curv"], "steps": steps, "states": len(self.reps), "start_lvl": self.lvl[0]}
            harm = None                                              # a TRUE harmonic (large dA -> brute-forced, no-return)
            if chain and s not in harm_srcs:                         # chain=False -> ignore harmonics, explore uniformly
                for c in ch:
                    if self.find(s) == self.find(c["cid"]):
                        continue
                    if c["large"] and c["aa"] >= harm_thresh and c["curv"] >= curv_thresh:  # actual no-return: |A∧A| & dA+A∧A
                        harm = c; break
            if harm is not None:                                     # reached a sub-goal -> CHAIN: continue FROM the harmonic frame
                subgoals.append((L[s], _name(harm["key"]), L[harm["cid"]], harm["curv"], self.lvl[harm["cid"]], harm["cid"]))
                log(f"     >>> HARMONIC #{len(subgoals)}: {L[s]} --{_name(harm['key'])}--> {L[harm['cid']]}  "
                    f"A∧A={harm['aa']:.1f} dA+A∧A={harm['curv']:.1f}  (env level={self.lvl[harm['cid']]})")
                harm_srcs.add(s)
                if blocked is not None:
                    expanded_q.discard(self.find(blocked))           # the EARLIER harmonic reopens (H2 can go to H1 or H3)
                blocked = s                                          # but THIS harmonic is now off-limits until the next one
                if len(subgoals) >= max_harmonics:
                    return {"completed": True, "harmonic": True, "subgoals": subgoals, "curv": harm["curv"],
                            "steps": steps, "states": len(self.reps), "start_lvl": self.lvl[0]}
                log(f"     >>> committing; continuing FROM {L[harm['cid']]} (forbidding prev harmonic {L[s]}; "
                    f"keeping {len(self.loops)} cached curls + {len(expanded_q)} basins)")
                frontier = [(-0.0, harm["cid"])]                     # RESTART best-first from the harmonic frame; caches persist
                continue
            nmerge = 0                                                # merge cheaply-reversible siblings (silent, counted)
            for a in range(len(ch)):
                for b in range(a + 1, len(ch)):
                    ci, cj = ch[a]["cid"], ch[b]["cid"]
                    if self.find(ci) == self.find(cj):
                        continue
                    rij = self.reaches(ci, cj); rji = self.reaches(cj, ci)
                    if rij and rji and rij[1] < min_action and rji[1] < min_action:
                        self.union(ci, cj); nmerge += 1
            scored = []
            for c in ch:                                             # score surviving cross-quotient edges
                if self.find(s) == self.find(c["cid"]) or c["cid"] == blocked:
                    continue                                         # intra-quotient (reversible/triangle) or the forbidden prev harmonic
                lp = wL if frozenset({s, c["cid"]}) in self.loops else 0.0   # de-prioritize the cached triangular path
                score = wH * c["aa"] + wF * c["dA"] + wP * self.depth[c["cid"]] \
                    - wV * qvisits[self.find(c["cid"])] - lp           # actual |A∧A| + dA (no-return emphasized via wH)
                qvisits[self.find(c["cid"])] += 1
                heapq.heappush(frontier, (-score, c["cid"])); scored.append((score, c))
            if scored:                                               # ONE clean line: the best next move
                bs, bc = max(scored, key=lambda x: x[0])
                log(f"     {L[s]} -> {L[bc['cid']]} BEST score={bs:.1f} via {_name(bc['key'])} "
                    f"(A∧A={bc['aa']:.1f} dA+A∧A={bc['curv']:.1f}){f'  [{nmerge} merged]' if nmerge else ''}")
        return {"completed": len(subgoals) > 0, "harmonic": len(subgoals) > 0, "subgoals": subgoals,
                "steps": steps, "states": len(self.reps), "reason": "budget/exhausted", "start_lvl": self.lvl[0]}


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--rollout", default="harmonic_frontier", choices=["harmonic_frontier", "harmonic_dfs", "mc"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=200)
    ap.add_argument("--max-harmonics", type=int, default=5)
    ap.add_argument("--min-action-threshold", type=float, default=1.5)
    ap.add_argument("--harmonic-threshold", type=float, default=20.0)
    ap.add_argument("--curvature-threshold", type=float, default=25.0)
    ap.add_argument("--da-mod", type=float, default=4.0, help="dA<this -> cheap inverse-head A∧A; >= -> brute-force")
    ap.add_argument("--da-large", type=float, default=12.0, help="dA>=this -> 'large' -> harmonic-stop eligible")
    args = ap.parse_args()
    cfg = Config()
    print(f"[{args.game}] mode={args.rollout} top-k={args.top_k}  (state id = 64-bit GNN frame hash; "
          f"A∧A: cheap<{args.da_mod}<=brute, large>={args.da_large})\n")
    s = Search(cfg, args.game, k=args.top_k); s.da_mod = args.da_mod; s.da_large = args.da_large
    t = time.time()
    if args.rollout == "harmonic_dfs":
        out = s.run_dfs(max_expand=args.max_depth, event_thresh=args.curvature_threshold)
    elif args.rollout == "mc":
        print("mc mode is superseded by harmonic_frontier; running harmonic_frontier instead.")
        out = s.run_frontier(max_expand=args.max_depth, max_harmonics=args.max_harmonics,
                             min_action=args.min_action_threshold)
    else:
        out = s.run_frontier(max_expand=args.max_depth, max_harmonics=args.max_harmonics,
                             min_action=args.min_action_threshold)
    dt = time.time() - t
    print()
    sg = out.get("subgoals", []); slvl = out.get("start_lvl", "?")
    finished = bool(out.get("level_up") or out.get("win"))
    if sg:                                                           # the chain of harmonic milestones (with env level)
        print(f"SUB-GOAL CHAIN ({len(sg)} harmonic{'s' if len(sg) != 1 else ''}, env level started at {slvl}):")
        for n, (src, act, dst, curv, lvl, cid) in enumerate(sg, 1):
            tag = "  <== LEVEL UP" if (n == len(sg) and finished) else ""
            print(f"  H{n}: {src} --{act}--> {dst}   (dA+A∧A={curv:.1f}, env level={lvl}){tag}")
    print(f"\nLEVEL FINISHED: {'YES ✓' if finished else f'NO — {len(sg)} harmonic milestone(s), env level never advanced past {slvl}'}")
    print(f"RESULT: {out['steps']} expansions, {out['states']} states, {dt:.1f}s"
          + ("" if finished else f"  (stopped: {out.get('reason', 'max-harmonics cap')})"))

    import json                                                       # save the chosen path's actions for replay/render
    target = sg[-1][5] if sg else max(range(len(s.depth)), key=lambda i: s.depth[i])
    acts = []
    for pid in s.path_to(target)[1:]:
        k = s.par_act.get(pid)
        acts.append([6, k[1], k[2]] if isinstance(k, tuple) else k)   # serialize: move int, or [6,x,y] click
    json.dump({"game": args.game, "actions": acts}, open(f"/tmp/actions_{args.game}.json", "w"))
    print(f"saved {len(acts)} path actions -> /tmp/actions_{args.game}.json")


if __name__ == "__main__":
    main()
