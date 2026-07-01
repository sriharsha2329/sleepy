"""solving_graph_deep/metrics_ls20.py — get the reward metrics right: dA, A∧A, dA+β·A∧A, and INTENSITY.

State identity = INPUT-GRAPH hash (perception node attrs + centroids + Mahalanobis), HUD dropped via the real
hud_regions (the accurate, detailed graph — not the latent which collapses).

Per directed transition (s --a--> s'):
  dA   = smae_pair(s, s')                          # semantic + Mahalanobis-position change (HUD excluded)
  A∧A  = model-straight 0 (reversible) UNLESS the transition is an EVENT:  aa != 0 AND z(dA) > 1
         where z(dA) = (dA - μ)/σ over ALL collected transitions. For events we BRUTE-FORCE the true A∧A =
         min over return-actions of smae_pair(s, return(s')) (0 = returns exactly, large = no-return holonomy).
  comb = dA + BETA·A∧A         (BETA = 10)

INTENSITY (= "sum of opacity", per node, the prior face_report metric): number of GREEN faces (reversible
cycle-basis loops) that STACK over the node — n is a vertex of the cycle OR geometrically inside it
(point-in-polygon) — i.e. the alpha-composited opacity 1-(1-0.18)^k it is exposed to. Anything part of a loop
is a face vertex, so its intensity is ≥ 1 (the floor). Overlapping loops push it higher.

Reward will be built from (dA, A∧A, intensity). This script makes those right and renders the green-INTENSITY
graph (faces stacked = opacity; node border/title = intensity) with NO variance/z labels.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.solving_graph_deep.metrics_ls20 [game]
"""
import os
os.environ.setdefault("OPERATION_MODE", "offline")
import gc
import sys
from collections import Counter

from curvature_wm_model import paths  # noqa: F401
sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from config import Config
from curvature_wm_model.hodge_flow.agent import Search, _name
from curvature_wm_model.hodge_flow.diagnose_lr import smae_pair
from curvature_wm_model.hodge_flow.rollout import branch_actions
from curvature_wm_model.hodge_flow.flow import TAU_REV
from curvature_wm_model.model.reward_metrics import faces_intensity, _order_face, BETA   # canonical intensity/faces (model)
from hud_mask import hud_regions, TOL

GAME = sys.argv[1] if len(sys.argv) > 1 else "ls20"
ALPHA = 0.18                                                          # per-face opacity (sum/composite -> intensity)
CYR, FOOTR = 3, 2
_REGIONS = hud_regions(GAME)
_ARC = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B",
        "#7FDBFF", "#870C25", "#1f3a93", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
_CMAP = ListedColormap(_ARC)


def _in_hud(cy, cx):
    return any(r0 - TOL <= cy <= r1 + TOL and c0 - TOL <= cx <= c1 + TOL for (r0, r1, c0, c1) in _REGIONS)


class InputHashSearch(Search):
    def _frame(self, g):
        items = []
        for n in g["nodes"]:
            cy, cx = float(n.get("cy", 0)), float(n.get("cx", 0))
            if _in_hud(cy, cx):
                continue
            items.append((str(n.get("type_hash")), tuple(n.get("color", [])), int(n.get("stab", 0)),
                          int(n.get("area_bin", 0)), round(cy, CYR), round(cx, CYR),
                          round(float(n.get("sx", 0)), FOOTR), round(float(n.get("sy", 0)), FOOTR)))
        items.sort()
        key = repr(items).encode()
        h = 1469598103934665603
        for b in key:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        _, _, H, M = super()._frame(g)
        return key, format(h, "016x"), H, M


def _brute_aa(s, u, v):
    """True A∧A of u->v: from snapshot at v, the MIN smae back to u over every return action (0 = reversible)."""
    env = s.env
    env.restore(s.snaps[v]); cs = env.snap(); best = float("inf")
    for (_bk, baid, bxy) in branch_actions(env, s.cfg, s.reps[v]):
        env.restore(cs); env.step(baid, bxy)
        d = smae_pair(s.reps[u], env.graph(), s.cfg)
        if d < best:
            best = d
    return 0.0 if best == float("inf") else float(best)


def _mask(fr):
    if fr is None:
        return None
    fr = np.asarray(fr).copy(); Hh, Ww = fr.shape[:2]; bgc = int(np.bincount(fr.ravel()).argmax())
    for (r0, r1, c0, c1) in _REGIONS:
        fr[int(r0 * Hh):min(int(np.ceil(r1 * Hh)) + 1, Hh), int(c0 * Ww):min(int(np.ceil(c1 * Ww)) + 1, Ww)] = bgc
    return fr


def main():
    s = InputHashSearch(Config(), GAME, k=3)
    s.run_frontier(max_expand=300, chain=False, verbose=False)
    nstates = len(s.reps)

    # ---- directed edges (strongest-dA action per (u,v)) ----
    de = {}
    for u in sorted(s.expanded):
        for c in s.children.get(u, []):
            v = c["cid"]
            if v == u:
                continue
            if (u, v) not in de or c["dA"] > de[(u, v)]["dA"]:
                de[(u, v)] = {"a": c["key"], "dA": float(c["dA"]), "aa": float(c["aa"])}

    # ---- μ, σ of dA over ALL collected transitions, then the z>1 EVENT gate (brute-force true A∧A) ----
    dA_all = np.array([e["dA"] for e in de.values()])
    mu, sd = float(dA_all.mean()), float(dA_all.std() or 1.0)
    thr = mu + sd
    n_bf = 0
    for (u, v), e in de.items():
        e["z"] = (e["dA"] - mu) / sd
        if e["z"] > 1.0 and e["aa"] == 0.0:                          # event w/o cheap aa -> brute-force the truth
            e["aa"] = _brute_aa(s, u, v); n_bf += 1
        e["event"] = bool(e["aa"] > 1e-6 and e["z"] > 1.0)          # A∧A model-straight 0 unless real event
        e["comb"] = e["dA"] + BETA * e["aa"]

    # ---- undirected edge set (max comb per edge) -> graph + layout ----
    und = {}
    for (u, v), e in de.items():
        k = (min(u, v), max(u, v)); und[k] = max(und.get(k, 0.0), e["comb"])

    # ---- layout (needed for face-exposure point-in-polygon) ----
    G = nx.Graph(); G.add_edges_from(und.keys())
    if G.number_of_nodes() == 0:
        print("no edges"); return
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = nx.kamada_kawai_layout(G)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    P = {n: (0.05 + 0.9 * (pos[n][0] - x0) / (x1 - x0 + 1e-9),
             0.05 + 0.9 * (pos[n][1] - y0) / (y1 - y0 + 1e-9)) for n in G.nodes}

    # ---- faces + INTENSITY from the canonical model-folder computation (corrected one-way-edge 0.5 rule) ----
    #      reversible(green) faces ×1 per node + 0.5 ONLY if the node is on a one-way edge (NOT charged-face members,
    #      which wrongly gave a reversible-only node like 48 a 1.5; with only reversible edges it must be 1.0).
    fi = faces_intensity(de, list(G.nodes))
    rev, green, charged, inten = fi["rev"], fi["green"], fi["charged"], fi["intensity"]
    maxc = max(inten.values()) if inten else 1

    # ===================== METRICS REPORT =====================
    comb_all = np.array([e["comb"] for e in de.values()])
    print(f"\n===== {GAME} reward metrics (INPUT-graph hash; β={BETA:.0f}) =====")
    print(f"states={nstates}  directed-edges={len(de)}  reversible-undirected={len(rev)}  green-faces={len(green)}")
    print(f"dA:        μ={mu:.3f}  σ={sd:.3f}   event gate z>1  =>  dA > {thr:.3f}   (brute-forced {n_bf} event edges)")
    print(f"dA+{BETA:.0f}·A∧A:  μ={comb_all.mean():.3f}  σ={comb_all.std():.3f}  max={comb_all.max():.1f}")
    events = sorted([(uv, e) for uv, e in de.items() if e["event"]], key=lambda x: -x[1]["comb"])
    print(f"\nEVENTS (A∧A≠0 AND z(dA)>1): {len(events)}")
    for (u, v), e in events[:12]:
        print(f"  {u:>3} --{_name(e['a']):<8}--> {v:<3}  dA={e['dA']:6.2f} (z={e['z']:+.2f})  A∧A={e['aa']:6.2f}  "
              f"dA+{BETA:.0f}A∧A={e['comb']:7.2f}")
    print(f"\nINTENSITY = reversible faces ×1 + on-charged-loop ×0.5   (reversible={len(green)}, charged={len(charged)}):")
    nloop = sum(1 for n in G.nodes if inten[n] > 0)
    print(f"  nodes on ≥1 loop (intensity>0) = {nloop}/{G.number_of_nodes()}   max intensity = {maxc:g}")
    hist = Counter(inten[n] for n in G.nodes)
    for k in sorted(hist):
        print(f"    intensity {k:>4g}: {hist[k]:>3} nodes")
    top = sorted(G.nodes, key=lambda n: -inten[n])[:10]
    print(f"  most-attached: " + ", ".join(f"{n}×{inten[n]:g}" for n in top))
    flagged = [n for n in (60, 49, 29, 38, 48, 118, 119, 120, 122, 131, 134, 138) if n in G]
    print(f"  flagged nodes: " + ", ".join(f"{n}×{inten[n]:g}" for n in flagged))

    # ===================== GREEN-INTENSITY RENDER (faces stacked = opacity; no variance/z) =====================
    fig = plt.figure(figsize=(18, 13)); bg = fig.add_axes([0, 0, 1, 1])
    bg.set_axis_off(); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    for cyc in charged:                                             # CHARGED (one-way) loops -> AMBER, half-weight
        if all(n in P for n in cyc):
            bg.add_patch(Polygon([P[n] for n in cyc], closed=True, facecolor="#F1C40F", alpha=0.06,
                                 edgecolor="#E67E22", lw=1.0, zorder=1))
    for cyc in green:                                                # reversible loops -> GREEN (stacking = intensity)
        if all(n in P for n in cyc):
            bg.add_patch(Polygon([P[n] for n in cyc], closed=True, facecolor="#2ECC40", alpha=ALPHA,
                                 edgecolor="#2ECC40", lw=2.0, zorder=1))
    def _chevron(u, v, color, lw, zo):                              # one small mid-edge arrow toward v (direction)
        (x0, y0), (x1, y1) = P[u], P[v]; dx, dy = x1 - x0, y1 - y0; L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-9:
            return
        ux, uy = dx / L, dy / L; px, py = -uy, ux; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; cl, cw = 0.007, 0.0045
        tx, ty = cx + ux * cl * 0.5, cy + uy * cl * 0.5
        bg.plot([cx - ux * cl * 0.5 + px * cw, tx, cx - ux * cl * 0.5 - px * cw],
                [cy - uy * cl * 0.5 + py * cw, ty, cy - uy * cl * 0.5 - py * cw],
                color=color, lw=lw, solid_capstyle="round", zorder=zo + 1)

    # EDGES (no z text):  BLUE = reversible (bidirectional)  ·  ORANGE = one-way  ·  RED = one-way HIGH jump (z(dA)>1)
    drawn = set()
    nbi = noneway = nhigh = 0
    for (u, v), e in de.items():
        if u not in P or v not in P:
            continue
        k = (min(u, v), max(u, v))
        if k in rev:                                                # reversible -> BLUE, no arrow (return implied)
            if k in drawn:
                continue
            drawn.add(k); nbi += 1
            bg.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#1f77b4", lw=1.1, alpha=0.85, zorder=2)
        else:                                                       # ONE-WAY -> orange, or RED if a high dA+βA∧A jump (z>1)
            high = e["z"] > 1.0
            col, lw = ("#d62728", 1.9) if high else ("#FF851B", 1.1)
            nhigh += high; noneway += (not high)
            bg.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=col, lw=lw, alpha=0.9, zorder=(4 if high else 3))
            _chevron(u, v, col, lw, 4 if high else 3)
    tile = 0.058
    for n in G.nodes:
        x, y = P[n]; ax = fig.add_axes([x - tile / 2, y - tile / 2, tile, tile])
        fr = _mask(s.frames.get(n))
        if fr is not None:
            ax.imshow(np.asarray(fr), cmap=_CMAP, vmin=0, vmax=15, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        shade = plt.cm.Greens(0.3 + 0.7 * inten[n] / maxc)          # border darker/thicker = more faces stacked
        for sp in ax.spines.values():
            sp.set_edgecolor(shade if inten[n] else "#222")
            sp.set_linewidth(0.7 + 2.3 * inten[n] / maxc)
        ax.set_title(f"{n}×{inten[n]}", fontsize=6, pad=0.5)
    bg.text(0.5, 0.99, f"{GAME} — INPUT-hash · {nstates} states · INTENSITY = reversible×1 + charged×0.5 (max {maxc:g}) · "
            f"GREEN=reversible loop · AMBER=charged (one-way) loop · edges BLUE=reversible ORANGE=one-way "
            f"RED=high jump z(dA)>1 · β={BETA:.0f}", fontsize=10, ha="center", va="top")
    out = paths.HERE / "solving_graph_deep" / f"inputhash_{GAME}_intensity.png"
    fig.savefig(out, dpi=130); plt.close(fig); del s; gc.collect()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
