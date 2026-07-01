"""green_viz.py — render ONLY the pure-reversible (GREEN/curl) loops of a rollout graph.

Logic kept pure: a face is GREEN only if EVERY edge around it is A∧A-reversible (edge_aa < TAU_REV).
To surface those loops at their natural size (pentagons/hexagons, not forced squares) we first DROP the
irreversible (ACT5/harmonic/EVENT) edges, then take cycle_basis on the REVERSIBLE-ONLY subgraph — so every
cycle we find is green by construction. The full graph is still laid out (so all frames are placed); the
irreversible edges are drawn faint for context, and only the green faces are shaded.

  ENVIRONMENTS_DIR=<dir> PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.green_viz g50t
"""
import os
os.environ.setdefault("OPERATION_MODE", "offline")
import gc
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from curvature_wm_model import paths  # noqa
sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))
from config import Config
from curvature_wm_model.hodge_flow.agent import Search
from curvature_wm_model.hodge_flow.flow import FlowGraph, TAU_REV
from hud_mask import hud_regions                                  # the SAME constant HUD region training masked

NAME = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT5", 6: "CLICK", 7: "UNDO"}


def _label(key):
    """Edge label: a move (int) -> its name; a click-on-object (6,x,y) -> CLICK@(col,row)."""
    if isinstance(key, tuple):
        return f"CLICK@({key[1]},{key[2]})"
    return NAME.get(key, str(key))


_ARC = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B",
        "#7FDBFF", "#870C25", "#1f3a93", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
_CMAP = ListedColormap(_ARC)
_POLY = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "g50t"
    max_expand = int(sys.argv[2]) if len(sys.argv) > 2 else 300       # optional 2nd arg: state-expansion cap
    cfg = Config()
    s = Search(cfg, game, k=3)
    s.run_frontier(max_expand=max_expand, chain=False, verbose=False)  # uniform coverage; ignore harmonics

    regions = hud_regions(game)                                  # PROPER HUD mask (same as training), not a crop
    fg = FlowGraph(cfg)
    fg.reps = list(range(len(s.reps)))
    for i in range(len(s.reps)):
        fr = s.frames.get(i)
        if fr is not None:
            fr = np.asarray(fr).copy()
            if fr.ndim >= 2:                                      # blank each detected HUD region to background
                Hh, Ww = fr.shape[:2]
                bgc = int(np.bincount(fr.ravel()).argmax())
                for (r0, r1, c0, c1) in regions:
                    y0, y1 = int(r0 * Hh), min(int(np.ceil(r1 * Hh)) + 1, Hh)
                    x0, x1 = int(c0 * Ww), min(int(np.ceil(c1 * Ww)) + 1, Ww)
                    fr[y0:y1, x0:x1] = bgc
        fg.frames[i] = fr
    for i in list(s.expanded):
        for c in s.children.get(i, []):
            fg.add(i, c["key"], c["cid"], c["dA"], c["aa"])
    nstates = len(s.reps)
    del s; gc.collect()

    # DIRECTED edges from the model (real transitions): (u,v) -> strongest-dA action + dA + A∧A.
    dedges = {}
    for u, am in fg.model.items():
        for a, (v, dA, aa) in am.items():
            if v == u or u not in fg.frames or v not in fg.frames:
                continue
            if (u, v) not in dedges or dA > dedges[(u, v)][1]:
                dedges[(u, v)] = (a, dA, aa)
    bidir = lambda u, v: (v, u) in dedges                            # both directions are REAL -> reversible

    G = nx.Graph(); G.add_edges_from(dedges.keys())
    if G.number_of_nodes() == 0:
        print("no edges"); return
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # PURE green: cycles of the BIDIRECTIONAL subgraph (both ways reachable -> reversible -> a real loop).
    # One-way events (e.g. ACT5 -> 0, no return) are NEVER in here, so they never make a triangle/face.
    H = nx.Graph()
    H.add_edges_from((u, v) for (u, v) in dedges if bidir(u, v) and u in G and v in G)
    green = [c for c in nx.cycle_basis(H) if len(c) >= 3]
    sizes = {}
    for c in green:
        sizes[len(c)] = sizes.get(len(c), 0) + 1

    pos = nx.kamada_kawai_layout(G)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    P = {n: (0.07 + 0.86 * (pos[n][0] - x0) / (x1 - x0 + 1e-9),
             0.07 + 0.86 * (pos[n][1] - y0) / (y1 - y0 + 1e-9)) for n in G.nodes}

    fig = plt.figure(figsize=(16, 12)); bg = fig.add_axes([0, 0, 1, 1])
    bg.set_axis_off(); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    for cyc in green:                                                 # ONLY green (reversible) faces
        bg.add_patch(Polygon([P[n] for n in cyc], closed=True, facecolor="#2ECC40", alpha=0.18,
                             edgecolor="#2ECC40", lw=3, zorder=1))

    def draw_edge(u, v, color, lw, two_way, z):
        """Line with ONE small chevron at the MIDDLE (travel direction). Bidirectional = a single small
        double-head (←→) at the middle; one-way = a single small head toward v."""
        (x0, y0), (x1, y1) = P[u], P[v]
        dx, dy = x1 - x0, y1 - y0; L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-9:
            return
        ux, uy = dx / L, dy / L; px, py = -uy, ux                   # unit along / unit perpendicular
        sh = 0.05                                                   # shrink ends to clear the frame tiles
        x0, y0, x1, y1 = x0 + ux * sh, y0 + uy * sh, x1 - ux * sh, y1 - uy * sh
        bg.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.8, zorder=z)
        cl, cw = 0.006, 0.004                                       # SMALL chevron (length / half-width)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2                       # ONE chevron, at the MIDDLE

        def chevron(sgn):
            tx, ty = cx + sgn * ux * cl * 0.5, cy + sgn * uy * cl * 0.5             # tip points travel dir
            b1x, b1y = cx - sgn * ux * cl * 0.5 + px * cw, cy - sgn * uy * cl * 0.5 + py * cw
            b2x, b2y = cx - sgn * ux * cl * 0.5 - px * cw, cy - sgn * uy * cl * 0.5 - py * cw
            bg.plot([b1x, tx, b2x], [b1y, ty, b2y], color=color, lw=lw, solid_capstyle="round", zorder=z + 1)
        if not two_way:                                           # ONLY one-way edges get an arrow;
            chevron(1.0)                                           # bidirectional = reversible -> no arrow needed

    def lab(u, v, txt, color, fs, z, bold=False):
        (x0, y0), (x1, y1) = P[u], P[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0; L = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / L, dx / L                                   # perpendicular unit
        off = 0.022                                                # shift label OFF the centered arrow
        bg.text(mx + px * off, my + py * off, txt, fontsize=fs, color=color, ha="center", va="center", zorder=z,
                fontweight=("bold" if bold else "normal"),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=(color if bold else "none"), alpha=0.8))

    LARGE = 20.0                                                      # dA+A∧A >= this = a "large jump" (event/no-return)
    njump = nbi = noneway = 0; drawn = set()
    for (u, v), (a, dA, aa) in dedges.items():
        if u not in P or v not in P or (u, v) in drawn:
            continue
        if bidir(u, v):                                              # BIDIRECTIONAL real transition -> blue, >><<
            if (v, u) in drawn:
                continue
            drawn.add((u, v)); drawn.add((v, u)); nbi += 1
            draw_edge(u, v, "#1f77b4", 1.0, True, 3)
            lab(u, v, f"{_label(a)}/{_label(dedges[(v, u)][0])}", "#1f77b4", 5.0, 4)
        else:                                                        # ONE-WAY real transition -> single direction >>>
            drawn.add((u, v))
            jump = dA + aa
            if jump >= LARGE:                                        # large event (e.g. ACT5): red, valued
                njump += 1; draw_edge(u, v, "#d62728", 1.4, False, 5)
                lab(u, v, f"{_label(a)}\ndA={dA:.0f} A∧A={aa:.0f}", "#d62728", 6.0, 6, bold=True)
            else:                                                   # one-way small passage: orange >>>
                noneway += 1; draw_edge(u, v, "#FF851B", 1.1, False, 5)
                lab(u, v, _label(a), "#FF851B", 5.0, 4)
    tile = 0.085
    for n in G.nodes:
        x, y = P[n]; ax = fig.add_axes([x - tile / 2, y - tile / 2, tile, tile])
        fr = fg.frames.get(n)
        if fr is not None:
            ax.imshow(np.asarray(fr), cmap=_CMAP, vmin=0, vmax=15, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#222"); sp.set_linewidth(1)
        ax.set_title(f"{n}", fontsize=7, pad=1)
    desc = ", ".join(f"{n}×{_POLY.get(k, str(k) + '-gon')}" for k, n in sorted(sizes.items())) or "none"
    bg.text(0.5, 0.985, f"{game} rollout (DIRECTED real transitions) — GREEN reversible loops ({len(green)}: {desc})  ·  "
            f"blue ↔ = bidirectional ({nbi})  ·  orange → = one-way ({noneway})  ·  "
            f"RED → = LARGE one-way event dA+A∧A≥{LARGE:.0f} ({njump}, never triangulated)",
            fontsize=9.5, ha="center", va="top")
    out = paths.HERE / "rollout_graphs_png" / f"green_{game}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out} | states={nstates} bidir={nbi} one-way={noneway} large-events={njump} "
          f"green-loops={len(green)} sizes={sizes}")


if __name__ == "__main__":
    main()
