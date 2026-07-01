"""solving_graph_deep/interactive_graph.py — DRAGGABLE interactive rollout graph (self-contained HTML, D3).

Same data as metrics_ls20 (INPUT-graph hash, β=10, bidirectional reversibility, intensity = # loops a node is
attached to) but rendered as an interactive force-directed graph you open in a browser:
  - drag any node (it pins where you drop it; "release pinned" re-floats them)
  - scroll to zoom, drag background to pan
  - edges: BLUE=reversible(bidirectional) · ORANGE=one-way · RED=one-way high jump z(dA)>1 (arrowed)
  - node ring/colour = intensity (# loops attached); toggle frame thumbnails on/off

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.solving_graph_deep.interactive_graph [game]
  then: open curvature_wm/solving_graph_deep/interactive_<game>.html
"""
import os
os.environ.setdefault("OPERATION_MODE", "offline")
import base64
import io
import json
import sys

from curvature_wm_model import paths  # noqa: F401
sys.path.insert(0, str(paths.REPO / "archive" / "cleanupv6"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import Config
from curvature_wm_model.hodge_flow.agent import _name
from curvature_wm_model.solving_graph_deep.metrics_ls20 import InputHashSearch, _brute_aa, _mask, _CMAP, BETA, _order_face

GAME = sys.argv[1] if len(sys.argv) > 1 else "ls20"

HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__GAME__ rollout</title>
<style>
 html,body{margin:0;height:100%;font-family:-apple-system,Helvetica,sans-serif;background:#fbfbfb;}
 #hud{position:fixed;top:8px;left:8px;z-index:10;background:rgba(255,255,255,.93);padding:8px 11px;
      border:1px solid #ccc;border-radius:7px;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.1);}
 #hud b{font-size:13px}
 .legend span{display:inline-block;margin-right:11px}
 .sw{display:inline-block;width:15px;height:3px;vertical-align:middle;margin-right:3px}
 svg{width:100vw;height:100vh;display:block;cursor:grab}
 .edge{stroke-opacity:.78}
 .node-label{font-size:7px;fill:#333;pointer-events:none;text-anchor:middle}
 button{font-size:11px;margin:4px 4px 0 0;cursor:pointer}
 #info{color:#1f77b4;font-weight:600}
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
</head><body>
<div id="hud">
 <b>__GAME__ rollout</b> — drag nodes (they pin) · scroll = zoom · drag bg = pan<br>
 <div class="legend">
   <span><i class="sw" style="background:#1f77b4"></i>reversible</span>
   <span><i class="sw" style="background:#FF851B"></i>one-way</span>
   <span><i class="sw" style="background:#d62728"></i>one-way high jump (z&gt;1)</span>
   <span><i class="sw" style="background:#F1C40F"></i>charged loop ×0.5</span>
 </div>
 <span>node ring = intensity (# loops attached); darker/thicker = more</span><br>
 <button id="toggleImg">toggle thumbnails</button>
 <button id="toggleFaces">toggle faces</button>
 <button id="release">release pinned</button>
 <button id="onlyLoops">toggle loop-only edges</button>
 <span id="info"></span>
</div>
<svg></svg>
<script>
const DATA = __DATA__;
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select("svg");
const root = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.08,8]).on("zoom", e=>root.attr("transform", e.transform)));
const sx=d3.scaleLinear().domain(d3.extent(DATA.nodes,d=>d.x)).range([90,W-90]);
const sy=d3.scaleLinear().domain(d3.extent(DATA.nodes,d=>d.y)).range([90,H-90]);
DATA.nodes.forEach(d=>{d.x=sx(d.x); d.y=sy(d.y);});
const byId=new Map(DATA.nodes.map(d=>[d.id,d]));
DATA.edges.forEach(e=>{e.source=byId.get(e.s); e.target=byId.get(e.t);});
const col={rev:"#1f77b4",one:"#FF851B",high:"#d62728"};
const greens=d3.scaleSequential(d3.interpolateGreens).domain([0,Math.max(1,DATA.maxc)]);

// CHARGED (one-way) loops behind, amber, half-weight; then GREEN reversible faces on top; both deform on drag
const cfaceSel=root.append("g").attr("class","cfaces").selectAll("polygon")
  .data(DATA.cfaces).join("polygon")
  .attr("fill","#F1C40F").attr("fill-opacity",0.06)
  .attr("stroke","#E67E22").attr("stroke-opacity",0.4).attr("stroke-width",0.8).attr("pointer-events","none");
const faceSel=root.append("g").attr("class","faces").selectAll("polygon")
  .data(DATA.faces).join("polygon")
  .attr("fill","#2ECC40").attr("fill-opacity",0.16)
  .attr("stroke","#2ECC40").attr("stroke-opacity",0.5).attr("stroke-width",1).attr("pointer-events","none");

const sim=d3.forceSimulation(DATA.nodes)
  .force("link",d3.forceLink(DATA.edges).id(d=>d.id).distance(42).strength(0.25))
  .force("charge",d3.forceManyBody().strength(-70))
  .force("collide",d3.forceCollide(15))
  .alpha(0.7).alphaDecay(0.018);

const link=root.selectAll("line").data(DATA.edges).join("line")
  .attr("class","edge").attr("stroke",d=>col[d.kind])
  .attr("stroke-width",d=>d.kind==="high"?2.4:(d.kind==="rev"?1.2:1.4));
// flow arrow = ONE chevron at the MIDDLE of each one-way edge (orange/red), pointing source->target
const chev=root.append("g").attr("class","chev").selectAll("path")
  .data(DATA.edges.filter(d=>d.kind!=="rev")).join("path")
  .attr("d","M-3.4,-3.4 L3.4,0 L-3.4,3.4").attr("fill","none")
  .attr("stroke",d=>col[d.kind]).attr("stroke-width",d=>d.kind==="high"?2.2:1.6)
  .attr("stroke-linecap","round").attr("stroke-linejoin","round").attr("pointer-events","none");

const TS=26;
const node=root.selectAll(".node").data(DATA.nodes).join("g").attr("class","node").call(drag(sim));
node.append("rect").attr("class","ring").attr("x",-TS/2-2).attr("y",-TS/2-2).attr("width",TS+4).attr("height",TS+4)
  .attr("rx",3).attr("fill","none").attr("stroke",d=>d.inten?greens(d.inten):"#bbb")
  .attr("stroke-width",d=>1+2.4*d.inten/Math.max(1,DATA.maxc));
node.append("image").attr("class","thumb").attr("href",d=>d.img)
  .attr("x",-TS/2).attr("y",-TS/2).attr("width",TS).attr("height",TS);
node.append("circle").attr("class","dot").attr("r",6).attr("fill",d=>d.inten?greens(d.inten):"#ddd")
  .attr("stroke","#333").attr("stroke-width",.5).style("display","none");
node.append("text").attr("class","node-label").attr("y",TS/2+8).text(d=>d.id+"×"+d.inten);
node.append("title").text(d=>"state "+d.id+" · attached to "+d.inten+" loop(s)");
node.on("mouseenter",(e,d)=>d3.select("#info").text("state "+d.id+" — "+d.inten+" loops"))
    .on("mouseleave",()=>d3.select("#info").text(""));

sim.on("tick",()=>{
  cfaceSel.attr("points",f=>f.map(id=>{const n=byId.get(id);return n.x+","+n.y;}).join(" "));
  faceSel.attr("points",f=>f.map(id=>{const n=byId.get(id);return n.x+","+n.y;}).join(" "));
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  chev.attr("transform",d=>{const mx=(d.source.x+d.target.x)/2,my=(d.source.y+d.target.y)/2,
    a=Math.atan2(d.target.y-d.source.y,d.target.x-d.source.x)*180/Math.PI;
    return `translate(${mx},${my}) rotate(${a})`;});
  node.attr("transform",d=>`translate(${d.x},${d.y})`);
});
function drag(sim){
  return d3.drag()
    .on("start",(e,d)=>{if(!e.active)sim.alphaTarget(0.18).restart();d.fx=d.x;d.fy=d.y;})
    .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y;})
    .on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);});   // stays pinned at fx/fy
}
let showImg=true, loopOnly=false;
d3.select("#toggleImg").on("click",()=>{showImg=!showImg;
  node.select(".thumb").style("display",showImg?null:"none");
  node.select(".ring").style("display",showImg?null:"none");
  node.select(".dot").style("display",showImg?"none":null);});
d3.select("#release").on("click",()=>{DATA.nodes.forEach(d=>{d.fx=null;d.fy=null;});
  sim.alphaTarget(0.3).restart();setTimeout(()=>sim.alphaTarget(0),900);});
d3.select("#onlyLoops").on("click",()=>{loopOnly=!loopOnly;
  link.style("display",d=>loopOnly&&d.kind!=="rev"?"none":null);
  chev.style("display",loopOnly?"none":null);});
let showFaces=true;
d3.select("#toggleFaces").on("click",()=>{showFaces=!showFaces;
  faceSel.style("display",showFaces?null:"none");cfaceSel.style("display",showFaces?null:"none");});
</script>
</body></html>"""


def _frame_b64(fr):
    fr = _mask(fr)
    if fr is None:
        return ""
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(np.asarray(fr), 0, 15), cmap=_CMAP, vmin=0, vmax=15, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    s = InputHashSearch(Config(), GAME, k=3)
    s.run_frontier(max_expand=300, chain=False, verbose=False)

    de = {}
    for u in sorted(s.expanded):
        for c in s.children.get(u, []):
            v = c["cid"]
            if v == u:
                continue
            if (u, v) not in de or c["dA"] > de[(u, v)]["dA"]:
                de[(u, v)] = {"a": c["key"], "dA": float(c["dA"]), "aa": float(c["aa"])}
    dA_all = np.array([e["dA"] for e in de.values()])
    mu, sd = float(dA_all.mean()), float(dA_all.std() or 1.0)
    for (u, v), e in de.items():
        e["z"] = (e["dA"] - mu) / sd
        if e["z"] > 1.0 and e["aa"] == 0.0:
            e["aa"] = _brute_aa(s, u, v)
        e["comb"] = e["dA"] + BETA * e["aa"]

    und = set((min(u, v), max(u, v)) for (u, v) in de)
    rev = set(k for k in und if (k[0], k[1]) in de and (k[1], k[0]) in de)     # bidirectional
    Hrev = nx.Graph(); Hrev.add_edges_from(rev)
    green = [_order_face(Hrev, c) for c in nx.minimum_cycle_basis(Hrev) if len(c) >= 3]   # reversible faces (weight 1)
    Hall = nx.Graph(); Hall.add_edges_from(und)
    charged = []
    for c in nx.minimum_cycle_basis(Hall):
        if len(c) < 3:
            continue
        cyc = _order_face(Hall, c)
        ed = [(min(cyc[i], cyc[(i + 1) % len(cyc)]), max(cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))]
        if any(e not in rev for e in ed):                                       # one-way edge -> charged loop
            charged.append(cyc)
    on_charged = set().union(*charged) if charged else set()
    G = nx.Graph(); G.add_edges_from(und)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    oneway_nodes = set()                                            # 0.5 ONLY for nodes on a UNIDIRECTIONAL arrow
    for (a, b) in (und - rev):
        oneway_nodes.add(a); oneway_nodes.add(b)
    inten = {n: sum(1 for cyc in green if n in cyc) + (0.5 if n in oneway_nodes else 0.0) for n in G.nodes}
    maxc = max(inten.values()) if inten else 1
    pos = nx.kamada_kawai_layout(G)

    nodes = [{"id": int(n), "inten": float(inten[n]), "x": float(pos[n][0]), "y": float(pos[n][1]),
              "img": _frame_b64(s.frames.get(n))} for n in G.nodes]
    edges = []
    drawn = set()
    for (u, v), e in de.items():
        if u not in G or v not in G:
            continue
        k = (min(u, v), max(u, v))
        if k in rev:
            if k in drawn:
                continue
            drawn.add(k); edges.append({"s": int(u), "t": int(v), "kind": "rev"})
        else:
            edges.append({"s": int(u), "t": int(v), "kind": "high" if e["z"] > 1.0 else "one",
                          "a": _name(e["a"]), "dA": round(e["dA"], 1), "comb": round(e["comb"], 1)})

    faces = [[int(n) for n in cyc] for cyc in green if all(n in G for n in cyc)]
    cfaces = [[int(n) for n in cyc] for cyc in charged if all(n in G for n in cyc)]
    data = {"game": GAME, "maxc": maxc, "nodes": nodes, "edges": edges, "faces": faces, "cfaces": cfaces}
    html = HTML.replace("__GAME__", GAME).replace("__DATA__", json.dumps(data))
    out = paths.HERE / "solving_graph_deep" / f"interactive_{GAME}.html"
    out.write_text(html)
    print(f"{GAME}: nodes={len(nodes)} edges={len(edges)} green-faces={len(green)} max-intensity={maxc}")
    print(f"saved {out}  ({out.stat().st_size//1024} KB)")
    print(f"open with:  open {out}")


if __name__ == "__main__":
    main()
