"""live_viewer.py — DYNAMIC web viewer: click Play and the Python backend plays the game online.

Instead of a static post-hoc HTML, this serves a live page backed by the real solver. When you press Play,
the backend constructs the 14/25-approach `Roller` (online_rollout_undo) and runs it; every probe, commit,
undo and level-up is streamed to the browser over Server-Sent Events (SSE), so you watch the search build the
state graph and play the game in real time. Pure stdlib (http.server) + D3 in the page — no extra deps.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.solving_graph_deep.live_viewer [port]
  then open http://localhost:8765  (auto-opens)

Controls in the page: game, target level, intensity-undo cap, brute-A∧A, human-clicks / grid-clicks (for
click games), wall-clock cap. The reward/escape/undo RULES are exactly the solver's — nothing is re-decided
in the browser; the page only renders what the backend streams.
"""
import os
os.environ.setdefault("OPERATION_MODE", "online")     # the viewer plays the REAL ARC-AGI API (OnlineEnv forces it too)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib                      # set a headless backend BEFORE anything imports pyplot
matplotlib.use("Agg")

import sys
import json
import queue
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from curvature_wm_model import paths            # sets up sys.path (source roots) — must precede hud_mask/config imports
try:                                             # load ARC_API_KEY (+ online config) so the standalone viewer can play online
    from dotenv import load_dotenv
    _envf = paths.HERE.parent / ".env"
    load_dotenv(_envf) if _envf.exists() else load_dotenv()
except Exception:
    pass
from hud_mask import hud_regions
from curvature_wm_model.solving_graph_deep import online_rollout_undo as oru
from curvature_wm_model.solving_graph_deep import metrics_ls20 as _mls
from curvature_wm_model.solving_graph_deep.online_rollout_undo import Roller, _name
from curvature_wm_model.solving_graph_deep.interactive_graph import _frame_b64

FALLBACK_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20",
                  "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87",
                  "tu93", "vc33", "wa30"]
# click-heavy games where the human-click teacher prior helps (move games are starved by it — keep it off there)
CLICK_GAMES = {"vc33", "lf52", "ft09", "s5i5", "tn36", "ka59", "sb26", "cn04", "sc25"}

RUN_LOCK = threading.Lock()            # one live rollout at a time (globals below are process-wide)
CURRENT = {"stop": None}               # the stop-Event of the session in flight (so a new Play / Stop can cancel it)


class StopRollout(Exception):
    pass


def _avail_games():
    """The ONLINE game ids (full, e.g. 'bp35-0a0ad940') so OnlineEnv/make get a valid id; falls back to the
    short names if the API is unreachable."""
    try:
        import arc_agi
        from arc_agi.base import OperationMode
        arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)
        envs = arc.get_environments()
        ids = []
        for e in (envs or []):
            gid = (getattr(e, "game_id", None) or getattr(e, "id", None)
                   or (e.get("game_id") if isinstance(e, dict) else None) or str(e))
            if gid:
                ids.append(str(gid))
        if ids:
            return sorted(set(ids))
    except Exception:
        pass
    return FALLBACK_GAMES


def _configure(params):
    """Set the solver's module-level knobs from the request (process-wide; guarded by RUN_LOCK)."""
    def b(k, d): return str(params.get(k, d)).lower() in ("1", "true", "on", "yes")
    def i(k, d):
        try: return int(params.get(k, d))
        except Exception: return d
    def f(k, d):
        try: return float(params.get(k, d))
        except Exception: return d
    oru.GAME = params.get("game", "ls20")
    oru.TARGET = i("target", 1)
    oru.MAX_STEPS = i("maxsteps", 200000)
    oru.WALL_SECS = f("wall", 180)
    oru.INTENSITY_MAX = f("intensity_max", 3)
    oru.BRUTE = False                       # forward stays env/brute; triangles use the MODEL (no brute-A∧A replay)
    oru.HUMAN_CLICKS = False                # NO human-run seeding — the trained model is the only prior
    oru.GRID_CLICK = b("grid_click", "0")   # optional coarse click-anywhere fallback (off by default)
    oru.INT_RECOMPUTE = i("int_recompute", 25)
    oru.ACTION_BUDGET0 = i("action_budget", 2)   # top-2 moves (escalates), exactly as the agent
    oru.CLICK_BUDGET0 = i("click_budget", 2)     # top-2 clicks (escalates)


class LiveRoller(Roller):
    """Roller that streams every node/probe/commit into a queue (the page renders them live)."""

    def __init__(self, game, q, stop):
        self._evq = q                               # NB: not self._q — that shadows Roller._q(h,a) used by rank_actions
        self._stop = stop
        self._seen_nodes = set()
        self._seen_edges = set()
        # ONLINE + model-guided, EXACTLY like agents/my_agent.py: OnlineEnv + live HUD + trained-model prior.
        from curvature_wm_model.online_env import OnlineEnv, detect_hud, detect_undo
        from curvature_wm_model.model.policy_prior import PolicyPrior
        from config import Config
        env = OnlineEnv(game, api_key=os.environ.get("ARC_API_KEY"))   # our own ARC-API instance
        detect_hud(env)                             # CURRENT HUD masking (hud_regions_from_frames, area 32..192)
        detect_undo(env)                            # capture undo (ACTION7 or symmetric-opposite) from the API
        _mls.GAME = game; _mls._REGIONS = env.regions    # keep metrics display masking in sync with the live HUD
        self.arc = env.arc                          # for the scorecard: self.arc.get_scorecard()
        prior = PolicyPrior(Config())               # the TRAINED model is the only action+click prior (no human seeding)
        super().__init__(game, policy=prior, env=env)

    # ---- helpers ----
    def _check_stop(self):
        if self._stop.is_set():
            raise StopRollout()

    def _alabel(self, a):
        nm = _name(a.aid)
        if a.type == "click" and a.xy:
            return f"{nm}@({int(a.xy[0])},{int(a.xy[1])})"
        return nm

    def _node(self, h, with_img=False):
        r = self.reg[h]
        d = {"id": r["id"], "lvl": int(r["lvl"]), "inten": round(float(self.intensity.get(h, 0.0)), 2)}
        if with_img:
            try: d["img"] = _frame_b64(r.get("frame"))
            except Exception: d["img"] = ""
        return d

    def _inten_map(self):
        return {str(self.reg[h]["id"]): round(float(self.intensity.get(h, 0.0)), 2) for h in self.reg}

    # ---- streamed hooks (every new state, every probe, every commit) ----
    def _register(self, g, snap, lvl, frame=None):
        h = super()._register(g, snap, lvl, frame)
        rid = self.reg[h]["id"]
        if rid not in self._seen_nodes:
            self._seen_nodes.add(rid)
            self._evq.put({"type": "node", **self._node(h)})
        return h

    def _on_probe(self, cur_h, a, p, registered):
        self._check_stop()
        src = self.reg[cur_h]["id"]
        if not registered:                          # unregistered probe: no-op (wall/dead click) OR death (game_over)
            self._evq.put({"type": "probe", "status": "death" if p.get("game_over") else "noop", "src": src,
                         "action": self._alabel(a), "dA": round(p["dA"], 2), "r": round(p["r"], 2)})
            return
        dst = self.reg[p["h2"]]["id"]
        status = "levelup" if p["lvl_event"] else ("useful" if self.is_useful(p) else "useless")
        if (src, dst) not in self._seen_edges:
            self._seen_edges.add((src, dst))
            self._evq.put({"type": "edge", "s": src, "t": dst, "kind": "explored",
                         "a": self._alabel(a), "dA": round(p["dA"], 2), "r": round(p["r"], 2)})
        self._evq.put({"type": "probe", "status": status, "src": src, "dst": dst,
                     "action": self._alabel(a), "dA": round(p["dA"], 2), "r": round(p["r"], 2),
                     "aa": round(p["A"], 2), "i_nxt": round(p["i_nxt"], 2)})

    def _after_commit(self, cur_h, a, p):
        self._check_stop()
        src = self.reg[cur_h]["id"]
        dst = self.reg[p["h2"]]["id"]
        ev = {"type": "commit", "src": src, "dst": dst,
              "kind": "levelup" if p["lvl_event"] else "committed",
              "action": self._alabel(a), "r": round(p["r"], 2), "dA": round(p["dA"], 2), "aa": round(p["A"], 2),
              "committed": len(self.path), "states": len(self.reg), "edges": len(self.graph.edge_map),
              "sigma": round(self.sigma, 2), "level": int(p["lvl2"]), "levelup": bool(p["lvl_event"]),
              "node": self._node(p["h2"], with_img=True), "intensity": self._inten_map()}
        try: ev["frame"] = _frame_b64(self.reg[p["h2"]].get("frame"))
        except Exception: ev["frame"] = ""
        self._evq.put(ev)


def _scorecard_summary(R):
    """The ARC scorecard for this play (arc.get_scorecard() -> .score + .games)."""
    try:
        sc = R.arc.get_scorecard()
        if sc is None:
            return None
        out = {"score": getattr(sc, "score", None)}
        games = getattr(sc, "games", None)
        try: out["games"] = len(games) if games is not None else None
        except Exception: out["games"] = None
        return out
    except Exception:
        return None


def _run_rollout(game, q, stop):
    try:
        q.put({"type": "status", "msg": f"loading {game} env + model…"})
        R = LiveRoller(game, q, stop)
        q.put({"type": "status", "msg": "playing", "game": game, "target": oru.TARGET})
        res = R.run()
        q.put({"type": "done", "solved": bool(res["solved"]), "committed": res["path_len"],
               "states": res["states"], "edges": res["edges"], "level_probes": res["level_probes"],
               "scorecard": _scorecard_summary(R)})
    except StopRollout:
        q.put({"type": "done", "stopped": True})
    except Exception as e:
        q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()[-1200:]})
    finally:
        q.put(None)                                  # sentinel -> close SSE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # quiet
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/games":
            self._send(200, "application/json", json.dumps({"games": _avail_games(),
                                                            "click_games": sorted(CLICK_GAMES)}).encode())
        elif u.path == "/stop":
            prev = CURRENT.get("stop")
            if prev: prev.set()
            self._send(200, "application/json", b'{"stopped":true}')
        elif u.path == "/play":
            self._play({k: v[0] for k, v in parse_qs(u.query).items()})
        else:
            self._send(404, "text/plain", b"not found")

    def _play(self, params):
        prev = CURRENT.get("stop")                   # cancel any session already in flight
        if prev:
            prev.set()
        with RUN_LOCK:
            stop = threading.Event()
            CURRENT["stop"] = stop
            _configure(params)
            game = params.get("game", "ls20")
            q = queue.Queue()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
            except Exception:
                CURRENT["stop"] = None
                return
            t = threading.Thread(target=_run_rollout, args=(game, q, stop), daemon=True)
            t.start()
            while True:
                ev = q.get()
                if ev is None:
                    break
                try:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ValueError):
                    stop.set()                       # browser closed the tab -> stop the rollout
                    break
            CURRENT["stop"] = None


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>live rollout viewer</title>
<style>
 :root{--bg:#0f1320;--pan:#171c2e;--ln:#2a3350;--tx:#dfe6f5;--mut:#8b97b8;--blue:#4f9dff;--red:#ff5d6c;--grn:#46d39a;}
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Helvetica,sans-serif;background:var(--bg);color:var(--tx)}
 #app{display:grid;grid-template-rows:auto 1fr 150px;height:100vh}
 header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;background:var(--pan);border-bottom:1px solid var(--ln)}
 header h1{font-size:14px;margin:0 10px 0 0;font-weight:700;letter-spacing:.3px}
 select,input,button{background:#0c1020;color:var(--tx);border:1px solid var(--ln);border-radius:6px;padding:4px 8px;font-size:12px}
 label{font-size:11px;color:var(--mut);display:inline-flex;gap:4px;align-items:center}
 button{cursor:pointer;font-weight:600}
 #play{background:var(--blue);color:#04122b;border:0} #stop{background:var(--red);color:#2b0410;border:0}
 button:disabled{opacity:.4;cursor:default}
 #stat{margin-left:auto;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
 #stat b{color:var(--tx)}
 main{display:grid;grid-template-columns:300px 1fr;min-height:0}
 #framewrap{border-right:1px solid var(--ln);background:#0a0e1a;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:10px;gap:8px}
 #frame{image-rendering:pixelated;width:260px;height:260px;border:1px solid var(--ln);border-radius:6px;background:#05070f}
 #act{font-size:12px;color:var(--mut);text-align:center;min-height:16px}
 #act b{color:var(--blue)}
 svg{width:100%;height:100%;display:block}
 .lnk{stroke-linecap:round}
 #log{overflow-y:auto;background:var(--pan);border-top:1px solid var(--ln);font-size:11px;font-family:ui-monospace,Menlo,monospace;padding:4px 8px}
 .le{padding:1px 4px;white-space:nowrap;color:var(--mut)}
 .le.c{color:var(--blue)} .le.u{color:var(--red);font-weight:700} .le.g{color:var(--grn)} .le.n{opacity:.5}
 .pill{display:inline-block;padding:0 5px;border-radius:8px;font-size:10px;margin-left:6px}
 .ok{background:#123a2a;color:var(--grn)} .bad{background:#3a1620;color:var(--red)}
</style>
<script src="https://d3js.org/d3.v7.min.js"></script></head><body>
<div id="app">
 <header>
  <h1>⚡ live rollout</h1>
  <label>game <select id="game"></select></label>
  <label>target L <input id="target" type="number" value="1" min="1" max="9" style="width:46px"></label>
  <label>undo&gt; <input id="imax" type="number" value="3" step="1" style="width:46px"></label>
  <label>wall s <input id="wall" type="number" value="180" step="10" style="width:56px"></label>
  <label><input id="grid" type="checkbox"> grid-clicks</label>
  <button id="play">▶ Play</button>
  <button id="stop" disabled>■ Stop</button>
  <span id="stat"></span>
 </header>
 <main>
  <div id="framewrap">
    <img id="frame" alt="game frame">
    <div id="act">press Play</div>
  </div>
  <svg></svg>
 </main>
 <div id="log"></div>
</div>
<script>
const $=id=>document.getElementById(id);
let es=null, nodes=[], links=[], nById=new Map(), lById=new Map(), lastDst=null, sim=null, sel={};
const COL={committed:"#4f9dff",levelup:"#ff5d6c",explored:"#39406a",rev:"#9ecae1"};

// ---- populate games ----
fetch("/games").then(r=>r.json()).then(d=>{
  const g=$("game"); d.games.forEach(n=>{const o=document.createElement("option");o.value=o.text=n;g.append(o);});
  const ls=d.games.find(n=>n.indexOf("ls20")===0); if(ls) g.value=ls;   // default to ls20 if present
});

// ---- D3 force graph (grows live) ----
const svg=d3.select("svg"), root=svg.append("g");
const eg=root.append("g"), ng=root.append("g");
svg.call(d3.zoom().scaleExtent([0.1,8]).on("zoom",e=>root.attr("transform",e.transform)));
function dims(){const r=svg.node().getBoundingClientRect();return [r.width,r.height];}
function initSim(){
  const [w,h]=dims();
  sim=d3.forceSimulation(nodes)
    .force("link",d3.forceLink(links).id(d=>d.id).distance(46).strength(.5))
    .force("charge",d3.forceManyBody().strength(-160))
    .force("center",d3.forceCenter(w/2,h/2))
    .force("collide",d3.forceCollide(18))
    .on("tick",tick);
}
function tick(){
  sel.l&&sel.l.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  sel.n&&sel.n.attr("transform",d=>`translate(${d.x},${d.y})`);
}
function restart(){
  sel.l=eg.selectAll("line").data(links,d=>d.key).join("line")
    .attr("class","lnk").attr("stroke",d=>COL[d.kind]||COL.explored)
    .attr("stroke-width",d=>d.kind==="committed"||d.kind==="levelup"?2.4:1)
    .attr("stroke-opacity",d=>d.kind==="explored"?.5:.95);
  sel.n=ng.selectAll("g.nd").data(nodes,d=>d.id).join(enter=>{
    const g=enter.append("g").attr("class","nd");
    g.append("circle").attr("class","ring");
    g.append("image").attr("class","img").attr("x",-13).attr("y",-13).attr("width",26).attr("height",26)
      .attr("preserveAspectRatio","none").style("image-rendering","pixelated");
    g.append("text").attr("y",22).attr("text-anchor","middle").attr("font-size",8).attr("fill","#8b97b8");
    return g;
  });
  sel.n.select("text").text(d=>d.id);
  sel.n.select("circle.ring").attr("r",d=>d.img?15:6)
    .attr("fill",d=>d.img?"none":(d.committed?COL.committed:"#222a44"))
    .attr("stroke",d=>d.cur?"#ffd166":(d.committed?COL.committed:ringCol(d)))
    .attr("stroke-width",d=>d.cur?3:(d.committed?2:1.2));
  sel.n.select("image.img").attr("href",d=>d.img||null).style("display",d=>d.img?null:"none");
  sim.nodes(nodes); sim.force("link").links(links); sim.alpha(.5).restart();
}
let maxI=1;
function ringCol(d){const t=Math.min(1,d.inten/Math.max(1,maxI));return d3.interpolateGreens(.25+.6*t);}
function addNode(o){ if(nById.has(o.id))return; const n={id:o.id,lvl:o.lvl,inten:o.inten||0,img:null,committed:false,cur:false};
  nById.set(o.id,n); nodes.push(n); restart(); }
function addLink(o){ const key=o.s+"->"+o.t; if(lById.has(key)){lById.get(key).kind=o.kind;restart();return;}
  addNode({id:o.s}); addNode({id:o.t}); const l={key,source:o.s,target:o.t,kind:o.kind};
  lById.set(key,l); links.push(l); restart(); }

// ---- log ----
function log(cls,txt){ const d=document.createElement("div"); d.className="le "+cls; d.textContent=txt;
  const L=$("log"); L.append(d); if(L.children.length>400)L.removeChild(L.firstChild); L.scrollTop=L.scrollHeight; }

// ---- event handling ----
function handle(ev){
  if(ev.type==="status"){ $("act").innerHTML=ev.msg==="playing"?`playing <b>${ev.game}</b> → L${ev.target}`:ev.msg;
    if(ev.msg!=="playing") log("n",ev.msg); return; }
  if(ev.type==="node"){ addNode(ev); return; }
  if(ev.type==="edge"){ addLink(ev); return; }
  if(ev.type==="probe"){
    if(ev.status==="death"){ log("u",`✗ probe ${ev.action}  dA=${ev.dA}  DEATH → penalized + pruned`); }
    else if(ev.status==="noop"){ log("n",`· probe ${ev.action}  dA=${ev.dA}  (no-op)`); }
    else if(ev.status==="levelup"){ log("u",`★ probe ${ev.action} → ${ev.dst}  dA=${ev.dA}  LEVEL-UP found`); }
    else if(ev.status==="useful"){ log("g",`✓ probe ${ev.action} → ${ev.dst}  r=${ev.r} dA=${ev.dA} A∧A=${ev.aa} i'=${ev.i_nxt}`); }
    return;
  }
  if(ev.type==="commit"){
    maxI=Math.max(maxI, ...Object.values(ev.intensity||{}).map(Number));
    if(ev.intensity) for(const [id,v] of Object.entries(ev.intensity)){ const n=nById.get(+id); if(n)n.inten=v; }
    addNode({id:ev.dst,lvl:ev.level}); addLink({s:ev.src,t:ev.dst,kind:ev.kind});
    const n=nById.get(ev.dst); n.committed=true; if(ev.node&&ev.node.img)n.img=ev.node.img;
    if(lastDst!==null)nById.get(lastDst)&&(nById.get(lastDst).cur=false);
    if(lastDst!==null && ev.src!==lastDst) log("u",`↩ undo → state ${ev.src}`);
    n.cur=true; lastDst=ev.dst;
    if(ev.frame)$("frame").src=ev.frame;
    $("act").innerHTML=`commit <b>${ev.action}</b> → state ${ev.dst}` + (ev.levelup?'  <span class="pill ok">LEVEL UP</span>':'');
    log(ev.levelup?"u":"c",`#${ev.committed} ${ev.src} —${ev.action}→ ${ev.dst}  r=${ev.r} dA=${ev.dA}`+(ev.levelup?`  ·LEVEL ${ev.level}!`:""));
    $("stat").innerHTML=`L<b>${ev.level}</b> · committed <b>${ev.committed}</b> · states <b>${ev.states}</b> · edges <b>${ev.edges}</b> · σ=${ev.sigma}`;
    restart(); return;
  }
  if(ev.type==="done"){
    const m = ev.stopped?"stopped":(ev.solved?`SOLVED ✓ (committed ${ev.committed}, ${ev.states} states)`:`stopped — not solved (${ev.states} states)`);
    $("act").innerHTML = ev.solved?'<span class="pill ok">SOLVED</span>':(ev.stopped?'stopped':'<span class="pill bad">not solved</span>');
    log(ev.solved?"g":"u","── "+m+" ──");
    if(ev.scorecard) log(ev.solved?"g":"n",`scorecard: score=${ev.scorecard.score} · games=${ev.scorecard.games}`);
    finish(); return;
  }
  if(ev.type==="error"){ log("u","ERROR: "+ev.msg); console.log(ev.tb); $("act").textContent="error"; finish(); return; }
}
function finish(){ if(es){es.close();es=null;} $("play").disabled=false; $("stop").disabled=true; }

// ---- controls ----
$("play").onclick=()=>{
  if(es)es.close();
  nodes=[];links=[];nById.clear();lById.clear();lastDst=null;maxI=1; eg.selectAll("*").remove();ng.selectAll("*").remove();
  $("log").innerHTML=""; $("frame").removeAttribute("src"); initSim();
  const q=new URLSearchParams({game:$("game").value,target:$("target").value,intensity_max:$("imax").value,
    wall:$("wall").value,grid_click:$("grid").checked?1:0});
  $("play").disabled=true; $("stop").disabled=false; $("act").textContent="connecting…";
  es=new EventSource("/play?"+q.toString());
  es.onmessage=e=>handle(JSON.parse(e.data));
  es.onerror=()=>{ log("n","(stream closed)"); finish(); };
};
$("stop").onclick=()=>{ fetch("/stop"); log("n","stop requested"); finish(); };
window.addEventListener("resize",()=>{ if(sim){const[w,h]=dims();sim.force("center",d3.forceCenter(w/2,h/2)).alpha(.2).restart();}});
</script></body></html>"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8765"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"live viewer → {url}   (Ctrl-C to stop)", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
