"""sweep.py — run the online rollout solver on every game's LEVEL 1 (autonomous batch).

Each game runs as a SUBPROCESS with a wall-clock timeout, so a hang/crash on one game is killed and the sweep
continues to the next. Per-game stdout -> sweep_logs/<game>.log; running summary -> sweep_results/summary.json
and summary.txt. Solver default: online_rollout_undo (DFS + intensity<=3 undo). All outputs stay in solving_graph_deep.

  python -m curvature_wm.solving_graph_deep.sweep                 # all games, level 1
  python -m curvature_wm.solving_graph_deep.sweep ar25 bp35       # specific games
  SOLVER=online_rollout      python -m ...sweep              # pick a different solver module
  MAXSTEPS=8000 TIMEOUT=300  python -m ...sweep              # tune budget / per-game timeout
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

from curvature_wm_model import paths  # noqa: F401

HERE = Path(__file__).resolve().parent
LOGD = HERE / "sweep_logs"; RESD = HERE / "sweep_results"
LOGD.mkdir(exist_ok=True); RESD.mkdir(exist_ok=True)

SOLVER = os.environ.get("SOLVER", "online_rollout_undo")
MAXSTEPS = os.environ.get("MAXSTEPS", "8000")
TIMEOUT = int(os.environ.get("TIMEOUT", "360"))
TARGET = os.environ.get("TARGET", "1")

ALL = sorted(p.name for p in paths.GRAPH_DATA.iterdir() if p.is_dir()) if paths.GRAPH_DATA.exists() else []
GAMES = [g for g in sys.argv[1:] if not g.startswith("-")] or ALL


def parse(out, g):
    solved = "solved=True" in out
    lines = [l for l in out.splitlines() if l.strip().startswith(f"{g}:")]
    return solved, (lines[-1].strip() if lines else "")


def main():
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONUNBUFFERED="1", OPERATION_MODE="offline")
    mod = f"curvature_wm.solving_graph_deep.{SOLVER}"
    results = {}
    print(f"=== sweep: solver={SOLVER} target=L{TARGET} maxsteps={MAXSTEPS} timeout={TIMEOUT}s games={len(GAMES)} ===", flush=True)
    for g in GAMES:
        t0 = time.time()
        logp = LOGD / f"{g}.log"
        try:
            with open(logp, "w") as lf:                         # stream to file -> partial output survives a kill
                p = subprocess.Popen([sys.executable, "-m", mod, g, MAXSTEPS, TARGET],
                                     stdout=lf, stderr=subprocess.STDOUT, env=env)
                try:
                    rc = p.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(); rc = None
            out = logp.read_text(errors="ignore")
            solved, summ = parse(out, g)
            status = "SOLVED" if solved else ("TIMEOUT" if rc is None else ("CRASH" if rc != 0 else "NOT_SOLVED"))
        except Exception as ex:
            logp.write_text(f"DRIVER ERROR: {ex!r}")
            status, summ = "ERROR", repr(ex)
        sec = round(time.time() - t0, 1)
        results[g] = {"status": status, "summary": summ, "sec": sec}
        print(f"  {g:>6}: {status:<10} {sec:>6.1f}s  {summ}", flush=True)
        (RESD / "summary.json").write_text(json.dumps(results, indent=2))
        n_solved = sum(1 for v in results.values() if v["status"] == "SOLVED")
        txt = [f"sweep solver={SOLVER} target=L{TARGET}  ({n_solved}/{len(results)} solved so far)"]
        for gg, v in results.items():
            txt.append(f"  {gg:>6}: {v['status']:<10} {v['sec']:>6.1f}s  {v['summary']}")
        (RESD / "summary.txt").write_text("\n".join(txt))
    n_solved = sum(1 for v in results.values() if v["status"] == "SOLVED")
    print(f"\n=== DONE: {n_solved}/{len(GAMES)} solved level {TARGET} ===", flush=True)
    print("\n".join(f"  {g:>6}: {v['status']:<10} {v['summary']}" for g, v in results.items()), flush=True)


if __name__ == "__main__":
    main()
