"""run_solver.py — verbose runner for the deep solver (online OR offline), using the SAME config as my_agent:
the trained model prior on the M1 GPU (MPS), top-2 combined ranking, A∧A brute only when dA>μ+σ. main.py
online swallows the solve trace, so use this to WATCH the search (every commit / state / level-up).

  OPERATION_MODE=offline python agents/run_solver.py bp35 offline
  python agents/run_solver.py bp35-0a0ad940 online        # loads .env for ARC_API_KEY
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))   # agents/
ROOT = os.path.dirname(HERE)                          # ARC-AGI-3-Agents/
sys.path.insert(0, ROOT)

game = sys.argv[1] if len(sys.argv) > 1 else "bp35"
online = (sys.argv[2].lower() if len(sys.argv) > 2 else "offline") == "online"
os.environ["OPERATION_MODE"] = "online" if online else "offline"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLBACKEND", "agg")
if online:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
else:
    os.environ.setdefault("ENVIRONMENTS_DIR", os.path.join(ROOT, "curvature_wm_model", "env", "environment_files"))

from curvature_wm_model.online_env import OnlineEnv, detect_hud, detect_undo
import curvature_wm_model.solving_graph_deep.online_rollout_v5 as oru
from curvature_wm_model.solving_graph_deep.online_rollout_v5 import _name
from curvature_wm_model.model.policy_prior import PolicyPrior
from config import Config

oru.MODEL_AA = os.environ.get("MODEL_AA", "1") == "1"          # v5: Hodge-flow A∧A (sub-goal detector), default ON
oru.TEACHER = os.environ.get("TEACHER", "1") == "1"           # v5: distinct-node-count navigation, default ON
oru.TARGET = int(os.environ.get("TARGET", "999"))            # v5: play THROUGH all levels by default
oru.MAX_STEPS = int(os.environ.get("MAX_STEPS", "30000"))
oru.INTENSITY_MAX = float(os.environ.get("INTENSITY_MAX", "2"))
oru.INT_RECOMPUTE = int(os.environ.get("INT_RECOMPUTE", "25"))
oru.WALL_SECS = float(os.environ.get("WALL_SECS", "0"))

gname = game if online else game.split("-")[0]
print(f"[run_solver] game={gname} online={online} MAX_STEPS={oru.MAX_STEPS} ... building env + model", flush=True)
env = OnlineEnv(game if online else gname, api_key=os.environ.get("ARC_API_KEY"), online=online)
detect_hud(env)                                                      # HUD detected from the LIVE opening frames (no human runs)
detect_undo(env)
prior = PolicyPrior(Config())                                        # trained model on MPS (the action+click prior)
print(f"[run_solver] model dev={prior.dev} | HUD regions={len(env.regions)} undo7={env.undo7} avail={env.actions()}", flush=True)


class _V(oru.Roller):
    """Verbose: prints every commit so the search is visible online or offline."""
    def _commit(self, cur_h, a, p, step):
        super()._commit(cur_h, a, p, step)
        act = _name(a.aid) + (f"@{tuple(int(v) for v in a.xy)}" if a.xy else "")
        print(f"  COMMIT #{len(self.path):3d} step{step:4d}: {act:16s} -> s{self.reg[p['h2']]['id']:<3d}"
              f" r={p['r']:7.1f} dA={p['dA']:6.1f} aa={p['A']:6.1f} lvl={p['lvl2']} | states={len(self.reg)}", flush=True)


R = _V(gname, policy=prior, env=env)
t0 = time.time()
res = R.run()
e = R.env
ac = getattr(e, "acount", 0); nf = getattr(e, "n_fwd", 0); nrs = getattr(e, "n_replay_steps", 0)
nrest = getattr(e, "n_restore", 0); nrep = getattr(e, "n_replay", 0); nnoop = getattr(e, "n_restore_noop", 0)
cheap_undo = ac - nf - nrs
print(f"[run_solver] DONE {time.time()-t0:.0f}s: solved={res.get('solved')} "
      f"committed={res.get('path_len')} states={res.get('states')} edges={res.get('edges')}", flush=True)
print(f"  ACTIONS acount={ac} = fwd(step) {nf} + cheap-undo {cheap_undo} + replay {nrs}  "
      f"|| restores={nrest} (noop {nnoop}, replay-fallback {nrep}, cheap {nrest-nnoop-nrep})  "
      f"|| MODEL_AA={oru.MODEL_AA} TEACHER={oru.TEACHER}", flush=True)
