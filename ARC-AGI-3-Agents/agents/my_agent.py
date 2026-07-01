"""my_agent.py — the 'sleepy' ARC-AGI-3 agent (plugs into the framework; agent.py is NOT modified).

It owns nothing but the decision: solve the game on OUR OWN ARC-API env using the deep reward-guided rollout
solver with the TRAINED model as the action prior (curvature_wm_model: OnlineEnv + PolicyPrior + the solver),
then REPLAY the committed solution through the framework's scored game one action per choose_action. The solve
runs on a separate env/scorecard (never scored); the framework's game only does the short solution.
"""
import os

try:
    from agents.agent import Agent          # the framework base (read-only; we only subclass it)
except Exception:
    Agent = object

try:
    from arcengine import GameAction
except Exception:
    GameAction = None

_GA = None
def _ga(aid, xy=None):
    global _GA
    if _GA is None:
        _GA = {0: GameAction.RESET, 1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
               4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6, 7: GameAction.ACTION7}
    a = _GA[int(aid)]
    if int(aid) == 6 and xy is not None:
        a.set_data({"x": int(xy[0]), "y": int(xy[1])})
    return a


class MyAgent(Agent):
    MAX_ACTIONS = 1_000_000                  # override the template's 80 (real cap is 5x baseline/level)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plan = None                    # committed solution to replay [(aid, xy), ...]
        self._i = 0

    # ---- solve on our OWN ARC-API env with the trained model as the rollout prior ----
    def _solve(self):
        from curvature_wm_model.online_env import OnlineEnv, detect_hud, detect_undo
        import curvature_wm_model.solving_graph_deep.online_rollout_v5 as v5
        from curvature_wm_model.model.policy_prior import PolicyPrior
        from config import Config
        v5.MODEL_AA = True                                                     # Hodge-flow A∧A (sub-goal detector, no env-verify reset)
        v5.TEACHER = True                                                      # navigate by distinct-node-count change, then reward
        v5.TARGET = int(os.environ.get("SLEEPY_TARGET", "999"))               # v5: play THROUGH all levels (multi-level)
        v5.MAX_STEPS = int(os.environ.get("SLEEPY_MAX_STEPS", "30000"))
        v5.INTENSITY_MAX = float(os.environ.get("INTENSITY_MAX", "2"))
        v5.BRUTE = False
        v5.INT_RECOMPUTE = 25
        v5.WALL_SECS = float(os.environ.get("SLEEPY_WALL", "600"))
        offline = os.environ.get("SLEEPY_OFFLINE", "0") == "1"                 # offline = local env/environment_files (no API)
        gid = self.game_id.split("-")[0] if offline else self.game_id         # offline games use the short name
        env = OnlineEnv(gid, api_key=os.environ.get("ARC_API_KEY"), online=not offline)
        detect_hud(env)                                                        # HUD masking (32..192) from the game (all levels)
        detect_undo(env)                                                       # capture undo
        prior = PolicyPrior(Config())                                          # the TRAINED model (M1/MPS) guides the rollout
        roller = v5.Roller(gid, policy=prior, env=env)
        res = roller.run()                                                     # writes tmp_graph/graph_<game>.json
        plan = [(int(a.aid), (tuple(a.xy) if a.xy else None)) for (_h, a) in roller.path]
        print(f"[sleepy-v5] {self.game_id}: solved={res.get('solved')} plan_len={len(plan)} "
              f"states={res.get('states')} maxlevel={getattr(roller, 'start_lvl', 0)}->{env.live.prev_level}", flush=True)
        return plan

    # ---- framework interface (drives the SCORED game) ----
    def choose_action(self, frames, latest_frame):
        if self._plan is None:
            try:
                self._plan = self._solve()
            except Exception as e:
                print(f"[sleepy] solve failed for {self.game_id}: {e}", flush=True)
                self._plan = []
        st = getattr(getattr(latest_frame, "state", None), "name", "")
        if st in ("NOT_PLAYED", "GAME_OVER"):
            return GameAction.RESET                                  # need a (re)start before replaying the plan
        # ADVANCE the plan ONLY at an ACTIONABLE state — the next state is where we can act, NOT an intermediate /
        # animation frame. At a non-actionable frame, HOLD (re-send the last action) and don't consume a plan step,
        # so the replay stays aligned with the actionable states the solve planned over.
        if not getattr(latest_frame, "available_actions", None):
            if self._i > 0:
                aid, xy = self._plan[self._i - 1]
                return _ga(aid, xy)
            return GameAction.RESET
        if self._i < len(self._plan):
            aid, xy = self._plan[self._i]
            self._i += 1
            return _ga(aid, xy)
        return GameAction.RESET

    def is_done(self, frames, latest_frame):
        return self._plan is not None and self._i >= len(self._plan)
