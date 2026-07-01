"""hud_final.py — STANDARD HUD-masking verification viz for all 25 offline games.

The canonical way to eyeball HUD masking after any change to hud_mask.py / detect_hud / _grid / strip_hud.
For each game it detects the HUD at game start (detect_hud -> blanks env.regions every frame), optionally
steps N random actions, then renders the MASKED frame (exactly what perception sees) with the captured HUD
bbox(es) outlined in orange + the pixel coords in each title. "bars only": square HUD (corner icons,
sb26/sk48 rule-blocks) is intentionally out of scope.

Run from the ARC-AGI-3-Agents dir with the project venv:
    python curvature_wm_model/perception/hud_final.py        # start frame (N=0)
    python curvature_wm_model/perception/hud_final.py 10     # after 10 random actions
Output -> $HUD_VIZ_OUT (default: cleanup11062026_attic/hud_viz/hud_final_<N>.png), kept OUT of the repo.
"""
import os, sys, random

random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                      # .../ARC-AGI-3-Agents
N = int(sys.argv[1]) if len(sys.argv) > 1 else 0                   # random actions before capturing the frame
os.environ["OPERATION_MODE"] = "offline"
EF = os.path.join(ROOT, "curvature_wm_model", "env", "environment_files")
os.environ.setdefault("ENVIRONMENTS_DIR", EF)
os.environ["MPLBACKEND"] = "agg"
sys.path.insert(0, ROOT)
import numpy as np, matplotlib; matplotlib.use("agg"); import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Rectangle                                                        # noqa: E402
from matplotlib.colors import ListedColormap                                                    # noqa: E402
from curvature_wm_model.online_env import OnlineEnv, detect_hud                                  # noqa: E402

OUT_DIR = os.environ.get("HUD_VIZ_OUT", "/Users/nuthsharsh/Documents/codes/cleanup11062026_attic/hud_viz")
os.makedirs(OUT_DIR, exist_ok=True)
PAL = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B",
       "#7FDBFF","#870C25","#FFFFFF","#555555","#999999","#DDDDDD","#39CCCC","#B10DC9"]
cm = ListedColormap(PAL)
games = sorted(os.listdir(EF))[:25]
fig, ax = plt.subplots(5, 5, figsize=(20, 20.5))
n_det = 0
for i, G in enumerate(games):
    a_ = ax[i // 5][i % 5]
    n = 0
    try:
        env = OnlineEnv(G, online=False); detect_hud(env)         # HUD mask captured ONCE at game start
        regs = env.regions
        for _ in range(N):                                        # optional: step N random actions
            avail = [a for a in env.actions()]
            if not avail:
                break
            a = random.choice(avail)
            if a == 6:
                env.step(6, (random.randint(0, 63), random.randint(0, 63)))
            else:
                env.step(a)
            n += 1
        fr = env.live.last_frame.astype(int).copy()               # MASKED frame after n actions (perception view)
    except Exception as e:
        fr = np.zeros((64, 64), int); regs = []; print(G, "ERR", e)
    a_.imshow(fr, cmap=cm, vmin=0, vmax=15, interpolation="nearest")
    for (B0, C0, B1, C1) in regs:
        a_.add_patch(Rectangle((C0 - .5, B0 - .5), C1 - C0 + 1, B1 - B0 + 1, fill=False, ec="orange", lw=2.5))
    tag = (" | ".join(f"[{r[0]},{r[1]},{r[2]},{r[3]}]" for r in regs)) if regs else "no HUD bar"
    if regs:
        n_det += 1
    a_.set_title(f"{G}  (+{n})  {tag}", fontsize=9)
    a_.set_xticks([]); a_.set_yticks([])
fig.suptitle(f"HUD masking — masked frame after {N} random actions + captured bbox (orange), {n_det}/25 detected — bars only",
             fontsize=15, y=0.995)
out = os.path.join(OUT_DIR, f"hud_final_{N}.png")
fig.savefig(out, dpi=78, bbox_inches="tight"); print("saved", out, "detected", n_det)
