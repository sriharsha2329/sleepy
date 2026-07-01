# curvature_wm

A clean, single, **~100K-weight** graph-transformer world model. No wake/sleep, no brain pool.
Learns from human runs across all games **except a held-out set** (`bp35`, `sc25`, `wa30`).

## What it learns (two tasks, one shared trunk)

A shared **graph-transformer trunk** (edge-biased attention — copied & edited from the old `Brain`)
encodes a state's object graph into per-node features. The *same* trunk encodes the current state
(for all heads) and the next state (for the inverse/click heads).

1. **Forward** — predict the **next latent state `z_next`** (the full latent, not Δz) from `(current
   state, action)`; for CLICK it also conditions on the clicked node's latent. Carries an `alive` head so
   births/deaths are represented. Loss = **RAW** next-latent error (MSE, or Huber via `--fwd_loss huber`).
   *Next state is masked from the input.*
2. **Action type** — predict the action id from `(current, next)`. *Action is masked from the input.*
3. **Click-on-node** — when the action is CLICK, predict **which node** was clicked (node localized via its
   edges). Part of the "next action" task.

The two task losses (forward; action = action-type CE + click CE) are combined with a **QRE balancer**
re-anchored on the **held-out games' skill** every `--eval_every` steps. Everything operates on the **raw
latent** (no graph high-pass / change-decomposition transform).

## Curvature (causal-ness) — read ANALYTICALLY, no extra model
Curvature **`r = (dA+A∧A)/Σ|z_current|`** is computed straight from the trained model — there is **no
separate ML head**. `dA = Σ smae(z_next−z_cur)`; `A∧A = min over return actions (stay + moves + click-on-node,
undo excluded) of smae(f_θ(z_next,b) − z_cur)` via the forward head. We read it two ways and compare:

- **`r(target)`** — uses the **actual** next state (real human transition).
- **`r(pred)`** — uses the **predicted** next state `ẑ = f_θ(z_t, a)`, so it's a pure function of `(current
  state, action)` (oblivious to the real next state). This replaces the old regression head: instead of an
  MLP guessing `r` from `(s,a)`, we plug the forward model's own next-state prediction into the same formula.
  The predicted state reuses the current edges/mask (the forward head predicts node latents only).

The gap `r(target) − r(pred)` is the **surprise**: large where the change was unforeseeable from `(s,a)` — the
candidate causal/event step. See `metrics/curvature.py` (`curvature_batch`, `curvature_batch_pred`,
`target_vs_pred`).

## Layout

```
curvature_wm/
  paths.py            # ALL path wiring; imports archived infra (config/perception/data) READ-ONLY
  splits.py           # ALL_GAMES / HELDOUT_GAMES / TRAIN_GAMES
  data/loader.py      # load_train / load_heldout -> Zp,Zc,Mp,Mc,EFp,EFc,a,ct
  model/trunk.py      # GraphTransformerTrunk (edge-biased attention)
  model/heads.py      # ForwardHead, InverseHead, ClickHead
  model/world_model.py# WorldModel: predict_next / predict_action / predict_click
  metrics/curvature.py# curvature_batch (target) + curvature_batch_pred, build_records, A∧A histogram, target_vs_pred
  training/qre.py     # QREBalancer
  training/train.py   # M2 trainer (two-task QRE loop, raw forward loss)
  eval/evaluate.py    # held-out report (fwd MAE + action/click t1/3/5, binned by actual change)
  tests/              # probe_lr_bp35, scale_analysis, test_curvature_metric
  checkpoints/        # *.pt (gitignored)
```

Heavy perception/featurize/`Config` substrate is imported **read-only** from `archive/cleanup/...` via
`paths.py`; the archived folders are never edited. Graph data is read from `archive/.../graph_data`.

## Run (from repo root, with the project venv)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.model.world_model              # param smoke (~109K base)
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.model.train --steps 2500 --fwd_loss huber --tag m2_raw100k
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.eval.evaluate --ckpt curvature_wm/checkpoints/m2_raw100k_2500.pt
# curvature: r(target) vs r(pred) on any games (incl. held-out)
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.metrics.curvature --ckpt curvature_wm/checkpoints/m2_raw100k_2500.pt --games bp35
# tests
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.tests.test_curvature_metric
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.tests.probe_lr_bp35           # 10L/10R on live bp35
```

## Deferred (later phases)
- **Rung-2 cross-game DML** — is an action causally useful (de-confounded τ̂)?
- **Stop-grad PPO policy** maximizing curvature `dA + A∧A`.
