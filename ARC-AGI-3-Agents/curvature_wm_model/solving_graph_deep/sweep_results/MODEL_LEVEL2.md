# Per-game model + level-2 online rollout (autonomous, 2026-06-25)

Adds a model-based layer on top of the online rollout, per the spec: per-game world model, predict st+1 from
(st,at) + at from (st,st+1), level-1 as the prior for level 2, fine-tune online every 30 steps while playing
level 2, only ONE layer per module tuned (M2 prior intact), no augmentation/noise.

## Base snapshot
`curvature_wm/checkpoints/base_snapshot/` holds the untouched base M2 models (m2_hud_2500, m2_aug_2500,
m2_raw100k_2500) so per-game retraining never loses them.

## Phase A — per-game models  (curvature_wm/model/train_per_game.py)
Warm-start from base M2 (m2_hud_2500), FINE-TUNE on ONE game's LEVEL-1 human transitions (LEVEL_MAX=0):
forward (st,at->st+1) + inverse (at from st,st+1) + click, with the shared trunk EDITABLE but only the **last
Linear of the trunk and of each head** trainable (~8.9k of 109k params; rest frozen = M2 prior). Short run
(150 steps), no augmentation, no noise. Saves `checkpoints/pergame_<game>.pt` — one retained model per game.
  Run: `python -m curvature_wm.model.train_per_game <game> 150`   (LEVEL_MAX=-1 = all levels; BASE=<ckpt> override)
  ls20 sanity: 480 level-1 transitions -> fwd_mae 0.020->0.019, inverse acc 0.97->0.99.

## Phase B — level-2 model rollout  (curvature_wm/solving_graph/online_rollout_model.py)
`ModelRoller(Roller)`: loads `pergame_<game>.pt` (level-1 prior), PLAYS level 2 via the undo rollout (TARGET=2,
continues past level 1, path cleared at the boundary so undo is never drained back into level 1), and FINE-TUNES
the model ONLINE every TRAIN_EVERY (=30) committed steps on the real (st,at,st+1) transitions it collects —
same one-layer-per-module light scheme. HUD masked everywhere (strip_hud on observe, drop_hud in the online-train
featurize). Click-on-node fully handled (inherited: enumeration, no-op-by-node/orbit, HUMAN_CLICKS teacher prior,
click_target). Saves the online-updated model to `pergame_<game>_lvl2.pt` + an HTML.
  Run: `WALL_SECS=160 TRAIN_EVERY=30 HUMAN_CLICKS=1 BRUTE=0 INT_RECOMPUTE=25 INTENSITY_MAX=99
        python -m curvature_wm.solving_graph.online_rollout_model <game> 50000 2`
  ls20 level-2 demo: loaded the prior, reached level 1 (committed 17), then explored level 2 while the model
  learned online (8 updates; forward MAE 0.016->0.014 on L1, loss 0.41->0.29 adapting to L2). Level 2 itself not
  solved within the window (big exploration, same wall as the L1-hard games).

## Status / next
- Per-game models being created for all 25 games (Phase A sweep) -> pergame_<game>.pt.
- Phase B mechanism verified working (per-game prior + online-every-30 + HUD + clicks + conservative undo).
- To actually SOLVE level 2: use the online-learned model to GUIDE action ordering (a forward/policy prior in
  rank_actions — the Roller already accepts an injected `policy`), not just learn in the background; that's the
  natural next step. Also synced to navigating_hodge (online_rollout_undo.py + online_rollout_model.py).
