# Online rollout — level-1 sweep, FINAL (autonomous run, 2026-06-25)

Solver: `online_rollout_undo.py` (DFS + restore-undo; reward-greedy dА+10·A∧A, no learned net; env has no
action-7 undo so "undo" = snapshot restore). Driver: `sweep.py` (subprocess/game + wall-clock timeout). Outputs in
solving_graph/. Working solvers backed up in archive/solvers_ls20_working/.

## RESULT: 14 / 25 games solve LEVEL 1
SOLVED (14): ar25, bp35, cd82, dc22, ft09, lf52, lp85, ls20, m0r0, r11l, sp80, su15, tu93, vc33
UNSOLVED (11): cn04, g50t, ka59, re86, s5i5, sb26, sc25, sk48, tn36, tr87, wa30
(per-solve HTML: solving_graph/online_undo_<game>_lvl1.html; logs: solving_graph/sweep_logs/<game>.log)

## How (3 passes, each adds knobs to online_rollout_undo; defaults preserve ls20)
- PASS 1  (BRUTE=0, INT_RECOMPUTE=25, CLICK_BUDGET0=6, INTENSITY_MAX=3)               -> 9
  ar25 bp35 cd82 dc22 lp85 ls20 m0r0 sp80 tu93
- PASS 2  (+ INTENSITY_MAX=99, GRID_CLICK=1: grid-click fallback)                     -> +3  (r11l, su15, ft09)
- PASS 3  (+ HUMAN_CLICKS=1: probe the click positions humans actually used)          -> +2  (vc33, lf52)

## Speed/coverage knobs (env vars on online_rollout_undo.py)
- BRUTE=0            : skip A∧A brute-force (level-up detected directly). ls20 result identical, big speedup.
- INT_RECOMPUTE=25   : recompute faces/intensity every N edges (was per-edge, O(E^2)). ls20 82s -> 31s.
- GRID_CLICK=1,GRID_N: probe a coarse click grid — recovers "click-to-start" games (env starts with only CLICK
  available + object-clicks no-op; start button isn't a perception object). Recovered ft09.
- HUMAN_CLICKS=1     : teacher prior — probe the most-used human click positions from graph_data. Recovered vc33
  (2 steps!) + lf52. The big lever for click games.
- INTENSITY_MAX      : undo back when intensity >= this (3 stay-on-rings; 99 never-undo-for-depth).
Re-run any games: `SOLVER=online_rollout_undo BRUTE=0 INT_RECOMPUTE=25 HUMAN_CLICKS=1 GRID_CLICK=1 INTENSITY_MAX=99
MAXSTEPS=15000 TIMEOUT=600 python -m curvature_wm.solving_graph.sweep <games...>`

## The 11 unsolved (honest)
Click usage in human runs (clicks/frames): g50t 0, re86 0, sk48 0, tr87 0, wa30 0 (MOVE-only);
ka59 19/442, cn04 67/411, sc25 139/305, sb26 230/257, s5i5 637/657, tn36 494/499 (CLICK).
- MOVE-only deep/big (g50t, re86, sk48, tr87, wa30): explore broadly / exhaust a sub-region with NO level-up probe
  within budget. The greedy doesn't find the deep goal path. NEXT: a learned policy/Q to GUIDE (not pure greedy),
  or goal-directed/complete search.
- CLICK games that human-clicks didn't crack (s5i5, tn36, ka59, sb26, cn04): need click SEQUENCES / deeper paths,
  not just the right click POSITION — the prior gets the positions but the level-up needs an ordered combo the
  greedy doesn't assemble. sc25 still 0-step even with human-clicks (its start ignores them — separate per-game
  env quirk). NEXT: order human clicks by the state they were used in (state-conditioned prior), or replay the
  human action sequence as a warm-start then explore from there.

Bottom line: the greedy online rollout + grid/human-click priors gets 14/25 level-1, fully reproducible. The
remaining 11 need a guided policy (the V/Q heads once they generalize) or state-conditioned click sequences.
