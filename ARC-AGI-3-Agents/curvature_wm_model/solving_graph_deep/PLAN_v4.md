# PLAN — online_rollout_v4.py (LANDMARK memory: kill the n² replay)

A FILE fork of online_rollout_v3.py but **PPO OFF** (we drop v3's online policy-gradient — net-negative + itself n²).
Keeps everything that works: the model-guided triangle chooser, intensity-gated loops (v2), the new reward, and the
**accurate perception/foot-key CLICK fix** (node-primary clicks via `forward_lookahead` + `state_dist_rows`, in the
shared `policy_prior.py`). v1/v2/v3 stay as checkpoints.

## Why (confirmed with user)
`acount ≈ n + n²`. The **n² term is REPLAY**: `env.restore` can't deepcopy the live env, so a cross-branch / failed
INV-undo does `reset + replay the AS-PLAYED path` (O(path)) every time → O(n²). The n term is the forwards. So the
lever is: **don't re-walk the whole history — anchor to a few important landmarks** (human "regions of interest").

## Landmark = region of interest (user's seed)
On each COMMIT, mark the destination state `h` a LANDMARK iff:
    (dA + A∧A) > μ + 2σ      (over the running dA+A∧A samples)   AND   intensity(h) < 1
i.e. a big-curvature / irreversible step that lands in a NON-loopy (low-intensity) area. Always include `h0` (root).
Keep `self.landmarks` as an ordered, capped set (K=5; evict the oldest/weakest). These are the only states we
"remember as important" — restores and backtracks anchor to them, not to all n states.

## The n² fix — landmark / shortest-path restore
`_restore(target_h)` replaces the raw `env.restore(target_snap)` at the expensive sites (backtrack, cross-branch):
1. If the env can cheaply INV-undo to target (target is a prefix of the current path) → do that (unchanged, cheap).
2. Else, build the **SHORTEST graph path** to target via the OnlineGraph (`nx.shortest_path`), preferring to start
   from the **nearest landmark that is a prefix of the current path** (reach it by cheap INV-undo), then replay only
   the short remaining segment landmark→target. Fallback: reset + replay the *de-looped* shortest path from h0
   (still shorter than the looping as-played path). → replay cost O(divergence / de-looped segment), not O(path).

## Landmark-guided backtrack (user-clarified)
When a node's actions are useless/exhausted, go to the **CLOSEST ANCESTOR that is a landmark** with untried actions,
via the **shortest path** to it (de-looped restore), and re-explore there. **If there is no such landmark, fall back
to the closest ancestor PARENT** with untried actions (the prior behavior). Replaces parent-by-parent crawling with
one jump and focuses re-exploration on ROIs.

## Implementation (edits to the v4 copy)
- Constants: default `PPO="0"` (drop it); add `LM_K=5`, `LM_SIGMA=2.0`, `LM_INTENSITY=1.0`.
- Track `self.daa_vals` (dA+A∧A per committed edge) for μ,σ; `self.landmarks` (ordered dict hash→snap).
- `_commit` (or `_step_check` accept): after recompute-intensity, call `_update_landmarks(h2, dA, aa, intensity)`.
- `_restore(target_h)`: the landmark/shortest-path restore above; route backtrack + cross-branch restores through it.
- `_run_triangle` backtrack block: pick the nearest landmark-with-untried-actions instead of popping to parent.
- Instrument: print #landmarks, replay_steps, acount.

## Test (ls20 first; user pessimistic about bp35)
Smoke ls20: expect SAME solve (24 states) with **replay_steps and acount down** (the n² term shrinks). Report the
before/after. Then bp35 as a stretch.

## Decisions / open
- K=5, μ+2σ, intensity<1 are the seeds — tunable via env knobs.
- "even unexplored edges" in the shortest path: the OnlineGraph only has observed edges; we use those (real,
  replayable). Truly-unexplored edges would need a model-predicted transition (later).

See [[triangle-rollout]], PLAN_v3.md (PPO, parked), PLAN_triangle.md.
