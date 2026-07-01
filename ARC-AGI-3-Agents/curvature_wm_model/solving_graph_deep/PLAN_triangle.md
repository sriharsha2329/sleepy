# PLAN — online_rollout_triangle.py (model-guided triangle rollout)

Duplicate of `online_rollout_undo.py`. Goal: replace the per-state **probe-everything escape-DFS** with a
**model-guided triangle** chooser that uses the world-model to pick the *least reversible* (most progress)
forward action, verifies reversibility in reality, avoids trivial undos and loops, and backtracks cheaply —
**without a huge rollout**.

## What the model already gives us (verified)
- **Forward** `WorldModel.predict_next(Zp,Mp,EFp,a,h_click) -> (z_next[B,N,od], alive[B,N])` — predicts next-state
  node **latents** (+ per-node alive).
- **Inverse** `predict_action(Zp,Mp,EFp,Zc,Mc,EFc) -> logits[B,n_actions]` → softmax = full action distribution
  (entropy available). Trained on **1-step** transitions.
- **Policy** `PolicyPrior.state_dist(g) -> (a_p[n_actions], c_p[N])` and `propose_scored()` — argmax / 2nd-argmax +
  click-node probs.
- **Return ranking** `PolicyPrior.rank_returns(g_from,g_to,cands)` — inverse heads rank which action maps g_from→g_to;
  used by `_model_return` which **env-verifies top-k** (this is the model-A∧A already in the file).

## THE GAP we are missing (most important)
The forward head predicts next-state **node latents only — not the next graph's structure** (edges `EF`, mask `M`,
node features). But the inverse/return heads need **featurized graphs (Z,M,EF) of BOTH states**. So the clean chain
"predict st+2 latent → run inverse(st+2→st)" is **not directly supported**. Two ways out:
- **(B-approx)** reuse st+1's `M,EF` for the predicted st+2 and feed `z_next` as its `Z` (assume structure ≈ unchanged
  for one step) → run inverse/return in latent space. Cheap, but an approximation; also needs `od == trunk-input dim`
  (must verify once).
- **(A-real)** don't predict st+2 at all for the *candidates*: take a **real 1-step env lookahead** for the top-k
  actions (the existing `probe()` already returns the real st+2 + dA + A∧A), then score reversibility from that.
  Exact, but costs k real steps per state (what we wanted to avoid).

Second gap: **the inverse is 1-step; "st+2 → st" is a 2-step query (OOD).** Use **1-step reversibility** (st+2 → st+1)
for the model score. For the **committed** action we still env-verify the true close (st+2 → st) via `rank_returns`
when we want certainty — but that is the ONLY place that costs env steps.

**DROP the `dA > μ+σ` event gate (your call — model reversibility is basically free).** In the undo solver A∧A was
only computed when `dA > μ+σ` purely to avoid the expensive *brute* replay. With the model the reversibility score is
one forward pass, so we compute it for **EVERY candidate, always** — no dA gate, no "event" notion driving it. The
env-verify (real replay) is reserved for the single committed action, not the candidates.

Third note (memory `r-pred-flat-forward-model`): the forward model is **mean-regressing** — it smooths the rare
events (level-ups, irreversible jumps) we care about. So we must **never trust predicted st+2 for reward**; only use
it to *rank which action to really try*, then **verify in reality**. This is consistent with the method (we do take
the chosen action for real and check reversibility).

## The metric you invited me to define — "least control / least reversible"
For a candidate next state s' (real or predicted) reached by action a from s:
```
irreversibility(s', s) =
    w1 * H(inverse_action_dist(s' -> s_prev))      # entropy of the 1-step reverse dist (MODEL, free): HIGH = no clear undo
  + w2 * model_return_score(s' -> s_prev)          # rank_returns top-1 reverse prob (MODEL, free): LOW = no-return
  - w3 * trivial_undo_penalty(a)                    # a == INV[last_action] and model CAN reverse -> push down
```
Both terms are **model-only (free), computed for every candidate, with no dA gate.** Real env-verify of reversibility
is done **once**, for the committed action (or skipped — "we can use the model" for that too, per you).
**HIGH irreversibility = we have "less control" to get back = real progress → PICK IT.** (Your phrase "more uniform
prediction → less control" = high reverse-entropy; the "less uncertain" wording is the one inconsistency, resolved
this way.) `w*` tunable; isolated in one function `_irreversibility(...)` so the criterion is a one-line flip.

## Per-state algorithm (the triangle loop)   — LOCKED with user
**No `probe()`, no real action for candidates.** Only the two *committed* forwards are real steps.
1. **Game start:** `detect_hud(env)` (+ `hud_final.py` to eyeball) and `detect_undo(env)` → available actions + INV
   table. (Already done in run_solver.)
2. **Forward 1 (real):** policy **argmax** action → REAL step → real st+1. Record the real 1-step reversibility
   **st ↔ st+1** (model inverse on the real pair — free).
3. **Predict st+2 candidates (MODEL latents, NO env steps):** from st+1 take the **top-5 actions (moves + clicks)**
   and **predict each st+2 LATENT** with `predict_next` (structure-approx: predicted latents + st+1's edges/mask).
   Skip candidates with (predicted) dst intensity `>= 2` (fall to the **2nd, then 3rd** ranked action), trivial undos
   (`a == INV[at]` unless the model says it is irreversible), and cache hits (no-op/loop; click no-op prior → 1e-3 &
   renormalise).
4. **Reversibility st+2 → st (MODEL, latents, free, EVERY candidate):** ask "can the model reliably predict an action
   that maps st+2 back to **st**?" via inverse/return heads. `irreversibility = w1·entropy(reverse_dist) −
   w2·return_prob`. (No dA gate — this is the whole point: free, so always computed.)
5. **Pick:** the st+2 that is **LEAST reliably reversible** (model can't confidently get back to st) = most progress.
6. **Commit (real):** take that one action for REAL → real st+2; record real 1-step reversibility **st+1 ↔ real
   st+2** (+ confirm it isn't a plain undo); `_recompute_intensity()` (every action is valuable).
7. **Sliding-window loop check (cheap, replaces big rollout):** ask the **model** "can st+4 reverse to st? st+5 to
   st+1?" If yes (confident reverse) → it's a loop → cache + backtrack.
8. **Budget + backtrack:** at a node try argmax → **2nd → 3rd → …** up to **5 actions**; if exhausted, restore to the
   **nearest parent with intensity < 2**, tie-break **highest dA + A∧A** (σ-bucket teacher already does dA+A∧A).

## Reuse map (from online_rollout_undo.py)
| Need | Reuse |
|---|---|
| real probe (do→observe→measure→undo) | `probe()` (515) |
| model-A∧A reversibility verify | `_model_return()` (347) / `rank_returns` |
| intensity + recompute-per-commit | `_recompute_intensity()` (431), `_commit` (683) |
| no-op move & loop-back cache | `_try_action` (391-429), `is_noop` (547) |
| click no-op by node+orbit | `bad_actions` + `_prior` (469) lower to 1e-3 |
| env restore / action-undo | `OnlineEnv.restore` (the if/elif INV undo) |
| backtrack by intensity + dA+A∧A | σ-bucket sort (630-635), INTENSITY undo (587) |
| HUD / undo detect at start | `detect_hud`,`detect_undo` in run_solver.py |

NEW: `_irreversibility()`, `_is_trivial_undo()`, `_window_reversible()` (model loop-check), and (Phase B)
`_forward_lookahead()` (predict_next + latent inverse) — likely a small method added to `PolicyPrior`.

## Phasing  — forward-lookahead is THE method (real-probe Phase A dropped per user)
0. **COMPAT PROBE (prerequisite, do FIRST):** load the model; from a real st+1, `predict_next` the top-5 st+2
   latents, then run inverse/return on (st+2-latent, st) with the structure-approx — confirm dims line up, it runs,
   and entropy / return-prob actually VARY across candidates (else the signal is useless). Gates everything.
1. **Implement `_forward_lookahead`** (likely on PolicyPrior): (graph st+1, action) → predicted st+2 latent;
   (st+2-latent, st_graph) → reverse action dist (entropy) + return-prob. 0 real steps.
2. **Rewrite `run()` selection** to the LOCKED loop (argmax real → predict top-5 st+2 → irreversibility → pick
   least-reversible → real-step winner → 2nd/3rd argmax → 5-budget → nearest intensity<2 parent), reusing
   intensity / caches / restore / INV-undo from the undo solver.
3. **Smoke** ls20 + bp35 (states / REAL-actions / solved); log gate, anti-undo, loop-cache counts.

## Smoke tests
- py_compile; import.
- COMPAT PROBE passes (latent inverse runs + discriminates).
- ls20 + bp35 solve; report REAL action count (should be ~path length — no per-candidate steps), states, gate/anti-undo/loop counts.

## Decisions (LOCKED with user)
1. Metric: `w1·entropy(reverse) − w2·return_prob`, model-only, no dA gate. ✓
2. Bridge: predict top-5 st+2 **latents** (NO real probe), model checks st+2→st reversibility on latents
   (structure-approx = predicted latents + st+1 edges/mask). Real steps ONLY for committed forwards. ✓
3. Node procedure: argmax → 2nd → 3rd → … (≤5 actions) → nearest intensity<2 parent (dA+A∧A tie-break);
   gate out intensity≥2 candidates. ✓
Remaining unknown → the COMPAT PROBE result (does the latent inverse chain run / discriminate). Gates implementation.
