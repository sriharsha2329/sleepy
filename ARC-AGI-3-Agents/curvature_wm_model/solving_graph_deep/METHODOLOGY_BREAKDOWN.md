# Focused Rollout — Methodology Breakdown (OLD vs CURRENT) — for a fresh methodology review

## 0. Context & goal
- Game: ARC-AGI-3 offline puzzle **ls20** (a maze/block puzzle, ~7 levels). The agent plays an offline env
  (`FastEnv`, snapshot/restore + `step(action)`), which returns a real per-object **perception graph** each step.
- A brain-plausible **world model** exists: a frozen **shared trunk** (graph-transformer encoder, `pooled(s)` =
  state embedding) + heads (forward / inverse / click). It is used here only as **priors**, never as ground truth.
- **Goal:** a *focused* rollout that **reaches the level-up (solves a level) without the state-space blow-up**.
  (A latent-hash rollout collapsed/merged states and exploded; switching to an **input-graph hash** gives ~140
  distinct, correct states for ls20 level 1 — see §1.)
- **Hard stance from the user:** *do NOT trust Q/RL to DRIVE the policy.* Priors only **rank**; the **structure**
  (DFS + backtrack + loop-filtering) explores. "Exploitation never works." Reward is computed from the **real
  env** (model-free transitions), not predicted.

## 1. Shared substrate (identical in OLD and CURRENT)
- **State identity = INPUT-GRAPH hash.** FNV-1a over, per non-HUD node, sorted:
  `(type_hash, color, stab, area_bin, centroid cy, cx, Mahalanobis sx, sy)`. HUD removed via the validated
  `hud_regions`/`drop_hud`. (The model's latent pooled hash collapses distinct states → NOT used for identity.)
- **Transitions are REAL.** For an action: `restore(snapshot) → step(a) → read the real next perception graph`.
  The model never predicts next-state or reward.
- **Reward per transition `s --a--> s'` (β=10):**
  `r = dA + 10·A∧A − 10·Δintensity − 10·intensity(s')`
  - `dA = smae_pair(s, s')` = semantic + Mahalanobis-position change over PLAYFIELD nodes (HUD excluded).
  - `A∧A` = no-return holonomy. **Model-straight 0** unless the transition is an EVENT: `dA > μ+σ`
    (running mean+std of dA over collected transitions) → then **brute-forced** = min `smae_pair(s, return(s'))`
    over all return actions (0 = returns exactly, large = no return).
  - `intensity(s)` = loop membership: `#reversible(green) faces containing s ×1  +  0.5 if s is on any charged
    (one-way) loop`. Recomputed every 10 steps (see triangle-fill).
- **Triangle-fill every 10 steps:** on the discovered edges so far, `minimum_cycle_basis` → minimal faces;
  classify each face **green** (every edge bidirectional/reversible) or **charged** (contains a one-way edge =
  a directed cycle with net circulation). Update `intensity` → feeds the reward of the next batch.
- **Heads trained per 10-step batch** on the **frozen trunk** (snapshot of base weights kept; never reset).
  Heads are PRIORS only.

## 2. OLD methodology — REACHED level 1 in **29 steps / 30 states** ✓
- **Action choice:** evaluate ALL available actions (step each in env, get real reward); take the **greedy
  argmax of REAL reward over FORWARD actions** (actions whose target is not already visited). A level-up is
  always taken.
- **Undo (the key part):** on a dead-end (no forward action), **undo exactly ONE step to the immediate parent**
  (`path.pop(); cur = path[-1]`) and try the parent's next action. True depth-first backtrack — **commits depth**.
- Q head was blended into the score at one point → it **HURT** (wandered) → removed; reward-only greedy solved it.
- **Outcome:** deep, committed, punched straight through to the level-up in 29 steps.

## 3. CURRENT methodology — CHURNS, never reaches (`maxlvl=0`) ✗
- **Action choice:** **policy-driven SAMPLING (no exploitation).** `π(s,·)` from a new **policy head**; sample
  ONE action from softmax over the **untried** actions. (First, untrained pass ≈ uniform.) Only the sampled
  action is stepped.
- **Undo:** on a dead-end, **jump to the GLOBALLY highest-reward frontier state** (any state with an untried
  action, ranked by its best transition reward) — undo along the path to it, or **teleport** to its snapshot.
- **PPO per 10-step batch:** MC return `G` with **γ=0.95** over the traversed segment (real rewards);
  **advantage = Q(s,a) − V(s)** where `Q(s,a)` = the action's **real** return and `V(s)` = mean over the state's
  available actions of `[real return where the action was taken this batch, else model Q estimate]`. Update the
  **Q head + policy head** (trunk frozen, base snapshotted). `intensity` + `σ` update each batch.
- **Outcome (4 episodes, ls20):** `maxlvl=0` every episode. It explores the ~36-state **start basin** + its
  ~12 green loops in the first ~60 steps, then **`undo` explodes (35 → 83 → 90 → 94)** — the last ~140 steps are
  almost pure backtracking through the already-explored core. **Never breaks out of the basin to the level.**

## 4. Side-by-side
| aspect | OLD (reached, 29 steps) | CURRENT (churns, maxlvl=0) |
|---|---|---|
| action pick | greedy **argmax real reward** over forward | **sample** from policy probs (no exploitation) |
| actions evaluated/step | ALL (step each) | only the sampled one |
| **undo / backtrack** | **one step to parent** (depth commit) | **jump to global best-reward frontier** (teleport) |
| Q/policy | none driving (Q hurt, removed) | policy samples; Q+policy trained as priors/value |
| depth behavior | commits a deep path → escapes basin | stays local → churns basin |
| reaches level | **YES (29)** | **NO** |

## 5. Failure analysis (current)
1. **Global-jump undo = the churn.** After the core is explored, every dead-end teleports to whatever core node
   has the best single reward, explores its last action (→ loop → dead-end), teleports again → `undo=90+`.
2. **No depth commitment** → never rolls a long path OUT of the start basin toward a distant level.
3. **Sparse reward trap:** the big reward is the level-up, which is never reached → the policy gets **no signal
   for which direction the level is** → can't learn to escape. (`dA` is the only dense "progress" signal.)
4. It "rolls out useless states" = the exhausted reversible core it keeps backtracking through.

## 6. Constraints / preferences the user has stated (must hold in any new methodology)
- **No exploitation** — sample actions from the policy distribution; priors only rank; structure explores.
- **State id = input-graph hash** (not latent).
- **Reward** exactly `dA + 10·A∧A − 10·Δintensity − 10·intensity(s')`.
- **A∧A** model-straight 0 unless `dA > μ+σ` → brute-force the real return.
- **Triangle/loop avoidance is LEARNED by the model** (reward penalizes intensity) — **NOT** a hardcoded penalty
  or fixed 20% bracket.
- **MC return γ=0.95** over the ~10-node traversed segment ("root up to 10 nodes, sum of MC").
- **Advantage = Q(s,a) − V(s) from REAL values where traversed; MODEL (Q head) only for not-taken actions.**
  V(s) gets more accurate as more of s's actions are tried and as triangles reveal more edges.
- **Update Q + policy(action) heads ONLY; freeze the shared trunk; snapshot base; never randomize/reset weights.**
- **Do NOT reset episodes** — accumulate discovered states across episodes ("same rollout, just filter more").
- **STOP when the level is reached.**
- **Undo is not an action** (bookkeeping to backtrack).
- **Visualization:** show the WHOLE run (the wandering/dead-ends, the "worst part"), and mark **explored vs
  not-explored** (frontier vs exhausted), as a develops-over-time HTML.

## 7. Open questions for the better methodology
- **Backtrack:** one-step-parent (depth, escapes basin — OLD) vs best-frontier (breadth, churns — CURRENT)?
  The user just said the OLD one-step-parent undo was "proper." Likely revert to it.
- How to **escape the start basin** to a distant level under a **sparse level-up reward + no-exploitation
  sampling**? (Use the dense `dA` progress gradient to commit depth?)
- How to **stop rolling out useless / exhausted / looped states** (prune them out of the sampling)?
- How should the **policy sampling** and the **DFS depth-commitment** coexist (sampling explores, but depth is
  needed to reach far)?
- Is per-batch PPO the right cadence, or does the sparse signal need something else (e.g., dA-shaped dense
  reward, frontier prioritized by progress, count-based novelty)?

## 8. Files (all in `curvature_wm/solving_graph/`)
- `metrics_ls20.py` — dA/A∧A/intensity metrics + the static input-hash graph (green/charged loops).
- `focused_rollout.py` — `Roll` (model-free rollout) + `emit_develop()` (the animated HTML) + `HTML_DEV` template.
- `focused_ppo.py` — `PPORoll` (the CURRENT policy-sampling PPO; QHead + PolicyHead; `run()` + `_ppo_update()`).
- `interactive_ls20.html`, `develop_ls20*.html` — the graphs. Checkpoints: `../checkpoints/focused_ppo_*.pt`.
