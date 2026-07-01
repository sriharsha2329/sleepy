# Hodge-Flow Rollout Framework

**gradient = `dA`  ·  curl = `A∧A` (triangle method)  ·  harmonic = where `dA` and the triangle method both fail**

**Status:** detailed plan only — no code, no training, no runs. Spec for approval.
**Folder:** `curvature_wm/hodge_flow/` (separate from `curvature_wm/rung3/`).
**Scope guardrail:** this layer only **recognizes** the agent's experience as a three-flow Hodge decomposition of the curvature, and from that **prescribes how to roll out**. It is **NOT** Rung-3 causal — no counterfactual `do(a)`, no ITE/advantage, no propensity. Rung-3 is a *separate later layer* that may consume these flows (§9). Built on the **position-aware curvature** (Mahalanobis), the settled foundation (§1, §5).

---

## 0. One-paragraph thesis

Every transition the agent makes carries a **curvature** `r = dA + A∧A`. Decompose the agent's *flow* over its state graph into three orthogonal parts. The **gradient** part is exactly `dA = z_next − z_current` — net forward transport, *measured directly*. The **curl** part is the reversible circulation, identified by the **triangle method gated by `A∧A`** — small loops you can walk and undo. The **harmonic** part is **the residual that neither `dA` (no net transport) nor the triangle method (not fillable / not reversible) can explain** — the genuine *no-return* obstruction. These three answer the only three questions a rollout needs: **where is progress (follow gradient), what is wasted looping (avoid curl), and what is an irreversible milestone (seek harmonic).**

---

## 1. Inputs we already have (no re-derivation here)

From the position-aware curvature already built in `rung3/diagnose_lr.py` (reused **read-only** by this folder):

| symbol | meaning | how obtained | status |
|---|---|---|---|
| `z` | position-aware state latent: semantic 62-d + **Mahalanobis centroid** (player position) | `node_latents` + Maha displacement; HUD row (cy ≥ 0.8) excluded | done |
| `s → s'` under `a` | observed transition | offline env (`online_v5.Live`), snapshot/fork via `envfast.FastEnv` | done |
| `dA(s,s') = z' − z` | **signed** forward change | `signed_gradient()` | done |
| `A∧A(s,s') = min_a (f(z',a) − z)` | **signed** best-return residual (`a*`=argmin‖·‖) | `signed_curl()` | done |
| `r = dA + A∧A` | curvature (magnitude) | `smae_pair` (+ Maha) | done |

> **Why position-aware is load-bearing (§5).** Without the Mahalanobis term, a moving player reads `dA ≈ 0` (the one-hot/graph channels can't see a ~6 px move). The flow decomposition would then be blind to navigation. Every flow below is computed on the position-aware curvature.

---

## 2. The three flows — the recognition

### 2.1 Gradient flow ≡ `dA` (net transport / progress) — MEASURED
- **Per edge:** `x_grad(e) = z' − z = dA` (signed). We do **not** least-squares solve for it — it is read directly (user: "the gradient is z_nextframe − z_currentframe").
- **Globally:** `x_grad = B₁ᵀ φ`, where `φ` is the **progress potential** (a scalar per state, e.g. distance-to-goal). The accumulated `dA` along a path *is* the change in `φ`; so `φ` is recovered from the measured `dA`, not invented.
- **Meaning:** real forward displacement — navigation steps and milestone jumps. Nonzero divergence `B₁ x_grad ≠ 0` = net transport.
- **Rollout role:** **FOLLOW** — move along `dA` toward higher progress (toward the goal).

### 2.2 Curl flow ≡ `A∧A` via the triangle method (reversible circulation) — TRIANGLE-FILLED
- **The triangle method:** for each triangle (3-cycle) in the state graph, **fill it** (add as a 2-cell of `B₂`) **iff all three of its edges are reversible**, i.e. `A∧A(edge) < τ_rev` (a return action exists). Curl = projection of the flow onto the filled cells (`B₂ψ`).
- **`A∧A` is the gate:** small `A∧A` ⇒ the loop returns ⇒ fill ⇒ curl. Large/UNKNOWN `A∧A` ⇒ leave unfilled (it falls into harmonic).
- **Boundary additivity (larger loops resolve into triangles):** a k-loop need not be its own cell. If its **chords exist as explored edges** it triangulates, and `curl(k-loop) = Σ curl(triangles)` because the shared chord cancels (opposite orientation in the two triangles):
  ```
  (s1→s2→s3→s1) + (s1→s3→s4→s1) = s1→s2→s3→s4→s1      chord s1↔s3 cancels
  ⇒ curl(4-loop) = curl(triA) + curl(triB)
  ```
  So the **triangle method already resolves any *triangulable* loop** — measure the small triangles to resolve the big loop.
- **Meaning:** reversible loops = "go around and come back" = **wasted circulation**.
- **Rollout role:** **AVOID** — these are no-progress loops; penalize so they are not repeated.

### 2.3 Harmonic ≡ where `dA` AND the triangle method both FAIL (no-return) — INFERRED
- **Definition:** `x_harm = x − x_grad − x_curl` (the residual).
- **It is exactly the double failure:** (a) `dA`/gradient fails — it is *not* net transport (a closed loop has zero gradient circulation); **and** (b) the triangle method fails — the loop is **not triangulable** (its chords don't exist, or a constituent edge is a no-return so its triangles can't be filled). What survives both is the harmonic.
- **Meaning:** the genuine **no-return** obstruction — "we don't know how to reach the last state from the initial (or get back)." Irreversible events / milestones. `β₁ = #` such holes.
- **Rollout role:** **SEEK** — these are the goals/sub-goals that advance the game irreversibly.

### 2.4 The decision table (the whole framework in one grid)

| flow | computed by | divergence | reversible? | what it is | ROLLOUT |
|---|---|---|---|---|---|
| **gradient** | `dA = z'−z` (measured) | ≠ 0 (net transport) | one-way progress | real forward step / milestone | **FOLLOW** toward goal |
| **curl** | triangle fill gated by `A∧A` | 0 (circulation) | **yes** (returns) | reversible wasted loop | **AVOID** (penalize, don't repeat) |
| **harmonic** | residual `x − grad − curl` | 0 (circulation) | **no** (no-return) | irreversible event / hole | **SEEK** (goal / sub-goal) |

---

## 3. The math (Helmholtz–Hodge on the state graph)

**Complex.** `V` = visited states deduped by position-aware curvature (`r < TOL`). `E` = observed transitions, oriented; signed flow `x(e)` (antisymmetric: `+w` if traversed `u→v` with `u<v`, else `−w`; net of both directions). `F` = the triangles filled by the `A∧A` gate (§2.2).

**Boundary operators.**
```
B₁ ∈ R^{|V|×|E|}   node–edge incidence (divergence);   B₁ x = net in-flow per node
B₂ ∈ R^{|E|×|F|}   edge–triangle incidence over filled (A∧A-reversible) triangles;   identity  B₁ B₂ = 0
L₁ = B₁ᵀB₁ + B₂B₂ᵀ                                            (1-Laplacian)
```

**Orthogonal decomposition.**
```
x = x_grad ⊕ x_curl ⊕ x_harm
x_grad = B₁ᵀφ           φ from the measured dA-progress potential        (NET TRANSPORT — dA)
x_curl = B₂ψ            ψ = (B₂ᵀB₂)⁺ B₂ᵀ x  over filled triangles         (REVERSIBLE — A∧A)
x_harm = x − x_grad − x_curl                                              (NO-RETURN — residual)
β₁     = |E| − rank(B₁) − rank(B₂) = # irreversible holes
```
Per-edge masses `g_e=|x_grad(e)|`, `c_e=|x_curl(e)|`, `h_e=|x_harm(e)|`; an edge's type = argmax. (Reuses `rung3/hodge.py` numerics: sparse `B₁/B₂`, `lsqr`; the `A∧A`-gated triangle fill already exists there.)

**Sign discipline (no absolute values until the very end).** `dA` and `A∧A` are kept **signed** per step (`signed_gradient`/`signed_curl`); the decomposition needs oriented flows. Magnitudes are taken only for the final per-edge masses and the energy split.

---

## 4. How to perform rollout (the core deliverable)

The rollout is a **wake → sleep → act** loop. The three flows map one-to-one onto the three things a rollout must do.

### 4.1 The loop
```
WAKE  (explore + measure):
  from the current state, branch real actions (snapshot/fork; no undo/stay);
  per step record the transition and compute  dA = z'−z (signed)  and  A∧A = min_a(f(z',a)−z) (signed);
  add nodes/edges to the accumulated state graph with the oriented flow x.

SLEEP (decompose + assign):
  fill A∧A-reversible triangles (triangle method, + chord-resolved larger loops) → B₂;
  Hodge-solve →  φ (progress potential / value),  curl edges (reversible loops),  harmonic edges (no-return);
  β₁ holes → the current set of no-return GOALS.

ACT  (shape the next rollout):
  π(a|s) ∝ exp( base(s,a)  + α·followGrad(s,a)  + γ·seekHarm(s,a)  − β·avoidCurl(s,a) ),   Σ_a π = 1
    followGrad : prefer the action that increases φ (descends distance-to-goal) — net progress
    seekHarm   : prefer the action that heads to the nearest harmonic no-return event (the goal)
    avoidCurl  : suppress actions on edges with high curl mass (reversible wasted loops)
```

### 4.2 Per-flow rollout rule
- **Gradient → FOLLOW.** Use `φ` as the value. Backward induction from each harmonic goal gives `dist_to_goal`; the rollout descends it (this is the reverse-curriculum: nearest-goal states fixed first).
- **Curl → AVOID.** Once the triangle method tags a reversible loop, mark its `(s,a)` and **penalize**. The penalty is **persistent across rollouts** (`P(s,a)` accumulates), so a loop found once is suppressed "next time and so on." Hard-tabu = `β → ∞`; soft = finite `β`.
- **Harmonic → SEEK.** Each harmonic edge / `β₁` hole is a **no-return milestone** = a sub-goal. The rollout aims for the nearest one; on reaching it, re-run WAKE from there → the next harmonic → **chain** the milestones into a full plan.

### 4.3 Chord exploration (resolve ambiguous loops)
When a loop is found but **not yet triangulable** (its chords/diagonals are unexplored), the rollout should **explore the chords**. Outcome decides the loop's nature:
- chords reversible ⇒ the loop triangulates to **curl** ⇒ avoid;
- a chord is a no-return (or chords genuinely absent) ⇒ the loop stays **harmonic** ⇒ it's a goal, seek.
This is the explore-vs-exploit knob, expressed in flow terms.

### 4.4 Persistence and convergence (informal)
- `avoidCurl` (`P`) only grows ⇒ revisited reversible loops vanish from the policy over rollouts.
- `φ` and the harmonic goal set sharpen as the graph fills ⇒ the path **gradient → harmonic** stabilizes.
- Target behavior: visit-count of any reversible loop → 0; the agent walks net-progress edges to successive no-return milestones.

---

## 5. Position-aware curvature (the foundation, restated)
- `dA` and `A∧A` are computed on `z` = semantic latent **+ Mahalanobis centroid displacement** (`‖Δpos·grid / clip(σ,1,20)‖`), HUD row excluded. Without it the gradient flow is blind to the player moving.
- This folder **reuses** `rung3/diagnose_lr.{signed_gradient, signed_curl, smae_pair}` and `rung3/hodge.py` read-only; it does **not** touch perception, `node_latents`, or M2.

---

## 6. Planned components (to build AFTER approval — none written yet)
```
curvature_wm/hodge_flow/
  PLAN.md        # this document
  flow.py        # accumulate state graph + oriented signed flow x from rollout transitions (reuse build_complex)
  gradient.py    # x_grad = dA (signed); progress potential φ (reverse-curriculum value from measured dA)
  curl.py        # triangle method: A∧A-gated triangle fill (B₂) + boundary-additivity for triangulable loops
  harmonic.py    # x_harm = x − grad − curl; β₁ holes; extract no-return goals/sub-goals
  rollout.py     # wake→sleep→act loop; policy shaping (follow grad / avoid curl / seek harmonic); persistent P
  tests/         # §7 smoke tests
```
Each is a *thin* orchestration over the existing `rung3` curvature + `hodge.py` primitives — the novelty is the **flow recognition + rollout rules**, not new numerics.

---

## 7. Validation (smoke tests — to RUN only on approval)
1. **Gradient = dA recovers motion:** on bp35 the gradient flow is nonzero exactly when the player moves (Maha), zero at the wall. (Already observed; formalize as a test.)
2. **Curl via triangle method:** a reversible 3-loop → curl-dominant; assert `‖B₁B₂‖<1e-6`.
3. **Boundary additivity:** a reversible 4-loop with its chord present → resolves to curl = sum of two triangle curls (matches direct projection).
4. **Harmonic = double failure:** a **non-triangulable** no-return loop (chord absent or a no-return edge) → harmonic-dominant; `β₁` counts the holes; a pure progress chain → harmonic ≈ 0.
5. **Rollout behavior:** on a toy graph with one reversible trap loop + one no-return goal, the shaped policy's loop-visit count → 0 and it reaches the goal; `P` persists across simulated rollouts.

---

## 8. Honest risks
1. **Triangulation depends on chord coverage.** A reversible loop whose chords are unexplored looks harmonic until §4.3 chord exploration runs — could mis-seek a "goal" that is actually a loop. Mitigation: explore chords before committing a hole to the goal set; report chord-coverage.
2. **`τ_rev` / `TOL` calibration.** On the position-aware scale (navigation `A∧A ≈ 0.67`, event `≈ 74`) the reversible/no-return gap is wide, but thresholds must be set from data, not assumed.
3. **Graph density.** Many states ⇒ many triangles; reuse `hodge.py`'s `MAX_EDGES`/`MAX_TRIS` loud guards.
4. **`φ` validity.** `φ` from measured `dA` is a true potential only if `dA` is near-conservative on the explored subgraph; the harmonic residual quantifies the deviation — report it.

---

## 9. Relationship to Rung-3 (kept strictly separate)
- **This layer = flow recognition + rollout** (descriptive + control): classify each transition as gradient/curl/harmonic and move accordingly. It needs **no counterfactuals**.
- **Rung-3 = causal counterfactual** (a *separate* `curvature_wm/rung3/` effort): `do(a')`, ITE/advantage, identification. It *could later* consume these flows (e.g. spend counterfactual budget only on the harmonic frontier), but **nothing causal is in this folder**.
- Boundary is deliberate: get the flow-driven rollout working and validated **first**; causal refinement is downstream and optional.
