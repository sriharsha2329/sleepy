# PLAN — online_rollout_v3.py (online MC policy-gradient on the policy head)

A FILE fork of online_rollout_triangle2.py (v2 = intensity-gated loops). v1/v2 kept as known-good checkpoints. v3
adds an ONLINE policy-gradient that fine-tunes ONLY the policy head, so the prior improves toward the discovered
SHORTEST path. MC returns (γ=0.95), NOT Q maximization.

## Per the user (locked intent)
- Snapshot the ORIGINAL model at start; FREEZE the trunk + forward/inverse/click heads; train ONLY the policy-head
  weights. The fine-tuned head becomes the new PRIOR for subsequent selection.
- **For EACH real action we take: compute (a) the SHORTEST path start(h0) → the current state, and (b) that action's
  REWARD.** (keep both per real action.)
- Reward `r = |dA| + 10·|A∧A| + 10·(Δintensity last→next)`.  [intensity-term sign → CONFIRM]
- MC returns along the shortest path, γ=0.95; REINFORCE-update the policy head.
- Off-path transitions get reward 0 — INCLUDING predicted-latent candidates the model maps back to a visited state.
- Do the update each step, once enough has been mapped (real + latent). Prior from MC, NOT Q maximization.

## Method
1. **Snapshot + freeze** at run start: keep the original trunk/heads state; `requires_grad=False` on trunk +
   forward/inverse/click; `optimizer = Adam(policy_head.parameters())`. (feasibility-checked first.)
2. **Per real action (in `_commit`)**: store `r = |dA| + 10|A∧A| + 10·(i_cur − i_nxt)` for that committed edge;
   recompute `SP = shortest_path(h0 → cur)` over the REAL edge graph (BFS / nx).
3. **MC update (each real action)**: for transitions ON SP compute `G_t = Σ_k γ^k r_{t+k}` (γ=0.95); off-path
   transitions (incl. latent loop-backs) contribute reward 0. ADVANTAGE `A_t = G_t − V(s_t)`, where V(s) is a value
   head on the SHARED FROZEN trunk (user: "Q(s,a) and V(s) can use the shared trunk of the snapshot"; Q(s,a) is the
   MC sample = G). Policy loss covers BOTH action AND click-node (user: "not only action but also click on node"):
   `L_π = −Σ_SP [log π(a_t|s_t) + 1[a_t=click]·log π(node_t|s_t)]·A_t`. Value loss `L_V = Σ_SP (V(s_t) − G_t)²`.
   `optimizer.step()` on POLICY head + VALUE head only (trunk + fwd/inv/click frozen; H detached).
4. The head forward must allow grad: run trunk under `no_grad`, **detach H**, run `policy_head(H)` WITH grad → logits
   → log π. (current `_forward` is all-no_grad; v3 adds a grad path.)
5. Selection (`_prior` / `rank_actions` / `forward_lookahead`) uses the live (updated) head each step.

## Reuse / New
- Reuse: v2 triangle loop + `_loop_ok`, `OnlineGraph` (real edges), intensity, `_model_return`, `forward_lookahead`,
  the `_commit` hook (`_after_commit`), env restore/INV-undo.
- New: model snapshot+freeze; policy-head optimizer; shortest-path (BFS h0→cur, nx already imported); per-action
  reward store; MC-return + REINFORCE update; `PolicyPrior` training methods (logits-with-grad + `update(traj)`).

## Decisions (locked with user)
1. Reward intensity sign: using `+10·(i_last − i_next)` (reward intensity DROP). [one-line flip if wrong]
2. Baseline = V(s) on the SHARED FROZEN trunk; advantage A=G−V; Q(s,a)=the MC return (G). ✓
   NOTE: the old ValueHead/ActionValueHead were REMOVED (policy_head.py:40) and are NOT in the archive — so v3 ADDS
   a minimal `ValueHead` (pooled trunk H → scalar, mirrors PolicyPriorHead), trained ONLINE via MC regression V→G.
3. Policy gradient on BOTH action and click-node log-probs. ✓
4. Update every real action ("each step"). ✓

## Feasibility — PASS (probe ran)
Fine-tuning ONLY the policy head online works (trunk frozen). Recipe: `with no_grad: H,_=model.encode(Z,M,E)`;
`H_det=H.detach()`; `a_logits,c_logits=head(H_det,pos,Mt)` (grad on head); `opt=Adam(params)`; `loss=−logπ·A`;
`opt.step()`. Verified: head params change, trunk+fwd/inv/click bit-identical, head grads nonzero / trunk grads None.
Snapshot = `copy.deepcopy(head.state_dict())`. The SAME mechanism adds the V head on the shared trunk. The click
head trains ONLY if the loss touches c_logits → the click term is included (Method 3).

## Smoke
- ls20 + bp35: prior improves across the run (states/actions trend DOWN vs v2); assert policy+value heads changed,
  trunk frozen; γ=0.95 MC returns correct on a tiny synthetic path.

See [[triangle-rollout]] (v2), PLAN_triangle.md.
