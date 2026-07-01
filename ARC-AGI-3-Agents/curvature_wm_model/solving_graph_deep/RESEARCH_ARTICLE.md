# A Gauge-Theoretic World Model for ARC-AGI-3: Group-Equivariant Perception, Yang–Mills Curvature Rewards, and Hodge Intensity for Online Graph Search

*Working draft — curvature_wm. References are real and listed at the end.*

## Abstract

We present a world model and search procedure for the ARC-AGI-3 interactive reasoning benchmark that is built from
three classical mathematical ideas, applied end-to-end to an agent that explores a deterministic puzzle
environment. (1) **Perception** is object-centric and *group-equivariant*: each frame is reduced to a graph of
objects whose identity is a symmetry-invariant signature, and whose temporal stability is established with a
*Weisfeiler–Lehman (WL)* relabelling run across time rather than within a single frame. (2) The **reward / local
geometry** that scores a transition is taken directly from the structure of a *Yang–Mills* gauge field: we treat
the action-induced change of state as a connection `A` and score a step by the field strength `F = dA + A∧A`, where
`dA` is the (semantic + positional) first-order change and `A∧A` is a no-return *holonomy* term measuring
irreversibility. (3) The accumulated transitions are *stitched* into a 1-complex (a graph), and a **Hodge /
combinatorial-Laplacian** view of that complex gives each state a scalar **intensity** — its membership in
reversible cycles ("how loop-heavy is this region") — which the agent uses to stay on the rim of attractor basins
and escape toward goals. We describe how these three pieces compose into an online, reward-guided graph search that
solves the first level of 14/25 ARC-AGI-3 games from a deterministic offline environment, and a per-game world
model (forward `s_t,a_t→s_{t+1}` and inverse `a_t←s_t,s_{t+1}`) that is lightly fine-tuned online while playing.

---

## 1. Introduction

ARC-AGI-3 frames are small integer grids that evolve under a discrete action set (movement, a generic CLICK on an
object, and an UNDO/reset). Levels are completed by reaching a goal configuration; the benchmark rewards an agent
that can *perceive structure*, *predict the consequences of actions*, and *plan/search* without enormous state-space
blow-up. Our system separates these concerns into a perception layer, a local-geometry (reward) layer, and a
global-topology (Hodge) layer, and ties them with an online graph search. The unifying thread is that each layer
borrows an established mathematical object — symmetry groups and WL colourings for perception, gauge curvature for
the reward, Hodge theory for the topology — rather than learning everything from scratch.

The design constraint throughout is **avoiding the latent/state blow-up**: a naive latent hash collapses distinct
states, while a full forward search explodes. We instead grow an *online skeleton* keyed by a symmetry-invariant
**input-graph hash**, and let the gauge reward + Hodge intensity decide which edges are worth keeping and which
direction is "out".

---

## 2. Perception: group theory and temporal Weisfeiler–Lehman

### 2.1 Object signatures are symmetry invariants
Each frame is segmented into objects; an object is described not by raw pixels but by a signature that is invariant
to the symmetries we do not care about (recolouring within an equivalence class, translation, small deformation):
`(type_hash, colour, stability, area_bin, centroid, Mahalanobis footprint)`. This is the geometric-deep-learning
principle — *encode the symmetry group of the domain into the representation so that nuisance transformations leave
the representation unchanged* (Bronstein et al., 2021; Cohen & Welling, 2016). Objects that are related by a
symmetry are grouped into **orbits** (`orbit` ids), the orbit being the group action's equivalence class; this is
what lets us treat "the same kind of object in a different place" as interchangeable (e.g. for click-target
equivalence: if clicking one member of an orbit is a no-op, the whole orbit is skipped).

### 2.2 State identity by an input-graph hash
A whole state is hashed by a canonical, order-independent FNV-1a over the sorted multiset of (HUD-removed) object
signatures. Because the signature is symmetry-invariant and the multiset is canonicalised, two frames that are the
"same configuration up to nuisance" collide, and genuinely different configurations do not. This **input-graph
hash** is the identity used everywhere downstream; it gives ~140 distinct states for a representative level where a
latent hash collapsed to ~36.

### 2.3 Temporal Weisfeiler–Lehman for object tracking and stability
The 1-dimensional WL test (Weisfeiler & Lehman, 1968) iteratively refines a node's colour by hashing the multiset
of its neighbours' colours; it is the canonical-form / isomorphism heuristic that also exactly characterises the
expressive power of message-passing GNNs (Shervashidze et al., 2011; Xu et al., 2019). We use WL **temporally**:
rather than refining colours over the spatial neighbourhood of a single frame, we propagate object identities
*across consecutive frames* and refine the per-object colour by the (spatial-neighbour ⊕ previous-frame) multiset.
This temporal WL relabelling is what stabilises object identity through motion and births/deaths — an object keeps
its identity (and orbit) across a step even as its position and local neighbourhood change — and it supplies the
`stability` channel used in the signature. The perception graph fed to the trunk is therefore a *temporally
WL-consistent* object graph, with edge features encoding the spatial relations between objects.

### 2.4 A group-equivariant trunk
The shared encoder is a small relation-biased graph transformer over the object graph (attention biased by edge
features; cf. Veličković et al., 2018 for graph attention). Because its inputs are the symmetry-invariant
signatures and a permutation-equivariant attention, the pooled state embedding inherits the desired invariances.
This trunk is reused across all heads (forward, inverse, click) and across games.

---

## 3. Local geometry: a Yang–Mills curvature reward `dA + 10·A∧A`

### 3.1 The gauge analogy
In Yang–Mills theory a gauge connection `A` (a Lie-algebra-valued 1-form) has a **field strength / curvature
2-form**
`F = dA + A∧A`
where `dA` is the exterior derivative (the abelian, first-order part) and `A∧A` is the non-abelian
self-interaction — the part that, for a non-commutative gauge group, makes parallel transport around a loop fail to
return to the identity, i.e. **holonomy** (Yang & Mills, 1954; Nakahara, 2003; Kobayashi & Nomizu, 1963; Atiyah &
Bott, 1983). The two terms are exactly the two things we want to score about an action: *how much it changes the
world locally*, and *whether that change can be undone*.

### 3.2 Reading the analogy into transitions
For a transition `s --a--> s'` we set:
- **`dA = smae_pair(s, s')`** — a semantic + Mahalanobis-position change over the playfield (HUD excluded). This is
  the abelian first-order change: how far the state moved under the action (the "exterior derivative" of the
  action-as-connection).
- **`A∧A` = a no-return holonomy.** Reversible moves (you can immediately step back) are "flat" — `A∧A = 0`. Only
  for *events* — transitions whose `dA` exceeds the running mean+σ (`z(dA) > 1`) — do we measure the true holonomy
  by brute force: the minimum distance back to `s` over all return actions from `s'` (0 = returns exactly = flat;
  large = no return = curved). This mirrors how the commutator term `A∧A` is precisely what obstructs returning
  around an infinitesimal loop.

The per-transition reward is then the field-strength magnitude (with `β = 10` to surface rare events out of the
~7–13 baseline), plus topological penalties introduced in §4 and a goal bonus:
```
r = dA + 10·A∧A − 10·max(0, intensity(s') − intensity(s)) − 10·intensity(s') − degree(s')   (+100 on level-up)
```
The `max(0,·)` makes *leaving* a high-intensity region free while *entering* one costly — the agent is pushed
"downhill in curvature/loopiness" toward the boundary of attractor basins. Level completion is a near-singular
event (`dA` jumps an order of magnitude, `z ≈ +18`), so it is detected directly and given a large flat bonus.

### 3.3 Why holonomy, not just `dA`
`dA` alone cannot distinguish a productive irreversible move (opening a door) from a large but reversible jiggle.
The `A∧A` holonomy term is what tags **bridges** — transitions you cannot undo — which are exactly the
level-advancing actions. In gauge-theory language, the agent is looking for places where the connection has real
curvature, not pure gauge.

---

## 4. Global topology: Hodge intensity, node-stitching, and the graph

### 4.1 Stitching transitions into a 1-complex
Every committed/observed transition `s→s'` adds a directed edge between two input-hash nodes; the accumulating
edges form a directed graph that we treat as the 1-skeleton of a cell complex. An edge is **reversible** iff both
directions `s→s'` and `s'→s` were observed (the "flat", `A∧A≈0` edges); one-way edges are the "charged" ones.

### 4.2 Faces and the combinatorial Hodge view
We attach 2-cells (faces) by computing the **minimum cycle basis** over the reversible subgraph (chordless minimal
loops; `nx.minimum_cycle_basis`, not the spanning-tree fundamental basis, which over-counts degree-2 chain nodes).
This makes the skeleton a 2-complex on which the **combinatorial / Hodge Laplacian** is defined, and the
Hodge decomposition splits 1-cochains (edge flows) into *gradient* (potential-driven progress), *curl*
(rotational / loop) and *harmonic* (global cycle) parts (Hodge, 1941; Lim, 2020; Jiang, Lim, Yao & Ye, 2011;
Schaub et al., 2020; Barbarossa & Sardellitti, 2020; Grady & Polimeni, 2010). The reversible cycles are precisely
the curl/harmonic-carrying 2-cells — the "loops you can walk around and come back".

### 4.3 Intensity as loop membership
We define a node's **intensity** as a scalar read-off of this topology:
```
intensity(n) = #{green (reversible) faces containing n}  +  0.5 · 1[n lies on a one-way (charged) edge]
```
i.e. a node deep inside many reversible loops (an attractor "basin core") has high intensity; a node on the rim or
on a one-way bridge has low intensity. The 0.5 is attached to the *one-way edge endpoints only*, not to every
member of a charged face — the charge lives on the arrow, not the region. Intensity is recomputed online as new
edges close new faces, and it feeds the reward (§3.2): the agent is rewarded for *radiating outward along the rings
of intensity 1/0.5 and escaping*, never for sinking into the loop-heavy core. This is the topological signal that
turned a wandering search into one that gets *out* of basins toward goals.

### 4.4 Online, not batch
Crucially, the graph, its faces, and the intensities are built **incrementally during interaction** — there is no
pre-built full graph and no post-hoc global shortest path. Each newly observed edge updates only its local
radius-2 closures; zero-gain (all-reversible, no-event) triangles are closed and never re-entered.

---

## 5. Putting it together: online reward-guided graph search

The agent runs a depth-first online rollout over the input-hash skeleton: it probes available actions
(`do → observe → measure → undo` via environment snapshot/restore), scores each by the gauge reward, commits to the
best *escape* edge (lowest next-intensity, toward the boundary), logs no-ops per (state, move) and per
(state, click-node/orbit) so wall-hits and dead clicks are never repeated, and backs out via restore when a region
exceeds an intensity ceiling or dead-ends. A human-action *teacher prior* supplies click positions for click-heavy
games (the productive click is usually one a human used), used only to order probing — the gauge reward always
decides. A per-game **world model** (the group-equivariant trunk + forward `s_t,a_t→s_{t+1}` and inverse
`a_t←s_t,s_{t+1}` heads) is warm-started from a base model, lightly fine-tuned per game on level-1 data (only one
layer per module, prior intact), and further fine-tuned *online every 30 steps* while playing later levels.

---

## 6. Results (summary)

On a deterministic offline ARC-AGI-3 environment (25 games):
- The online gauge+Hodge rollout completes **level 1 of 14/25 games**, including click-to-start and click-heavy
  games once a coarse-grid / human-click prior is added.
- Remaining failures are (a) move-only games whose goal is too deep for greedy local search, and (b) click games
  whose goal needs an *ordered* click sequence, not just the right positions.
- The per-game world model trains stably with the one-layer-per-module light fine-tune (forward MAE and inverse
  accuracy improve in ~150 steps) and continues to reduce loss online while exploring a new level — demonstrating
  the "training-while-playing" loop, though the current bottleneck for level 2 is search/exploration rather than
  the model.

---

## 7. Discussion and future work

The three classical objects play distinct, complementary roles: **group theory + temporal WL** give a perception
whose *identities are stable and symmetry-aware*; **Yang–Mills curvature** gives a *local* score that separates
irreversible progress (holonomy) from reversible jiggle; **Hodge theory** gives a *global* topological coordinate
(intensity) that tells the agent where the basins are and which way is out. The clearest next lever is to let the
online-learned forward/inverse model *guide* the search (action-ordering prior), turning the model from a passive
online learner into an active planner — and to replace position-only click priors with state-conditioned click
*sequences* for the games that need ordered interaction.

---

## References

1. Weisfeiler, B., & Lehman, A. A. (1968). *A reduction of a graph to a canonical form and an algebra arising
   during this reduction.* Nauchno-Technicheskaya Informatsia, 2(9), 12–16.
2. Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011).
   *Weisfeiler–Lehman graph kernels.* Journal of Machine Learning Research, 12, 2539–2561.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How powerful are graph neural networks?* (GIN). ICLR.
4. Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). *Geometric Deep Learning: Grids, Groups,
   Graphs, Geodesics, and Gauges.* arXiv:2104.13478.
5. Cohen, T., & Welling, M. (2016). *Group Equivariant Convolutional Networks.* ICML.
6. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention
   Networks.* ICLR.
7. Yang, C. N., & Mills, R. L. (1954). *Conservation of isotopic spin and isotopic gauge invariance.* Physical
   Review, 96(1), 191–195.
8. Nakahara, M. (2003). *Geometry, Topology and Physics* (2nd ed.). IOP Publishing. (Field strength
   `F = dA + A∧A`.)
9. Kobayashi, S., & Nomizu, K. (1963). *Foundations of Differential Geometry, Vol. I.* Wiley. (Connections,
   curvature, holonomy.)
10. Atiyah, M. F., & Bott, R. (1983). *The Yang–Mills equations over Riemann surfaces.* Phil. Trans. R. Soc. Lond.
    A, 308, 523–615.
11. Hodge, W. V. D. (1941). *The Theory and Applications of Harmonic Integrals.* Cambridge University Press.
12. Lim, L.-H. (2020). *Hodge Laplacians on graphs.* SIAM Review, 62(3), 685–715.
13. Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). *Statistical ranking and combinatorial Hodge theory.*
    Mathematical Programming, 127(1), 203–244.
14. Schaub, M. T., Benson, A. R., Horn, P., Lippner, G., & Jadbabaie, A. (2020). *Random walks on simplicial
    complexes and the normalized Hodge 1-Laplacian.* SIAM Review, 62(2), 353–391.
15. Barbarossa, S., & Sardellitti, S. (2020). *Topological signal processing over simplicial complexes.* IEEE
    Transactions on Signal Processing, 68, 2992–3007.
16. Grady, L. J., & Polimeni, J. R. (2010). *Discrete Calculus: Applied Analysis on Graphs for Computational
    Science.* Springer.
17. Chollet, F. (2019). *On the Measure of Intelligence.* arXiv:1911.01547. (ARC.)
