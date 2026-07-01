"""Object-state vertices for the Hodge complex — clustered from the GNN per-node LATENTS.

Per design (GNN-first): the latent state is the GNN output, so an object's identity is a cluster of its GNN
per-object embedding `H[i]`. We use the latents AS-IS — no flicker-masking of the input (user's call:
"use the latents"). Revisited object-states collapse to one vertex (k-means in GNN-latent space); that
collapse is what closes the cycles the Hodge decomposition needs.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.cluster      # synthetic smoke
"""
from __future__ import annotations

import numpy as np


def _sqdist(X, C, chunk=4096):
    """[n,k] squared distances, chunked over rows to bound memory."""
    out = np.empty((len(X), len(C)), np.float32)
    c2 = (C * C).sum(1)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        out[s:s + chunk] = (x * x).sum(1, keepdims=True) - 2.0 * x @ C.T + c2[None]
    return out


def assign(C, X):
    """Nearest-center id for each row of X."""
    if len(X) == 0:
        return np.zeros(0, np.int64)
    return _sqdist(X, C).argmin(1).astype(np.int64)


def _kmeanspp(X, k, rng):
    """k-means++ seeding — avoids the local optima that random init falls into."""
    n = len(X)
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
    return C


def _lloyd(X, C, iters):
    a = _sqdist(X, C).argmin(1)
    for _ in range(iters):
        for c in range(len(C)):
            m = a == c
            if m.any():
                C[c] = X[m].mean(0)
        a = _sqdist(X, C).argmin(1)
    inertia = float(_sqdist(X, C)[np.arange(len(X)), a].sum())
    return C, a, inertia


def kmeans(X, k, iters=25, seed=0, n_init=4):
    """k-means++ init + n_init restarts, keep lowest inertia. Returns centers C[k,d], assignment a[n]."""
    k = min(k, len(X))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C0 = _kmeanspp(X, k, rng)
        C, a, inertia = _lloyd(X, C0.copy(), iters)
        if best is None or inertia < best[2]:
            best = (C, a, inertia)
    return best[0], best[1]


def object_states(Hp, Hc, Mp, Mc, M=100, seed=0):
    """Cluster alive per-object GNN latents (from both prev and cur frames) into M object-states.

    Hp, Hc : [T, N, d] GNN per-node latents for the prev / cur frame of each transition.
    Mp, Mc : [T, N] bool alive masks.
    Returns: C [M', d] centers, op [T, N] / oc [T, N] object-state id per slot (-1 where dead).
    """
    Mp = Mp.astype(bool); Mc = Mc.astype(bool)
    # Defense-in-depth (review blocker): never cluster a zero-vector latent. A truncated/dead-but-masked slot
    # is a zero row; if it slipped through it would collapse hundreds of distinct objects into one origin
    # cluster. AND the alive mask with a nonzero-latent test so such phantoms are dropped, not clustered.
    Mp = Mp & (np.abs(Hp).sum(-1) > 0)
    Mc = Mc & (np.abs(Hc).sum(-1) > 0)
    pool = np.concatenate([Hp[Mp], Hc[Mc]], 0).astype(np.float32)
    if len(pool) == 0:
        raise ValueError("no alive (nonzero) object latents to cluster")
    if not np.isfinite(pool).all():                      # fail loud on NaN/inf rather than silently collapse
        bad = int((~np.isfinite(pool)).any(1).sum())
        raise ValueError(f"non-finite GNN latents in cluster pool: {bad}/{len(pool)} rows — sanitize the GNN output")
    C, _ = kmeans(pool, M, seed=seed)
    op = np.full(Mp.shape, -1, np.int64)
    oc = np.full(Mc.shape, -1, np.int64)
    op[Mp] = assign(C, Hp[Mp].astype(np.float32))
    oc[Mc] = assign(C, Hc[Mc].astype(np.float32))
    return C, op, oc


# ----------------------------------------------------------------------------- synthetic smoke
def _smoke():
    rng = np.random.default_rng(0)
    T, N, d, K = 200, 6, 64, 5
    centers = rng.normal(0, 5, (K, d)).astype(np.float32)        # K true object-states
    Hp = np.zeros((T, N, d), np.float32); Hc = np.zeros((T, N, d), np.float32)
    Mp = np.zeros((T, N), bool); Mc = np.zeros((T, N), bool)
    truep = np.full((T, N), -1, np.int64)
    for t in range(T):
        nlive = rng.integers(2, N + 1)
        for i in range(nlive):
            kp = rng.integers(0, K)
            Hp[t, i] = centers[kp] + rng.normal(0, 0.3, d); Mp[t, i] = True; truep[t, i] = kp
            # cur: same object-state most of the time (static), sometimes transitions
            kc = kp if rng.random() < 0.7 else rng.integers(0, K)
            Hc[t, i] = centers[kc] + rng.normal(0, 0.3, d); Mc[t, i] = True

    C, op, oc = object_states(Hp, Hc, Mp, Mc, M=K, seed=0)
    P = []
    P.append(("recovered K centers", len(C) == K))
    # purity: each recovered cluster should map ~1:1 to a true state on prev frames
    al = Mp & (truep >= 0)
    pred = op[al]; true = truep[al]
    # majority-true per predicted cluster
    pur = 0
    for c in range(len(C)):
        sel = pred == c
        if sel.any():
            pur += np.bincount(true[sel]).max()
    purity = pur / al.sum()
    P.append((f"clustering purity high ({purity:.2f})", purity > 0.9))
    P.append(("op/oc = -1 exactly on dead slots", bool((op[~Mp] == -1).all() and (oc[~Mc] == -1).all())))
    P.append(("op/oc >= 0 on alive slots", bool((op[Mp] >= 0).all() and (oc[Mc] >= 0).all())))
    # static objects (kp==kc) should mostly keep op==oc after clustering
    ok = all(v for _, v in P)
    print("CLUSTER SMOKE (object-states from GNN latents):")
    for n, v in P:
        print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    _smoke()
