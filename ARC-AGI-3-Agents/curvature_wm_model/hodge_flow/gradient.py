"""gradient.py — gradient flow ≡ dA, and the progress potential φ (Hodge-Flow §2.1 / §4.2).

The gradient is the net-transport part, measured directly as dA = z_next − z_current (kept in the FlowGraph as
the signed edge flow). For rollout we also need the scalar PROGRESS POTENTIAL φ: the rollout descends φ toward a
no-return goal. φ(node) = graph distance to the nearest goal (reverse BFS over the directed model) — a monotone
progress coordinate that the accumulated dA traces out. Lower φ = closer to the goal = "follow the gradient."
"""
from __future__ import annotations

from collections import defaultdict, deque


def progress_potential(fg, goal_ids):
    """φ(node) = # model-steps to the nearest goal (reverse BFS from goals over u --a--> v). Unreachable → absent.
    Following decreasing φ = net forward progress (the gradient) toward a no-return goal."""
    radj = defaultdict(set)                                   # v -> {u : exists u--a-->v}
    for u, am in fg.model.items():
        for a, (v, _dA, _aa) in am.items():
            if v != u:
                radj[v].add(u)
    phi = {g: 0 for g in goal_ids}
    q = deque(goal_ids)
    while q:
        v = q.popleft()
        for u in radj[v]:
            if u not in phi:
                phi[u] = phi[v] + 1
                q.append(u)
    return phi


def gradient_edge_mass(dec):
    """Per-edge gradient magnitude g_e (net transport) from a decomposition, or None if no decomposition."""
    return None if dec is None else dec["g_e"]
