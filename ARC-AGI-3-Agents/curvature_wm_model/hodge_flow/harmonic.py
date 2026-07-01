"""harmonic.py — the no-return goals: where dA and the triangle method BOTH fail (Hodge-Flow §2.3).

Two faces of "no-return", both surfaced here:
  - NO-RETURN EVENTS (bridges): a transition with large dA AND large A∧A — net transport you cannot undo. In a
    one-way game these are gradient *bridges* (no cycle), so the simplicial harmonic is ~0; the A∧A flag is what
    identifies them. These are the milestones the rollout SEEKS.  → `no_return_goals`.
  - NO-RETURN LOOPS (true holes): circulation that is neither net transport (dA fails) nor triangulable
    (the triangle method fails) → the simplicial harmonic component, beta1 > 0.  → `harmonic_loop_edges`.
"""
from __future__ import annotations


def no_return_goals(fg, tau_event):
    """Goal NODES = the target of a no-return EVENT transition: A∧A ≥ tau_event AND dA ≥ tau_event (irreversible,
    big change). Navigation (small dA / small A∧A) never qualifies. These are the irreversible milestones to seek."""
    goals = set()
    for u, am in fg.model.items():
        for a, (v, dA, aa) in am.items():
            if v != u and aa >= tau_event and dA >= tau_event:
                goals.add(v)
    return goals


def harmonic_loop_edges(dec, frac=0.5):
    """Undirected edges whose flow is HARMONIC-dominant — circulation around an unfillable (non-triangulable,
    irreversible) hole. Empty in a strictly one-way game (beta1=0). Returns set of (lo,hi)."""
    if dec is None:
        return set()
    g, c, h = dec["g_e"], dec["c_e"], dec["h_e"]
    return {e for i, e in enumerate(dec["edges"]) if h[i] / (g[i] + c[i] + h[i] + 1e-9) > frac}


def beta1(dec):
    return None if dec is None else dec.get("beta1")
