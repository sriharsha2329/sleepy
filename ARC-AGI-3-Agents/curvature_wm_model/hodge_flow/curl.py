"""curl.py — the triangle method: A∧A-gated fill + Hodge decomposition; identify reversible-loop (curl) edges.

Curl ≡ the reversible circulation, found by filling triangles whose every edge is reversible (A∧A < TAU_REV) and
projecting the flow onto them (rung3.hodge with the A∧A switch). Boundary additivity (PLAN §2.2): a triangulable
larger loop is the sum of its triangle curls, so this resolves any triangulable loop. Curl-dominant edges are the
reversible WASTED loops the rollout must avoid.

Note (bp35): a pure back-and-forth on ONE edge is a 2-cycle, not a triangle — it shows as ~0 NET flow (cancels),
not as Hodge curl. Such reversible 2-cycles are caught by the per-edge A∧A + revisit penalty in rollout.py; the
Hodge curl here catches the 3+ node (triangulable) reversible loops.
"""
from __future__ import annotations

from curvature_wm_model.hodge_flow.hodge import hodge_decompose
from curvature_wm_model.hodge_flow import hodge_loops as hl


def decompose(fg, method="cycle"):
    """Hodge-decompose the FlowGraph's oriented flow with the A∧A fill. Returns dec (grad/curl/harm per edge +
    energy split) or None if no edges.
      method='triangle' — fill A∧A-reversible TRIANGLES only (fast; misses non-triangulable 4+ loops → harmonic).
      method='cycle'    — fill A∧A-reversible CYCLES via the cycle basis (catches 2D-grid 4-loops as CURL). DEFAULT,
                          because real games (ls20/g50t/…) are 2D and their reversible loops are NOT triangles."""
    nodes, edges, x, rev = fg.hodge_inputs()
    if len(edges) == 0:
        return None
    if method == "cycle":
        filled = hl.reversible_fill(nodes, edges, rev)
        dec = hl.hodge_decompose_filled(nodes, edges, x, filled)
        dec["beta1"] = None
    else:
        dec = hodge_decompose(nodes, edges, x, reversible_edges=rev)
    dec["reversible"] = rev
    return dec


def curl_edges(fg, dec, frac=0.5):
    """Undirected edges whose flow is CURL-dominant (reversible triangulable loops → avoid). Returns set of (lo,hi)."""
    if dec is None:
        return set()
    g, c, h = dec["g_e"], dec["c_e"], dec["h_e"]
    out = set()
    for i, e in enumerate(dec["edges"]):
        if c[i] / (g[i] + c[i] + h[i] + 1e-9) > frac:
            out.add(e)
    return out


def energy(dec):
    return None if dec is None else (dec["e_grad"], dec["e_curl"], dec["e_harm"])
