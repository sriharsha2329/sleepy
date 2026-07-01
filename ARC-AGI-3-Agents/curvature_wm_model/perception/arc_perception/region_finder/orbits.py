"""Context-refined equivalence classes for candidates.

NOTE on naming:  what this module computes is NOT a group orbit.  It is an
equivalence partition of the candidate set, defined by the tuple

    (kind, κ_local, ctx_signature(region))

where κ_local is itself the canonical representative of the patch's *D4-orbit*
(computed in Phase 1C).  Hence the choice of name:

  * "D4-orbit"  — the genuine group orbit; lives in Phase 1C as κ_local.
  * "class"     — the engineering name for this Phase 1D equivalence class.
                  Not a group orbit; just a partition.

Refined equivalence:
    p ~ q   iff   (kind, κ_local, ctx_signature(region(p))) =
                  (kind, κ_local, ctx_signature(region(q)))

Class keys are assigned deterministically by sorted hash:  K0001, K0002, …
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable


def _class_hash(candidate, region) -> str:
    """Visualization hash — uses depth-2 ctx_signature; distinguishes positions."""
    ctx = ""
    if region is not None:
        ctx = (region.features.get("context_signature_d2")
               or region.features.get("context_signature")
               or "")
    raw = f"{candidate.kind}|{candidate.local_signature}|{ctx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _class_hash_evidence(candidate, region) -> str:
    """Evidence hash for the click_feedback layer.

    Priority order (most stable across frames first):
      C2:  `persistent_id` from the region tracker — survives cursor
           movement and other WL-cascading perturbations.  This is what
           makes click evidence actually accumulate across frames.
      C3:  fall back to `graph_orbit_id` (Weisfeiler-Leman fixed-point)
           when no persistent track exists yet.
      C1:  fall back to `context_signature_evidence` (pos-invariant depth-1)
           as a last resort.

    Why C2 must beat C3 here: WL is a graph hash — when the cursor moves,
    its neighbour relations change, those neighbours' colours change, and
    the change cascades through the fixed-point iteration.  Two
    *identical* clicks on the same screen pixel of the same region can
    therefore land in different WL orbits frame-to-frame, fragmenting
    evidence.  `persistent_id` is matched by mask IoU + colour, so it
    survives that cascade.
    """
    ctx = ""
    if region is not None:
        ctx = (region.features.get("persistent_id")
               or region.features.get("graph_orbit_id")
               or region.features.get("context_signature_evidence")
               or "")
    raw = f"{candidate.kind}|{candidate.local_signature}|{ctx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def group_candidates_into_classes(
    candidates: Iterable,
    regions: Iterable,
) -> dict:
    """Mutate each candidate to set `class_key`; return class → metadata dict."""
    by_region = {r.region_id: r for r in regions}
    buckets: dict[str, list] = defaultdict(list)
    for c in candidates:
        h = _class_hash(c, by_region.get(c.region_id))
        buckets[h].append(c)

    classes: dict[str, dict] = {}
    for i, h in enumerate(sorted(buckets.keys())):
        members = buckets[h]
        class_key = f"K{i + 1:04d}"
        for c in members:
            c.class_key = class_key
            # `class_key_hash` is stable across frames (depends only on
            # (kind, κ_local, ctx)); the renumbered class_key is not.
            # The UI uses the hash for colour stability.
            c.features["class_key_hash"] = h
            # Position-invariant counterpart for the click_feedback evidence
            # layer (fix C1).  See _class_hash_evidence.
            c.features["class_key_hash_evidence"] = _class_hash_evidence(
                c, by_region.get(c.region_id)
            )
        classes[class_key] = {
            "class_key_hash": h,
            "size": len(members),
            "candidates": [c.candidate_id for c in members],
        }
    return classes


def class_size_histogram(classes: dict) -> dict[int, int]:
    """How many classes of each size — useful as a debug stat."""
    hist: dict[int, int] = defaultdict(int)
    for meta in classes.values():
        hist[meta["size"]] += 1
    return dict(hist)
