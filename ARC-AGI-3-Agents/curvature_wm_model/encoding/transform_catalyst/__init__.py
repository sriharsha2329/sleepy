"""transform_catalyst — represent each transition as a TRANSFORMATION (not a next-state),
group-coordinatized (Lie generators), holonomy-tracked (orbit-graph loops), correspondence-free.

The structural priors (genuine change τ, causal non-commutativity curvature, progress) are used as
CATALYSTS — support-pruning + reliability-ranked QRE for click candidates, decision-time bias, and
replay priority — NEVER as reward and NEVER in the value/TD target. See SPEC.md for the hard
constraints (each one encodes a measured failure we will not re-import)."""
