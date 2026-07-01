"""GATE (build-order step 2) — is per-object change SPARSE/bimodal, and does Module-1 τ preserve it?

The whole approach rests on one premise: in a transition almost every object is unchanged and a few
carry the genuine change. If that's false, "predict the change" buys nothing. We measure CHANGE
CONCENTRATION = share of total per-object change mass held by the single most-changed object
(per transition); high = bimodal/sparse. We report it for raw ‖Δz‖ (the data premise) and for
τ = ‖P_⊥Δz‖ from Module 1 (mechanics check). Trained-generator τ (symmetry removed) lands after
Module 2; with random init we only assert τ is computable and does not destroy the concentration.

  PYTORCH_ENABLE_MPS_FALLBACK=1 ../../../.venv/bin/python -m transform_catalyst.verify_tau
"""
from __future__ import annotations

import numpy as np
import torch

from .repro import seed_all
from .data_adapter import load_pairs, obj_dim
from .lie_nodes import NodeLie


def _concentration(per_obj_change: np.ndarray, matched: np.ndarray):
    """top-1 object's share of total change mass, per transition (only transitions with real change)."""
    c = per_obj_change * matched
    tot = c.sum(1); top1 = c.max(1)
    sel = tot > 1e-6
    conc = top1[sel] / np.clip(tot[sel], 1e-9, None)
    return conc, sel


def main():
    seed_all(0)
    data, cfg = load_pairs(n_files=40)
    Zp, Zc, Mp, Mc = data["Zp"], data["Zc"], data["Mp"], data["Mc"]
    matched = (Mp & Mc).astype(np.float32)                 # pid soft-evidence pairing (diagnostic only)
    dz = Zc - Zp
    raw = np.linalg.norm(dz, axis=-1)                      # [T,N] raw per-object change
    print(f"[data] {Zp.shape[0]} transitions | N_max={Zp.shape[1]} obj_dim={obj_dim(cfg)} | "
          f"mean matched objs/transition={matched.sum(1).mean():.1f}", flush=True)

    conc, sel = _concentration(raw, matched)
    frac_zero = ((raw < 1e-6) & matched.astype(bool))[matched.astype(bool)].mean()
    print(f"[premise] raw ‖Δz‖ top-1 concentration: mean={conc.mean():.2f} median={np.median(conc):.2f} "
          f"(p90={np.quantile(conc,0.9):.2f}) | fraction of matched objects with ~0 change={frac_zero:.2f}",
          flush=True)
    print(f"[premise] => {'SPARSE/bimodal: predict-the-change is well-founded' if np.median(conc) > 0.5 else 'NOT clearly sparse — reconsider'}",
          flush=True)

    # Module-1 mechanics: τ computable at object granularity; concentration preserved (random gens)
    lie = NodeLie(obj_dim(cfg))
    with torch.no_grad():
        tau = lie.tau(torch.from_numpy(Zp), torch.from_numpy(dz)).numpy()
    tconc, _ = _concentration(tau, matched)
    gate = lie.gate().tolist()
    print(f"[module1] τ=‖P_⊥Δz‖ top-1 concentration (init gens): mean={tconc.mean():.2f} "
          f"median={np.median(tconc):.2f} | gate(open if≥{lie.ridge**0.5:.3f})={[round(g,3) for g in gate]}",
          flush=True)
    print(f"[module1] τ vs raw concentration delta={np.median(tconc)-np.median(conc):+.2f} "
          f"(trained generators should push this further by removing symmetry-only change)", flush=True)


if __name__ == "__main__":
    main()
