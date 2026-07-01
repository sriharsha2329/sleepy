"""MODULE 1 (replacement) — DISCRETE HODGE decomposition of the transition over the object graph.

The continuous-Lie generators went inert: the equivariance test is persistence-dominated (f≈identity
commutes with everything) so every gate pinned at 0.73 and the curl layer was effectively L2. The fix
is to drop the LEARNED continuous generators for the DETERMINISTIC discrete Hodge split on the relation
graph (nodes = objects, edges = relations). There is no gate to collapse, and it is category-correct
for a discrete grid world (the recon flagged exactly this).

0-cochain (node-field) Hodge — the genuine-change projector that replaces P_⊥:
    Δz = persistence(Δz)  ⊕  δ
    persistence = D⁻¹W Δz   (graph consensus; a GLOBAL uniform transform is in ker ⇒ pure symmetry)
    δ = (I − D⁻¹W) Δz       genuine differentiated change: where an object moves DIFFERENTLY from its
                            graph neighbours.  Uniform shift ⇒ δ = 0, deterministically (no gate).

1-cochain (edge-flow) REVERSE Hodge — recover the potential from the flow:
    f_{ij} = (s_i+s_j)·(dx+dy)_{ij}·W_{ij}      s=‖δ‖; (dx,dy) ANTISYMMETRIC ⇒ f carries genuine curl
    solve  L₀ φ = div f      (graph Poisson, conjugate gradient, MPS-native)   ← "reverse": flow→potential
    φ_i  = how far object i sits along the transformation's gradient flow — a CLICK localisation feature.
    ρ = 1 − ⟨φ, div f⟩ / ‖f‖²  ∈ [0,1] : 0 = pure gradient/transport, 1 = cyclic/rotational.

No learned parameters (nn.Module only for device placement + API symmetry with NodeLie). Graceful
degradation: edges=None ⇒ W=0 ⇒ persistence = global mean ⇒ δ = Δz − mean (the no-graph fallback).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data_adapter import N_RELS


class NodeHodge(nn.Module):
    def __init__(self, d: int, geo_off: int = N_RELS, cg_iters: int = 12):
        super().__init__()
        self.d = int(d)
        self.geo = int(geo_off)          # index of dx within the edge feature vector
        self.cg = int(cg_iters)

    # ----------------------------------------------------------------- graph
    def _adj(self, edges, mask) -> torch.Tensor:
        """Symmetric, self-loop-free adjacency W[B,N,N] from relation presence; dead nodes dropped."""
        B, N = mask.shape
        mf = mask.float()
        if edges is None:
            return torch.zeros(B, N, N, device=mask.device, dtype=mf.dtype)
        W = (edges[..., :N_RELS].abs().sum(-1) > 0).float()              # [B,N,N]
        W = W * (mf[:, :, None] * mf[:, None, :])
        W = torch.maximum(W, W.transpose(-1, -2))
        W = W * (1.0 - torch.eye(N, device=W.device, dtype=W.dtype))
        return W

    def _smooth(self, x, W, mask) -> torch.Tensor:
        """Graph consensus D⁻¹W x; a live but ISOLATED node falls back to the global live mean."""
        mf = mask.float()[..., None]                                      # [B,N,1]
        deg = W.sum(-1, keepdim=True)                                     # [B,N,1]
        sm = torch.einsum("bij,bjd->bid", W, x) / deg.clamp(min=1.0)
        gm = (x * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        iso = (deg <= 0) & mask.bool()[..., None]
        sm = torch.where(iso, gm.expand_as(sm), sm)
        return sm * mf

    # ----------------------------------------------------------------- 0-form Hodge (genuine change)
    def transverse(self, z, dz, edges=None, mask=None) -> torch.Tensor:
        """δ = (I − D⁻¹W) Δz — genuine differentiated change (persistence removed). [...,d] -> [...,d]."""
        W = self._adj(edges, mask)
        return (dz - self._smooth(dz, W, mask)) * mask[..., None]

    def highpass(self, z, edges=None, mask=None) -> torch.Tensor:
        """z with graph consensus removed — a global-offset-invariant node descriptor (retrieval)."""
        W = self._adj(edges, mask)
        return (z - self._smooth(z, W, mask)) * mask[..., None]

    def tau(self, z, dz, edges=None, mask=None) -> torch.Tensor:
        """τ = ‖δ‖ per object. Bimodal expected: the genuinely-changed object high, the rest ≈ 0."""
        return self.transverse(z, dz, edges, mask).norm(dim=-1)

    def persistence_share(self, z, dz, edges=None, mask=None) -> float:
        """‖persistence‖² / ‖Δz‖² ∈ [0,1] — fraction of the raw change that is pure global symmetry."""
        W = self._adj(edges, mask)
        sm = self._smooth(dz, W, mask)
        w = mask[..., None]
        pers = float((sm.pow(2) * w).sum()); tot = float((dz.pow(2) * w).sum())
        return pers / max(tot, 1e-9)

    # ----------------------------------------------------------------- 1-form REVERSE Hodge (potential)
    def _matvec(self, W, x, mask) -> torch.Tensor:
        """Combinatorial Laplacian L₀ x = D x − W x, masked. x[B,N]."""
        deg = W.sum(-1)
        return (deg * x - torch.einsum("bij,bj->bi", W, x)) * mask

    def _zero_mean(self, x, mask) -> torch.Tensor:
        m = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return (x - m) * mask

    def potential(self, z, dz, edges, mask):
        """Reverse Hodge: φ with L₀φ = div f (CG), plus the rotational fraction ρ. -> (φ[B,N], ρ[B]).

        f is the antisymmetric edge flow of the genuine change weighted by the (antisymmetric) geometry,
        so it carries real curl; φ is the recovered gradient potential and ρ the non-gradient share."""
        m = mask.to(z.dtype)
        W = self._adj(edges, mask)
        s = self.transverse(z, dz, edges, mask).norm(dim=-1)             # [B,N] genuine-change magnitude
        gp = edges[..., self.geo] + edges[..., self.geo + 1]             # dx+dy, antisymmetric [B,N,N]
        F = (s[:, :, None] + s[:, None, :]) * gp * W                     # antisymmetric by construction
        f = 0.5 * (F - F.transpose(-1, -2))
        b = self._zero_mean(f.sum(-1) * m, mask)                         # div f ⟂ constants
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum(1, keepdim=True)
        for _ in range(self.cg):                                         # CG — matmul only (MPS-native)
            Ap = self._matvec(W, p, m)
            alpha = rs / (p * Ap).sum(1, keepdim=True).clamp(min=1e-12)
            x = self._zero_mean(x + alpha * p, mask)
            r = r - alpha * Ap
            rs_new = (r * r).sum(1, keepdim=True)
            p = r + (rs_new / rs.clamp(min=1e-12)) * p
            rs = rs_new
        grad_E = (x * b).sum(1)                                          # φᵀL₀φ = ‖gradient part‖²
        flow_E = 0.5 * (f * f).sum((1, 2))                              # ‖f‖² (each edge once)
        rot = (1.0 - grad_E / flow_E.clamp(min=1e-9)).clamp(0.0, 1.0)
        return x, rot
