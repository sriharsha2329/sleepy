"""MODULE 1 — node-level Lie projection with a gate that BITES.

Generators act on the OBJECT-STATE latent z ∈ R^d (HARD CONSTRAINT 2: never the pool). The
transformation factors γ = g∘δ; this module reads the symmetry part g via the generators and
exports the genuine part δ = P_⊥Δz (transverse residual) and its norm τ = ‖P_⊥Δz‖.

Why Lie (not finite groups): the generators are LEARNED by gradient descent, which needs a tangent
(exp(t·G) smooth in G); a finite group has no tangent. The discrete grid symmetries (D4, etc.) are
recovered as exp(t*·G) at special t — the continuous generator is the learnable superset.

Gate fix: a SOFT column scale is ~inoperative (the ridge solve is ~scale-invariant to column
scaling, so a distrusted generator g_k≈0.07 still deletes ~83% of genuine change). HARD-DROP
columns with g_k < √ridge ≈ 0.032 so they truly leave the span ⇒ P_⊥ ignores them (graceful
degradation: all gates closed ⇒ P_⊥ → I ⇒ τ → ‖Δz‖, the no-symmetry fallback).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def expm_taylor(A: torch.Tensor, order: int = 8) -> torch.Tensor:
    """exp(A) via truncated Taylor — MPS-native (matmul/add only). Exact for A=t·ξ, ‖A‖≈0.15
    (order-8 remainder ~1e-13). Differentiable; supports a leading [m,...] batch of matrices."""
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).expand_as(A)
    E = I.clone(); term = I.clone()
    for k in range(1, order + 1):
        term = (term @ A) / k
        E = E + term
    return E


class NodeLie(nn.Module):
    def __init__(self, d: int, m: int = 6, margin: float = 0.25, ridge: float = 1e-3):
        super().__init__()
        self.d, self.m = int(d), int(m)
        self.margin, self.ridge = float(margin), float(ridge)
        self.W = nn.Parameter(torch.randn(self.m, d, d) * 0.02)
        self.register_buffer("equiv_ema", torch.full((self.m,), 1.0))   # EMA of relative equiv error

    def generators(self) -> torch.Tensor:
        """G_k = (W_k − W_kᵀ)/‖·‖_F (skew ⇒ exp orthogonal). -> [m,d,d]."""
        G = self.W - self.W.transpose(-1, -2)
        return G / G.flatten(1).norm(dim=1).clamp_min(1e-6)[:, None, None]

    def gate(self) -> torch.Tensor:
        """g_k = σ((margin − r̂_k)/margin) ∈ (0,1); r̂_k = EMA of the SCALE-FREE relative equiv error."""
        return torch.sigmoid((self.margin - self.equiv_ema) / max(self.margin, 1e-6))

    @torch.no_grad()
    def update_equiv(self, per_k: torch.Tensor, rate: float = 0.1) -> None:
        self.equiv_ema.mul_(1 - rate).add_(rate * per_k.detach().to(self.equiv_ema.device))

    def tangent(self, z: torch.Tensor) -> torch.Tensor:
        """Orbit tangent A(z)=[G_1 z|…|G_m z] with distrusted columns HARD-DROPPED. z[...,d] -> [...,d,m]."""
        A = torch.einsum("kij,...j->...ik", self.generators(), z)        # [...,d,m]
        keep = (self.gate() >= self.ridge ** 0.5).to(A.dtype)            # {0,1} per generator
        return A * keep.view(*([1] * (A.dim() - 1)), -1)

    def transverse(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """P_⊥Δz, the genuine (non-symmetry) part. [...,d] -> [...,d].

        MPS-native projection via modified Gram-Schmidt on the (gated) ≤m columns — pure
        matmul/norm, NO torch.linalg.solve (which falls back to CPU at ~1.8s/call on MPS). A
        closed-gate column has ~0 norm and is skipped, so this is exactly the ridge-projection in
        the limit the hard gate already enforces (drop ⇒ leave the span)."""
        A = self.tangent(z)                                              # [...,d,m]
        par = torch.zeros_like(dz)
        qs = []
        for k in range(self.m):
            v = A[..., k]
            for q in qs:                                                # orthogonalize vs kept basis
                v = v - (v * q).sum(-1, keepdim=True) * q
            nrm = v.norm(dim=-1, keepdim=True)
            q = (v / nrm.clamp_min(1e-6)) * (nrm > 1e-4)                # unit; degenerate ⇒ 0
            par = par + (dz * q).sum(-1, keepdim=True) * q
            qs.append(q)
        return dz - par

    def tau(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """τ = ‖P_⊥Δz‖ per object -> [...]. Expected bimodal: active object high, rest ≈ 0."""
        return self.transverse(z, dz).norm(dim=-1)

    def equivariance(self, z, forward_fn, t: float = 0.15, mask=None):
        """Per-generator equivariance abs_k=‖f(exp(tξ_k)z)−exp(tξ_k)f(z)‖², rel_k scale-free. Trains
        the generators against the NODE-LEVEL forward model (MODULE 2). z[...,d]; optional live mask."""
        G = self.generators()
        fz = forward_fn(z)
        w = None if mask is None else mask.unsqueeze(-1).to(fz.dtype)

        def _mean(x):
            return x.mean() if w is None else (x * w).sum() / (w.sum().clamp(min=1.0) * x.shape[-1])

        ab, rl = [], []
        for k in range(self.m):
            E = expm_taylor(t * G[k])
            fzt = forward_fn(z @ E.t()); efz = fz @ E.t()
            num = _mean((fzt - efz).pow(2))
            den = 0.5 * (_mean(fzt.pow(2)) + _mean(efz.pow(2))) + 1e-8
            ab.append(num); rl.append(num / den)
        return torch.stack(ab), torch.stack(rl)
