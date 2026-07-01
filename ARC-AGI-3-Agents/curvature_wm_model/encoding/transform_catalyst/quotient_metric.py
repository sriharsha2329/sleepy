"""Lie-group QUOTIENT metric — symmetry-invariant state similarity for retrieval.

d_G(z, z') = min_θ ‖exp(Σ_k θ_k G_k) z − z'‖  (distance to the orbit; metric on ℝ^d/G).
G_k skew ⇒ ρ(θ)=exp(Σθ_k G_k) ∈ SO(d), so d_G is symmetric and the radial part ‖z‖ is preserved
(symmetry can't change the norm). First-order: d_G ≈ ‖P_⊥(z−z')‖ with P_⊥ from the orbit tangent
A(z)=[G_k z] — the transverse norm IS the first-order quotient distance. Gauss-Newton refines the
exact geodesic for non-nearby pairs. Only GATED generators enter (gauge); none ⇒ P_⊥→I ⇒ plain L2.
"""
from __future__ import annotations

import torch

from .lie_nodes import expm_taylor


def _batched_gs(A: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the columns of A [b,d,m] (modified Gram-Schmidt; degenerate col -> 0). MPS-native."""
    b, d, m = A.shape
    Q = torch.zeros_like(A)
    for k in range(m):
        v = A[:, :, k]
        for j in range(k):
            v = v - (v * Q[:, :, j]).sum(-1, keepdim=True) * Q[:, :, j]
        n = v.norm(dim=-1, keepdim=True)
        Q[:, :, k] = torch.where(n > 1e-4, v / n.clamp_min(1e-6), torch.zeros_like(v))
    return Q


@torch.no_grad()
def lie_quotient_dist(Zq: torch.Tensor, Zmem: torch.Tensor, lie, ridge: float = 1e-3,
                      q_chunk: int = 128) -> torch.Tensor:
    """First-order quotient distance d_G(z_q, z_mem) ≈ ‖P_⊥(z_q−z_mem)‖, P_⊥ at each query's tangent.
    Zq [B,d], Zmem [N,d] -> [B,N]. States differing only by a learned symmetry are ≈0; the radial
    (norm) difference is preserved. Gauge: only gated generators; none ⇒ exact L2 (graceful)."""
    G = lie.generators()
    keep = lie.gate() >= ridge ** 0.5
    G = G[keep]
    if G.shape[0] == 0:
        return torch.cdist(Zq, Zmem)                                  # no symmetry ⇒ P_⊥ = I
    out = Zq.new_empty(Zq.shape[0], Zmem.shape[0])
    for s in range(0, Zq.shape[0], q_chunk):
        zq = Zq[s:s + q_chunk]                                        # [b,d]
        A = torch.einsum("kij,bj->bik", G, zq)                        # [b,d,m] orbit tangent
        Q = _batched_gs(A)                                            # [b,d,m] orthonormal
        D = zq[:, None, :] - Zmem[None, :, :]                         # [b,N,d]
        par = torch.einsum("bnm,bdm->bnd", torch.einsum("bnd,bdm->bnm", D, Q), Q)
        out[s:s + q_chunk] = (D - par).norm(dim=-1)                   # ‖P_⊥ Δ‖
    return out


@torch.no_grad()
def lie_quotient_refine(zq: torch.Tensor, Zt: torch.Tensor, lie, ridge: float = 1e-3,
                        iters: int = 3) -> torch.Tensor:
    """Exact geodesic d_G(z_q, z_t) for a small candidate set via Gauss-Newton orbit alignment.
    zq [d], Zt [K,d] -> [K]. Use to RE-RANK the linearized top-K (non-nearby pairs)."""
    G = lie.generators()[lie.gate() >= ridge ** 0.5]                  # [m,d,d] gated
    m = G.shape[0]
    K = Zt.shape[0]
    if m == 0:
        return (Zt - zq).norm(dim=-1)
    theta = torch.zeros(K, m, device=zq.device)
    zr = zq.expand(K, -1).clone()                                     # ρ(θ) z_q, starts at z_q
    eye = torch.eye(m, device=zq.device)
    for _ in range(iters):
        J = torch.stack([(G[k] @ zr.t()).t() for k in range(m)], -1)  # [K,d,m] tangent at zr
        r = (zr - Zt).unsqueeze(-1)                                   # [K,d,1]
        JtJ = (J.transpose(1, 2) @ J + ridge * eye).cpu()            # tiny m×m; solve on CPU (no MPS fallback)
        Jtr = (J.transpose(1, 2) @ r).cpu()
        theta = theta - torch.linalg.solve(JtJ, Jtr).squeeze(-1).to(zq.device)
        E = expm_taylor((theta[:, :, None, None] * G).sum(1))         # [K,d,d]
        zr = torch.einsum("kij,j->ki", E, zq)
    return (zr - Zt).norm(dim=-1)
