"""Shared graph-transformer trunk — copied & edited from world_model.Brain's relation-biased attention.

Encodes ONE state's object graph (action-free) into per-node features H[B,N,d] + a pooled vector.
The SAME trunk is applied to the current state (for all heads) and to the next state (for the inverse +
click heads) — that is the "common trunk".

Differences from Brain: action conditioning is removed from the trunk (it now lives in ForwardHead, since
the trunk is shared by action-free heads); stacked `n_blocks` of pre-LN edge-biased attention + FFN.
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401  (sets sys.path for transform_catalyst)

import torch
import torch.nn as nn

from transform_catalyst.data_adapter import obj_dim, EDGE_DIM


class _EdgeBiasedBlock(nn.Module):
    """One pre-LN edge-biased self-attention block (single head) + FFN, both residual. Edge features bias the
    scores. Attention is normalized with SINKHORN (doubly-stochastic: 2D row+col normalization), not softmax
    (which is 1D row-only) — so the attention matrix is balanced over both queries and keys."""

    def __init__(self, d, ffn_mult=2, sinkhorn_iters=3):
        super().__init__()
        self.d = d; self.sinkhorn_iters = sinkhorn_iters
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.ebias = nn.Linear(EDGE_DIM, 1)                         # edge features -> additive attention bias
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, h, mask, edges):
        x = self.ln1(h)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        sc = (q @ k.transpose(-1, -2)) / self.d ** 0.5 + self.ebias(edges).squeeze(-1)   # relation-biased scores
        km = mask[:, None, :]                                       # valid-key mask [B,1,n]
        A = (sc - sc.amax(-1, keepdim=True)).exp() * km             # stable exp; dead keys -> 0
        for _ in range(self.sinkhorn_iters):                       # SINKHORN: alternate row & col normalization (2D)
            A = A / (A.sum(-1, keepdim=True) + 1e-6)               #   row-normalize (over keys)
            A = A * km                                             #   keep dead keys at 0
            A = A / (A.sum(-2, keepdim=True) + 1e-6)               #   col-normalize (over queries)
        A = A / (A.sum(-1, keepdim=True) + 1e-6)                   # final row-norm -> convex weights for A@v
        h = h + A @ v                                             # attention residual
        h = h + self.ffn(self.ln2(h))                            # FFN residual
        return h


class GraphTransformerTrunk(nn.Module):
    """enc -> n_blocks x edge-biased attention -> (per-node H, pooled). Action-free, shared by all heads."""

    def __init__(self, cfg, d=64, n_blocks=2, max_nodes=24, ffn_mult=2, in_dim=None):
        super().__init__()
        self.od = obj_dim(cfg); self.d = d; self.max_nodes = max_nodes
        self.in_dim = in_dim if in_dim is not None else self.od    # in_dim>od ⇒ augmented latent (Mahalanobis pos)
        self.enc = nn.Linear(self.in_dim, d)
        self.blocks = nn.ModuleList([_EdgeBiasedBlock(d, ffn_mult) for _ in range(n_blocks)])

    def forward(self, Z, mask, edges):
        B, N, _ = Z.shape
        live = mask.any(0)
        n = int(live.nonzero().max()) + 1 if bool(live.any()) else 1
        n = min(n, self.max_nodes)
        Zs, ms, es = Z[:, :n], mask[:, :n], edges[:, :n, :n]
        h = self.enc(Zs)
        for blk in self.blocks:
            h = blk(h, ms, es)
        h = h * ms[..., None]
        pooled = h.sum(1) / ms.sum(1, keepdim=True).clamp(min=1.0)
        H = Z.new_zeros(B, N, self.d); H[:, :n] = h                 # scatter back to full N (dead slots = 0)
        return H, pooled
