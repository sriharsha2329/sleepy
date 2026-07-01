"""Muon optimizer — DeepSeek/Moonshot "scalable Muon" recipe.

For each 2D weight matrix: momentum buffer -> Newton–Schulz orthogonalization (O ≈ U Vᵀ of the momentum)
-> update scaled by 0.2·√(max(d_out,d_in)) so its RMS matches Adam, + decoupled weight decay. Non-matrix
params (embeddings, biases, norms, thin 1-column matrices) should be optimized with Adam separately.

Refs: Keller Jordan (Muon); Moonshot "Muon is Scalable for LLM Training" (the 0.2·√max scale + wd).
"""
from __future__ import annotations

import torch


def _newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize G via 5 quintic Newton–Schulz iterations (returns ≈ U Vᵀ of G)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.t()
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom, lr, ns = grp["momentum"], grp["lr"], grp["ns_steps"]
            nest, wd = grp["nesterov"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get("buf")
                if buf is None:
                    buf = st["buf"] = torch.zeros_like(p)
                buf.mul_(mom).add_(p.grad)
                upd = p.grad.add(buf, alpha=mom) if nest else buf      # Nesterov momentum
                o = _newtonschulz5(upd, ns)
                if wd:
                    p.mul_(1.0 - lr * wd)                              # decoupled weight decay
                p.add_(o, alpha=-lr * 0.2 * (max(p.shape) ** 0.5))     # Moonshot RMS-matching scale


def split_muon_adam(named_params):
    """Route params: 2D matrices (min dim >= 2, not embeddings) -> Muon; everything else -> Adam."""
    muon, adam = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 2 and min(p.shape) >= 2 and "act" not in name and "emb" not in name:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam
