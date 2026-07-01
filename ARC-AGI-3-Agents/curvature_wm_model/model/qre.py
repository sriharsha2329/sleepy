"""QRE-balanced multi-task weighting — COPIED from qre_balance.py (kept self-contained in the new folder).

w = softmax_β( log(held-out skill) + λ·grad-alignment ), iterated to the QRE fixed point, then floored so
no task starves. `total_loss` optionally scales grad-normalized task losses so w = true share of trunk influence.
"""
from __future__ import annotations

import numpy as np
import torch


class QREBalancer:
    def __init__(self, n_heads, lam=5.0, beta=1.0, floor=0.1, iters=12):
        self.K = n_heads; self.lam = lam; self.beta = beta; self.floor = floor; self.iters = iters
        self.w = np.ones(n_heads) / n_heads

    @staticmethod
    def alignment(grads):
        """agree_k = mean_j cos(g_k, g_j): how each task's trunk-gradient aligns with the others."""
        G = torch.stack([g / (g.norm() + 1e-9) for g in grads])
        C = G @ G.T
        return ((C.sum(1) - 1.0) / max(len(grads) - 1, 1)).cpu().numpy()

    def weights(self, skills, agree=None):
        """skills: held-out skill per task (the UNWRITABLE anchor). agree: optional grad-alignment [K]."""
        rel = np.clip(np.asarray(skills, float), 1e-3, None)
        base = np.log(rel); base -= base.max()
        a = np.zeros(self.K) if agree is None else np.asarray(agree, float)
        w = np.exp(self.beta * base); w /= w.sum()
        for _ in range(self.iters):
            v = self.beta * (base + self.lam * a); v -= v.max()
            w = np.exp(v); w /= w.sum()
        w = np.maximum(w, self.floor); w /= w.sum()
        self.w = w; return w

    def total_loss(self, losses, grad_norm=True):
        """Σ_k w_k L_k. If grad_norm, divide each L_k by its detached scale so w = true influence share."""
        ws = torch.tensor(self.w, dtype=losses[0].dtype, device=losses[0].device)
        if grad_norm:
            scaled = [w * (L / (L.detach().abs() + 1e-6)) for w, L in zip(ws, losses)]
        else:
            scaled = [w * L for w, L in zip(ws, losses)]
        return sum(scaled)
