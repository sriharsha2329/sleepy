"""WorldModel — the shared trunk + the three heads wired together.

  predict_next(current state, action)         -> (z_next, alive)     [forward; next state masked from input]
  predict_action(current, next)               -> action-type logits  [inverse; action masked from input]
  predict_click(current, next)                -> node logits          [click-on-node; edges from current state]

`encode(Z,M,E)` runs the shared trunk on ONE state. Current state = (Zp,Mp,EFp); next state = (Zc,Mc,EFc).
No wake/sleep, no brain pool. `python -m curvature_wm.model.world_model` self-tests param_count (~100K).
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401

import torch
import torch.nn as nn

from curvature_wm_model.model.trunk import GraphTransformerTrunk
from curvature_wm_model.model.heads import ForwardHead, InverseHead, ClickHead


class WorldModel(nn.Module):
    def __init__(self, cfg, d=64, n_blocks=2, max_nodes=24, ffn_mult=2, in_dim=None):
        super().__init__()
        self.cfg = cfg
        self.trunk = GraphTransformerTrunk(cfg, d, n_blocks, max_nodes, ffn_mult, in_dim=in_dim)
        self.forward_head = ForwardHead(cfg, d, od=in_dim)        # predicts the (augmented) latent when in_dim>od
        self.inverse_head = InverseHead(cfg, d)
        self.click_head = ClickHead(cfg, d)

    def encode(self, Z, M, E):
        """Shared trunk on one state -> (per-node H[B,N,d], pooled[B,d])."""
        return self.trunk(Z, M.float() if M.dtype != torch.float32 else M, E)

    def predict_next(self, Zp, Mp, EFp, a, h_click=None):
        H, pooled = self.encode(Zp, Mp, EFp)
        return self.forward_head(Zp, H, pooled, a, h_click)          # (z_next [B,N,od], alive [B,N])

    def predict_action(self, Zp, Mp, EFp, Zc, Mc, EFc):
        Hc, _ = self.encode(Zp, Mp, EFp)
        Hn, _ = self.encode(Zc, Mc, EFc)
        union = (Mp.bool() | Mc.bool()).float()
        return self.inverse_head(Hc, Hn, union)                     # [B,n_actions]

    def predict_click(self, Zp, Mp, EFp, Zc, Mc, EFc):
        Hc, _ = self.encode(Zp, Mp, EFp)
        Hn, _ = self.encode(Zc, Mc, EFc)
        return self.click_head(Hc, Hn, EFp, Mp.float())             # [B,N] over current-state nodes

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        base = n(self.trunk) + n(self.forward_head) + n(self.inverse_head) + n(self.click_head)
        return {"trunk": n(self.trunk), "forward": n(self.forward_head), "inverse": n(self.inverse_head),
                "click": n(self.click_head), "base_total": base, "total": self.param_count()}


if __name__ == "__main__":                                          # smoke test: shapes + param budget
    from config import Config
    cfg = Config()
    m = WorldModel(cfg)
    bd = m.param_breakdown()
    print("param breakdown:", {k: f"{v:,}" for k, v in bd.items()})
    B, N, od = 8, 32, m.trunk.od
    Zp = torch.randn(B, N, od); Zc = torch.randn(B, N, od)
    Mp = torch.zeros(B, N, dtype=torch.bool); Mp[:, :6] = True
    Mc = torch.zeros(B, N, dtype=torch.bool); Mc[:, :7] = True       # a birth at slot 6
    from transform_catalyst.data_adapter import EDGE_DIM
    EFp = torch.randn(B, N, N, EDGE_DIM); EFc = torch.randn(B, N, N, EDGE_DIM)
    a = torch.randint(0, cfg.n_actions, (B,))
    h_click = Zp[torch.arange(B), 0]
    zn, alive = m.predict_next(Zp, Mp, EFp, a, h_click)
    al = m.predict_action(Zp, Mp, EFp, Zc, Mc, EFc)
    cl = m.predict_click(Zp, Mp, EFp, Zc, Mc, EFc)
    print(f"z_next={tuple(zn.shape)} alive={tuple(alive.shape)} action={tuple(al.shape)} click={tuple(cl.shape)}")
    (zn.pow(2).mean() + al.pow(2).mean() + cl.clamp(-10, 10).pow(2).mean()).backward()
    print("backward OK; total params =", f"{bd['total']:,}")
