"""The three heads — copied & edited from world_model.py (Brain decoder, InverseHead, ClickHead).

All read the shared trunk features (dim d), not raw object latents.
  ForwardHead  : (current trunk feats, action[, clicked-node latent]) -> NEXT latent state z_next + alive.
                 Built from Brain.dec + Brain.alive + act embedding + click_enc. Never sees z_next (masking).
  InverseHead  : (current feats, next feats) -> action-type logits. Never sees the action (masking).
  ClickHead    : (current feats, next feats, edges) -> which NODE was clicked (node localized via its edges).
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401

import torch
import torch.nn as nn

from transform_catalyst.data_adapter import obj_dim, EDGE_DIM

CLICK = 6                                                          # action id for click (matches world_model.py / cfg)


class ForwardHead(nn.Module):
    """Predict the NEXT latent state z_next DIRECTLY (not Δz) from the current trunk features + action.
    For CLICK, also conditions on the clicked node's latent h_click (click-as-features). Carries Brain's
    `alive` head so births/deaths (a slot becoming alive/dead next) are represented."""

    def __init__(self, cfg, d=64, od=None):
        super().__init__()
        self.od = od if od is not None else obj_dim(cfg); self.d = d   # od>obj_dim ⇒ predict the augmented latent
        self.act = nn.Embedding(cfg.n_actions, d)                  # action-TYPE embedding
        self.click_enc = nn.Linear(self.od, d)                    # clicked object's latent -> action cond
        self.dec = nn.Sequential(nn.Linear(self.od + 3 * d, d), nn.GELU(), nn.Linear(d, self.od))
        self.alive = nn.Linear(d, 1)                              # P(node live in next state)

    def forward(self, Z_cur, H_cur, pooled_cur, a, h_click=None):
        B, N, _ = Z_cur.shape
        act = self.act(a)                                          # [B,d]
        if h_click is not None:
            act = act + (a == CLICK).float()[:, None] * self.click_enc(h_click)
        actb = act[:, None, :].expand(B, N, self.d)
        pooledb = pooled_cur[:, None, :].expand(B, N, self.d)
        z_next = self.dec(torch.cat([Z_cur, H_cur, pooledb, actb], -1))      # [B,N,od] full next latent
        alive = self.alive(H_cur + actb).squeeze(-1)                          # [B,N]
        return z_next, alive


class InverseHead(nn.Module):
    """(current feats, next feats) -> action-type logits. Action is NOT an input (masked).
    Adapted from world_model.InverseHead with trunk features H in place of raw latents Z."""

    def __init__(self, cfg, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.out = nn.Linear(d, cfg.n_actions)

    def forward(self, H_cur, H_next, mask):
        h = self.net(torch.cat([H_cur, H_next - H_cur], -1))
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.out(pooled)


class ClickHead(nn.Module):
    """Click localization: which NODE (identified by its edges) was clicked, from (current, next) trunk
    features + edges. Adapted from world_model.ClickHead (node head; edge structure summarized via esum)."""

    def __init__(self, cfg, d=64):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(2 * d + EDGE_DIM, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, H_cur, H_next, edges, mask):
        esum = (edges * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None].clamp(min=1.0)   # [B,N,EDGE_DIM]
        feat = torch.cat([H_cur, H_next - H_cur, esum], -1)
        return self.node(feat).squeeze(-1).masked_fill(~mask.bool(), -1e9)                            # [B,N]
