"""policy_head.py — the policy-prior head model π(a | st): a single mini-transformer over the shared trunk's
per-node features H(st) fused with the Mahalanobis position block (st+1 masked). Outputs the action-type prior
and the click-target (per-node) prior. This is the ONLY new model component; the trunk + M2 heads are reused.

Model + TRAINING now live together in curvature_wm/model/ (trained by model/train.py --stage policy).
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401

import torch
import torch.nn as nn

CLICK = 6                                                            # action id for click (matches heads.py / cfg)


class PolicyPriorHead(nn.Module):
    """mini-transformer( [ H(st) ⊕ pos ] ) -> (action logits [B,n_actions], click logits [B,N]).
    `pos` is the per-node Mahalanobis position+centroid block (un-squashed); st+1 never enters."""

    def __init__(self, cfg, d=64, pos_dim=4, n_heads=4, ffn_mult=2):
        super().__init__()
        self.inp = nn.Linear(d + pos_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=ffn_mult * d,
                                           batch_first=True, activation="gelu", dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, num_layers=1)        # "a single mini transformer is enough"
        self.act_out = nn.Linear(d, cfg.n_actions)
        self.click_out = nn.Linear(d, 1)

    def forward(self, H_cur, pos, mask):
        pad = ~mask.bool()
        x = self.inp(torch.cat([H_cur, pos], -1))
        h = self.tf(x, src_key_padding_mask=pad)
        m = mask.float()[..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        a_logits = self.act_out(pooled)
        click_logits = self.click_out(h).squeeze(-1).masked_fill(pad, -1e9)
        return a_logits, click_logits

class ValueHead(nn.Module):
    """V(s): a scalar state-value baseline on the SHARED FROZEN trunk's pooled per-node features (+ pos block),
    added back for the v3 ONLINE MC policy-gradient (advantage A = G - V(s)). Trained online via MC regression
    V->G; the trunk + forward/inverse/click/policy heads stay as-is. A small MLP (no transformer) is enough."""

    def __init__(self, cfg, d=64, pos_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + pos_dim, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, H_cur, pos, mask):
        m = mask.float()[..., None]
        feat = torch.cat([H_cur, pos], -1)
        pooled = (feat * m).sum(1) / m.sum(1).clamp(min=1.0)        # mean-pool the valid nodes
        return self.net(pooled).squeeze(-1)                         # [B]

# NOTE: ActionValueHead (Q) is intentionally NOT added — v3 uses the MC return as the Q estimate (no Q-maximization).
