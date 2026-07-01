"""MODULE 2 — transverse-residual head (gated short-circuit; NO μ/σ; predict δ, not next-state).

HARD CONSTRAINT 3: the target is δ = P_⊥(z' − z) (≈0 for unchanged objects), never ẑ'≈z'. The
"predict next state" step is the GATED short-circuit z_next = (1−g)·z + g·c (bounded — nothing to
inflate, no variance head to game), and δ̂ = P_⊥(z_next − z) is the genuine-change prediction.

Generators (Module 1) are trained by the equivariance loss against THIS node-level gated forward
(not a pooled proxy). Target P_⊥ is detached so the head regresses to the current transverse while
the generators are shaped by equivariance — the target doesn't chase its own moving frame.

pid pairing is SOFT EVIDENCE for forming the δ target on matched objects; births/deaths simply
aren't in `matched`, so the head is supervised only where correspondence is real (graceful).

Gates:
  * residual beats persistence on CHANGED objects: rel = ‖δ̂−δ‖²/‖δ‖² < 1 (persistence ⇒ δ̂=0 ⇒ 1)
  * trained-τ concentration sharpens vs the init-generator baseline (the decisive premise test)

  PYTORCH_ENABLE_MPS_FALLBACK=1 ../../../.venv/bin/python -m transform_catalyst.residual_head --steps 800 --device mps
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .repro import seed_all
from .data_adapter import load_pairs, obj_dim
from .lie_nodes import NodeLie


class ResidualHead(nn.Module):
    def __init__(self, cfg, d: int = 24, n_heads: int = 2, m: int = 6):
        super().__init__()
        od = obj_dim(cfg)
        self.enc = nn.Linear(od, d)
        self.act = nn.Embedding(cfg.n_actions, d)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.dec = nn.Linear(d, od)                       # "commit" content (object-latent space)
        self.Wg = nn.Linear(2 * od, od)                   # stay/commit gate (bounded; no σ to game)
        self.lie = NodeLie(od, m=m)

    def forward_next(self, Z, mask, a):
        """Gated short-circuit next-state prediction z_next [B,N,od]."""
        h = self.enc(Z) + self.act(a)[:, None, :]
        q, k = self.Wq(h), self.Wk(h)
        v = self.Wv(self.enc(Z))
        sc = (q @ k.transpose(-1, -2)) / h.shape[-1] ** 0.5
        sc = sc.masked_fill(~mask[:, None, :].bool(), -1e9)
        ztil = sc.softmax(-1) @ v
        c = self.dec(ztil)
        g = torch.sigmoid(self.Wg(torch.cat([Z, c], -1)))
        return (1 - g) * Z + g * c

    def residual(self, Z, mask, a):
        """δ̂ = P_⊥(z_next − z) — the genuine-change prediction."""
        z_next = self.forward_next(Z, mask, a)
        return self.lie.transverse(Z, z_next - Z)


def _concentration(per_obj, matched):
    c = per_obj * matched
    tot = c.sum(1); top1 = c.max(dim=1).values; sel = tot > 1e-6
    return (top1[sel] / tot[sel].clamp(min=1e-9)) if bool(sel.any()) else c.new_zeros(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--files", type=int, default=40)
    ap.add_argument("--lam-equiv", type=float, default=1.0)
    args = ap.parse_args()

    seed_all(0)
    dev = torch.device(args.device)
    data, cfg = load_pairs(n_files=args.files)
    Zp = torch.from_numpy(data["Zp"]).to(dev); Zc = torch.from_numpy(data["Zc"]).to(dev)
    Mp = torch.from_numpy(data["Mp"]).to(dev); Mc = torch.from_numpy(data["Mc"]).to(dev)
    A = torch.from_numpy(data["a"]).to(dev)
    T = Zp.shape[0]
    head = ResidualHead(cfg).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    print(f"[mod2] {T} transitions obj_dim={obj_dim(cfg)} params={sum(p.numel() for p in head.parameters())} "
          f"| training {args.steps} on {dev}", flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, T, (args.batch,), device=dev)
        z, zc, mp, mc, a = Zp[idx], Zc[idx], Mp[idx], Mc[idx], A[idx]
        matched = (mp & mc).unsqueeze(-1).float()
        dhat = head.residual(z, mp, a)
        with torch.no_grad():
            dtgt = head.lie.transverse(z, zc - z)         # detached target (current transverse frame)
        l_delta = ((dhat - dtgt) ** 2 * matched).sum() / (matched.sum().clamp(min=1.0) * z.shape[-1])
        ab, rl = head.lie.equivariance(z, lambda zz: head.forward_next(zz, mp, a), mask=mp.float())
        loss = l_delta + args.lam_equiv * ab.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        head.lie.update_equiv(rl)
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                # residual beats persistence on CHANGED matched objects
                dn = (dtgt ** 2).sum(-1)                  # ‖δ‖² per object
                ch = (dn > dn[(mp & mc)].median()) & (mp & mc)
                num = (((dhat - dtgt) ** 2).sum(-1) * ch).sum()
                den = (dn * ch).sum().clamp(min=1e-9)
                rel = float(num / den)
                tau = head.lie.tau(z, zc - z)
                conc = float(_concentration(tau, (mp & mc).float()).mean())
                sps = step / (time.time() - t0)
            print(f"[{step:4d}/{args.steps}] {sps:4.1f}it/s | Lδ={float(l_delta):.4f} equiv={float(ab.mean()):.4f} "
                  f"| residual rel-err(changed)={rel:.3f} (<1 beats persistence) | trained-τ conc={conc:.2f} "
                  f"| gate={[round(g,3) for g in head.lie.gate().tolist()]}", flush=True)
    print(f"[mod2] done {time.time()-t0:.0f}s. Decisive: trained-τ conc vs init 0.45 (Gate 1); "
          f"residual rel-err <1 = predicts genuine change.", flush=True)


if __name__ == "__main__":
    main()
