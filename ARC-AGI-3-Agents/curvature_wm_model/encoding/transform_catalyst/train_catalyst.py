"""INTEGRATED catalyst trainer — causal, interventional, model-based RL in ONE growing pool, run
together under the federation's own wake/sleep + adenosine + GROWTH methodology.

  Module 1 (Hodge δ) + 2 (residual δ): SHARED discrete-Hodge frame — genuine change = (I−D⁻¹W)Δz,
                                       persistence (global symmetry) removed deterministically (no gate)
  Module 3 (reverse Hodge potential) : φ from L₀φ=div f (graph Poisson) + rotational fraction ρ
  Module 4 (CLICK catalyst)          : node + attached edges + φ (where predicted change concentrates)
  Module 5 (TScore)                  : replay priority / decision bias
  Module 6 (inverse (sₜ,sₜ₊₁)→aₜ)    : the DIRECT action readout (causal RL)
  Epistemic guard                    : routed-brain disagreement on ŝₜ₊₁ — down-weight intervals
                                       the pool is extrapolating (no acting on hallucinations)

Decision (inference) is interventional: for each action a, forward predicts ŝₜ₊₁(a); the inverse
scores consistency p(a | sₜ, ŝₜ₊₁(a)); the epistemic guard weights by confidence. Structural
signals never touch a value/TD target (constraint 1).

  PYTORCH_ENABLE_MPS_FALLBACK=1 ../../../.venv/bin/python -m transform_catalyst.train_catalyst --steps 4000 --device mps
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .repro import seed_all
from .data_adapter import load_pairs, obj_dim, EDGE_DIM, N_RELS
from .hodge_nodes import NodeHodge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from federation.growth import maybe_grow                          # REUSE the real growth controller


# ----------------------------------------------------------------- Modules 1+2: catalyst-brain
class CatalystBrain(nn.Module):
    """UNIFIED brain ('do all'): per-object FORWARD (z_next, the gradient part) + pooled CLAIMS
    (γ delta-kinds / prog / mag) + value Q(s,a) (MC, action-from-current-state). Curl (Lie) +
    harmonic are SHARED at the pool. The claim/value targets are what give each brain something
    to specialize on (so they stop collapsing to rank-1 and the pool actually grows)."""
    def __init__(self, cfg, d: int = 24, seed: int = 0):
        super().__init__()
        od = obj_dim(cfg); self.d = d; n_dk = getattr(cfg, "n_delta_kinds", 12)
        self.enc = nn.Linear(od, d); self.act = nn.Embedding(cfg.n_actions, d)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.dec = nn.Linear(d, od); self.Wg = nn.Linear(2 * od, od)
        self.h_gamma = nn.Linear(d, n_dk); self.h_prog = nn.Linear(d, 1)
        self.h_mag = nn.Linear(d, 2); self.h_value = nn.Linear(d, 1)   # value Q(s,a) — MC return
        g = torch.Generator().manual_seed(1234 + seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.add_(0.01 * torch.randn(p.shape, generator=g))

    def forward(self, Z, mask, a):
        h = self.enc(Z) + self.act(a)[:, None, :]                 # action-conditioned node tokens
        q, k = self.Wq(h), self.Wk(h); v = self.Wv(self.enc(Z))
        sc = (q @ k.transpose(-1, -2)) / h.shape[-1] ** 0.5
        sc = sc.masked_fill(~mask[:, None, :].bool(), -1e9)
        ztil = sc.softmax(-1) @ v
        c = self.dec(ztil); g = torch.sigmoid(self.Wg(torch.cat([Z, c], -1)))
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)   # claim readout
        mag = self.h_mag(pooled)
        # "feat" = the per-object hidden h (diverse across objects) — NOT the attention output ztil,
        # which collapses to rank-1 (uniform attention) and falsely trips the S2 erank-collapse alarm,
        # resetting every brain before it can ever become a split-candidate (the no-growth bug).
        return {"z_next": (1 - g) * Z + g * c, "feat": h,
                "gamma": self.h_gamma(pooled), "prog": self.h_prog(pooled).squeeze(-1),
                "mag_mean": mag[:, 0], "mag_logvar": mag[:, 1].clamp(-6.0, 6.0),
                "value": self.h_value(pooled).squeeze(-1)}


class CatalystRouter(nn.Module):
    def __init__(self, cfg, cap=64, k=2, e0=8):
        super().__init__()
        self.cap, self.k, self.n = cap, k, e0
        self.W = nn.Parameter(torch.randn(cap, obj_dim(cfg)) * 0.02)

    def forward(self, Z, mask):
        s = (Z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        logits = s @ self.W[:self.n].t()
        k = min(self.k, self.n)
        g, idx = torch.softmax(logits, -1).topk(k, dim=-1)
        return idx, g / g.sum(-1, keepdim=True)

    def add_brain(self, parent_idx=None):
        with torch.no_grad():
            if self.n < self.cap and parent_idx is not None:
                self.W[self.n] = self.W[parent_idx] + 0.02 * torch.randn_like(self.W[0])
        self.n = min(self.n + 1, self.cap)

    def remove_brain(self, j):
        with torch.no_grad():
            self.W.data = torch.cat([self.W.data[:j], self.W.data[j + 1:], self.W.data[j:j + 1]], 0)
        self.n -= 1


class CatalystPool(nn.Module):
    def __init__(self, cfg, e0=8, cap=64, k=2, m=6):
        super().__init__()
        self.brains = nn.ModuleList([CatalystBrain(cfg, seed=i) for i in range(e0)])
        self.router = CatalystRouter(cfg, cap, k, e0)
        self.hodge = NodeHodge(obj_dim(cfg), geo_off=N_RELS)  # SHARED discrete-Hodge frame (Module 1)
        dd = self.brains[0].d
        # CLICK = node + ATTACHED-EDGE summary + reverse-Hodge potential φ (where the predicted change concentrates)
        self.click_head = nn.Sequential(nn.Linear(dd + EDGE_DIM + 1, dd), nn.GELU(), nn.Linear(dd, 1))

    def forward(self, Z, mask, a, edges=None):
        """STACKED bmm over all alive brains (no Python loop) — dense [E,B,N,·], peak ~17MB at cap=64.
        Each brain's Linears are stacked into [E,·] and run as batched einsums; routing selects the
        top-k per row and gate-blends. Memory-bounded by E·B·N·N (NOT the OOM-prone broadcast)."""
        idx, gates = self.router(Z, mask)                     # [B,k]
        B, N, od = Z.shape; E = len(self.brains); d = self.brains[0].d; bs = self.brains
        SW = lambda att: torch.stack([getattr(b, att).weight for b in bs])   # [E,out,in] (one stack/layer)
        SB = lambda att: torch.stack([getattr(b, att).bias for b in bs])     # [E,out]
        he = torch.einsum('bno,edo->ebnd', Z, SW('enc')) + SB('enc')[:, None, None, :]   # enc(Z) [E,B,N,d]
        ua = torch.stack([b.act.weight for b in bs])[:, a]                                # [E,B,d]
        h = he + ua[:, :, None, :]
        q = torch.einsum('ebnd,efd->ebnf', h, SW('Wq')) + SB('Wq')[:, None, None, :]
        kk = torch.einsum('ebnd,efd->ebnf', h, SW('Wk')) + SB('Wk')[:, None, None, :]
        v = torch.einsum('ebnd,efd->ebnf', he, SW('Wv')) + SB('Wv')[:, None, None, :]
        sc = torch.einsum('ebnf,ebmf->ebnm', q, kk) / d ** 0.5
        sc = sc.masked_fill(~mask[None, :, None, :].bool(), -1e9)
        ztil = torch.einsum('ebnm,ebmf->ebnf', sc.softmax(-1), v)
        c = torch.einsum('ebnf,eof->ebno', ztil, SW('dec')) + SB('dec')[:, None, None, :]
        Ze = Z[None].expand(E, B, N, od)
        g = torch.sigmoid(torch.einsum('ebnx,eox->ebno', torch.cat([Ze, c], -1), SW('Wg')) + SB('Wg')[:, None, None, :])
        z_next = (1 - g) * Ze + g * c                                                     # [E,B,N,od]
        pooled = (h * mask[None, :, :, None]).sum(2) / mask.sum(1)[None, :, None].clamp(min=1.0)   # [E,B,d]
        gamma = torch.einsum('ebd,ekd->ebk', pooled, SW('h_gamma')) + SB('h_gamma')[:, None, :]
        prog = (torch.einsum('ebd,ekd->ebk', pooled, SW('h_prog')) + SB('h_prog')[:, None, :]).squeeze(-1)
        magh = torch.einsum('ebd,ekd->ebk', pooled, SW('h_mag')) + SB('h_mag')[:, None, :]   # [E,B,2]
        value = (torch.einsum('ebd,ekd->ebk', pooled, SW('h_value')) + SB('h_value')[:, None, :]).squeeze(-1)
        # ---- route-select top-k per row + gate-blend ----
        ar = torch.arange(B, device=Z.device)[:, None]
        def pick(x):                                          # x [E,B,...] -> blended [B,...] (+ the [B,k,...] sel)
            sel = x.transpose(0, 1)[ar, idx]                  # [B,k,...]
            gsh = gates.view(B, idx.shape[1], *([1] * (sel.dim() - 2)))
            return (gsh * sel).sum(1), sel
        znext, zsel = pick(z_next); feat, _ = pick(h)
        gamma, _ = pick(gamma); prog, _ = pick(prog)
        mm, _ = pick(magh[..., 0]); ml, _ = pick(magh[..., 1]); val, _ = pick(value)
        dz = znext - Z
        dhat = self.hodge.transverse(Z, dz, edges, mask)                  # genuine change (persistence removed)
        epi = (zsel.var(dim=1).mean(-1) * mask).sum(1) / mask.sum(1).clamp(min=1.0)        # ensemble disagreement
        counts = torch.bincount(idx.flatten(), minlength=E).float()
        # ---- CLICK head: node + ATTACHED-EDGE summary + reverse-Hodge potential φ(predicted change) ----
        click_logit = None
        if edges is not None:
            esum = (edges * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None].clamp(min=1.0)  # [B,N,edge]
            with torch.no_grad():                                         # φ is a deterministic geometric feature
                phi, _ = self.hodge.potential(Z, dz, edges, mask)         # [B,N] (uses predicted change ⇒ deployable)
            click_in = torch.cat([feat, esum, phi[..., None]], -1)
            click_logit = self.click_head(click_in).squeeze(-1).masked_fill(~mask.bool(), -1e9)
        return {"dhat": dhat, "z_next": znext, "feat": feat, "epi": epi, "counts": counts,
                "gamma": gamma, "prog": prog, "mag_mean": mm, "mag_logvar": ml.clamp(-6.0, 6.0),
                "value": val, "click_logit": click_logit, "idx": idx, "gates": gates}

    def add_brain(self, init_from, noise=0.01):
        child = copy.deepcopy(self.brains[init_from])
        with torch.no_grad():
            for p in child.parameters():
                p.add_(noise * torch.randn_like(p))
        self.brains.append(child)

    def remove_brain(self, j):
        self.brains = nn.ModuleList([b for i, b in enumerate(self.brains) if i != j])


# ----------------------------------------------------------------- Module 6: inverse head
class InverseHead(nn.Module):
    """Read aₜ from the transition (sₜ, sₜ₊₁) — the direct action readout for causal RL. Trained on
    real (Zp,Zc,a); at inference fed (sₜ, ŝₜ₊₁(a)) per candidate action for interventional scoring."""

    def __init__(self, cfg, d=32):
        super().__init__()
        od = obj_dim(cfg)
        self.net = nn.Sequential(nn.Linear(2 * od, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.out = nn.Linear(d, cfg.n_actions)

    def forward(self, Zp, Zn, mask):
        h = self.net(torch.cat([Zp, Zn - Zp], -1))                  # current + transformation
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.out(pooled)


class ClickInverseHead(nn.Module):
    """Read the CLICKED NODE from the transition (zₜ, zₜ₊₁) — the click analogue of InverseHead.
    PER-NODE (not pooled): for each node it reads its own state zₜ, its change zₜ₊₁−zₜ, and its
    ATTACHED-EDGE summary → a click logit. The clicked node is the one that CHANGED, which the
    transition reveals (the click analogue of reading the action from the change). Offline upper
    bound — uses observed zₜ₊₁, exactly like the action inverse; deploy uses the s-only click head."""

    def __init__(self, cfg, d=48):
        super().__init__()
        od = obj_dim(cfg)
        self.net = nn.Sequential(nn.Linear(2 * od + EDGE_DIM, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.out = nn.Linear(d, 1)

    def forward(self, Zp, Zn, edges, mask):
        esum = (edges * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None].clamp(min=1.0)  # [B,N,edge]
        h = self.net(torch.cat([Zp, Zn - Zp, esum], -1))                                            # per-node + edges
        return self.out(h).squeeze(-1).masked_fill(~mask.bool(), -1e9)                              # [B,N]


class ClickDetectHead(nn.Module):
    """'Should I click?' detector — click vs NOT-click, from the CURRENT STATE only (deployable, no s′).
    Per-node clickability on [zₜ, attached-edge summary] → max+mean pool → one click-vs-not logit. A
    BINARY, class-balanced decision, so it will actually predict 'click' (unlike the multi-class action
    inverse, where click loses the argmax to moves → 0.7% recall). ADDITIVE: separate head + separate
    BCE; touches NONE of the Hodge machinery (δ / highpass retrieval / nav-Φ / persist-rot)."""

    def __init__(self, cfg, d=48):
        super().__init__()
        od = obj_dim(cfg)
        self.node = nn.Sequential(nn.Linear(od + EDGE_DIM, d), nn.GELU(), nn.Linear(d, 1))
        self.out = nn.Linear(2, 1)                                   # combine max + mean clickability

    def forward(self, Zp, edges, mask):
        esum = (edges * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None].clamp(min=1.0)
        s = self.node(torch.cat([Zp, esum], -1)).squeeze(-1)        # [B,N] per-node clickability
        mf = mask.float()
        mx = s.masked_fill(~mask.bool(), -1e9).max(1).values        # strongest clickable node
        mn = (s * mf).sum(1) / mf.sum(1).clamp(min=1.0)             # mean over live nodes
        return self.out(torch.stack([mx, mn], -1)).squeeze(-1)      # [B] P(click) logit


# ----------------------------------------------------------------- S1-S5 dashboard (compact)
class CatalystDashboard:
    def __init__(self, n):
        self.hist = [deque(maxlen=8) for _ in range(n)]
        self.erank = [1.0] * n; self.erank_peak = [0.0] * n; self.var = [0.0] * n; self.count = [0] * n

    def grow(self, j):
        self.hist.append(deque(self.hist[j], maxlen=8)); self.erank.append(self.erank[j])
        self.erank_peak.append(0.0); self.var.append(self.var[j]); self.count.append(0)

    def drop(self, j):
        for L in (self.hist, self.erank, self.erank_peak, self.var, self.count):
            L.pop(j)

    def verdict(self, j):
        h = list(self.hist[j])
        plateau = h[-1] if h else 0.0                          # current held-out risk (S1) — no manual window
        erank, peak = self.erank[j], self.erank_peak[j]
        declining = peak > 0 and erank < 0.8 * peak
        v = {"s1_plateau": plateau, "s3_min_cos": -self.var[j], "s2_erank": erank, "n_j": self.count[j] * 50}
        if self.count[j] <= 1:
            v["verdict"] = "starved"
        elif (erank < 0.10 * 24) or (declining and erank < 0.3 * 24):
            v["verdict"] = "plasticity-reset"
        elif plateau > 0 and self.var[j] > 0:                  # overloaded: risk × interference (no sleep-count rule)
            v["verdict"] = "split-candidate"                  # maybe_grow ranks by plateau×interference, budgeted
        else:
            v["verdict"] = "healthy"
        return v


def save_checkpoint(path, pool, inverse, click_inverse, click_detect, M_click, cfg, args, step, sleeps, metrics):
    """Persist the full catalyst (pool+inverse+click_inverse+click_detect+click memory) + config + metrics.
    The pool grows, so n_brains/router_n are stored to rebuild the right-sized CatalystPool before
    load_state_dict. Writes a human-readable <path>.json sidecar so the weights are self-documenting."""
    import json
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg_fields = {k: getattr(cfg, k) for k in
                  ("n_actions", "n_colors", "n_stab", "n_areabin", "grid", "n_delta_kinds")
                  if hasattr(cfg, k)}
    ckpt = {"pool": pool.state_dict(), "inverse": inverse.state_dict(),
            "click_inverse": click_inverse.state_dict(), "click_detect": click_detect.state_dict(),
            "M_click": M_click.detach().cpu(),
            "n_brains": len(pool.brains), "router_n": int(pool.router.n),
            "obj_dim": obj_dim(cfg), "cfg": cfg_fields, "args": vars(args),
            "step": step, "sleeps": sleeps, "metrics": metrics}
    torch.save(ckpt, p)
    side = {"step": step, "sleeps": sleeps, "n_brains": len(pool.brains),
            "obj_dim": obj_dim(cfg), "cfg": cfg_fields, "args": vars(args), "metrics": metrics}
    p.with_suffix(".json").write_text(json.dumps(side, indent=2))
    return str(p)


def _conc(per_obj, matched):
    c = per_obj * matched; tot = c.sum(1); top1 = c.max(dim=1).values; sel = tot > 1e-6
    return float((top1[sel] / tot[sel].clamp(min=1e-9)).mean()) if bool(sel.any()) else float("nan")


# ----------------------------------------------------------------- training loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps"); ap.add_argument("--files", type=int, default=40)
    ap.add_argument("--e0", type=int, default=8); ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--aden-target", type=float, default=500.0)
    ap.add_argument("--aden-min", type=int, default=150); ap.add_argument("--aden-max", type=int, default=1200)
    ap.add_argument("--max-min", type=float, default=30.0)
    ap.add_argument("--save", default="runs/catalyst_clickqre.pt",
                    help="checkpoint path (rolling each sleep + final); '' to disable")
    args = ap.parse_args()

    seed_all(0); dev = torch.device(args.device)
    data, cfg = load_pairs(n_files=args.files)
    Zp = torch.from_numpy(data["Zp"]).to(dev); Zc = torch.from_numpy(data["Zc"]).to(dev)
    Mp = torch.from_numpy(data["Mp"]).to(dev); Mc = torch.from_numpy(data["Mc"]).to(dev)
    A = torch.from_numpy(data["a"]).to(dev); T = Zp.shape[0]
    GAM = torch.from_numpy(data["gamma_t"]).to(dev); PROG = torch.from_numpy(data["prog"]).to(dev)
    MAG = torch.from_numpy(data["mag"]).to(dev); GRET = torch.from_numpy(data["Gret"]).to(dev)
    EFp = torch.from_numpy(data["EFp"])                              # CPU [T,N,N,edge] — moved per-batch (mem-light)
    CT = torch.from_numpy(data["ct"]).to(dev)                        # clicked slot per transition (-1 if not click)
    PHI = torch.from_numpy(data["phi"]).to(dev)                      # dense progress potential Φ (trajectory position)
    cut = max(256, T // 10)

    # CLICKED-NODE FEATURE MEMORY (retrieval leg): prev-frame latents of the node humans actually clicked,
    # from the MEMORY region [cut,T) only (no eval leak). Clicks have no stable node-id across states, so
    # retrieval matches by FEATURE — "which live node looks like the ones humans click".
    _cmask = CT.clone(); _cmask[:cut] = -1                           # restrict to memory region
    _cidx = (_cmask >= 0).nonzero(as_tuple=True)[0]
    if _cidx.numel() > 3000:
        _cidx = _cidx[torch.randperm(_cidx.numel(), device=dev)[:3000]]
    M_click = (Zp[_cidx, CT[_cidx]] if _cidx.numel() > 0
               else torch.zeros(1, obj_dim(cfg), device=dev))       # [Nc, od]

    # class-balanced inverse CE + majority baseline (ACTION judged vs majority, not random)
    cnt = torch.bincount(A[cut:], minlength=cfg.n_actions).float() + 1.0
    freq = cnt / cnt.sum(); class_w = (1.0 / freq.sqrt()); class_w = (class_w / class_w.mean()).to(dev)
    maj = torch.bincount(A[:cut], minlength=cfg.n_actions).float()
    maj_top1 = float(100 * maj.max() / maj.sum())

    pool = CatalystPool(cfg, e0=args.e0, cap=args.cap).to(dev)
    inverse = InverseHead(cfg).to(dev)
    click_inverse = ClickInverseHead(cfg).to(dev)                  # (zₜ,zₜ₊₁)+edges -> clicked NODE
    click_detect = ClickDetectHead(cfg).to(dev)                    # state -> P(click vs not), deployable
    p_click = float((A[cut:] == 6).float().mean())                 # click base rate (memory region)
    det_pos_w = torch.tensor((1.0 - p_click) / max(p_click, 1e-6), device=dev)  # balance the BCE
    dash = CatalystDashboard(args.e0)
    opt = torch.optim.Adam(list(pool.parameters()) + list(inverse.parameters())
                           + list(click_inverse.parameters()) + list(click_detect.parameters()), lr=2e-3)
    print(f"[catalyst] {T} trans obj_dim={obj_dim(cfg)} E0={args.e0} cap={args.cap} dev={dev} | "
          f"majority ACTION top1={maj_top1:.1f}% | legs: inv+L2+Hodge+nav(retrieval do(a)), NO value-Q; "
          f"CLICK QRE: s-only+(s,s')-inv+retrieval, detector(click/not), joint(action×node); "
          f"wake/sleep+growth", flush=True)

    aden = ema = 0.0; since = sleeps = 0; t0 = time.time(); metrics = {}
    count_acc = torch.zeros(len(pool.brains), device=dev)                     # on-device routed counts (S5)
    for step in range(1, args.steps + 1):
        if (time.time() - t0) > args.max_min * 60:
            break
        idx = torch.randint(cut, T, (args.batch,), device=dev)
        z, zc, mp, mc, a = Zp[idx], Zc[idx], Mp[idx], Mc[idx], A[idx]
        edges = EFp[idx.cpu()].to(dev)                                                  # CPU index -> MPS
        out = pool(z, mp, a, edges=edges); matched = (mp & mc).unsqueeze(-1).float()
        with torch.no_grad():
            dtgt = pool.hodge.transverse(z, zc - z, edges, mp)            # discrete-Hodge genuine change
        gt, pgt, mgt = GAM[idx], PROG[idx], MAG[idx]
        l_delta = ((out["dhat"] - dtgt) ** 2 * matched).sum() / (matched.sum().clamp(min=1.0) * z.shape[-1])
        l_inv = F.cross_entropy(inverse(z, zc, (mp & mc).float()), a, weight=class_w)   # Module 6 on REAL transition
        l_gamma = F.binary_cross_entropy_with_logits(out["gamma"], gt)                  # γ delta-kinds
        l_prog = F.binary_cross_entropy_with_logits(out["prog"], pgt)                   # level-up
        l_mag = 0.5 * (torch.exp(-out["mag_logvar"]) * (out["mag_mean"] - mgt) ** 2 + out["mag_logvar"]).mean()
        clk = CT[idx]; isc = clk >= 0                                                   # CLICK transitions
        l_click = (F.cross_entropy(out["click_logit"][isc], clk[isc]) if bool(isc.any())
                   else out["click_logit"].sum() * 0.0)                                 # s-only: which node (node+edges)
        ci_logit = click_inverse(z, zc, edges, mp)                                      # (zₜ,zₜ₊₁)+edges -> node
        l_click_inv = (F.cross_entropy(ci_logit[isc], clk[isc]) if bool(isc.any())
                       else ci_logit.sum() * 0.0)                                       # click analogue of the inverse
        det_logit = click_detect(z, edges, mp)                                          # state -> P(click vs not)
        l_detect = F.binary_cross_entropy_with_logits(det_logit, (a == 6).float(), pos_weight=det_pos_w)
        # NO value-Q here at all — composition routes AROUND the churn-eroded value head (not through a
        # learned value). Progress lives in Φ, computed in SLEEP and READ at decision time (the nav leg).
        task = l_delta + l_inv + l_gamma + l_prog + l_mag + l_click + l_click_inv + l_detect
        opt.zero_grad(); task.backward(); opt.step()
        count_acc = count_acc + out["counts"]                                # on-device (no per-step .item())
        cur = float(l_delta.detach()); ema = 0.99 * ema + 0.01 * cur if ema else cur
        aden += cur / max(ema, 1e-6); since += 1
        if step % 100 == 0:                                                   # WAKE heartbeat
            sps = step / (time.time() - t0)
            print(f"[{step:5d}] wake {sps:4.1f}it/s | δ={float(l_delta):.4f} γ={float(l_gamma):.3f} "
                  f"clk={float(l_click):.3f} inv={float(l_inv):.3f} aden={aden:.0f}/{args.aden_target:.0f} "
                  f"brains={len(pool.brains)}", flush=True)
        if since >= args.aden_min and (aden >= args.aden_target or since >= args.aden_max):
            sleeps += 1
            dash.count = [int(x) for x in count_acc.tolist()]                # one sync (S5 counts) at sleep
            ev = slice(0, cut)
            with torch.no_grad():
                ze, zce, mpe, mce, ae = Zp[ev], Zc[ev], Mp[ev], Mc[ev], A[ev]
                edges_ev = EFp[ev].to(dev)
                oe = pool(ze, mpe, ae, edges=edges_ev)
                dte = pool.hodge.transverse(ze, zce - ze, edges_ev, mpe)      # discrete-Hodge genuine change
                me = (mpe & mce).unsqueeze(-1).float()
                for j in range(len(pool.brains)):
                    rows = (oe["idx"] == j).any(1)
                    if int(rows.sum()) > 0:
                        zj = pool.brains[j](ze[rows], mpe[rows], ae[rows])["z_next"]
                        pr = ((pool.hodge.transverse(ze[rows], zj - ze[rows], edges_ev[rows], mpe[rows]) - dte[rows]) ** 2 * me[rows]
                              ).sum((1, 2)) / (me[rows].sum((1, 2)).clamp(min=1.0) * ze.shape[-1])  # per-row δ
                        dash.hist[j].append(float(pr.mean()))                  # S1 risk
                        dash.var[j] = float(pr.std()) if pr.numel() > 1 else 0.0  # S3 interference (split signal)
                        f = oe["feat"][rows][mpe[rows].bool()]
                        if f.shape[0] > 2:
                            s = torch.linalg.svdvals((f - f.mean(0)).cpu())
                            p = s / s.sum().clamp(min=1e-9)
                            er = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
                            dash.erank[j] = er; dash.erank_peak[j] = max(dash.erank_peak[j], er)

            class _Med:
                def brain_kappa(self, j):
                    h = list(dash.hist[j]); return 1.0 / (1.0 + (float(np.mean(h)) if h else 1.0))
            before = len(pool.brains)
            logs = maybe_grow(pool, pool.router, dash, _Med(), max_brains=args.cap, min_child_n=1)
            for ln in logs:
                if ln.startswith("SPLIT brain"): dash.grow(int(ln.split()[2].rstrip(":")))
                elif ln.startswith("PRUNE brain"): dash.drop(int(ln.split()[2].rstrip(":")))
            count_acc = torch.zeros(len(pool.brains), device=dev)            # resize after growth
            # ---- Hodge diagnostics (replace the inert Lie gates): persistence share + rotational fraction ----
            with torch.no_grad():
                persist = pool.hodge.persistence_share(ze, zce - ze, edges_ev, mpe)   # ‖global symmetry‖²/‖Δz‖²
                _, rotf = pool.hodge.potential(ze, zce - ze, edges_ev, mpe)           # reverse-Hodge rotational frac
                rot = float(rotf.mean())
                tau = pool.hodge.tau(ze, zce - ze, edges_ev, mpe); conc = _conc(tau, me.squeeze(-1))
                dn = (dte ** 2).sum(-1); ch = (dn > dn[(mpe & mce)].median()) & (mpe & mce)
                rel = float((((oe["dhat"] - dte) ** 2).sum(-1) * ch).sum() / (dn * ch).sum().clamp(min=1e-9))
                # ===== ACTION — DO ALL methods as POLICIES (probabilities), scored by the likelihood
                # each puts on the HUMAN action vs majority. Human runs only, NO counterfactuals. =====
                def _summ(Z, M): return (Z * M[..., None]).sum(1) / M.sum(1, keepdim=True).clamp(min=1.0)
                def _hsumm(sel):                          # offset-invariant Hodge descriptor (highpass mean+|mean|)
                    parts = []
                    for s in range(0, sel.shape[0], 1024):
                        ii = sel[s:s + 1024]; mi = Mp[ii]
                        hp = pool.hodge.highpass(Zp[ii], EFp[ii.cpu()].to(dev), mi)
                        den = mi.sum(1, keepdim=True).clamp(min=1.0)
                        parts.append(torch.cat([(hp * mi[..., None]).sum(1) / den,
                                                (hp.abs() * mi[..., None]).sum(1) / den], -1))
                    return torch.cat(parts, 0)
                mperm = torch.randperm(T - cut, device=dev)[:4000] + cut
                tr_s = _summ(Zp[mperm], Mp[mperm]); tr_a = A[mperm]; tr_sh = _hsumm(mperm)
                sub = slice(0, min(256, cut)); zs, zcs, mps_, mcs, as_ = Zp[sub], Zc[sub], Mp[sub], Mc[sub], A[sub]
                ev_s = _summ(zs, mps_); B = zs.shape[0]; K = 24
                ev_sh = _hsumm(torch.arange(B, device=dev))
                pi_maj = (maj / maj.sum()).to(dev)[None].expand(B, -1)                       # majority distribution
                pi_inv = torch.softmax(inverse(zs, zcs, (mps_ & mcs).float()), -1)            # (1) inverse (s,s')->a
                def _retr_pol(dist):                                                          # k-NN human-action vote
                    nbr = dist.topk(K, largest=False).indices
                    sc = torch.zeros(B, cfg.n_actions, device=dev)
                    sc.scatter_add_(1, tr_a[nbr], torch.ones_like(tr_a[nbr], dtype=sc.dtype))
                    return sc / sc.sum(1, keepdim=True).clamp(min=1e-9)
                pi_l2 = _retr_pol(torch.cdist(ev_s, tr_s))                                    # (2) L2 retrieval
                pi_hodge = _retr_pol(torch.cdist(ev_sh, tr_sh))                               # (3) Hodge-highpass retrieval
                # (4) NAVIGATION — RETRIEVAL-CONDITIONED-ON-ACTION do(a) (the Lie-spirit approach, realized
                # via HODGE not the inert continuous Lie, and via RETRIEVAL not model sampling — per the
                # 'see similar human states instead of sampling' constraint): for each action a, take the
                # Hodge-invariant nearest human transitions that ACTUALLY TOOK a, and score a by their
                # realized progress (Φ + return). Action-DISCRIMINATIVE by construction (per-action
                # neighbour sets) — this fixes the model-forward collapse that pinned nav at uniform. No
                # forward model, no Lie generators, Hodge only for the symmetry-invariant matching.
                mem_score = PHI[mperm] + GRET[mperm]                                          # grounded outcome (dense Φ + return)
                gsub = mem_score.mean().expand(B, cfg.n_actions).clone()                      # prior for unseen actions
                for a_ in range(cfg.n_actions):
                    am = tr_a == a_; na = int(am.sum())
                    if na >= K:
                        nb = torch.cdist(ev_sh, tr_sh[am]).topk(K, largest=False).indices     # Hodge-nearest same-action
                        gsub[:, a_] = mem_score[am][nb].mean(1)
                    elif na > 0:
                        gsub[:, a_] = mem_score[am].mean()
                pi_nav = torch.softmax(gsub, -1)
                pi_nav = (1.0 - rot) * pi_nav + rot * (1.0 / cfg.n_actions)                   # gate on curl (rot)
                # Nash/QRE fusion, BASELINE-RELATIVE: weight ∝ max(0, κ_m − κ_majority), measured on a
                # held-out half. Parameter-free (majority is the natural baseline) and DO-NO-HARM — a
                # leg that can't beat majority gets weight 0, so adding weak legs never dilutes the fusion.
                pis = torch.stack([pi_inv, pi_l2, pi_hodge, pi_nav], 0)                       # [4,B,nA] (NO value head)
                M_ = pis.shape[0]; half = B // 2
                kap = torch.stack([pis[m, :half].gather(1, as_[:half, None]).mean() for m in range(M_)])
                kmaj = pi_maj[:half].gather(1, as_[:half, None]).mean()                       # majority baseline κ
                wr = (kap - kmaj).clamp(min=0.0)
                w = wr / wr.sum() if float(wr.sum()) > 1e-9 else kap / kap.sum().clamp(min=1e-9)
                pi_qre = (w[:, None, None] * pis).sum(0)
                def _lik(p, sl=slice(None)): return float(p[sl].gather(1, as_[sl, None]).mean())
                lik = {"maj": _lik(pi_maj), "inv": _lik(pi_inv),
                       "L2": _lik(pi_l2), "Hodge": _lik(pi_hodge), "nav": _lik(pi_nav),
                       "mean": _lik(pis.mean(0), slice(half, None)),
                       "QRE": _lik(pi_qre, slice(half, None))}                                # both on held-out half
                qre_w = {"inv": round(float(w[0]), 2), "L2": round(float(w[1]), 2),
                         "Hodge": round(float(w[2]), 2), "nav": round(float(w[3]), 2)}
                # ACTION t1/3/5 (federation-comparable) from the QRE-fused policy on the held-out half
                qh, ah = pi_qre[half:], as_[half:]
                ark = (qh > qh.gather(1, ah[:, None])).sum(1)
                a_top = [100 * float((ark < kk).float().mean()) for kk in (1, 3, 5)]
                # ===== CLICK — same multi-leg QRE/Nash treatment as ACTIONS, over NODES. Three legs:
                # (s) s-only head [deployable], (s,s') click-inverse [offline upper bound], and node-feature
                # RETRIEVAL ("which live node looks like the ones humans click"). Baseline-relative QRE with
                # CHANCE (1/#live) as the baseline (no majority node exists). Then JOINT-compose with the
                # action decision: P(action=click) × P(node | click). =====
                ctc = CT[ev]; isc = ctc >= 0
                if bool(isc.any()):
                    mci = mpe[isc]; cti = ctc[isc]; zci = ze[isc]; Nce = int(isc.sum())
                    cl = oe["click_logit"][isc]                                   # (1) s-only head
                    civ = click_inverse(ze, zce, edges_ev, mpe)[isc]             # (2) (s,s') inverse
                    flat = zci.reshape(-1, zci.shape[-1]); kk_ = min(K, M_click.shape[0]); parts = []
                    for s in range(0, flat.shape[0], 8192):                       # (3) node-feature retrieval (chunked)
                        parts.append(torch.cdist(flat[s:s + 8192], M_click).topk(kk_, largest=False).values.mean(1))
                    retr = (-torch.cat(parts)).reshape(zci.shape[0], zci.shape[1])
                    def _pn(lg): return torch.softmax(lg.masked_fill(~mci.bool(), -1e9), -1)
                    cpis = torch.stack([_pn(cl), _pn(civ), _pn(retr)], 0)         # [3,Nce,N] node distributions
                    chf = max(1, Nce // 2)
                    ckap = torch.stack([cpis[m, :chf].gather(1, cti[:chf, None]).mean() for m in range(3)])
                    cchance = (1.0 / mci[:chf].sum(1).float().clamp(min=1.0)).mean()   # chance baseline κ
                    cwr = (ckap - cchance).clamp(min=0.0)
                    cw = cwr / cwr.sum() if float(cwr.sum()) > 1e-9 else ckap / ckap.sum().clamp(min=1e-9)
                    cqre = (cw[:, None, None] * cpis).sum(0)
                    qh2, th2 = cqre[chf:], cti[chf:]
                    crk = (qh2 > qh2.gather(1, th2[:, None])).sum(1)
                    c_top = [100 * float((crk < kk).float().mean()) for kk in (1, 3, 5)]
                    chance = 100.0 / float(mci.sum(1).float().mean())
                    def _clk(p): return float(p[chf:].gather(1, th2[:, None]).mean())
                    click_lik = {"s": _clk(_pn(cl)), "inv": _clk(_pn(civ)), "retr": _clk(_pn(retr)), "QRE": _clk(cqre)}
                    cqre_w = {"s": round(float(cw[0]), 2), "inv": round(float(cw[1]), 2), "retr": round(float(cw[2]), 2)}
                    # CLICK DETECTOR (deployable, state-only) + JOINT = detector says click AND node top-1 right
                    det_ev = click_detect(ze, edges_ev, mpe).sigmoid()           # [cut] P(click) on all eval
                    det_c = det_ev[isc]                                          # on click transitions
                    det_rec = float((det_c[chf:] > 0.5).float().mean())         # recall P(detect | click), held-out
                    dpred = det_ev > 0.5
                    det_prec = float(isc[dpred].float().mean()) if bool(dpred.any()) else float("nan")
                    joint_t1 = 100 * float(((det_c[chf:] > 0.5) & (crk < 1)).float().mean())
                else:
                    c_top = [float("nan")] * 3; chance = float("nan"); joint_t1 = float("nan")
                    det_rec = det_prec = float("nan")
                    click_lik = {"s": float("nan"), "inv": float("nan"), "retr": float("nan"), "QRE": float("nan")}
                    cqre_w = {}
            sps = step / (time.time() - t0)
            for ln in logs:                                                   # full create/prune/edit events
                print(f"        {ln}", flush=True)
            print(f"[{step:5d}] SLEEP#{sleeps} {sps:4.1f}it/s brains {before}->{len(pool.brains)} | "
                  f"δrel(chg)={rel:.3f} τconc={conc:.2f} HODGE persist={persist:.2f} rot={rot:.2f} | "
                  f"ACTION t1/3/5(QRE)={a_top[0]:.0f}/{a_top[1]:.0f}/{a_top[2]:.0f} lik "
                  f"inv={lik['inv']:.3f} L2={lik['L2']:.3f} Hodge={lik['Hodge']:.3f} nav={lik['nav']:.3f} "
                  f"QRE={lik['QRE']:.3f} maj={lik['maj']:.3f} w={qre_w}", flush=True)
            print(f"        CLICK t1/3/5(QRE)={c_top[0]:.0f}/{c_top[1]:.0f}/{c_top[2]:.0f} (chance~{chance:.0f}) "
                  f"lik s={click_lik['s']:.2f} inv={click_lik['inv']:.2f} retr={click_lik['retr']:.2f} "
                  f"QRE={click_lik['QRE']:.2f} w={cqre_w} | DETECT rec={det_rec:.2f} prec={det_prec:.2f} "
                  f"| JOINT(click&node) t1={joint_t1:.0f}", flush=True)
            metrics = {"action_t135": a_top, "action_lik": lik, "action_qre_w": qre_w, "action_maj": maj_top1,
                       "click_t135": c_top, "click_chance": chance, "click_lik": click_lik,
                       "click_qre_w": cqre_w, "detect_recall": det_rec, "detect_prec": det_prec,
                       "joint_t1": joint_t1, "delta_rel": rel, "tau_conc": conc,
                       "hodge_persist": persist, "hodge_rot": rot, "brains": len(pool.brains)}
            if args.save:                                                  # ROLLING save — latest converged weights
                save_checkpoint(args.save, pool, inverse, click_inverse, click_detect, M_click, cfg, args, step, sleeps, metrics)
            aden = 0.0; since = 0
    if args.save:                                                          # FINAL save
        sp = save_checkpoint(args.save, pool, inverse, click_inverse, click_detect, M_click, cfg, args, step, sleeps, metrics)
        print(f"[catalyst] saved checkpoint -> {sp} (+ .json sidecar)", flush=True)
    print(f"[catalyst] done {sleeps} sleeps, brains={len(pool.brains)}, {step} steps, "
          f"{(time.time()-t0)/60:.1f}min | ACTION judged vs majority {maj_top1:.1f}%", flush=True)


if __name__ == "__main__":
    main()
