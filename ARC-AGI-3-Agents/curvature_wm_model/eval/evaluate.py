"""Evaluate a trained curvature_wm checkpoint on the HELD-OUT games (never seen in training).

CORE VIEW — everything as a function of how much the frame ACTUALLY changed (binned on the x-axis):
  x-axis (bins) = MAE of the raw change |z_next - z_cur|.
  per bin: fwd_MAE = MAE between PREDICTED and ACTUAL next frame (y-axis); action t1/t3/t5;
           click t1/t3/t5 (exact node identified by its edges, CLICK rows only).
Plus OVERALL + per-game fwd MAE and an ACTION-by-id histogram (collapse check).

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.eval.evaluate --ckpt curvature_wm/checkpoints/m2_raw100k_2500.pt
"""
from __future__ import annotations

import argparse

from curvature_wm_model import paths  # noqa: F401

import numpy as np
import torch

from config import Config
from curvature_wm_model.data.loader import load_heldout, load_split
from curvature_wm_model.model.world_model import WorldModel
from curvature_wm_model.splits import HELDOUT_GAMES
from curvature_wm_model.model.train import _to_tensors, _batch, CLICK

_IDX = ("Zp", "Zc", "Mp", "Mc", "EFp", "EFc", "a", "ct")


@torch.no_grad()
def _per_row(model, b, dev):
    Zp, Zc, Mp, Mc, EFp, EFc, a, ct = (b[k] for k in _IDX)
    B = Zp.shape[0]
    hclick = Zp[torch.arange(B, device=dev), ct.clamp(min=0)] * (ct >= 0).float()[:, None]
    union = (Mp.bool() | Mc.bool()).float()
    denom = union.sum(1).clamp(min=1.0)

    zn, _ = model.predict_next(Zp, Mp, EFp, a, hclick)

    def mae(x):
        return (x.abs().mean(-1) * union).sum(1) / denom
    c = mae(Zc - Zp)                                                # x: actual change |z_next - z_cur|
    e = mae(zn - Zc)                                                # y: forward error |pred - actual|

    al = model.predict_action(Zp, Mp, EFp, Zc, Mc, EFc)
    arank = (al > al.gather(1, a[:, None])).sum(1)
    pred_a = al.argmax(1)
    clk = model.predict_click(Zp, Mp, EFp, Zc, Mc, EFc)
    crank = (clk > clk.gather(1, ct.clamp(min=0)[:, None])).sum(1)
    isc = (a == CLICK) & (ct >= 0)
    return tuple(x.cpu().numpy() for x in (c, e, arank, pred_a, crank, isc))


def _tk(rank, sel, ks=(1, 3, 5)):
    r = rank[sel]
    return "/".join(f"{(r < k).mean():.2f}" for k in ks) if len(r) else "  n/a   "


@torch.no_grad()
def evaluate(ckpt_path, max_per_game=1500, bs=512, nbins=10, hud=False):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cfg = Config()
    ck = torch.load(ckpt_path, map_location=dev)
    model = WorldModel(cfg, d=ck.get("d", 64), n_blocks=ck.get("n_blocks", 2)).to(dev)
    model.load_state_dict(ck["state_dict"], strict=False); model.eval()   # strict=False: ckpt has no curvature_head

    # HUD-removed heldout (matches a --hud-trained model); else the raw heldout
    raw = load_split(HELDOUT_GAMES, cfg, max_per_game, tag="heldout_hud", hud=True) if hud else load_heldout(cfg, max_per_game)
    t = _to_tensors(raw); games = raw["game"].astype(str); a_all = raw["a"].astype(int)
    T = t["a"].shape[0]
    c = np.zeros(T); e = np.zeros(T)
    arank = np.zeros(T, int); pred_a = np.zeros(T, int); crank = np.zeros(T, int); isc = np.zeros(T, bool)
    for s in range(0, T, bs):
        idx = torch.arange(s, min(s + bs, T)); sl = slice(s, min(s + bs, T))
        c[sl], e[sl], arank[sl], pred_a[sl], crank[sl], isc[sl] = _per_row(model, _batch(t, idx, dev), dev)

    allsel = np.ones(T, bool)
    print(f"HELD-OUT eval | {ckpt_path}  (transitions={T:,}, chance action t1=1/{cfg.n_actions}={1.0/cfg.n_actions:.3f})\n")

    # ---- OVERALL + per game ----
    print("            fwd_MAE | action t1/t3/t5 | click t1/t3/t5 (n)")
    print(f"  OVERALL   {e.mean():.4f} |   {_tk(arank, allsel)}   |  {_tk(crank, isc)} ({int(isc.sum())})")
    for g in HELDOUT_GAMES:
        gs = games == g; gc = gs & isc
        print(f"  {g:<8}  {e[gs].mean():.4f} |   {_tk(arank, gs)}   |  {_tk(crank, gc)} ({int(gc.sum())})")

    # ---- ACTION by id (histogram level) ----
    print("\nACTION by id (OVERALL):  id  true_share  recall@1  pred_share")
    for aid in range(cfg.n_actions):
        tsel = a_all == aid; nt = int(tsel.sum())
        if not nt and (pred_a == aid).sum() == 0:
            continue
        recall = (arank[tsel] < 1).mean() if nt else float("nan")
        print(f"    {aid:>2}{'(click)' if aid == CLICK else '':<7} {nt / T:.3f}      {recall:.3f}     {(pred_a == aid).mean():.3f}")

    # ---- CORE: bins of the actual change ----
    print(f"\nBy actual change  x=|z_next - z_cur|  ({nbins} bins):")
    print("  bin            x_range          n    mean_x   fwd_MAE | action t1/t3/t5 | click t1/t3/t5 (n)")
    edges = np.quantile(c, np.linspace(0, 1, nbins + 1))
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (c >= lo) & (c <= hi) if i == nbins - 1 else (c >= lo) & (c < hi)
        if not sel.sum():
            continue
        cl = sel & isc
        tag = "smallest" if i == 0 else ("largest" if i == nbins - 1 else "")
        print(f"  {i+1:>2} {tag:<8} [{lo:.4f},{hi:.4f}] {int(sel.sum()):>5d}  {c[sel].mean():.4f}  {e[sel].mean():.4f} | "
              f"  {_tk(arank, sel)}   |  {_tk(crank, cl)} ({int(cl.sum())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_per_game", type=int, default=1500)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--hud", action="store_true", help="evaluate on HUD-removed heldout (for a --hud-trained model)")
    args = ap.parse_args()
    evaluate(args.ckpt, args.max_per_game, nbins=args.nbins, hud=args.hud)


if __name__ == "__main__":
    main()
