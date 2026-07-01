"""Per-transition curvature  r = (dA + A∧A) / Σ|z_current|  over human runs, computed ANALYTICALLY from the
model — NO separate ML "curvature head". We read curvature two ways and compare them:

  r(target)  uses the ACTUAL next state z_{t+1} (real human transition, real action).
  r(pred)    uses the PREDICTED next state ẑ_{t+1} = f_theta(z_t, a) from the forward head — so it is a pure
             function of (current state, taken action), oblivious to the real next state. This REPLACES the
             old regression head: instead of an MLP guessing r from (s,a), we plug the forward model's own
             next-state prediction into the same curvature formula.

All residuals use SMAE (smooth-L1) instead of MAE (|·|): small per-element residuals (static flicker) are
squashed quadratically toward ~0, while real changes stay ~linear — so no-op/static steps register ~0.

For a "next state" Znext (=z_{t+1} for target, =ẑ_{t+1} for pred), with smae(e) = smooth_l1(e; beta=SMAE_BETA):

  dA   = sum over (alive nodes, features) of smae(Znext - z_current)    (change relative to the current state)

  A∧A  = min over RETURN actions b of   sum smae(f_theta(Znext, b) - z_current)
         f_theta = the forward head; apply each b at Znext, measure SMAE distance back to z_current. The
         minimising b RETURNS Znext -> z_current (model-based min-return holonomy). Candidates b:
            - "stay" (no-op): return by not moving -> distance = dA  (a no-op step is trivially reversible)
            - movement actions: every action id except generic-click (6) and undo/reset (7)
            - click-on-node i, for each alive node i (the node with its edges) — h_click = that node's latent
         small A∧A => some action (or staying) undoes the step (reversible); large => nothing returns.

  curvature = dA + A∧A

For r(pred) the forward head predicts node latents only (not edges/births), so the predicted next state reuses
the CURRENT state's edges/mask (EFp, Mp) — keeping r(pred) entirely a function of the current state + action.
Znext is encoded by the trunk ONCE; each candidate b only re-runs the (cheap) forward head. No env replay.

The gap r(target) - r(pred) is the SURPRISE: where they agree the change was foreseeable (reversible/expected);
where r(target) >> r(pred) the change was not predicted from (s,a) — the candidate causal/event step.

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.curvature \
      --ckpt curvature_wm/checkpoints/m2_raw100k_2500.pt --games bp35 --max_per_game 300
"""
from __future__ import annotations

import argparse

from curvature_wm_model import paths  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from curvature_wm_model.data.loader import load_split
from curvature_wm_model.model.world_model import WorldModel

CLICK = 6
UNDO = 7
SMAE_BETA = 2.0      # smooth-L1 knob: residuals < beta squashed (quadratic). beta=2 > max residual(~1) => ~MSE


def move_ids(n_actions):
    """movement return actions = every action except generic-click and undo/reset."""
    return [a for a in range(n_actions) if a not in (CLICK, UNDO)]


@torch.no_grad()
def _curvature_core(model, Zp, union, Znext, Mnext, EFnext, dev, n_click=None, return_dists=False):
    """dA + A∧A for a generic "next state" (Znext,Mnext,EFnext), with residuals summed over `union` nodes.
    Returns dA[B], AA[B], total[B], best[B] (index into labels), labels[list], optionally D[n_cand,B]."""
    B, N, od = Znext.shape

    def smae_back(zpred):                                               # SMAE (smooth-L1) of residual vs z_current
        s = F.smooth_l1_loss(zpred, Zp, beta=SMAE_BETA, reduction="none")   # squashes small (flicker) residuals
        return (s.sum(-1) * union).sum(-1)                            # [B] sum over features, masked over nodes

    dA = smae_back(Znext)                                              # change of Znext relative to z_current

    H, pooled = model.encode(Znext, Mnext, EFnext)                     # encode the next state ONCE; reuse per action
    dists, labels = [dA], ["stay"]                                     # "stay" = no-op return (you didn't move): dist = dA
    for b in move_ids(model.cfg.n_actions):                            # ---- movement actions ----
        bv = torch.full((B,), b, dtype=torch.long, device=dev)
        zb, _ = model.forward_head(Znext, H, pooled, bv)              # f_theta(Znext, b)
        dists.append(smae_back(zb)); labels.append(f"move{b}")
    nclick = N if n_click is None else min(n_click, N)                 # ---- click on each node (node w/ edges) ----
    clickv = torch.full((B,), CLICK, dtype=torch.long, device=dev)
    for i in range(nclick):
        zi, _ = model.forward_head(Znext, H, pooled, clickv, Znext[:, i])    # h_click = node i's latent
        di = smae_back(zi)
        di = torch.where(Mnext[:, i].bool(), di, torch.full_like(di, float("inf")))  # invalid where node i is dead
        dists.append(di); labels.append(f"click@{i}")
    D = torch.stack(dists, 0)                                          # [n_cand, B]
    AA, best = D.min(0)
    out = (dA, AA, dA + AA, best, labels)
    return (*out, D) if return_dists else out


@torch.no_grad()
def curvature_batch(model, Zp, Mp, EFp, Zc, Mc, EFc, dev, n_click=None, return_dists=False):
    """r(target): curvature from the ACTUAL next state (Zc,Mc,EFc); residuals over the union of alive nodes."""
    union = (Mp.bool() | Mc.bool()).float()
    return _curvature_core(model, Zp, union, Zc, Mc, EFc, dev, n_click, return_dists)


@torch.no_grad()
def curvature_batch_pred(model, Zp, Mp, EFp, a, dev, h_click=None, n_click=None, return_dists=False):
    """r(pred): curvature from the PREDICTED next state ẑ = f_theta(Zp, a). A pure function of (current state,
    taken action) — never sees the real next state. The forward head predicts node latents only, so the
    predicted state reuses the CURRENT edges/mask (EFp, Mp); residuals are summed over the current alive nodes."""
    zhat, _ = model.predict_next(Zp, Mp, EFp, a, h_click)              # predicted next latent state ẑ
    union = Mp.float()
    return _curvature_core(model, Zp, union, zhat, Mp, EFp, dev, n_click, return_dists)


@torch.no_grad()
def build_records(model, data, dev, bs=256, n_click=None):
    """Per transition: dA/A∧A/curv/best for BOTH r(target) [actual next state] and r(pred) [predicted next
    state, function of (current state, action) only]. r = curvature / Σ|z_current| (same normalizer for both)."""
    g = lambda k, f=True: torch.from_numpy(data[k]).float() if f else torch.from_numpy(data[k])  # noqa: E731
    Zp, Zc = g("Zp"), g("Zc"); Mp, Mc = g("Mp", False), g("Mc", False)
    EFp, EFc = g("EFp"), g("EFc"); a = data["a"].astype(np.int64); T = len(a)
    at = torch.from_numpy(a)
    ct = torch.from_numpy(data["ct"].astype(np.int64)) if "ct" in data else None
    dA = np.zeros(T); AA = np.zeros(T); best = np.zeros(T, np.int64)                 # target (actual next state)
    dAp = np.zeros(T); AAp = np.zeros(T); bestp = np.zeros(T, np.int64)             # pred (predicted next state)
    zl1 = np.zeros(T); labels = []
    for s in range(0, T, bs):
        sl = slice(s, min(s + bs, T))
        zp, mp = Zp[sl].to(dev), Mp[sl].to(dev); efp = EFp[sl].to(dev)
        mc = Mc[sl].to(dev); ab = at[sl].to(dev)
        hclick = None
        if ct is not None:
            cb = ct[sl].to(dev)
            hclick = zp[torch.arange(zp.shape[0], device=dev), cb.clamp(min=0)] * (cb >= 0).float()[:, None]
        dab, aab, _, bestb, labels = curvature_batch(
            model, zp, mp, efp, Zc[sl].to(dev), mc, EFc[sl].to(dev), dev, n_click)
        dpb, apb, _, bestpb, _ = curvature_batch_pred(model, zp, mp, efp, ab, dev, hclick, n_click)
        dA[sl] = dab.cpu().numpy(); AA[sl] = aab.cpu().numpy(); best[sl] = bestb.cpu().numpy()
        dAp[sl] = dpb.cpu().numpy(); AAp[sl] = apb.cpu().numpy(); bestp[sl] = bestpb.cpu().numpy()
        un = (mp.bool() | mc.bool()).float()
        zl1[sl] = ((zp.abs().sum(-1)) * un).sum(-1).cpu().numpy()                    # Σ|z_current| over union
    curv = (dA + AA).astype(np.float32); curvp = (dAp + AAp).astype(np.float32)
    z = zl1.astype(np.float32) + 1e-6
    return {"Zp": data["Zp"], "a": a, "game": data["game"], "zcur_l1": zl1.astype(np.float32), "labels": np.array(labels),
            "dA": dA.astype(np.float32), "AA": AA.astype(np.float32), "curv": curv, "best": best,
            "r_target": (curv / z).astype(np.float32),
            "dA_pred": dAp.astype(np.float32), "AA_pred": AAp.astype(np.float32), "curv_pred": curvp, "best_pred": bestp,
            "r_pred": (curvp / z).astype(np.float32)}


def _cat(label):
    return "click" if str(label).startswith("click@") else "move"


def aa_histogram(AA, best, labels, nbins=10):
    """Histogram of A∧A (x-axis) with, per bin, what REVERSES the step: move actions vs click-on-node,
    and the most common specific reversers."""
    AA = np.asarray(AA); T = len(AA)
    best_lab = np.array([labels[int(b)] for b in best])
    cat = np.array([_cat(x) for x in best_lab])
    edges = np.quantile(AA, np.linspace(0, 1, nbins + 1))
    print("\nA∧A histogram (x = A∧A; which action reverses z_next -> z_current):")
    print("  bin            A∧A_range            n    mean_AA | reversed_by  move% / click% | top reversers")
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (AA >= lo) & (AA <= hi) if i == nbins - 1 else (AA >= lo) & (AA < hi)
        n = int(sel.sum())
        if not n:
            continue
        mv = 100.0 * np.mean(cat[sel] == "move"); ck = 100.0 * np.mean(cat[sel] == "click")
        vals, counts = np.unique(best_lab[sel], return_counts=True)
        top = "  ".join(f"{vals[j]}({counts[j]})" for j in np.argsort(-counts)[:3])
        tag = "smallest" if i == 0 else ("largest" if i == nbins - 1 else "")
        print(f"  {i+1:>2} {tag:<8} [{lo:8.2f},{hi:8.2f}] {n:>5d}  {AA[sel].mean():7.2f} |  {mv:5.0f} / {ck:5.0f}     | {top}")
    mv = 100.0 * np.mean(cat == "move"); ck = 100.0 * np.mean(cat == "click")
    n_moves = sum(1 for x in labels if str(x).startswith("move"))
    move_lbls = [str(x) for x in labels if str(x).startswith("move")]
    print(f"  OVERALL: reversed by move {mv:.0f}%  /  click-on-node {ck:.0f}%   "
          f"(candidates: {n_moves} moves {move_lbls} + one click per alive node; undo/action-7 excluded)")


def _rankcorr(x, y):
    """(Pearson, Spearman) between x and y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pear = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 else 0.0
    return pear, spear


def target_vs_pred(rec, nbins=10):
    """Compare r(target) [actual next state] vs r(pred) [predicted next state, from (s,a) only], binned by the
    ACTUAL change dA. Where r(pred) < r(target) the change was NOT foreseeable from (s,a) -> surprise/event."""
    rt, rp, dA = rec["r_target"], rec["r_pred"], rec["dA"]
    pear, spear = _rankcorr(rt, rp)
    print("\nr(target) vs r(pred)   [pred = curvature of the forward model's predicted next state; (s,a) only]")
    print(f"  correlation: Pearson={pear:.3f}  Spearman={spear:.3f}")
    q = lambda x: (x.mean(), np.median(x), np.quantile(x, 0.95), x.max())  # noqa: E731
    print("            mean    p50    p95    max")
    print("  r(target) %.3f  %.3f  %.3f  %.3f" % q(rt))
    print("  r(pred)   %.3f  %.3f  %.3f  %.3f" % q(rp))
    print(f"\n  binned by ACTUAL change dA ({nbins} bins) — gap = r(target) - r(pred) = surprise:")
    print("  bin            dA_range          n   mean_dA | r(target)  r(pred)    gap")
    edges = np.quantile(dA, np.linspace(0, 1, nbins + 1))
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (dA >= lo) & (dA <= hi) if i == nbins - 1 else (dA >= lo) & (dA < hi)
        n = int(sel.sum())
        if not n:
            continue
        tag = "smallest" if i == 0 else ("largest" if i == nbins - 1 else "")
        gt, gp = rt[sel].mean(), rp[sel].mean()
        print(f"  {i+1:>2} {tag:<8} [{lo:7.2f},{hi:7.2f}] {n:>5d}  {dA[sel].mean():7.2f} |   {gt:6.3f}    {gp:6.3f}  {gt-gp:+6.3f}")


def _load_model(ckpt, cfg, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = WorldModel(cfg, d=ck.get("d", 64), n_blocks=ck.get("n_blocks", 2)).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval()   # strict=False: ckpt may carry stale curvature_head keys
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="curvature_wm/checkpoints/m2_raw100k_2500.pt")
    ap.add_argument("--games", nargs="+", default=["bp35"])
    ap.add_argument("--max_per_game", type=int, default=300)
    ap.add_argument("--n_click", type=int, default=None, help="cap click-node candidates (default: all nodes)")
    ap.add_argument("--save", default=None, help="optional .npz path for the (state, action, curvature) records")
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cfg = Config()
    model = _load_model(args.ckpt, cfg, dev)
    data = load_split(args.games, cfg, args.max_per_game, tag="curv_" + "_".join(args.games))
    rec = build_records(model, data, dev, n_click=args.n_click)
    dA, AA, curv, best, labels = rec["dA"], rec["AA"], rec["curv"], rec["best"], rec["labels"]
    print(f"curvature over {args.games} | transitions={len(dA):,}  (model={args.ckpt})")
    print("  [r(target): from ACTUAL next state]")
    print(f"  dA   : mean={dA.mean():.3f}  p50={np.median(dA):.3f}  p95={np.quantile(dA,0.95):.3f}  max={dA.max():.3f}")
    print(f"  A∧A  : mean={AA.mean():.3f}  p50={np.median(AA):.3f}  p95={np.quantile(AA,0.95):.3f}  max={AA.max():.3f}")
    print(f"  curv : mean={curv.mean():.3f}  p50={np.median(curv):.3f}  p95={np.quantile(curv,0.95):.3f}")
    print("  highest-curvature transitions (state index, action, dA, A∧A, curv, best-return):")
    for j in np.argsort(-curv)[:8]:
        print(f"    idx={j:5d}  a={int(rec['a'][j])}  dA={dA[j]:.3f}  A∧A={AA[j]:.3f}  curv={curv[j]:.3f}  best={labels[best[j]]}")
    aa_histogram(AA, best, labels)
    target_vs_pred(rec)
    if args.save:
        np.savez(args.save, **{k: v for k, v in rec.items() if k != "Zp"}, Zp=rec["Zp"])
        print(f"  saved records -> {args.save}")


if __name__ == "__main__":
    main()
