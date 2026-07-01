"""Train the curvature_wm world model + policy head — BOTH STAGES, one entry point, one folder.

The model (shared trunk + heads) and its training now live together under curvature_wm/model/.

Stage 'world' (step 1): the shared TRUNK + forward/inverse/click heads, two QRE-balanced tasks —
  forward : predict next latent state z_next (+ alive/birth-death)   [next state masked from input]
  action  : action type (CE) + clicked node (CE on CLICK rows)        [action & click masked from input]
  The QRE balancer (find-the-shared-trunk) re-anchors the two task weights every --eval_every steps on the
  HELD-OUT games' skill (the unwritable signal). Saves checkpoints/{tag}_{steps}.pt.

Stage 'policy' (step 2): FREEZE that trunk, train ONLY the PolicyPriorHead pi(a | s_t) to predict the NEXT
  action from the CURRENT state alone (s_{t+1} never enters). Saves checkpoints/policy_head_{tag}.pt and
  reports action t1/t3/t5 + same-edge click t1/t3/t5. Reuses the world stage's HUD-removed caches.

Both stages exclude sk48/sb26 and HUD-remove every game's constant HUD region when --hud is set.

  # both stages, HUD-removed (the standard run):
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.model.train --stage both --hud --steps 2500 --tag m2_hud
  # only one stage:
  ... --stage world --hud --steps 2500 --tag m2_hud
  ... --stage policy --hud --tag m2_hud --trunk_ckpt curvature_wm/checkpoints/m2_hud_2500.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from curvature_wm_model import paths  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from transform_catalyst.data_adapter import N_RELS, obj_dim
from curvature_wm_model.data.loader import load_train, load_heldout, load_split
from curvature_wm_model.data.latents import augment_Z, aug_dim
from curvature_wm_model.splits import ALL_GAMES, TRAIN_GAMES, HELDOUT_GAMES
from curvature_wm_model.model.world_model import WorldModel
from curvature_wm_model.model.policy_head import PolicyPriorHead
from hud_mask import EXCLUDE                           # curvature_wm/perception (on sys.path via paths)
from curvature_wm_model.model.qre import QREBalancer
from curvature_wm_model.model.muon import Muon, split_muon_adam

CLICK = 6
_KEYS = ("Zp", "Zc", "Mp", "Mc", "EFp", "EFc", "a", "ct")


def _to_tensors(d):
    """numpy arrays -> CPU torch tensors (batches are moved to the device per step)."""
    t = {}
    for k in _KEYS:
        v = d[k]
        if v.dtype == bool:
            t[k] = torch.from_numpy(v)
        elif np.issubdtype(v.dtype, np.integer):
            t[k] = torch.from_numpy(v.astype(np.int64))
        else:
            t[k] = torch.from_numpy(v.astype(np.float32))
    return t


def _batch(t, idx, dev):
    return {k: t[k][idx].to(dev) for k in _KEYS}


def losses_and_metrics(model, b, dev, fwd_huber=False, fwd_huber_delta=1.0):
    """Returns (L_fwd, L_action, metrics). Masking is structural: each head only receives its allowed inputs.
    Forward loss = RAW next-latent error (MSE, or Huber if fwd_huber). Action = CE; click = CE on CLICK rows."""
    Zp, Zc, Mp, Mc, EFp, EFc, a, ct = (b["Zp"], b["Zc"], b["Mp"], b["Mc"], b["EFp"], b["EFc"], b["a"], b["ct"])
    B = Zp.shape[0]
    # clicked-node latent for forward conditioning (zeroed where not a valid click)
    hclick = Zp[torch.arange(B, device=dev), ct.clamp(min=0)] * (ct >= 0).float()[:, None]
    union = (Mp.bool() | Mc.bool()).float()
    denom = union.sum().clamp(min=1.0)

    # ---- forward task: predict next latent state z_next; RAW latent error ----
    zn, alive = model.predict_next(Zp, Mp, EFp, a, hclick)
    fwd_el = F.huber_loss(zn, Zc, delta=fwd_huber_delta, reduction="none") if fwd_huber else (zn - Zc) ** 2
    fwd_loss = (fwd_el.sum(-1) * union).sum() / denom
    abce = F.binary_cross_entropy_with_logits(alive, Mc.float(), reduction="none")
    alive_bce = (abce * union).sum() / denom
    L_fwd = fwd_loss + alive_bce

    # ---- action task: action type + click-on-node ----
    al = model.predict_action(Zp, Mp, EFp, Zc, Mc, EFc)
    action_ce = F.cross_entropy(al, a)
    clk = model.predict_click(Zp, Mp, EFp, Zc, Mc, EFc)
    crows = (a == CLICK) & (ct >= 0)
    click_ce = F.cross_entropy(clk[crows], ct[crows]) if bool(crows.any()) else torch.zeros((), device=dev)
    L_action = action_ce + click_ce

    with torch.no_grad():
        fwd_mae = ((zn - Zc).abs().mean(-1) * union).sum() / denom                  # raw next-latent MAE (objective)
        action_acc = (al.argmax(-1) == a).float().mean()
        click_acc = (clk[crows].argmax(-1) == ct[crows]).float().mean() if bool(crows.any()) else torch.tensor(0.0)
        m = {"fwd_loss": float(fwd_loss), "fwd_mae": float(fwd_mae), "alive_bce": float(alive_bce),
             "action_ce": float(action_ce), "action_acc": float(action_acc),
             "click_ce": float(click_ce), "click_acc": float(click_acc), "n_click": int(crows.sum())}
    return L_fwd, L_action, m


@torch.no_grad()
def heldout_skill(model, t_ho, dev, n=512, fwd_huber=False, fwd_huber_delta=1.0):
    model.eval()
    idx = torch.randperm(t_ho["a"].shape[0])[:min(n, t_ho["a"].shape[0])]
    _, _, m = losses_and_metrics(model, _batch(t_ho, idx, dev), dev, fwd_huber, fwd_huber_delta)
    model.train()
    fwd_skill = float(np.exp(-m["fwd_loss"]))              # smaller forward loss -> skill -> 1
    action_skill = max(m["action_acc"], 1e-3)              # top-1 accuracy
    return [fwd_skill, action_skill], m


# =====================================================================================================
# STAGE 1 — world model (shared trunk, QRE-balanced forward + action tasks)
# =====================================================================================================
def train_world(args, cfg, dev):
    print(f"[curvature_wm] STAGE world | device={dev} | loading data... (aug={args.aug}, hud={args.hud})", flush=True)
    if args.hud:                                                    # remove per-game CONSTANT HUD region
        tg = [g for g in ALL_GAMES if g not in EXCLUDE]   # NO held-out games: train on ALL 23 (bp35/sc25/wa30 folded in)
        print(f"  HUD-REMOVE: dropping each game's constant HUD region; excluding {sorted(EXCLUDE)} "
              f"-> {len(tg)} train games, heldout={HELDOUT_GAMES}", flush=True)
        dtr = load_split(tg, cfg, args.max_per_game, tag="train_hud", hud=True)
        dho = load_split(HELDOUT_GAMES, cfg, args.max_per_game, tag="heldout_hud", hud=True)
    elif args.aug:                                                   # hold out the games we visualize/test
        HO = ["ls20", "bp35", "g50t"]
        train_games = [g for g in ALL_GAMES if g not in HO]
        print(f"  AUG: held out {HO}; Mahalanobis position folded INTO the latent (trunk becomes position-aware)", flush=True)
        dtr = load_split(train_games, cfg, args.max_per_game, tag="augtr")
        dho = load_split(HO, cfg, args.max_per_game, tag="aughd")
        for dd in (dtr, dho):
            dd["Zp"] = augment_Z(dd["Zp"], cfg); dd["Zc"] = augment_Z(dd["Zc"], cfg)
    else:
        dtr = load_train(cfg, args.max_per_game); dho = load_heldout(cfg, args.max_per_game)
    tr = _to_tensors(dtr); ho = _to_tensors(dho)
    T = tr["a"].shape[0]
    print(f"  train transitions={T:,} | heldout={ho['a'].shape[0]:,} | clicks(train)={int((tr['a']==CLICK).sum()):,}"
          f" | latent dim={tr['Zp'].shape[-1]}", flush=True)

    model = WorldModel(cfg, d=args.d, n_blocks=args.n_blocks, in_dim=aug_dim(cfg) if args.aug else None).to(dev)
    print(f"  params: { {k: f'{v:,}' for k,v in model.param_breakdown().items()} }", flush=True)
    muon_p, adam_p = split_muon_adam(model.named_parameters())
    opt_m = Muon(muon_p, lr=args.muon_lr, weight_decay=0.01)
    opt_a = torch.optim.Adam(adam_p, lr=args.lr)
    print(f"  optimizer = Muon({len(muon_p)} matrices, lr={args.muon_lr}) + Adam({len(adam_p)} others, lr={args.lr}) "
          f"| Sinkhorn attention", flush=True)
    qre = QREBalancer(2, floor=0.15)                                # find-the-shared-trunk: balance fwd vs action

    fwd_huber = args.fwd_loss == "huber"
    print(f"  forward loss = RAW latent [{args.fwd_loss}"
          f"{f' delta={args.fwd_huber_delta}' if fwd_huber else ''}] | QRE shared-trunk balancer (floor=0.15)", flush=True)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, T, (args.bs,))
        L_fwd, L_action, m = losses_and_metrics(model, _batch(tr, idx, dev), dev, fwd_huber, args.fwd_huber_delta)
        loss = qre.total_loss([L_fwd, L_action], grad_norm=True)
        opt_m.zero_grad(); opt_a.zero_grad(); loss.backward(); opt_m.step(); opt_a.step()

        if step % args.eval_every == 0 or step == 1:
            skills, hm = heldout_skill(model, ho, dev, fwd_huber=fwd_huber, fwd_huber_delta=args.fwd_huber_delta)
            w = qre.weights(skills)
            print(f"[{step:5d}] train: fwd_mae={m['fwd_mae']:.4f} act_acc={m['action_acc']:.3f} "
                  f"click_acc={m['click_acc']:.3f}(n={m['n_click']}) | HELDOUT: fwd_mae={hm['fwd_mae']:.4f} "
                  f"act_acc={hm['action_acc']:.3f} click_acc={hm['click_acc']:.3f} | "
                  f"QRE w(fwd,act)=[{w[0]:.2f},{w[1]:.2f}]", flush=True)

    ckdir = paths.HERE / "checkpoints"; ckdir.mkdir(exist_ok=True)
    ckpath = ckdir / f"{args.tag}_{args.steps}.pt"
    torch.save({"state_dict": model.state_dict(), "d": args.d, "n_blocks": args.n_blocks}, ckpath)
    print(f"saved world model -> {ckpath}", flush=True)
    return ckpath


# =====================================================================================================
# STAGE 2 — policy prior head pi(a | s_t) on the FROZEN shared trunk
# =====================================================================================================
def _load_frozen_trunk(ckpt, cfg, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = WorldModel(cfg, d=ck["d"], n_blocks=ck["n_blocks"]).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _edge_sig(EFp, Mp):
    return ((EFp[..., :N_RELS] * Mp[:, None, :, None].float()).sum(2) * 10).round()


def _tuple(d):
    return (torch.from_numpy(d["Zp"]).float(), torch.from_numpy(d["Mp"]).bool(),
            torch.from_numpy(d["EFp"]).float(), torch.from_numpy(d["a"]).long(), torch.from_numpy(d["ct"]).long())


def _slice(d, sel):
    return (torch.from_numpy(d["Zp"][sel]).float(), torch.from_numpy(d["Mp"][sel]).bool(),
            torch.from_numpy(d["EFp"][sel]).float(), torch.from_numpy(d["a"][sel]).long(),
            torch.from_numpy(d["ct"][sel]).long())


@torch.no_grad()
def coverage(head, model, data, dev, pos0, ks=(1, 3, 5), chunk=512):
    Z, Mp, EFp, a, ct = data
    T = Z.shape[0]; K = max(ks)
    a_hit = {k: 0 for k in ks}; c_hit = {k: 0 for k in ks}; n = 0; nc = 0
    for i in range(0, T, chunk):
        sl = slice(i, i + chunk); Zb = Z[sl].to(dev); Mb = Mp[sl].to(dev)
        H, _ = model.encode(Zb, Mb, EFp[sl].to(dev))
        al, cl = head(H, Zb[:, :, pos0:pos0 + 4], Mb)
        ab = a[sl].to(dev); topa = al.topk(K, -1).indices
        for k in ks:
            a_hit[k] += (topa[:, :k] == ab[:, None]).any(-1).sum().item()
        n += ab.shape[0]
        ctb = ct[sl].to(dev); clk = (ab == CLICK) & (ctb >= 0)
        if clk.any():
            sig = _edge_sig(EFp[sl].to(dev), Mb); topc = cl.topk(K, -1).indices
            for s in torch.where(clk)[0].tolist():
                tsig = sig[s, ctb[s]]
                rank = next((r + 1 for r, j in enumerate(topc[s].tolist()) if torch.equal(sig[s, j], tsig)), None)
                if rank is not None:
                    for k in ks:
                        if rank <= k:
                            c_hit[k] += 1
            nc += int(clk.sum().item())
    return ({k: a_hit[k] / max(n, 1) for k in ks}, {k: c_hit[k] / max(nc, 1) for k in ks}, n, nc)


def train_policy(args, cfg, dev, trunk_ckpt):
    od = obj_dim(cfg); pos0 = od - 5                                  # foot block (px,py,sx,sy) in the basic latent
    print(f"[curvature_wm] STAGE policy | trunk={trunk_ckpt.name} FROZEN | heldout={HELDOUT_GAMES} "
          f"| excluded={sorted(EXCLUDE)}", flush=True)
    model = _load_frozen_trunk(trunk_ckpt, cfg, dev)
    if args.hud:
        tg = [g for g in ALL_GAMES if g not in EXCLUDE]   # NO held-out games: train on ALL 23 (bp35/sc25/wa30 folded in)
        dtr = load_split(tg, cfg, args.max_per_game, tag="train_hud", hud=True)       # reuse world-stage caches
        dho = load_split(HELDOUT_GAMES, cfg, args.max_per_game, tag="heldout_hud", hud=True)
    else:
        dtr = load_train(cfg, args.max_per_game); dho = load_heldout(cfg, args.max_per_game)
    Z, Mp, EFp, a, ct = _tuple(dtr)
    T = Z.shape[0]
    print(f"  human transitions: {T:,} | latent dim {Z.shape[-1]} | pos block dims [{pos0}:{pos0+4}]", flush=True)

    head = PolicyPriorHead(cfg, d=model.trunk.d).to(dev)
    print(f"  policy head: {sum(p.numel() for p in head.parameters()):,} params (only these train)", flush=True)
    opt = torch.optim.Adam(head.parameters(), lr=args.policy_lr)
    g = torch.Generator().manual_seed(0)
    for step in range(1, args.policy_steps + 1):
        idx = torch.randint(0, T, (args.bs,), generator=g); Mb = Mp[idx].to(dev); Zb = Z[idx].to(dev)
        with torch.no_grad():
            H, _ = model.encode(Zb, Mb, EFp[idx].to(dev))
        a_logits, click_logits = head(H, Zb[:, :, pos0:pos0 + 4], Mb)
        tgt_a = a[idx].to(dev)
        loss = F.cross_entropy(a_logits, tgt_a)
        clk = (tgt_a == CLICK) & (ct[idx].to(dev) >= 0)
        if clk.any():
            loss = loss + F.cross_entropy(click_logits[clk], ct[idx].to(dev)[clk])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.policy_log_every == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}", flush=True)

    out = paths.HERE / "checkpoints" / f"policy_head_{args.tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "d": model.trunk.d, "n_actions": cfg.n_actions, "hud": args.hud}, out)
    print(f"saved policy head -> {out}\n", flush=True)

    evals = [(f"TRAIN ({Z.shape[0]:,})", _tuple(dtr))]
    games = dho["game"].astype(str)
    for gh in HELDOUT_GAMES:
        sel = games == gh
        if sel.any():
            evals.append((f"HELD-OUT {gh}", _slice(dho, sel)))
    for name, data in evals:
        ac, cc, n, nc = coverage(head, model, data, dev, pos0)
        print(f"{name:20s}: action  t1={ac[1]:.2f} t3={ac[3]:.2f} t5={ac[5]:.2f}  (n={n:,})")
        print(f"{' '*20}  click   t1={cc[1]:.2f} t3={cc[3]:.2f} t5={cc[5]:.2f}  (same-edge node, n_click={nc:,})")
    return head


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["world", "policy", "both"], default="both",
                    help="world = shared-trunk world model (QRE); policy = pi(a|s_t) on frozen trunk; both = world then policy")
    # --- world-stage ---
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=2)
    ap.add_argument("--max_per_game", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3, help="Adam lr (embeddings/biases/norms)")
    ap.add_argument("--muon_lr", type=float, default=0.02, help="Muon lr (2D weight matrices)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--fwd_loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--fwd_huber_delta", type=float, default=1.0)
    ap.add_argument("--aug", action="store_true",
                    help="AUGMENTED latent: append un-squashed Mahalanobis position to node_latents (trunk position-aware)")
    ap.add_argument("--hud", action="store_true",
                    help="remove each game's CONSTANT HUD region from train+heldout graphs; excludes sk48/sb26")
    ap.add_argument("--tag", type=str, default="cwm")
    # --- policy-stage ---
    ap.add_argument("--policy_steps", type=int, default=2500)
    ap.add_argument("--policy_lr", type=float, default=1e-3)
    ap.add_argument("--policy_log_every", type=int, default=500)
    ap.add_argument("--trunk_ckpt", type=str, default=None,
                    help="(stage=policy) frozen world-model checkpoint; defaults to checkpoints/{tag}_{steps}.pt")
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cfg = Config()

    ckpath = None
    if args.stage in ("world", "both"):
        ckpath = train_world(args, cfg, dev)
    if args.stage in ("policy", "both"):
        if ckpath is not None:                                       # just-trained world model
            trunk = ckpath
        elif args.trunk_ckpt:                                        # explicit frozen trunk
            trunk = Path(args.trunk_ckpt)
        else:                                                        # default: checkpoints/{tag}_{steps}.pt
            trunk = paths.HERE / "checkpoints" / f"{args.tag}_{args.steps}.pt"
        train_policy(args, cfg, dev, trunk)


if __name__ == "__main__":
    main()
