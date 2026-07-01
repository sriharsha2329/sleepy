"""Rung-3 state encoding — GNN FIRST, latents SECOND.

Order (per design): the trained GNN world model (M2) is the encoder. The per-frame perception features
(`node_latents`) are only its INPUT — they are NOT the latent state. The **latent state of a frame is the
GNN trunk's output** (per-node H and the pooled vector). Everything downstream (object-state clustering,
the Hodge complex, counterfactual scoring) operates on these GNN latents, never on raw node_latents.

First-frame convention: a reset frame has no preceding frame or action. We treat its **preceding action as
UNDO (7)** and self-pair the frame with itself — the reset frame is, semantically, "what an undo/reset lands
you on". This replaces the live env's dummy action `0` (online_v5.py:202) and gives a well-defined latent for
frame 0. (The trunk is action-free, so this does not change the trunk latent itself; it makes the transition
interface honest and gives the right `a` to any action-conditioned head / the SCM prefix.)

  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m curvature_wm.hodge_flow.encode      # synthetic smoke
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401  (wires config / data / transform_catalyst, read-only)

import numpy as np
import torch

from config import Config
from data import featurize_transition
import transform_catalyst.data_adapter as cda
from curvature_wm_model.model.world_model import WorldModel

UNDO = 7
CKPT = "curvature_wm/checkpoints/m2_raw100k_2500.pt"


def build_gnn(ckpt: str = CKPT, dev=None):
    """STEP 1 — build the GNN (load the trained M2 world model). The GNN is the encoder.

    max_nodes is forced to cfg.n_max (32): the shared default is 24, which silently TRUNCATES live slots
    24..31 and scatters their H as zero -> a fake zero object-state on ~33% of bp35 frames (review blocker).
    max_nodes is a runtime slicing bound only (no weight has that dim), so the checkpoint loads unchanged.
    Caveat: the ckpt was trained at 24, so slots >=24 are OOD until a retrain — but processing them (OOD,
    nonzero) is strictly better than collapsing hundreds of distinct objects into one origin cluster."""
    dev = dev or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    cfg = Config()
    ck = torch.load(ckpt, map_location=dev)
    m = WorldModel(cfg, d=ck.get("d", 64), n_blocks=ck.get("n_blocks", 2), max_nodes=cfg.n_max).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False)   # strict=False: ckpt may carry stale curvature_head keys
    m.eval()
    return m, dev


def state_feat(cur_graph, cfg, prev_graph=None, prev_action=None):
    """Featurize ONE state into GNN-input arrays.

    cur_graph / prev_graph are object graphs {"nodes": [...], "edges": [...]} (env or perception output).
    First frame (prev_graph is None) -> self-pair the frame with itself, preceding action = UNDO.
    """
    if prev_graph is None:                              # ---- first-frame convention ----
        prev_graph, prev_action = cur_graph, UNDO
    if prev_action is None:
        prev_action = UNDO
    tr = {"prev_nodes": prev_graph["nodes"], "cur_nodes": cur_graph["nodes"],
          "prev_edges": prev_graph.get("edges", []), "cur_edges": cur_graph.get("edges", []),
          "a": int(prev_action), "click": None, "deltas": [], "level": 0, "r_ext": 0.0, "done": False}
    return featurize_transition(tr, cfg)


@torch.no_grad()
def latent(model, feat, dev, cfg=None, tag: str = "cur"):
    """STEP 2 — build the latent FROM the GNN. Returns (H [N,d] per-node, pooled [d]).

    The state latent is the GNN trunk output, not raw node_latents. `tag` selects which frame of the
    featurized pair to encode ("cur" = the state itself; for a first-frame self-pair, prev==cur)."""
    cfg = cfg or Config()
    Z = torch.from_numpy(cda.node_latents(feat, tag, cfg)).float()[None].to(dev)
    M = torch.from_numpy(feat[f"mask_{tag}"]).bool()[None].to(dev)
    E = torch.from_numpy(cda.edge_feats(feat, tag, cfg)).float()[None].to(dev)
    H, pooled = model.encode(Z, M, E)
    return H[0], pooled[0]


@torch.no_grad()
def encode_first_frame(model, graph, dev, cfg=None):
    """Convenience: GNN latent of a RESET frame (no preceding action -> UNDO self-pair)."""
    cfg = cfg or Config()
    return latent(model, state_feat(graph, cfg), dev, cfg, tag="cur")


# ----------------------------------------------------------------------------- synthetic smoke
def _toy_graph(n=3, seed=0):
    rng = np.random.default_rng(seed)
    nodes = [{"pid": k, "cy": float(rng.integers(0, 64)), "cx": float(rng.integers(0, 64)),
              "color": [int(k % 4)], "stab": 3, "area_bin": 2, "type_hash": f"t{k}", "k": 5,
              "px": float(rng.integers(0, 64)), "py": float(rng.integers(0, 64)), "sx": 4.0, "sy": 4.0}
             for k in range(n)]
    return {"nodes": nodes, "edges": []}


def _smoke():
    cfg = Config()
    dev = torch.device("cpu")
    model, _ = build_gnn(dev=dev)
    g = _toy_graph(3)

    # first frame: preceding action must be UNDO, self-paired
    ff = state_feat(g, cfg)                                   # prev_graph=None -> first-frame path
    P = []
    P.append(("first-frame preceding action == UNDO", int(ff["a"]) == UNDO))
    P.append(("first-frame self-pair (prev mask == cur mask)", bool((ff["mask_prev"] == ff["mask_cur"]).all())))
    H, pooled = latent(model, ff, dev, cfg)
    P.append(("GNN latent shapes", H.shape == (cfg.n_max, model.trunk.d) and pooled.shape == (model.trunk.d,)))
    P.append(("latent finite", bool(torch.isfinite(pooled).all())))

    # a normal step uses the real preceding frame+action; latent must still be GNN output
    g2 = _toy_graph(3, seed=1)
    sf = state_feat(g2, cfg, prev_graph=g, prev_action=3)
    P.append(("step preceding action preserved", int(sf["a"]) == 3))
    H2, p2 = latent(model, sf, dev, cfg)
    P.append(("step latent shapes", p2.shape == (model.trunk.d,)))

    # the latent is the GNN's output, not the raw node_latents (different dim: d vs od)
    P.append(("latent dim d != raw od (GNN-produced, not raw features)", model.trunk.d != model.trunk.od))

    ok = all(v for _, v in P)
    print("ENCODE SMOKE (GNN-first, undo first-frame):")
    for n, v in P:
        print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    _smoke()
