"""policy_prior.py — the trained policy head as an ACTION PRIOR over a LIVE perception graph.

Loads the shared trunk (m_*.pt, forward+inverse+click+QRE) + PolicyPriorHead (policy_head_*.pt) and, given a
state's per-object graph, returns the model's RANKED action recommendations:
    score(move a)      = P(a | s)
    score(click obj i) = P(CLICK | s) · P(node i | s)
so the model "sees the state and recommends the action". Drop-in for the solver's `policy`: the Roller calls
`policy.propose(graph, available, k)` (same (key, aid, xy) shape as branch_actions) — replacing the uniform prior.

No value/Q heads. Featurization matches training exactly: base node_latents (NOT aug) + the foot pos block
Z[:, od-5 : od-1] (pos0 = od-5, as in model/train.py --stage policy).
"""
from __future__ import annotations

from curvature_wm_model import paths  # noqa: F401

CLICK = 6


def _default_ckpts():
    """Pick a MATCHED (trunk, policy-head) pair — prefer m_all, fall back to m_hud. Never mix a trunk with a
    head trained on a DIFFERENT trunk (the head's features are trunk-specific)."""
    ckdir = paths.HERE / "checkpoints"
    for tag in ("m_all", "m_hud"):
        trunk, head = ckdir / f"{tag}_2500.pt", ckdir / f"policy_head_{tag}.pt"
        if trunk.exists() and head.exists():
            return trunk, head
    return None, None


class PolicyPrior:
    def __init__(self, cfg, trunk_ckpt=None, head_ckpt=None, dev=None):
        import torch
        from transform_catalyst.data_adapter import obj_dim
        from curvature_wm_model.model.world_model import WorldModel
        from curvature_wm_model.model.policy_head import PolicyPriorHead
        self.cfg = cfg
        # device priority: CUDA > CPU > MPS. CUDA (submission GPU) first; else CPU (preferred over MPS for
        # correctness). MPS is used ONLY if dev='mps' is passed explicitly or DEVICE=mps is set.
        import os as _os
        _envdev = _os.environ.get("DEVICE", "").lower()
        def _cuda_usable():
            try:
                if not torch.cuda.is_available():
                    return False
                _t = torch.zeros(1, device="cuda"); _ = (_t + _t).cpu()    # exercise a kernel: old GPUs (P100 sm_60) raise here
                return True
            except Exception:
                return False
        if dev:
            self.dev = dev
        elif _envdev in ("cuda", "cpu", "mps"):
            self.dev = torch.device(_envdev)
        elif _cuda_usable():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")                                 # CUDA absent/unusable -> CPU (preferred over MPS)
        self.od = obj_dim(cfg)
        self.pos0 = self.od - 5                          # foot block (px,py,sx,sy) inside the base latent
        dt, dh = _default_ckpts()
        trunk_ckpt = trunk_ckpt or dt
        head_ckpt = head_ckpt or dh
        tk = torch.load(trunk_ckpt, map_location=self.dev)
        self.model = WorldModel(cfg, d=tk["d"], n_blocks=tk["n_blocks"]).to(self.dev)
        self.model.load_state_dict(tk["state_dict"], strict=False)
        self.model.eval()
        hk = torch.load(head_ckpt, map_location=self.dev)
        self.head = PolicyPriorHead(cfg, d=hk["d"]).to(self.dev)
        self.head.load_state_dict(hk["head"])
        self.head.eval()
        self.trunk_ckpt = str(trunk_ckpt)
        self.head_ckpt = str(head_ckpt)

    def _forward(self, graph):
        """trunk + policy head on the state graph -> (a_p [n_actions], c_p [N nodes], feat). Featurization
        matches training: base node_latents (non-aug) + foot pos block Z[:, pos0:pos0+4]."""
        import torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat
        feat = _feat(graph, graph, self.cfg)            # self-transition: featurize the current state
        Z = cda.node_latents(feat, "cur", self.cfg)     # BASE latent (the trunk was trained on this, non-aug)
        M = feat["mask_cur"].astype(bool)
        with torch.no_grad():
            Zt = torch.from_numpy(Z).float()[None].to(self.dev)
            Mt = torch.from_numpy(M)[None].to(self.dev)
            Et = torch.from_numpy(cda.edge_feats(feat, "cur", self.cfg)).float()[None].to(self.dev)
            H, _ = self.model.encode(Zt, Mt, Et)
            a_logits, c_logits = self.head(H, Zt[:, :, self.pos0:self.pos0 + 4], Mt)
            a_p = torch.softmax(a_logits[0], -1).cpu().numpy()
            c_p = torch.softmax(c_logits[0], -1).cpu().numpy()
        return a_p, c_p, feat

    def state_dist(self, graph):
        """(a_p, c_p): the model's action-type prior and per-node click prior — used by the solver's _prior:
        P(move a)=a_p[a.aid]; P(click node i)=a_p[CLICK]*c_p[i] (i = graph node index = solver's a.node)."""
        a_p, c_p, _ = self._forward(graph)
        return a_p, c_p

    def state_dist_rows(self, graph):
        """Like state_dist but ALSO returns foot2row {(round(px,2),round(py,2)): latent_row} so the caller indexes
        c_p by the clicked node's FOOT — NOT enumerate(g['nodes']) order, which align_slots re-ranks (the row
        misalignment that broke v3 clicks). c_p and Z share this foot-keyed row order."""
        import numpy as np
        a_p, c_p, feat = self._forward(graph)
        idx = np.where(feat["mask_cur"].astype(bool))[0]
        foot = feat["foot_cur"]
        foot2row = {(round(float(foot[i, 0]), 2), round(float(foot[i, 1]), 2)): int(i) for i in idx}
        return a_p, c_p, foot2row

    def propose_scored(self, graph, available, k=None):
        """[(score, key, action_id, click_xy)] sorted desc — the model's ranked recommendations for `available`."""
        import numpy as np
        from curvature_wm_model.hodge_flow.diagnose_lr import HUD_THRESH
        a_p, c_p, feat = self._forward(graph)
        M = feat["mask_cur"].astype(bool)
        cands = []
        for a in available:
            if a == CLICK:
                for i in np.where(M)[0]:
                    if feat["pos_cur"][i, 0] >= HUD_THRESH:          # never click the HUD/score region
                        continue
                    px = int(round(float(feat["foot_cur"][i, 0])))
                    py = int(round(float(feat["foot_cur"][i, 1])))
                    cands.append((float(a_p[CLICK]) * float(c_p[i]), (CLICK, px, py), CLICK, (px, py)))
            else:
                cands.append((float(a_p[a]), a, a, None))
        cands.sort(key=lambda c: -c[0])
        return cands[:k] if k else cands

    def propose(self, graph, available, k=5):
        """[(key, aid, xy)] — top-k action recommendations (same shape as branch_actions)."""
        return [(key, aid, xy) for (_s, key, aid, xy) in self.propose_scored(graph, available, k)]

    def rank_returns(self, g_from, g_to, cands):
        """Model-based A∧A (triangles use the model): order the return candidates by how strongly the INVERSE
        head (+ CLICK head) believes each action maps g_from -> g_to. g_from = the post-action state s'; g_to =
        the origin s. cands = [(key, aid, xy)] (branch_actions shape); returns the SAME tuples, best-return
        first. The solver then env-VERIFIES the top-k (so forward stays env/brute; only the *choice* of which
        triangles to close is the model's)."""
        import numpy as np, torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat
        feat = _feat(g_from, g_to, self.cfg)        # transition s'(prev) -> s(cur) = the inverse head's direction
        def _st(side):
            Z = torch.from_numpy(cda.node_latents(feat, side, self.cfg)).float()[None].to(self.dev)
            M = torch.from_numpy(feat["mask_" + side].astype(bool))[None].to(self.dev)
            E = torch.from_numpy(cda.edge_feats(feat, side, self.cfg)).float()[None].to(self.dev)
            return Z, M, E
        Zp, Mp, EFp = _st("prev")   # s'
        Zc, Mc, EFc = _st("cur")    # s
        with torch.no_grad():
            a_p = torch.softmax(self.model.predict_action(Zp, Mp, EFp, Zc, Mc, EFc)[0], -1).cpu().numpy()
            c_p = torch.softmax(self.model.predict_click(Zp, Mp, EFp, Zc, Mc, EFc)[0], -1).cpu().numpy()
        foot = feat["foot_prev"]; idx_from = np.where(feat["mask_prev"].astype(bool))[0]
        def _score(key, aid, xy):
            if aid != CLICK:
                return float(a_p[aid]) if 0 <= aid < len(a_p) else 0.0
            if xy is None or len(idx_from) == 0:
                return float(a_p[CLICK]) * 1e-6
            d = (foot[idx_from, 0] - float(xy[0])) ** 2 + (foot[idx_from, 1] - float(xy[1])) ** 2
            ni = int(idx_from[int(np.argmin(d))])           # nearest s' node to the click -> its c_p weight
            return float(a_p[CLICK]) * (float(c_p[ni]) if ni < len(c_p) else 1e-6)
        return sorted(cands, key=lambda c: -_score(c[0], c[1], c[2]))

    def return_conf(self, g_from, g_to):
        """Hodge-flow A∧A WITHOUT env steps: the inverse head's MAX confidence that some action maps g_from -> g_to.
        High => a return is likely (reversible, low A∧A); low => no return (irreversible, high A∧A). g_from = the
        post-action state s'; g_to = the origin s. No branch_actions, no env.step, no reset — pure model read."""
        import numpy as np, torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat
        feat = _feat(g_from, g_to, self.cfg)
        def _st(side):
            Z = torch.from_numpy(cda.node_latents(feat, side, self.cfg)).float()[None].to(self.dev)
            M = torch.from_numpy(feat["mask_" + side].astype(bool))[None].to(self.dev)
            E = torch.from_numpy(cda.edge_feats(feat, side, self.cfg)).float()[None].to(self.dev)
            return Z, M, E
        Zp, Mp, EFp = _st("prev"); Zc, Mc, EFc = _st("cur")
        with torch.no_grad():
            a_p = torch.softmax(self.model.predict_action(Zp, Mp, EFp, Zc, Mc, EFc)[0], -1).cpu().numpy()
        return float(a_p.max())

    def forward_lookahead(self, g_from, g_back, cands):
        """MODEL-only TRIANGLE lookahead — NO env stepping. For each candidate action, PREDICT the next-state
        latent st+2 from g_from via predict_next, then score how reversible st+2 -> g_back is via the inverse head:
        returns {key: (rev_entropy, rev_maxprob)}. HIGH entropy / LOW maxprob = the model cannot confidently get
        back = LESS reversible = more progress. cands = [(key, aid, xy, node)]. For a click, the conditioning is the
        clicked NODE's own latent (node-primary; perception + centroid + Mahalanobis), located by EXACT foot(px,py)→
        latent-row — NOT enumerate(g['nodes']) order, which align_slots re-ranks (that mismatch was the bp35
        regression; fixed). Verified by
        scratchpad/probe_latent_chain*.py: predict_next out-dim == trunk in-dim (62), the signal varies across
        candidates, z2 latents are distinct (real action-conditioned signal). Structure-approx: the predicted st+2
        reuses g_from's mask/edges (one-step structure ~ unchanged) — confirmed sufficient by the probe."""
        import numpy as np, torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat

        def _st(g):
            feat = _feat(g, g, self.cfg)
            Z = torch.from_numpy(cda.node_latents(feat, "cur", self.cfg)).float()[None].to(self.dev)
            M = torch.from_numpy(feat["mask_cur"].astype(bool))[None].to(self.dev)
            E = torch.from_numpy(cda.edge_feats(feat, "cur", self.cfg)).float()[None].to(self.dev)
            return Z, M, E, feat
        Z1, M1, EF1, feat1 = _st(g_from)            # st+1: predict st+2 FROM here (latents)
        Z0, M0, EF0, _ = _st(g_back)                # st:   the state to reverse BACK to
        foot = feat1["foot_cur"]; idx = np.where(feat1["mask_cur"].astype(bool))[0]
        # Map the clicked NODE to its LATENT ROW by its EXACT foot (px,py) key. NOT enumerate(g['nodes']) order:
        # align_slots re-ranks nodes (paired, area) into latent rows, so enumerate-index != latent-row — that mismatch
        # made half of bp35's clicks condition on the WRONG node (the regression). foot_cur shares Z1's slot space, so
        # the foot key is bijective and correct.
        foot2row = {(round(float(foot[i, 0]), 2), round(float(foot[i, 1]), 2)): int(i) for i in idx}

        def _hclick(node, xy):
            if xy is None:
                return None
            i = foot2row.get((round(float(xy[0]), 2), round(float(xy[1]), 2)), -1)   # exact clicked-node foot -> row
            if not (0 <= i < Z1.shape[1]):                     # fallback: nearest foot (xy not exactly on a node foot)
                if idx.size == 0:
                    return None
                d = (foot[idx, 0] - float(xy[0])) ** 2 + (foot[idx, 1] - float(xy[1])) ** 2
                i = int(idx[int(np.argmin(d))])
            return Z1[0, i][None]                              # the clicked NODE's own latent conditions predict_next
        out = {}
        with torch.no_grad():
            for cand in cands:
                key, aid, xy = cand[0], cand[1], cand[2]
                node = cand[3] if len(cand) > 3 else None      # node id (pid); absent for v1/v2 3-tuples -> px,py fallback
                try:
                    at = torch.tensor([int(aid)], dtype=torch.long, device=self.dev)
                    hc = _hclick(node, xy) if int(aid) == CLICK else None
                    z2, _alive = self.model.predict_next(Z1, M1, EF1, at, hc)       # predicted st+2 latent (no env step)
                    a_rev = torch.softmax(self.model.predict_action(z2, M1, EF1, Z0, M0, EF0)[0], -1).cpu().numpy()
                    out[key] = (float(-(a_rev * np.log(a_rev + 1e-12)).sum()), float(a_rev.max()))
                except Exception:
                    out[key] = (0.0, 1.0)                       # predict failed -> treat as fully reversible (worst)
        return out

    # ---- v3: ONLINE MC policy-gradient on the policy head (trunk + fwd/inv/click FROZEN) ----
    def train_setup(self, lr=1e-3):
        """Snapshot the original policy head, add a fresh ValueHead on the FROZEN trunk, and build an optimizer over
        ONLY the policy + value heads. Idempotent (no-op if already set up). Feasibility verified in
        scratchpad/feas_policy_head_online.py: head params update, trunk/fwd/inv/click bit-identical."""
        import copy, torch
        from curvature_wm_model.model.policy_head import ValueHead
        if getattr(self, "opt", None) is not None:
            return
        self.snapshot = copy.deepcopy(self.head.state_dict())          # original prior (rollback reference)
        self.value = ValueHead(self.cfg, d=self.head.act_out.in_features).to(self.dev)
        for p in self.model.parameters():                             # FREEZE trunk + forward/inverse/click
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(list(self.head.parameters()) + list(self.value.parameters()), lr=lr)
        self.head.train(); self.value.train()

    def _logits_value_grad(self, graph):
        """Grad path: trunk under no_grad -> detach H -> policy head (a/c logits) + value head (V), grad on the HEADS
        only. Returns (a_logits[na], c_logits[N], V scalar, foot2row)."""
        import numpy as np, torch
        import transform_catalyst.data_adapter as cda
        from curvature_wm_model.hodge_flow.diagnose_lr import _feat
        feat = _feat(graph, graph, self.cfg)
        Z = torch.from_numpy(cda.node_latents(feat, "cur", self.cfg)).float()[None].to(self.dev)
        M = torch.from_numpy(feat["mask_cur"].astype(bool))[None].to(self.dev)
        E = torch.from_numpy(cda.edge_feats(feat, "cur", self.cfg)).float()[None].to(self.dev)
        with torch.no_grad():
            H, _ = self.model.encode(Z, M, E)                        # FROZEN trunk
        Hd = H.detach()
        pos = Z[:, :, self.pos0:self.pos0 + 4]
        a_logits, c_logits = self.head(Hd, pos, M)
        V = self.value(Hd, pos, M)
        idx = np.where(feat["mask_cur"].astype(bool))[0]; foot = feat["foot_cur"]
        foot2row = {(round(float(foot[i, 0]), 2), round(float(foot[i, 1]), 2)): int(i) for i in idx}
        return a_logits[0], c_logits[0], V[0], foot2row

    def update(self, traj):
        """ONE MC policy-gradient step over traj=[(graph, aid, xy, G)]. Advantage A=G-V(s) (V detached for the policy
        term); policy loss covers the ACTION and (for clicks) the CLICK-NODE; value loss=(V-G)^2 (MC regression).
        Off-path transitions are simply absent from `traj` (reward 0 -> excluded). Returns the scalar loss."""
        import torch
        if not traj or getattr(self, "opt", None) is None:
            return 0.0
        pol, val = [], []
        for (graph, aid, xy, G) in traj:
            a_logits, c_logits, V, foot2row = self._logits_value_grad(graph)
            logp = torch.log_softmax(a_logits, -1)[int(aid)]
            if int(aid) == CLICK and xy is not None:                 # joint: + log P(node @ clicked foot)
                i = foot2row.get((round(float(xy[0]), 2), round(float(xy[1]), 2)), -1)
                if 0 <= i < c_logits.shape[0]:
                    logp = logp + torch.log_softmax(c_logits, -1)[i]
            adv = float(G) - V.detach()
            pol.append(-logp * adv)
            val.append((V - float(G)) ** 2)
        loss = torch.stack(pol).sum() + torch.stack(val).sum()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.detach())
