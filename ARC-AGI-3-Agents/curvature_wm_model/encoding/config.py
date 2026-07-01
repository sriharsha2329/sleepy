"""Config for the brain-plausible ARC agent (ARC_agent_methodology). MPS, .venv."""
from __future__ import annotations

from dataclasses import dataclass

# Perception is a SIBLING feature (cleanup/perception), imported as the installed
# `arc_perception` package — not nested here. Expose its packages under the legacy
# top-level names the world-model code imports (`region_finder`, `perception`).
import sys as _sys
try:
    import arc_perception.region_finder as _rf
    import arc_perception.core as _core
    _sys.modules.setdefault("region_finder", _rf)
    _sys.modules.setdefault("perception", _core)
except ImportError:  # arc_perception not installed -> caller must add it to the env
    pass

import torch


@dataclass
class BrainConfig:
    # --- dims (reuse the self_model_agent world-model width) ---
    d: int = 64                 # model width / belief width
    d_h: int = 48               # type_hash (WL class) embedding
    heads: int = 4
    n_layers: int = 3           # encoder depth (structural-biased attention)
    n_max: int = 32             # objects/slots N_max
    n_actions: int = 8          # action ids 0..7 (6 = click)
    th_vocab: int = 4096
    n_colors: int = 16
    n_stab: int = 9             # D4 stabilizer classes (symmetry type tau) 1..8 ; 0 pad
    n_areabin: int = 16
    n_relations: int = 4
    de_rel: int = 16
    n_delta_kinds: int = 12
    grid: int = 64
    d_ff: int = 256

    # --- self_model_agent SelfModel/ValueEnsemble compatibility (REUSE world model) ---
    tau: float = 0.99           # EMA target encoder decay
    lambda_inv: float = 1.0
    lambda_reg: float = 1.0
    lambda_topo: float = 0.5
    lambda_click: float = 0.5
    lambda_match: float = 0.5
    lambda_pres: float = 0.5
    k_val: int = 4              # value ensemble heads (for imagine_one / leverage)
    n0: int = 5
    v0: float = 0.5
    rho: float = 0.5
    theta_prune: float = 0.1
    tau_prune: float = 0.5
    dead_c: float = 1.0
    sleep_pressure: float = 200.0

    # --- §6 Hebbian continuum memory ---
    mem_levels: int = 3         # working, episodic, mid (semantic = slow weights theta)
    mem_tau0: float = 2.0       # base timescale; ladder tau_l = tau0 * ladder**l
    mem_ladder: float = 8.0     # geometric ladder ratio (levels do not alias)
    mem_eta: float = 0.5        # Hebbian write rate
    belief_dim: int = 64        # GRU-gated belief state width (== d)

    # --- §7 multi-expert ring + QRE ---
    # More clusters = finer specialization by game-state TYPE (with structural routing): click-
    # heavy states and move states reach DIFFERENT experts, so training on one game doesn't
    # overwrite the others' experts (interference resistance). Only helps WITH structural routing
    # (ring_coord on the quotient signature) — with belief routing every state hit one set.
    n_experts: int = 12
    ring_vnodes: int = 32       # virtual nodes per expert on the consistent-hashing ring
    ring_N: int = 4             # |R(z)| clockwise successors = consensus set size
    qre_lambda: float = 4.0     # quantal-response temperature (lambda->inf = hard Nash)
    qre_iters: int = 8          # fixed-point iterations

    # --- §8 adenosine (Process S) ---
    aden_eta: float = 0.05      # pressure accrual rate
    aden_max: float = 50.0      # A_max
    aden_theta_up: float = 8.0  # sleep trigger
    aden_theta_dn: float = 1.0  # wake threshold
    aden_lambda_M: float = 0.1  # weight on synaptic work ||dM||^2 in load
    sleep_passes: int = 30
    aden_refractory: int = 20   # min wake steps between sleeps — caps "sleep storms" in the
                                # high-surprise (calibration) region so GRPO can't over-update
                                # and collapse the policy. Pressure keeps accruing; sleep waits.

    # --- §9 replay ---
    buffer_cap: int = 50000
    prio_lambda_delta: float = 1.0
    prio_lambda_cov: float = 0.2
    prio_lambda_stab: float = 0.3   # up-weight symmetric (large-stabilizer) states (§9.2 pivotal)

    # --- §9.2 Lie continuous symmetry ---
    lie_m: int = 6                  # number of candidate infinitesimal generators
    lie_t: float = 0.15             # one-parameter-subgroup step for the equivariance loss
    lie_ridge: float = 1e-3         # ridge in the transverse pseudo-inverse solve
    lie_margin: float = 0.25        # relative-equiv gate threshold (scale-free graceful degrade)
    beta_int: float = 0.5           # intrinsic-reward weight on ‖P⊥Δz‖ (symmetry-aware curiosity)
    lambda_equiv: float = 1.0       # equivariance-loss weight in sleep consolidation
    lambda_nov: float = 0.3         # count-based novelty reward λ/√count(orbit) — dense shaping
                                    # toward unseen QUOTIENT states (WL orbits), not symmetric copies

    # --- §10 counterfactual ---
    # NOTE: rung-3 abduction (seed-pinning) is DEFERRED — the reused imagine_one re-encodes from
    # the graph and cannot ingest the abducted residual u, so seeds don't change the imagined
    # outcome (they cancel in the paired baseline). We therefore run cf_seeds=1 and treat the
    # sibling comparison as rung-2 INTERVENTIONAL (still a real advantage signal now that the
    # value head is trained). Restore K>1 once a seed-conditioned forward rollout exists.
    cf_seeds: int = 1
    cf_siblings: int = 4        # counterfactual sibling actions per pivotal step

    # --- §11 GRPO ---
    grpo_group: int = 4         # G rollouts per state
    grpo_horizon: int = 3       # imagined latent rollout depth
    grpo_clip: float = 0.2      # eps_c
    grpo_beta_kl: float = 0.05  # KL to BC reference
    grpo_entropy: float = 0.05  # entropy floor — prevents policy collapse to the click majority
    grpo_weight: float = 0.3    # down-weight GRPO vs the (human-plausibility) BC anchor. GRPO's
                                # value reward favors the high-δ click (noisy-TV curiosity trap);
                                # BC must dominate the policy in imitation or it collapses to click.
    surprise_bonus: float = 0.5 # intrinsic reward weight (within-group variance early)

    # --- training ---
    gamma: float = 0.99
    lr: float = 3e-4
    lr_router: float = 5e-5     # router/encoder trained slowly (avoid routing drift)
    batch: int = 16
    seed: int = 0

    @property
    def edge_dim(self) -> int:
        return self.de_rel


# Alias so any reused self_model_agent code doing `from config import Config` resolves to the
# (superset) BrainConfig when brain_agent's config wins the import.
Config = BrainConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
