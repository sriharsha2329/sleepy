"""Step 1 (HARD CONSTRAINT 5) — reproducibility scaffolding. Nothing below it is measurable
without this: order-dependent eval gave non-reproducible numbers before.

  * seed_all          — torch / numpy / python / mps seeds (k-means seeding lives in holonomy.py)
  * frozen_eval(*mods)— eval with test-time Hebbian/memory writes OFF + no_grad (eval must not learn)

The "pinned encoder" is the frozen object codebook (deterministic argmin over per-object features);
there is no learned encoder to drift, so pinning = using that frozen codebook, which the data
adapter already does.
"""
from __future__ import annotations

import contextlib
import random

import numpy as np
import torch


def seed_all(seed: int = 0) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


@contextlib.contextmanager
def frozen_eval(*modules):
    """Eval with weights frozen AND test-time writes disabled. Any module exposing a `no_write`
    flag (Hebbian memories) has it set True for the duration; restored on exit."""
    prev_train = [(m, getattr(m, "training", False)) for m in modules]
    prev_write = [(m, getattr(m, "no_write", None)) for m in modules]
    for m in modules:
        if hasattr(m, "eval"):
            m.eval()
        if hasattr(m, "no_write"):
            m.no_write = True
    try:
        with torch.no_grad():
            yield
    finally:
        for m, t in prev_train:
            if hasattr(m, "train"):
                m.train(t)
        for m, w in prev_write:
            if w is not None:
                m.no_write = w
