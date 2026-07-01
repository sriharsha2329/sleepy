"""curvature_wm — a clean ~100K-weight graph-transformer world model.

Single shared trunk over the current state's object graph + three heads:
  forward (predict next latent state z_next), inverse (predict action type), click (predict clicked node).
No wake/sleep, no brain pool. Trains from human runs across all games except a held-out set.
All path wiring lives in this package's `paths.py`; the archived infra under archive/ is imported READ-ONLY.
"""
