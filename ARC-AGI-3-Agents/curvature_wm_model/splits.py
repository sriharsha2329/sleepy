"""Game-level train / held-out split. Held-out games are excluded from ALL training so we measure
cross-game generalization (not just held-out transitions inside training games).

Held-out picked by graph_data coverage: bp35 (canonical, 14 runs) + sc25 (15) + wa30 (14) — solid eval
coverage while the large lp85 (54) stays in training.
"""
from curvature_wm_model import paths  # noqa: F401  (sets sys.path; exposes GRAPH_DATA)

ALL_GAMES = sorted([p.name for p in paths.GRAPH_DATA.iterdir() if p.is_dir()]) if paths.GRAPH_DATA.exists() else []
HELDOUT_GAMES = ["bp35", "sc25", "wa30"]
TRAIN_GAMES = [g for g in ALL_GAMES if g not in HELDOUT_GAMES]


def is_heldout(game: str) -> bool:
    return game in HELDOUT_GAMES
