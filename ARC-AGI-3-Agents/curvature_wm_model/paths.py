"""sys.path + offline-env wiring for curvature_wm (now SELF-CONTAINED).

The perception API and the graph->latent encoding substrate were CONSOLIDATED into this folder
(no longer imported from archive/). Importing this module is idempotent and makes the following
resolve from the LOCAL copies:

  from config import Config                                            # encoding/config.py
  from transform_catalyst.data_adapter import obj_dim, node_latents, edge_feats, EDGE_DIM, N_RELS
  from data import iter_transitions, featurize_transition             # encoding/data.py
  from featurize import align_slots, click_target_slot               # encoding/featurize.py
  import arc_perception        (PerceptionLayer, ...)                  # perception/arc_perception
  import graph_extract         (frame -> per-object graph)            # perception/graph_extract.py

It also wires OFFLINE game play (no network): OPERATION_MODE=offline and ENVIRONMENTS_DIR pointed at
the local games in env/environment_files, so `arc_agi.Arcade().make(<game>)` loads them locally.

The big DATA now lives under data/ too (gitignored): data/human_runs (~6.5 GB raw recordings) and
data/graph_data (~4.3 GB per-object perception graphs, produced by perception/graph_extract).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent              # curvature_wm/
REPO = HERE.parent                                  # repo root

PERC = HERE / "perception"                          # arc_perception package + graph_extract.py
ENC = HERE / "encoding"                             # config.py, data.py, featurize.py, transform_catalyst/
ENV_FILES = HERE / "env" / "environment_files"      # local OFFLINE games (arc_agi reads via ENVIRONMENTS_DIR)

# Big DATA is SHARED with the source tree (NOT duplicated into this copy): recordings (HUD detection) +
# per-object perception graphs (training inputs). REPO = repo root; caches live under curvature_wm/data.
HUMAN_RUNS = REPO / "curvature_wm" / "data" / "human_runs"          # SHARED raw recordings
GRAPH_DATA = REPO / "curvature_wm" / "data" / "graph_data"          # SHARED per-object perception graphs

# --- local substrate on sys.path (source roots, exactly like the archive dirs they replace) ---
for _p in (str(PERC), str(ENC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- default to the ARC-AGI API (online); OnlineEnv also forces it. ENVIRONMENTS_DIR kept for any offline use. ---
os.environ.setdefault("OPERATION_MODE", "online")
os.environ.setdefault("ENVIRONMENTS_DIR", str(ENV_FILES))
