"""Dataset: train/test split, streaming transitions, slot featurization, collate.

Trains on the per-object graphs in graph_data/, EXCLUDING the 25 per-game top runs
(test set from perception_layer/perception_data/top_runs.json). Streaming
IterableDataset with a shuffle buffer keeps memory light over ~170k transitions.
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from featurize import align_slots, click_target_slot, delta_multihot

THIS = Path(__file__).resolve().parent                      # curvature_wm/encoding
CWM = THIS.parent                                           # curvature_wm/
# Per-object graphs are DATA (gitignored), consolidated under curvature_wm/data/.
GRAPH_DATA = CWM / "data" / "graph_data"
TOP_RUNS = CWM.parent / "archive" / "cleanup" / "perception" / "perception_data" / "top_runs.json"


def build_splits(graph_dir=GRAPH_DATA, top_runs_json=TOP_RUNS):
    """(train_files, test_files). Test = the 25 per-game top-run uuids."""
    test_uuids = set()
    try:
        tr = json.loads(Path(top_runs_json).read_text())
        for v in tr.values():
            test_uuids.add(Path(v["recording"]).name.split(".")[0])
    except Exception:
        pass
    files = sorted(glob.glob(str(Path(graph_dir) / "*" / "*.jsonl")))
    train = [f for f in files if Path(f).stem not in test_uuids]
    test = [f for f in files if Path(f).stem in test_uuids]
    return train, test


def _r_ext(prev_level, cur_level, deltas):
    if cur_level > prev_level:
        return 1.0
    for d in deltas or []:
        sub = (d.get("detail", {}) or {}).get("type", "")
        if d.get("kind") == "death" or sub == "death_reset":
            return -1.0
    return 0.0


def iter_transitions(path, level_max=None):
    """Yield transitions. If level_max is set, only transitions whose PREV frame is at
    level_idx <= level_max (e.g. level_max=0 -> level-1 gameplay + its completion)."""
    rows = []
    with open(path) as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("meta"):
                continue
            rows.append(o)
    for t in range(1, len(rows)):
        prev, cur = rows[t - 1], rows[t]
        if level_max is not None and int(prev.get("level_idx", 0)) > level_max:
            continue
        yield {
            "prev_nodes": prev["nodes"], "cur_nodes": cur["nodes"],
            "prev_edges": prev["edges"], "cur_edges": cur["edges"],
            "a": int(cur.get("action", -1)), "click": cur.get("click"),
            "deltas": cur.get("deltas", []), "level": int(cur.get("level_idx", 0)),
            "r_ext": _r_ext(int(prev.get("level_idx", 0)),
                            int(cur.get("level_idx", 0)), cur.get("deltas")),
            "done": (t == len(rows) - 1),
        }


def featurize_transition(tr, cfg):
    arr = align_slots(tr["prev_nodes"], tr["cur_nodes"],
                      tr["prev_edges"], tr["cur_edges"], cfg)
    a = tr["a"]
    is_click = (a == 6) and (tr["click"] is not None)
    ct, cxy, c_dm, c_dn = click_target_slot(arr["pos_prev"], arr["mask_prev"],
                                            tr["click"] if is_click else None, cfg.grid,
                                            foot_prev=arr["foot_prev"])  # Step-1 §1.3.1 Mahalanobis i†
    click_mask = bool(is_click and ct >= 0)
    arr["a"] = np.int64(max(0, min(cfg.n_actions - 1, a if a >= 0 else 0)))
    arr["delta_multihot"] = delta_multihot(tr["deltas"], cfg.n_delta_kinds)
    arr["click_target"] = np.int64(ct if click_mask else -1)
    arr["click_xy"] = np.asarray(cxy, np.float32)
    arr["click_mask"] = np.bool_(click_mask)
    # Step-1 offset channel: Mahalanobis-normalized within-footprint offset δ̃ + geometric
    # confidence p_geo = exp(-d_M²/18) (attenuation, not rejection — far clicks whisper)
    arr["click_dnorm"] = np.asarray(c_dn if click_mask else (0.0, 0.0), np.float32)
    arr["click_pgeo"] = np.float32(np.exp(-(c_dm ** 2) / 18.0) if click_mask else 1.0)
    arr["r_ext"] = np.float32(tr["r_ext"])
    arr["done"] = np.bool_(tr["done"])
    arr["level"] = np.int64(tr["level"])
    return arr


class GraphTransitionStream(IterableDataset):
    def __init__(self, files, cfg, shuffle_buf=4096, seed=0, level_max=None):
        self.files = list(files)
        self.cfg = cfg
        self.buf = shuffle_buf
        self.seed = seed
        self.epoch = 0
        self.level_max = level_max

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        files = list(self.files)
        rng.shuffle(files)
        wi = get_worker_info()
        if wi is not None:
            files = files[wi.id::wi.num_workers]
        buf = []
        for fp in files:
            for tr in iter_transitions(fp, self.level_max):
                feat = featurize_transition(tr, self.cfg)
                if len(buf) < self.buf:
                    buf.append(feat)
                else:
                    j = rng.randrange(len(buf))
                    yield buf[j]
                    buf[j] = feat
        rng.shuffle(buf)
        yield from buf


_LONG = {"stab_prev", "stab_cur", "areabin_prev", "areabin_cur", "wl_prev", "wl_cur",
         "edge_rel_prev", "edge_rel_cur", "a", "click_target", "level"}
_BOOL = {"mask_prev", "mask_cur", "click_mask", "done"}


def collate(batch, cfg=None):
    out = {}
    for k in batch[0]:
        vals = np.stack([b[k] for b in batch])
        t = torch.as_tensor(vals)
        if k in _LONG:
            t = t.long()
        elif k in _BOOL:
            t = t.bool()
        else:
            t = t.float()
        out[k] = t
    return out
