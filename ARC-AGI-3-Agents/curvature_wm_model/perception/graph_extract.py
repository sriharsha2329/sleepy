"""Per-object perception-graph extractor for the self-model.

For every human_runs/<game>/<uuid>.recording.jsonl, emit one JSONL of per-frame
PER-OBJECT graphs to  self_model_agent/graph_data/<game>/<uuid>.jsonl.  Each frame:

  node : {pid, color[list], stab, area, area_bin, cy, cx, type_hash, orbit, k}
         (cy,cx = centroid normalized to [0,1]; pid = GaugeTracker persistent id
          for cross-frame identity bootstrap; k = WL-class multiplicity this frame)
  edge : {src, dst, relation, dx, dy, dist}     (src/dst are pids; dx,dy centroid offset)
  plus : {t, frame_idx, level_idx, action, click, deltas}

Reuses the existing region_finder + perception (WL + GaugeTracker) stack — NO pixels
leave this stage; downstream the self-model consumes only these graphs.

Usage:  ./.venv/bin/python self_model_agent/graph_extract.py --jobs 8
        --game bp35   (one game)   --force
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent                      # curvature_wm/perception
CWM = THIS.parent                                           # curvature_wm/
# Big inputs/outputs are DATA (gitignored), consolidated under curvature_wm/data/.
HUMAN_RUNS = CWM / "data" / "human_runs"                     # raw recordings
OUT_ROOT = CWM / "data" / "graph_data"                       # per-object perception graphs

# Perception is imported as the installed `arc_perception` package (sibling feature),
# exposed under the legacy names `region_finder` / `perception` used in _imports().
try:
    import arc_perception.region_finder as _rf
    import arc_perception.core as _core
    sys.modules.setdefault("region_finder", _rf)
    sys.modules.setdefault("perception", _core)
except ImportError:
    pass

logging.disable(logging.CRITICAL)


def _action_id(raw) -> int:
    if isinstance(raw, bool):
        return -1
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else -1
    return -1


def _frame_grid(data):
    fr = data.get("frame")
    if not isinstance(fr, list) or not fr:
        return None
    import numpy as np
    g = fr[-1] if isinstance(fr[0], list) else fr
    a = np.array(g, dtype=np.int32)
    while a.ndim > 2:
        a = a[0]
    return a if a.ndim == 2 else None


def _process_frame(arr, tracker, mods):
    """Run perception on one frame → (nodes, edges, deltas) for OBJECT regions."""
    (estimate_background_label, segment_grid, build_regions, build_region_graph,
     assign_graph_orbits, type_hash, area_bin) = mods
    H, W = arr.shape
    bg = int(estimate_background_label(arr))
    raw, label_map = segment_grid(arr, background_label=bg, min_area=3)
    regions = build_regions(raw, label_map)
    G, edges = build_region_graph(regions, label_map, near_threshold=3.0)
    assign_graph_orbits(regions, G)
    assignments, deltas = tracker.update(regions, edges=edges)

    # Per-object SPATIAL FOOTPRINT (the bbox + x/y variance graph_data dropped). Each node is
    # a REGION, not a point: a cell maps to the node whose footprint contains it (Mahalanobis),
    # which guards against the naive nearest-centroid bug (a tiny node 6.5σ away).
    import numpy as np
    from region_finder.regions import region_mask
    foot = {}     # region_id -> (px, py, sx, sy)  in PIXELS (mask centroid + std)
    for _ridx, _r in enumerate(regions):
        _idx = raw[_ridx].get("_index") if isinstance(raw[_ridx], dict) else None
        _mask = (label_map == _idx) if _idx is not None else region_mask(_ridx, label_map)
        _ys, _xs = np.where(_mask)
        if len(_xs):
            foot[_r.region_id] = (float(_xs.mean()), float(_ys.mean()),
                                  float(_xs.std()), float(_ys.std()))
        else:
            foot[_r.region_id] = (float(_r.centroid[1]), float(_r.centroid[0]), 0.0, 0.0)

    objs = [r for r in regions
            if r.kind not in ("background_component", "background", "noise")]
    # WL-class multiplicity (the "xk" orbit-aggregated count) this frame
    th_of = {}
    counts = {}
    for r in objs:
        th = type_hash(r)
        th_of[r.region_id] = th
        counts[th] = counts.get(th, 0) + 1

    def pid_of(rid, fidx):
        return assignments.get(rid) or f"f{fidx}_{rid}"

    fidx = getattr(tracker, "_frame_idx_hint", 0)
    cmap = {}      # region_id -> (cy, cx) normalized
    nodes = []
    for r in objs:
        cy = float(r.centroid[0]) / max(1, H)
        cx = float(r.centroid[1]) / max(1, W)
        cmap[r.region_id] = (cy, cx)
        th = th_of[r.region_id]
        px, py, sx, sy = foot[r.region_id]
        by0, bx0, by1, bx1 = r.bbox      # [ymin, xmin, ymax, xmax]
        nodes.append({
            "pid": pid_of(r.region_id, fidx),
            "rid": r.region_id,
            "color": list(r.color) if r.color else [],
            "stab": int(r.features.get("object_stabilizer", 1)),
            "area": int(r.area),
            "area_bin": int(area_bin(r.area)),
            "cy": round(cy, 4), "cx": round(cx, 4),
            # --- per-object spatial footprint (pixels): mask centroid, x/y std, bbox ---
            "px": round(px, 2), "py": round(py, 2),
            "sx": round(sx, 2), "sy": round(sy, 2),
            "bbox": [int(by0), int(bx0), int(by1), int(bx1)],
            "type_hash": th,
            "orbit": r.features.get("graph_orbit_id", ""),
            "k": counts[th],
        })
    obj_ids = {r.region_id for r in objs}
    rid2pid = {r.region_id: pid_of(r.region_id, fidx) for r in objs}
    out_edges = []
    for e in edges:
        if e.source not in obj_ids or e.target not in obj_ids:
            continue
        sy, sx = cmap[e.source]
        ty, tx = cmap[e.target]
        feat = e.features or {}
        out_edges.append({
            "src": rid2pid[e.source],
            "dst": rid2pid[e.target],
            "relation": e.relation,
            "dx": round(tx - sx, 4), "dy": round(ty - sy, 4),
            "dist": round(float(feat.get("distance", ((tx - sx) ** 2 + (ty - sy) ** 2) ** 0.5)), 4),
        })
    return nodes, out_edges, [{"kind": d.kind, "detail": d.detail} for d in deltas]


def extract_trajectory(in_path: Path, out_path: Path, mods, GaugeTracker) -> dict:
    import numpy as np  # noqa: F401
    with in_path.open() as f:
        lines = f.readlines()
    summary = None
    try:
        last = json.loads(lines[-1]).get("data", {})
        if "won" in last:
            summary = last
    except Exception:
        pass
    summary = summary or {"won": False, "total_actions": len(lines)}
    traj_id = in_path.stem.split(".")[0]
    tracker = GaugeTracker(graveyard_ttl=30, history_limit=-1)

    rows = []
    prev_levels = 0
    for li, line in enumerate(lines):
        try:
            data = json.loads(line).get("data", {})
        except Exception:
            continue
        if "frame" not in data:
            continue
        arr = _frame_grid(data)
        if arr is None:
            continue
        tracker._frame_idx_hint = li
        ai = data.get("action_input", {}) or {}
        adata = ai.get("data", {}) or {}
        cx, cy = adata.get("x"), adata.get("y")
        click = ([int(cx), int(cy)]
                 if isinstance(cx, (int, float)) and isinstance(cy, (int, float)) else None)
        levels = int(data.get("levels_completed", 0))
        nodes, edges, deltas = _process_frame(arr, tracker, mods)
        rows.append({
            "t": len(rows), "frame_idx": li, "level_idx": levels,
            "action": _action_id(ai.get("id", -1)), "click": click,
            "nodes": nodes, "edges": edges, "deltas": deltas,
        })
        prev_levels = levels

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(json.dumps({
            "meta": True, "traj_id": traj_id, "n_frames": len(rows),
            "won": bool(summary.get("won", False)),
            "levels_completed": int(summary.get("levels_completed", prev_levels)),
            "total_actions": int(summary.get("total_actions", len(rows))),
            "source": str(in_path),
        }) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"traj_id": traj_id, "n_frames": len(rows), "out_path": str(out_path)}


def _imports():
    # region_finder / perception are aliased to arc_perception at module load (top of file)
    from region_finder.image_ops import estimate_background_label, segment_grid
    from region_finder.regions import build_regions
    from region_finder.graph import build_region_graph
    from perception import assign_graph_orbits, GaugeTracker
    from perception.wl import type_hash, area_bin
    mods = (estimate_background_label, segment_grid, build_regions,
            build_region_graph, assign_graph_orbits, type_hash, area_bin)
    return mods, GaugeTracker


def _alias_perception():
    """Expose arc_perception's packages as `region_finder` / `perception`. Must run in
    EACH process (loky workers don't re-run module-top code)."""
    if "region_finder" not in sys.modules:
        import arc_perception.region_finder as _rf
        import arc_perception.core as _core
        sys.modules["region_finder"] = _rf
        sys.modules["perception"] = _core


def _build_one(in_path: str, out_path: str, force: bool) -> dict:
    _alias_perception()                      # loky workers start fresh
    if (not force) and Path(out_path).exists() and Path(out_path).stat().st_size > 0:
        try:
            meta = json.loads(Path(out_path).open().readline())
            if meta.get("meta") and meta.get("source") == in_path:
                return {"traj_id": meta["traj_id"], "n_frames": meta["n_frames"],
                        "out_path": out_path, "cached": True}
        except Exception:
            pass
    mods, GaugeTracker = _imports()
    try:
        return extract_trajectory(Path(in_path), Path(out_path), mods, GaugeTracker)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        return {"traj_id": Path(in_path).stem[:8], "n_frames": 0,
                "out_path": out_path, "error": str(e)}


def main():
    from joblib import Parallel, delayed   # only needed for the parallel extractor CLI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    games = ([args.game] if args.game
             else sorted(p.name for p in HUMAN_RUNS.iterdir() if p.is_dir()))
    tasks = []
    for g in games:
        for f in sorted((HUMAN_RUNS / g).glob("*.recording.jsonl")):
            uuid = f.stem.split(".")[0]
            tasks.append((str(f), str(OUT_ROOT / g / f"{uuid}.jsonl")))

    print(f"Extracting per-object graphs for {len(tasks)} runs, {args.jobs} jobs ...",
          flush=True)
    results = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(_build_one)(i, o, args.force) for i, o in tasks)
    n_err = sum(1 for r in results if r and r.get("error"))
    tot = sum(r["n_frames"] for r in results if r)
    print(f"\nDone: {len(results)-n_err}/{len(results)} runs, {tot} frames, "
          f"{n_err} errors. Out → {OUT_ROOT}")


if __name__ == "__main__":
    main()
