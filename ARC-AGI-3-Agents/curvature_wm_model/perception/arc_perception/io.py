"""Loaders for human-run recordings (.recording.jsonl) and convenience runners.

A recording is a JSONL file where most lines carry a per-frame `data.frame` grid
plus `data.action_input` (action id + optional click x,y) and `data.levels_completed`.
The last line may be a trajectory summary with `won` / `levels_completed`.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def action_id(raw) -> int:
    """Normalise an action id ('ACTION6', 6, ...) to int; -1 if absent."""
    if isinstance(raw, bool):
        return -1
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else -1
    return -1


def iter_recording_frames(path, max_frames: Optional[int] = None) -> Iterator[dict]:
    """Yield per-frame dicts from a .recording.jsonl in order.

    Each item: {frame: np.ndarray[H,W], frame_idx: int, action: int,
                click: [x,y]|None, levels: int}
    """
    path = Path(path)
    n = 0
    with path.open() as f:
        for line_idx, line in enumerate(f):
            if max_frames is not None and n >= max_frames:
                return
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data = obj.get("data", obj)
            if not isinstance(data, dict) or "frame" not in data:
                continue
            ff = data["frame"]
            if not (isinstance(ff, list) and ff):
                continue
            grid = ff[-1] if isinstance(ff[0], list) else ff
            arr = np.array(grid, dtype=np.int32)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim != 2:
                continue
            ai = data.get("action_input", {}) or {}
            adata = ai.get("data", {}) or {}
            cx, cy = adata.get("x"), adata.get("y")
            click = ([int(cx), int(cy)]
                     if isinstance(cx, (int, float)) and isinstance(cy, (int, float))
                     else None)
            yield {
                "frame": arr,
                "frame_idx": line_idx,
                "action": action_id(ai.get("id", -1)),
                "click": click,
                "levels": int(data.get("levels_completed", 0)),
            }
            n += 1


def recording_meta(path) -> dict:
    """Best-effort trajectory summary (won, levels_completed) from the last line."""
    path = Path(path)
    try:
        with path.open() as f:
            lines = f.readlines()
        last = json.loads(lines[-1]).get("data", {})
        if "won" in last:
            return {
                "won": bool(last.get("won", False)),
                "levels_completed": int(last.get("levels_completed", 0)),
                "total_actions": int(last.get("total_actions", len(lines))),
            }
    except Exception:
        pass
    return {"won": False, "levels_completed": 0, "total_actions": 0}


def list_recordings(human_runs_dir, game: str) -> list[Path]:
    """All .recording.jsonl paths for a game under a human_runs directory."""
    return [Path(p) for p in sorted(
        glob.glob(str(Path(human_runs_dir) / game / "*.recording.jsonl")))]
