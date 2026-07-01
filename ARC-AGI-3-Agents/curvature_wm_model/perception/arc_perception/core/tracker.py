"""Gauge-consistent temporal tracker using 2-WL pair-space matching.

Tracks ALL frames for the entire game session.  History is kept in memory
until reset() is called (game stop / game switch).  Objects that disappear
stay in a graveyard indefinitely — if they reappear later with the same
type_hash near the same position, their old PID is resurrected.

Pipeline per frame:
  1. Extract (type_hash, colour, centroid) for each region
  2. Match against active objects: decomposed Hungarian by type-hash group
  3. Match leftovers against graveyard (resurrection)
  4. Match remaining leftovers by colour (mutation detection)
  5. Births / deaths / mutations / moves emitted as DeltaEvents
  6. Full frame snapshot stored in history
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .wl import type_hash as compute_type_hash, area_bin, pair_match_decomposed


@dataclass
class TrackedObject:
    pid: str
    type_hash: str
    shape_hash: str
    colour: tuple
    area_bin: int
    centroid: tuple[float, float]
    lifetime: int = 1
    last_seen_frame: int = 0


@dataclass
class DeltaEvent:
    kind: str          # "birth" | "death" | "mutation" | "move" | "resurrect"
                       # | "state_change" | "transition"
                       # | "edge_birth" | "edge_death"
    pid: str
    detail: dict = field(default_factory=dict)


@dataclass
class FrameSnapshot:
    frame_idx: int
    assignments: dict[str, str]       # region_id → pid
    deltas: list[DeltaEvent]
    objects: dict[str, str]            # pid → type_hash (lightweight)


class GaugeTracker:
    """Gauge-consistent tracker with full game history.

    Tracks every frame from game start to game stop.  reset() clears
    everything (call on game switch).
    """

    def __init__(self, hash_mismatch_cost: float = 50.0,
                 colour_mismatch_cost: float = 200.0,
                 max_match_cost: float = 100.0,
                 large_transition_threshold: int = 3,
                 graveyard_ttl: int = 30,
                 history_limit: int = 0):
        """
        graveyard_ttl: drop dead objects from graveyard after this many
                       frames missing (prevents O(N) growth of matching cost).
        history_limit: max FrameSnapshots to retain (0 = unlimited).  Set
                       to e.g. 100 to bound memory during long sessions.
        """
        self._active: dict[str, TrackedObject] = {}
        self._graveyard: dict[str, TrackedObject] = {}
        self._history: list[FrameSnapshot] = []
        self._next_id: int = 1
        self._frame_idx: int = 0
        self._hash_mismatch = hash_mismatch_cost
        self._colour_mismatch = colour_mismatch_cost
        self._max_cost = max_match_cost
        self._large_threshold = large_transition_threshold
        self._level_start_sig: tuple = ()
        self._level_idx: int = 0
        self._prev_edges: set[tuple] = set()
        self._graveyard_ttl = graveyard_ttl
        self._history_limit = history_limit

    def _new_pid(self) -> str:
        pid = f"P{self._next_id:04d}"
        self._next_id += 1
        return pid

    def _extract(self, region) -> dict:
        colour = tuple(region.color) if region.color else ()
        centroid = (float(region.centroid[0]), float(region.centroid[1]))
        thash = compute_type_hash(region)
        abin = area_bin(region.area)
        shash = region.features.get("canonical_shape_hash", "")
        return {
            "type_hash": thash,
            "shape_hash": shash,
            "colour": colour,
            "area_bin": abin,
            "centroid": centroid,
        }

    def update(self, regions: list,
              edges: list | None = None) -> tuple[dict[str, str], list[DeltaEvent]]:
        """Match current-frame regions to tracked objects.

        Args:
            regions: list of Region objects from perception pipeline.
            edges:   list of EdgeRecord objects (source, target, relation).
                     Used for edge-level delta tracking.

        Returns (assignments, deltas).
        """
        self._frame_idx += 1
        curr_info: dict[str, dict] = {}
        for r in regions:
            if r.kind in ("background", "background_component", "noise"):
                continue
            curr_info[r.region_id] = self._extract(r)

        active_list = list(self._active.values())
        curr_entries = list(curr_info.items())
        assignments: dict[str, str] = {}
        deltas: list[DeltaEvent] = []

        # --- Pass 1: match against active objects ---
        if active_list and curr_entries:
            old_objs = [{"type_hash": o.type_hash, "colour": o.colour,
                         "centroid": o.centroid} for o in active_list]
            new_objs = [{"type_hash": info["type_hash"], "colour": info["colour"],
                         "centroid": info["centroid"]}
                        for _, info in curr_entries]

            results = pair_match_decomposed(
                old_objs, new_objs,
                hash_mismatch_cost=self._hash_mismatch,
                colour_mismatch_cost=self._colour_mismatch,
                max_match_cost=self._max_cost,
            )

            matched_old: set[int] = set()
            matched_new: set[int] = set()

            for old_idx, new_idx, cost in results:
                if old_idx is not None and new_idx is not None:
                    old = active_list[old_idx]
                    rid, info = curr_entries[new_idx]
                    matched_old.add(old_idx)
                    matched_new.add(new_idx)
                    assignments[rid] = old.pid
                    self._emit_change(deltas, old, info)
                    self._active[old.pid] = self._make_tracked(
                        old.pid, info, old.lifetime + 1)

            # Move unmatched old → graveyard
            for i, old in enumerate(active_list):
                if i not in matched_old:
                    deltas.append(DeltaEvent("death", old.pid, {
                        "type_hash": old.type_hash, "colour": old.colour,
                        "last_centroid": old.centroid,
                    }))
                    self._graveyard[old.pid] = old
                    del self._active[old.pid]

            # Collect unmatched new for pass 2
            unmatched_new = [(j, curr_entries[j])
                             for j in range(len(curr_entries))
                             if j not in matched_new]
        elif not active_list:
            unmatched_new = list(enumerate(curr_entries))
        else:
            unmatched_new = []
            for old in active_list:
                deltas.append(DeltaEvent("death", old.pid, {
                    "type_hash": old.type_hash, "colour": old.colour}))
                self._graveyard[old.pid] = old
            self._active.clear()

        # --- Pass 2: match unmatched new against graveyard (resurrection) ---
        still_unmatched = []
        if unmatched_new and self._graveyard:
            grave_list = list(self._graveyard.values())
            grave_objs = [{"type_hash": g.type_hash, "colour": g.colour,
                           "centroid": g.centroid} for g in grave_list]
            new_objs = [{"type_hash": info["type_hash"], "colour": info["colour"],
                         "centroid": info["centroid"]}
                        for _, (_, info) in unmatched_new]

            results = pair_match_decomposed(
                grave_objs, new_objs,
                hash_mismatch_cost=self._hash_mismatch,
                colour_mismatch_cost=self._colour_mismatch,
                max_match_cost=self._max_cost,
            )

            matched_grave: set[int] = set()
            matched_unew: set[int] = set()

            for gi, ni, cost in results:
                if gi is not None and ni is not None:
                    grave = grave_list[gi]
                    orig_j, (rid, info) = unmatched_new[ni]
                    matched_grave.add(gi)
                    matched_unew.add(ni)
                    assignments[rid] = grave.pid
                    frames_missing = self._frame_idx - grave.last_seen_frame
                    deltas.append(DeltaEvent("resurrect", grave.pid, {
                        "type_hash": info["type_hash"],
                        "frames_missing": frames_missing,
                    }))
                    self._active[grave.pid] = self._make_tracked(
                        grave.pid, info, grave.lifetime + 1)
                    del self._graveyard[grave.pid]

            still_unmatched = [unmatched_new[ni]
                               for ni in range(len(unmatched_new))
                               if ni not in matched_unew]
        else:
            still_unmatched = unmatched_new

        # --- Pass 3: remaining unmatched new → births ---
        for _, (rid, info) in still_unmatched:
            pid = self._new_pid()
            assignments[rid] = pid
            deltas.append(DeltaEvent("birth", pid, {
                "type_hash": info["type_hash"],
                "colour": info["colour"],
            }))
            self._active[pid] = self._make_tracked(pid, info, 1)

        # Edge-level delta tracking (quotient-graph edges by type_hash)
        if edges is not None:
            rid_to_type = {}
            for r in regions:
                if r.region_id in curr_info:
                    rid_to_type[r.region_id] = curr_info[r.region_id]["type_hash"]
            curr_edges: set[tuple] = set()
            for e in edges:
                ta = rid_to_type.get(e.source)
                tb = rid_to_type.get(e.target)
                if ta and tb and ta != tb:
                    key = (min(ta, tb), max(ta, tb), e.relation)
                    curr_edges.add(key)
            edge_births = curr_edges - self._prev_edges
            edge_deaths = self._prev_edges - curr_edges
            for eb in edge_births:
                deltas.append(DeltaEvent("edge_birth", "EDGE", {
                    "source_type": eb[0], "target_type": eb[1],
                    "relation": eb[2],
                }))
            for ed in edge_deaths:
                deltas.append(DeltaEvent("edge_death", "EDGE", {
                    "source_type": ed[0], "target_type": ed[1],
                    "relation": ed[2],
                }))
            self._prev_edges = curr_edges

        # Classify large transitions: level_change vs death_reset
        n_births = sum(1 for d in deltas if d.kind == 'birth')
        n_deaths = sum(1 for d in deltas if d.kind == 'death')
        n_resurrect = sum(1 for d in deltas if d.kind == 'resurrect')
        is_large = (n_births + n_deaths) >= self._large_threshold

        transition_type = "none"
        if is_large:
            curr_sig = tuple(sorted(
                obj.type_hash for obj in self._active.values()))

            if not self._level_start_sig:
                # First frame — set as level start
                self._level_start_sig = curr_sig
                transition_type = "level_start"
            elif n_resurrect > n_births:
                # More resurrections than fresh births → death/reset
                transition_type = "death_reset"
            else:
                # Compare against level-start signature
                s_old = set(self._level_start_sig)
                s_new = set(curr_sig)
                union = s_old | s_new
                similarity = len(s_old & s_new) / len(union) if union else 1.0
                if similarity > 0.7:
                    transition_type = "death_reset"
                else:
                    transition_type = "level_change"
                    self._level_idx += 1
                    self._level_start_sig = curr_sig

            deltas.append(DeltaEvent("transition", "GAME", {
                "type": transition_type,
                "births": n_births,
                "deaths": n_deaths,
                "resurrections": n_resurrect,
                "level_idx": self._level_idx,
            }))

        # Evict stale objects from graveyard (TTL-bounded)
        if self._graveyard_ttl > 0:
            stale = [pid for pid, obj in self._graveyard.items()
                     if self._frame_idx - obj.last_seen_frame > self._graveyard_ttl]
            for pid in stale:
                del self._graveyard[pid]

        # Lightweight snapshot — only store {pid → type_hash}, not deep copies.
        # Full TrackedObject state is in self._active (current frame only).
        if self._history_limit != -1:
            snapshot = FrameSnapshot(
                frame_idx=self._frame_idx,
                assignments=dict(assignments),
                deltas=list(deltas),
                objects={pid: obj.type_hash
                         for pid, obj in self._active.items()},
            )
            self._history.append(snapshot)
            if self._history_limit > 0 and len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit:]

        return assignments, deltas

    def _make_tracked(self, pid: str, info: dict, lifetime: int) -> TrackedObject:
        return TrackedObject(
            pid=pid, type_hash=info["type_hash"],
            shape_hash=info.get("shape_hash", ""),
            colour=info["colour"], area_bin=info["area_bin"],
            centroid=info["centroid"], lifetime=lifetime,
            last_seen_frame=self._frame_idx,
        )

    def _emit_change(self, deltas: list, old: TrackedObject, info: dict):
        dr = info["centroid"][0] - old.centroid[0]
        dc = info["centroid"][1] - old.centroid[1]
        new_shape = info.get("shape_hash", "")
        if old.type_hash != info["type_hash"]:
            deltas.append(DeltaEvent("mutation", old.pid, {
                "old_hash": old.type_hash,
                "new_hash": info["type_hash"],
                "colour": info["colour"],
            }))
        elif old.shape_hash and new_shape and old.shape_hash != new_shape:
            deltas.append(DeltaEvent("state_change", old.pid, {
                "old_shape": old.shape_hash,
                "new_shape": new_shape,
                "type_hash": old.type_hash,
            }))
        elif abs(dr) + abs(dc) > 0.5:
            deltas.append(DeltaEvent("move", old.pid, {
                "dy": dr, "dx": dc,
            }))

    def lifetime(self, pid: str) -> int:
        obj = self._active.get(pid) or self._graveyard.get(pid)
        return obj.lifetime if obj else 0

    @property
    def frame_count(self) -> int:
        return self._frame_idx

    @property
    def history(self) -> list[FrameSnapshot]:
        return self._history

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def graveyard_count(self) -> int:
        return len(self._graveyard)

    def summary(self) -> dict:
        """Current tracker state for API/UI."""
        return {
            "frame_count": self._frame_idx,
            "active": self.active_count,
            "graveyard": self.graveyard_count,
            "total_pids_issued": self._next_id - 1,
            "history_frames": len(self._history),
        }

    def reset(self):
        self._active.clear()
        self._graveyard.clear()
        self._history.clear()
        self._next_id = 1
        self._frame_idx = 0
        self._level_start_sig = ()
        self._level_idx = 0
        self._prev_edges = set()
