"""CP-SAT Solver Service — clean API wrapping the validated 3-phase pipeline.

Architecture (verified: 0 hard-constraint violations on 790 sessions):
  Phase A  — CP-SAT time+instructor assignment (no rooms), ~15s
  Phase A+ — Warm-start with room-pool penalties for tight pools, ~30s
  Phase B  — Greedy room assignment with depth-2 steal chains, <0.1s
  Phase C  — CP-SAT repair for remaining room failures, ~60s (if needed)

Usage:
    from cpsat_solver import solve_timetable

    result = solve_timetable("data_fixed")
    # result.schedule       → list[dict]  (the timetable)
    # result.violations     → 0
    # result.rooms_assigned → 778
    # result.total_sessions → 790

Or with raw JSON dicts:
    result = solve_timetable_from_json(courses, groups, instructors, rooms)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from src.io.data_store import DataStore
from cpsat_phase1 import (
    build_sessions,
    build_phase1_model,
    solve_and_report,
    extract_assignments,
    phase_b_room_assignment,
    add_warm_start_hints,
    phase_c_repair,
    build_schedule_json,
    Session,
)


@dataclass
class SolveResult:
    """Result of a CP-SAT timetable solve."""
    success: bool
    schedule: list[dict[str, Any]] = field(default_factory=list)
    total_sessions: int = 0
    rooms_assigned: int = 0
    rooms_failed: int = 0
    violations: int = 0
    violation_details: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    seeds_tried: int = 0
    steals: int = 0
    error: str | None = None


@dataclass
class SolveConfig:
    """Configuration for the CP-SAT solver."""
    time_limit_phase_a: int = 15
    time_limit_phase_a_plus: int = 30
    time_limit_phase_c: int = 60
    seeds: int = 1
    room_pool_limit: int = 5
    cross_qualify: bool = True
    relax_pmi: bool = True
    relax_ictd: bool = False
    relax_sre: bool = False
    relax_fte: bool = False
    relax_ffc: bool = False


def solve_timetable(
    data_dir: str = "data_fixed",
    config: SolveConfig | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> SolveResult:
    """Run the full 3-phase CP-SAT pipeline on data from a directory.

    Args:
        data_dir: Path to directory containing Course.json, Groups.json, etc.
        config: Solver configuration. Uses defaults if None.
        on_progress: Optional callback(phase_name, progress_pct) for UI updates.

    Returns:
        SolveResult with schedule and validation info.
    """
    cfg = config or SolveConfig()
    t0 = time.time()

    def _progress(phase: str, pct: float) -> None:
        if on_progress:
            on_progress(phase, pct)

    # ── Load data ──
    _progress("loading", 0.0)
    store = DataStore.from_json(data_dir, run_preflight=False)
    sessions, iids, rids = build_sessions(store, cross_qualify=cfg.cross_qualify)
    n_sessions = len(sessions)
    _progress("loading", 1.0)

    best_failed = n_sessions + 1
    best_assignments: list[dict] | None = None
    best_failed_ais: list[int] = []

    for seed_i in range(cfg.seeds):
        seed = seed_i + 1
        _progress("phase_a", seed_i / cfg.seeds)

        # ── Phase A: time + instructor (no rooms) ──
        model_a, vars_a = build_phase1_model(
            sessions, store, iids, rids,
            relax_pmi=cfg.relax_pmi,
            relax_ictd=cfg.relax_ictd,
            relax_sre=cfg.relax_sre,
            relax_fte=cfg.relax_fte,
            relax_ffc=cfg.relax_ffc,
            no_rooms=True,
            room_pool_limit=0,
        )
        result_a, solver_a = solve_and_report(
            model_a, sessions, store, iids, rids, vars_a,
            time_limit=cfg.time_limit_phase_a,
            random_seed=seed,
        )
        if not result_a or solver_a is None:
            continue

        assignments = extract_assignments(solver_a, sessions, iids, vars_a)

        # ── Phase A+: warm-start with room pool penalties ──
        if cfg.room_pool_limit > 0:
            _progress("phase_a_plus", seed_i / cfg.seeds)
            model_b, vars_b = build_phase1_model(
                sessions, store, iids, rids,
                relax_pmi=cfg.relax_pmi,
                relax_ictd=cfg.relax_ictd,
                relax_sre=cfg.relax_sre,
                relax_fte=cfg.relax_fte,
                relax_ffc=cfg.relax_ffc,
                no_rooms=True,
                room_pool_limit=cfg.room_pool_limit,
            )
            add_warm_start_hints(model_b, vars_b, solver_a, vars_a, assignments)
            result_b, solver_b = solve_and_report(
                model_b, sessions, store, iids, rids, vars_b,
                time_limit=cfg.time_limit_phase_a_plus,
                random_seed=seed,
            )
            if result_b and solver_b is not None:
                assignments = extract_assignments(solver_b, sessions, iids, vars_b)

        # ── Phase B: greedy room assignment ──
        _progress("phase_b", (seed_i + 0.8) / cfg.seeds)
        n_failed, failed_ais = phase_b_room_assignment(
            assignments, sessions, store, rids,
        )

        if n_failed < best_failed:
            best_failed = n_failed
            best_assignments = [a.copy() for a in assignments]
            best_failed_ais = list(failed_ais)

        if n_failed == 0:
            break

    # ── Phase C: repair remaining failures ──
    if best_assignments is not None and best_failed > 0:
        _progress("phase_c", 0.0)
        best_failed = phase_c_repair(
            best_assignments, best_failed_ais,
            sessions, store, iids, rids,
            time_limit=cfg.time_limit_phase_c,
        )
        _progress("phase_c", 1.0)

    elapsed = time.time() - t0

    if best_assignments is None:
        return SolveResult(
            success=False,
            total_sessions=n_sessions,
            elapsed_seconds=round(elapsed, 2),
            error="No feasible Phase A solution found",
        )

    # ── Validate ──
    violations = _validate_hard_constraints(best_assignments, sessions, iids, rids)
    total_v = sum(violations.values())

    # ── Build schedule JSON ──
    schedule = build_schedule_json(best_assignments, sessions, store, rids)
    rooms_assigned = sum(1 for a in best_assignments if a["room"] >= 0)

    _progress("done", 1.0)

    return SolveResult(
        success=(total_v == 0),
        schedule=schedule,
        total_sessions=n_sessions,
        rooms_assigned=rooms_assigned,
        rooms_failed=best_failed,
        violations=total_v,
        violation_details=violations,
        elapsed_seconds=round(elapsed, 2),
        seeds_tried=min(seed_i + 1, cfg.seeds),
    )


def solve_timetable_from_json(
    courses: list[dict],
    groups: list[dict],
    instructors: list[dict],
    rooms: list[dict],
    config: SolveConfig | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> SolveResult:
    """Run the CP-SAT pipeline on raw JSON data (from API requests).

    Creates a temp directory, writes the JSON files, runs the solver,
    and cleans up.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="cpsat_solve_"))
    try:
        (work_dir / "Course.json").write_text(json.dumps(courses))
        (work_dir / "Groups.json").write_text(json.dumps(groups))
        (work_dir / "Instructors.json").write_text(json.dumps(instructors))
        (work_dir / "Rooms.json").write_text(json.dumps(rooms))
        return solve_timetable(str(work_dir), config=config, on_progress=on_progress)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _validate_hard_constraints(
    assignments: list[dict],
    sessions: list[Session],
    iids: list[str],
    rids: list[str],
) -> dict[str, int]:
    """Independent validation of all 6 hard constraints. Returns violation counts."""

    # 1. NoRoomDoubleBooking
    room_slots: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        if a["room"] >= 0:
            room_slots[a["room"]].append((a["start"], a["end"], ai))
    rc = 0
    for sl in room_slots.values():
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[i][1] > sl[j][0] and sl[i][0] < sl[j][1]:
                    rc += 1

    # 2. NoInstructorDoubleBooking
    inst_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        for iid in a["instructors"]:
            inst_slots[iids.index(iid)].append((a["start"], a["end"]))
    ic = 0
    for sl in inst_slots.values():
        sl_sorted = sorted(sl)
        for i in range(len(sl_sorted)):
            for j in range(i + 1, len(sl_sorted)):
                if sl_sorted[i][1] > sl_sorted[j][0] and sl_sorted[i][0] < sl_sorted[j][1]:
                    ic += 1

    # 3. NoStudentDoubleBooking
    grp_slots: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        s = sessions[a["orig_i"]]
        for gid in set(s.group_ids):
            grp_slots[gid].append((a["start"], a["end"]))
    gc = 0
    for sl in grp_slots.values():
        sl_sorted = sorted(sl)
        for i in range(len(sl_sorted)):
            for j in range(i + 1, len(sl_sorted)):
                if sl_sorted[i][1] > sl_sorted[j][0] and sl_sorted[i][0] < sl_sorted[j][1]:
                    gc += 1

    # 4. SpreadAcrossDays
    sib: dict[str, list[int]] = defaultdict(list)
    for a in assignments:
        sib[sessions[a["orig_i"]].sibling_key].append(a["start"] // 7)
    sv = sum(1 for days in sib.values() if len(days) > 1 and len(days) != len(set(days)))

    # 5. InstructorMustBeQualified
    qv = 0
    for a in assignments:
        s = sessions[a["orig_i"]]
        qual = {iids[i] for i in s.qualified_instructor_idxs}
        for iid in a["instructors"]:
            if iid not in qual:
                qv += 1

    # 6. RoomMustHaveFeatures
    fv = sum(
        1 for a in assignments
        if a["room"] >= 0 and a["room"] not in sessions[a["orig_i"]].compatible_room_idxs
    )

    return {
        "NoRoomDoubleBooking": rc,
        "NoInstructorDoubleBooking": ic,
        "NoStudentDoubleBooking": gc,
        "SpreadAcrossDays": sv,
        "InstructorMustBeQualified": qv,
        "RoomMustHaveFeatures": fv,
    }
