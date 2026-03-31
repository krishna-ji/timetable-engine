#!/usr/bin/env python3
"""Two-Phase Schedule Solver.

Phase 1: CP-SAT solves time slot + instructor assignment (SRE relaxed).
Phase 2: Greedy room assignment using interval coloring.

This decomposition avoids the exponential room×session interaction that
makes the monolithic model intractable.

Usage:
    python solve_schedule.py --data-dir data_fixed --export schedule.json
    python solve_schedule.py --data-dir data_fixed --time-limit 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ortools.sat.python import cp_model

from cpsat_oracle import (
    Session,
    build_model,
    build_sessions,
)
from src.io.data_store import DataStore


def phase1_solve(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 300,
    relax_fca: bool = False,
    relax_ictd: bool = False,
) -> dict | None:
    """Two-step Phase 1:
      Step A: Solve without SRE (fast) → get time+instructor+room hints
      Step B: Re-solve with full SRE, using Step A as warm-start hint

    Returns solution dict or None if infeasible/timeout.
    """
    print("\n" + "=" * 72)
    print("  PHASE 1A: Quick solve (no room NoOverlap)")
    print("=" * 72)

    # Step A: Fast solve without SRE
    model_a, vars_a = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_sre=True, relax_fca=relax_fca, relax_ictd=relax_ictd,
    )

    solver_a = cp_model.CpSolver()
    solver_a.parameters.max_time_in_seconds = 60
    solver_a.parameters.num_workers = 8

    status_a = solver_a.Solve(model_a)
    if status_a not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Step A failed: {status_a}")
        return None

    mi_a = vars_a["model_indices"]
    print(f"  Step A: OPTIMAL in {solver_a.wall_time:.2f}s")

    # Extract Step A solution values
    hint_values = {}
    for mi, orig_i in enumerate(mi_a):
        hint_values[orig_i] = {
            "start": solver_a.value(vars_a["start"][mi]),
            "inst": solver_a.value(vars_a["instructor"][mi]),
            "room": solver_a.value(vars_a["room"][mi]),
        }

    # Step B: Full solve with SRE, using Step A as hint
    print("\n" + "=" * 72)
    print("  PHASE 1B: Full solve with room NoOverlap (warm-started)")
    print("=" * 72)

    model_b, vars_b = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_sre=False, relax_fca=relax_fca, relax_ictd=relax_ictd,
    )

    mi_b = vars_b["model_indices"]

    # Add hints from Step A
    hints_added = 0
    for mi, orig_i in enumerate(mi_b):
        if orig_i in hint_values:
            h = hint_values[orig_i]
            model_b.add_hint(vars_b["start"][mi], h["start"])
            model_b.add_hint(vars_b["instructor"][mi], h["inst"])
            model_b.add_hint(vars_b["room"][mi], h["room"])
            hints_added += 1

    print(f"  Hints added: {hints_added}")

    proto = model_b.proto
    print(f"  Variables: {len(proto.variables)}, Constraints: {len(proto.constraints)}")

    solver_b = cp_model.CpSolver()
    solver_b.parameters.max_time_in_seconds = time_limit
    solver_b.parameters.num_workers = 8
    solver_b.parameters.log_search_progress = True

    t0 = time.time()
    status_b = solver_b.Solve(model_b)
    elapsed = time.time() - t0

    STATUS_NAMES = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN (timed out)",
    }

    print(f"\n  Phase 1B result: {STATUS_NAMES.get(status_b, 'UNKNOWN')}")
    print(f"  Elapsed: {elapsed:.2f}s | Branches: {solver_b.num_branches:,} | Conflicts: {solver_b.num_conflicts:,}")

    if status_b not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fall back to Step A solution (no room guarantees)
        print("  Falling back to Step A solution (room conflicts possible)...")
        solution = {
            "model_indices": mi_a,
            "impossible": vars_a["impossible_sessions"],
            "assignments": [],
        }
        for mi, orig_i in enumerate(mi_a):
            s = sessions[orig_i]
            solution["assignments"].append({
                "session_idx": orig_i, "mi": mi,
                "start": solver_a.value(vars_a["start"][mi]),
                "end": solver_a.value(vars_a["end"][mi]),
                "instructor_idx": solver_a.value(vars_a["instructor"][mi]),
                "room_idx": solver_a.value(vars_a["room"][mi]),
                "day": solver_a.value(vars_a["day"][mi]),
                "duration": s.duration,
                "compatible_rooms": s.compatible_room_idxs,
            })
        return solution

    # Extract Step B solution  
    solution = {
        "model_indices": mi_b,
        "impossible": vars_b["impossible_sessions"],
        "assignments": [],
    }
    for mi, orig_i in enumerate(mi_b):
        s = sessions[orig_i]
        solution["assignments"].append({
            "session_idx": orig_i, "mi": mi,
            "start": solver_b.value(vars_b["start"][mi]),
            "end": solver_b.value(vars_b["end"][mi]),
            "instructor_idx": solver_b.value(vars_b["instructor"][mi]),
            "room_idx": solver_b.value(vars_b["room"][mi]),
            "day": solver_b.value(vars_b["day"][mi]),
            "duration": s.duration,
            "compatible_rooms": s.compatible_room_idxs,
        })

    print(f"  Assigned {len(solution['assignments'])} sessions (rooms verified)")

    return solution

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS_NAMES = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN (timed out)",
    }

    print(f"\n  Phase 1 result: {STATUS_NAMES.get(status, 'UNKNOWN')}")
    print(f"  Elapsed: {elapsed:.2f}s | Branches: {solver.num_branches:,} | Conflicts: {solver.num_conflicts:,}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Extract solution
    solution = {
        "model_indices": model_indices,
        "impossible": impossible_sessions,
        "assignments": [],
    }

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        solution["assignments"].append({
            "session_idx": orig_i,
            "mi": mi,
            "start": solver.value(start_vars[mi]),
            "end": solver.value(end_vars[mi]),
            "instructor_idx": solver.value(inst_vars[mi]),
            "room_idx": solver.value(room_vars[mi]),
            "day": solver.value(day_vars[mi]),
            "duration": s.duration,
            "compatible_rooms": s.compatible_room_idxs,
        })

    print(f"  Assigned {len(solution['assignments'])} sessions")
    if impossible_sessions:
        print(f"  ({len(impossible_sessions)} impossible sessions excluded)")

    return solution


def phase2_room_assignment(
    solution: dict,
    sessions: list[Session],
    room_ids: list[str],
    store: DataStore,
) -> list[dict] | None:
    """Phase 2: Assign rooms via CP-SAT with NoOverlap per room.

    Uses the Phase 1 room assignments as hints, but allows re-assignment
    to resolve conflicts. For sessions where the Phase 1 room is conflict-free,
    keeps that assignment.

    Returns list of complete assignments or None if completely fails.
    """
    print("\n" + "=" * 72)
    print("  PHASE 2: Room Assignment (CP-SAT with NoOverlap)")
    print("=" * 72)

    assignments = solution["assignments"]
    t0 = time.time()

    result = _cpsat_full_room_assignment(assignments, sessions, room_ids, store)
    elapsed = time.time() - t0

    if result is not None:
        # Count constraint violations
        print(f"\n  Phase 2 result: ALL {len(assignments)} sessions assigned rooms")
        print(f"  Elapsed: {elapsed:.2f}s")
        return result

    # CP-SAT room assignment failed — fall back to using Phase 1 rooms directly
    print("  CP-SAT room assignment INFEASIBLE — using Phase 1 rooms (with conflicts)")

    qts = store.qts
    instructor_ids = list(store.instructors.keys())

    # Build schedule from Phase 1 assignments directly
    schedule = []
    for ai, a in enumerate(assignments):
        s = sessions[a["session_idx"]]
        ridx = a["room_idx"]
        iidx = a["instructor_idx"]
        start_q = a["start"]
        day_str, time_str = qts.quanta_to_time(start_q)
        schedule.append({
            "session_index": a["session_idx"],
            "course_id": s.course_id,
            "course_type": s.course_type,
            "group_ids": s.group_ids,
            "instructor_id": instructor_ids[iidx],
            "instructor_name": store.instructors[instructor_ids[iidx]].name,
            "room_id": room_ids[ridx],
            "room_name": store.rooms[room_ids[ridx]].name if hasattr(store.rooms[room_ids[ridx]], 'name') else room_ids[ridx],
            "start_quanta": start_q,
            "duration": a["duration"],
            "day": day_str,
            "time": time_str,
        })

    return schedule


def _cpsat_room_repair(
    failed_indices: list[int],
    assignments: list[dict],
    room_occupied: dict[int, list[tuple[int, int]]],
    room_assignments: dict[int, int],
    room_ids: list[str],
) -> bool:
    """Use CP-SAT to repair the failed room assignments."""
    model = cp_model.CpModel()

    fail_vars = {}
    for ai in failed_indices:
        a = assignments[ai]
        compat = a["compatible_rooms"]
        if not compat:
            return False
        rv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(compat), f"room_{ai}"
        )
        fail_vars[ai] = rv

    # No overlap between failed sessions and existing schedule
    for ai in failed_indices:
        a = assignments[ai]
        for aj in failed_indices:
            if aj <= ai:
                continue
            b = assignments[aj]
            # Check if time intervals overlap
            if a["start"] < b["end"] and b["start"] < a["end"]:
                model.add(fail_vars[ai] != fail_vars[aj])

        # Check against existing assignments
        for aj, ridx in room_assignments.items():
            b = assignments[aj]
            if a["start"] < b["end"] and b["start"] < a["end"]:
                # This room is occupied, prevent same room
                model.add(fail_vars[ai] != ridx)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for ai in failed_indices:
            ridx = solver.value(fail_vars[ai])
            room_assignments[ai] = ridx
            room_occupied[ridx].append((assignments[ai]["start"], assignments[ai]["end"]))
        return True
    return False


def _cpsat_full_room_assignment(
    assignments: list[dict],
    sessions: list[Session],
    room_ids: list[str],
    store: DataStore,
) -> list[dict] | None:
    """Full CP-SAT room assignment using NoOverlap per room."""
    model = cp_model.CpModel()
    n = len(assignments)

    room_vars = []
    for ai, a in enumerate(assignments):
        compat = a["compatible_rooms"]
        if not compat:
            return None
        rv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(compat), f"room_{ai}"
        )
        room_vars.append(rv)

    # Build per-room NoOverlap using optional intervals
    # This is the proper SRE encoding but only for room assignment
    # (time slots are fixed, so intervals have fixed start/duration)
    room_possible: dict[int, list[int]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        for ridx in a["compatible_rooms"]:
            room_possible[ridx].append(ai)

    for ridx, ai_list in room_possible.items():
        if len(ai_list) <= 1:
            continue
        opt_ivs = []
        for ai in ai_list:
            a = assignments[ai]
            pres = model.new_bool_var(f"rp_{ai}_{ridx}")
            model.add(room_vars[ai] == ridx).only_enforce_if(pres)
            model.add(room_vars[ai] != ridx).only_enforce_if(~pres)
            # Fixed interval — start and duration are constants
            opt_iv = model.new_optional_fixed_size_interval_var(
                a["start"], a["duration"], pres, f"riv_{ai}_{ridx}"
            )
            opt_ivs.append(opt_iv)
        model.add_no_overlap(opt_ivs)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Room CP-SAT result: {status} (INFEASIBLE={cp_model.INFEASIBLE})")
        return None

    qts = store.qts
    instructor_ids = list(store.instructors.keys())

    schedule = []
    for ai, a in enumerate(assignments):
        s = sessions[a["session_idx"]]
        ridx = solver.value(room_vars[ai])
        iidx = a["instructor_idx"]
        start_q = a["start"]
        day_str, time_str = qts.quanta_to_time(start_q)

        schedule.append({
            "session_index": a["session_idx"],
            "course_id": s.course_id,
            "course_type": s.course_type,
            "group_ids": s.group_ids,
            "instructor_id": instructor_ids[iidx],
            "instructor_name": store.instructors[instructor_ids[iidx]].name,
            "room_id": room_ids[ridx],
            "room_name": store.rooms[room_ids[ridx]].name if hasattr(store.rooms[room_ids[ridx]], 'name') else room_ids[ridx],
            "start_quanta": start_q,
            "duration": a["duration"],
            "day": day_str,
            "time": time_str,
        })

    return schedule


def print_schedule_summary(schedule: list[dict], store: DataStore) -> None:
    """Print a human-readable summary of the schedule."""
    print("\n" + "=" * 72)
    print("  SCHEDULE SUMMARY")
    print("=" * 72)

    # Per-day breakdown
    day_sessions: dict[str, list[dict]] = defaultdict(list)
    for entry in schedule:
        day_sessions[entry["day"]].append(entry)

    for day in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        entries = day_sessions.get(day, [])
        total_q = sum(e["duration"] for e in entries)
        print(f"\n  {day}: {len(entries)} sessions, {total_q} quanta")

    # Instructor utilization
    inst_load: dict[str, int] = defaultdict(int)
    for entry in schedule:
        inst_load[entry["instructor_name"]] += entry["duration"]

    print(f"\n  Instructor utilization (top 15):")
    for name, load in sorted(inst_load.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * load
        print(f"    {name:30s}  {load:3d}q  {bar}")

    # Room utilization
    room_load: dict[str, int] = defaultdict(int)
    for entry in schedule:
        room_load[entry["room_id"]] += entry["duration"]

    print(f"\n  Room utilization (top 15):")
    for rid, load in sorted(room_load.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * (load // 2)
        print(f"    {rid:10s}  {load:3d}q  {bar}")

    # Constraint verification
    print(f"\n  Quick constraint check:")
    # CTE: group no-overlap
    group_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entry in schedule:
        for gid in entry["group_ids"]:
            group_intervals[gid].append((entry["start_quanta"], entry["start_quanta"] + entry["duration"]))

    cte_violations = 0
    for gid, intervals in group_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                cte_violations += 1

    # FTE: instructor no-overlap
    inst_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entry in schedule:
        inst_intervals[entry["instructor_id"]].append(
            (entry["start_quanta"], entry["start_quanta"] + entry["duration"])
        )

    fte_violations = 0
    for iid, intervals in inst_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                fte_violations += 1

    # SRE: room no-overlap
    room_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entry in schedule:
        room_intervals[entry["room_id"]].append(
            (entry["start_quanta"], entry["start_quanta"] + entry["duration"])
        )

    sre_violations = 0
    for rid, intervals in room_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                sre_violations += 1

    print(f"    CTE (group no-overlap):      {cte_violations} violations")
    print(f"    FTE (instructor no-overlap): {fte_violations} violations")
    print(f"    SRE (room no-overlap):       {sre_violations} violations")

    total_v = cte_violations + fte_violations + sre_violations
    if total_v == 0:
        print(f"\n  ✓ ZERO hard-constraint violations — schedule is FEASIBLE!")
    else:
        print(f"\n  ✗ {total_v} total violations — schedule needs work")

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-Phase Schedule Solver"
    )
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Phase 1 solver time limit (seconds)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export schedule to JSON file")
    parser.add_argument("--relax-fca", action="store_true",
                        help="Relax instructor availability constraint")
    parser.add_argument("--relax-ictd", action="store_true",
                        help="Relax same-day dispersion constraint")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    # Build sessions
    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    # Phase 1: Time + Instructor
    solution = phase1_solve(
        sessions, store, instructor_ids, room_ids,
        time_limit=args.time_limit,
        relax_fca=args.relax_fca,
        relax_ictd=args.relax_ictd,
    )

    if solution is None:
        print("\n  Phase 1 FAILED — no time+instructor assignment found.")
        print("  Try increasing --time-limit or relaxing constraints.")
        sys.exit(1)

    # Phase 2: Room Assignment
    schedule = phase2_room_assignment(solution, sessions, room_ids, store)

    if schedule is None:
        print("\n  Phase 2 FAILED — room assignment infeasible.")
        print("  Not enough compatible rooms for simultaneous sessions.")
        sys.exit(1)

    # Summary
    print_schedule_summary(schedule, store)

    # Export
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
        print(f"\n  Schedule exported to: {args.export}")
        print(f"  Total entries: {len(schedule)}")

    print("\nDone!")
    sys.exit(0)


if __name__ == "__main__":
    main()
