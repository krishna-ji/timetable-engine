#!/usr/bin/env python3
"""Selective-SRE Schedule Solver.

Strategy: NoOverlap on tight-pool rooms only (all practical/both rooms +
specialized lecture rooms) + cumulative capacity for generic lecture rooms.
Then Phase 2 assigns generic lecture rooms greedily.

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


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def identify_tight_rooms(
    sessions: list[Session],
    model_indices: list[int],
    threshold: int = 10,
) -> set[int]:
    """Return room indices that appear in any session pool of size <= threshold."""
    tight = set()
    for orig_i in model_indices:
        s = sessions[orig_i]
        if len(s.compatible_room_idxs) <= threshold:
            tight.update(s.compatible_room_idxs)
    return tight


def add_selective_sre(
    model: cp_model.CpModel,
    vars_dict: dict,
    sessions: list[Session],
    tight_rooms: set[int],
    room_ids: list[str],
) -> None:
    """Add NoOverlap constraints only for tight-pool rooms."""
    model_indices = vars_dict["model_indices"]
    start_vars = vars_dict["start"]
    end_vars = vars_dict["end"]
    room_vars = vars_dict["room"]

    # Build room → [model indices that can use it]
    room_possible: dict[int, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for ridx in s.compatible_room_idxs:
            if ridx in tight_rooms:
                room_possible[ridx].append(mi)

    n_intervals = 0
    n_constraints = 0
    for ridx in sorted(tight_rooms):
        mi_list = room_possible.get(ridx, [])
        if len(mi_list) <= 1:
            continue
        opt_ivs = []
        for mi in mi_list:
            orig_i = model_indices[mi]
            s = sessions[orig_i]
            pres = model.new_bool_var(f"tsre_{mi}_{ridx}")
            model.add(room_vars[mi] == ridx).only_enforce_if(pres)
            model.add(room_vars[mi] != ridx).only_enforce_if(~pres)
            opt_iv = model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi],
                pres, f"toiv_{mi}_{ridx}"
            )
            opt_ivs.append(opt_iv)
            n_intervals += 1
        model.add_no_overlap(opt_ivs)
        n_constraints += 1

    print(f"  Selective SRE: {n_constraints} NoOverlap constraints, "
          f"{n_intervals} optional intervals "
          f"(for {len(tight_rooms)} tight rooms)")


def add_theory_cumulative(
    model: cp_model.CpModel,
    vars_dict: dict,
    sessions: list[Session],
    capacity: int,
) -> None:
    """Add cumulative: at most `capacity` theory sessions at any quantum."""
    model_indices = vars_dict["model_indices"]
    interval_vars = vars_dict["interval"]

    theory_intervals = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        if s.course_type == "theory":
            theory_intervals.append(interval_vars[mi])

    if theory_intervals:
        demands = [1] * len(theory_intervals)
        model.add_cumulative(theory_intervals, demands, capacity)
        print(f"  Theory cumulative: at most {capacity} theory sessions "
              f"at any quantum ({len(theory_intervals)} sessions)")


def add_practical_cumulative(
    model: cp_model.CpModel,
    vars_dict: dict,
    sessions: list[Session],
    capacity: int,
) -> None:
    """Add cumulative: at most `capacity` practical sessions at any quantum."""
    model_indices = vars_dict["model_indices"]
    interval_vars = vars_dict["interval"]

    prac_intervals = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        if s.course_type == "practical":
            prac_intervals.append(interval_vars[mi])

    if prac_intervals:
        demands = [1] * len(prac_intervals)
        model.add_cumulative(prac_intervals, demands, capacity)
        print(f"  Practical cumulative: at most {capacity} practical sessions "
              f"at any quantum ({len(prac_intervals)} sessions)")


# ────────────────────────────────────────────────────────────────────
# Phase 1: Solve with selective SRE
# ────────────────────────────────────────────────────────────────────

def phase1_solve(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 300,
    relax_fca: bool = False,
    relax_ictd: bool = False,
    pool_threshold: int = 10,
) -> dict | None:
    """Phase 1: Build model with selective SRE and solve."""
    print("\n" + "=" * 72)
    print("  PHASE 1: CP-SAT with selective SRE")
    print("=" * 72)

    # Build base model without SRE
    model, vars_dict = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_sre=True,  # We add our own selective SRE
        relax_fca=relax_fca,
        relax_ictd=relax_ictd,
    )

    model_indices = vars_dict["model_indices"]

    # Identify tight rooms and add selective SRE
    tight_rooms = identify_tight_rooms(sessions, model_indices, pool_threshold)
    add_selective_sre(model, vars_dict, sessions, tight_rooms, room_ids)

    # Count non-tight rooms by type
    non_tight_lecture = 0
    for ridx, rid in enumerate(room_ids):
        if ridx not in tight_rooms:
            feat = store.rooms[rid].room_features
            if feat == "lecture":
                non_tight_lecture += 1

    print(f"  Non-tight rooms: {len(room_ids) - len(tight_rooms)} "
          f"(all lecture, {non_tight_lecture} rooms)")

    # Theory cumulative
    theory_capacity = sum(
        1 for rid in room_ids
        if store.rooms[rid].room_features in ("lecture", "both")
    )
    add_theory_cumulative(model, vars_dict, sessions, theory_capacity)

    # Practical cumulative
    practical_capacity = sum(
        1 for rid in room_ids
        if store.rooms[rid].room_features in ("practical", "both")
    )
    add_practical_cumulative(model, vars_dict, sessions, practical_capacity)

    # Model stats
    proto = model.proto
    print(f"\n  Variables: {len(proto.variables)}, "
          f"Constraints: {len(proto.constraints)}")

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
    print(f"  Elapsed: {elapsed:.2f}s | Branches: {solver.num_branches:,} "
          f"| Conflicts: {solver.num_conflicts:,}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Extract solution
    solution = {
        "model_indices": model_indices,
        "impossible": vars_dict["impossible_sessions"],
        "tight_rooms": tight_rooms,
        "assignments": [],
    }
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        solution["assignments"].append({
            "session_idx": orig_i,
            "mi": mi,
            "start": solver.value(vars_dict["start"][mi]),
            "end": solver.value(vars_dict["end"][mi]),
            "instructor_idx": solver.value(vars_dict["instructor"][mi]),
            "room_idx": solver.value(vars_dict["room"][mi]),
            "day": solver.value(vars_dict["day"][mi]),
            "duration": s.duration,
            "compatible_rooms": s.compatible_room_idxs,
        })

    print(f"  Assigned {len(solution['assignments'])} sessions")
    if solution["impossible"]:
        print(f"  ({len(solution['impossible'])} impossible sessions excluded)")

    return solution


# ────────────────────────────────────────────────────────────────────
# Phase 2: Room fix-up for non-tight rooms
# ────────────────────────────────────────────────────────────────────

def phase2_room_fixup(
    solution: dict,
    sessions: list[Session],
    room_ids: list[str],
    store: DataStore,
) -> list[dict]:
    """Phase 2: Fix room conflicts for non-tight rooms via greedy assignment.

    Tight rooms already have NoOverlap from Phase 1 — conflict-free.
    Non-tight rooms (generic lecture rooms) may conflict — resolve greedily.
    """
    print("\n" + "=" * 72)
    print("  PHASE 2: Room fix-up for non-tight rooms")
    print("=" * 72)

    assignments = solution["assignments"]
    tight_rooms = solution["tight_rooms"]

    # Build room timeline
    room_timeline: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        ridx = a["room_idx"]
        room_timeline[ridx].append((a["start"], a["end"], ai))

    # Find conflicts in non-tight rooms
    conflicting_ais: set[int] = set()
    for ridx, intervals in room_timeline.items():
        if ridx in tight_rooms:
            continue
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                conflicting_ais.add(intervals[i][2])
                conflicting_ais.add(intervals[i + 1][2])

    if not conflicting_ais:
        print("  No conflicts — all room assignments valid!")
    else:
        print(f"  {len(conflicting_ais)} sessions need room reassignment...")

        # Non-tight rooms
        non_tight_idxs = sorted(
            ridx for ridx in range(len(room_ids))
            if ridx not in tight_rooms
        )

        # Build occupancy (excluding conflicting sessions)
        occupancy: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ai, a in enumerate(assignments):
            ridx = a["room_idx"]
            if ai not in conflicting_ais:
                occupancy[ridx].append((a["start"], a["end"]))

        # Greedy reassignment (hardest first)
        reassigned = 0
        failed = 0
        for ai in sorted(conflicting_ais,
                         key=lambda x: len(assignments[x]["compatible_rooms"])):
            a = assignments[ai]
            compat_nt = [
                ridx for ridx in a["compatible_rooms"]
                if ridx not in tight_rooms
            ]
            assigned = False
            for ridx in compat_nt:
                ok = all(
                    a["start"] >= oe or a["end"] <= os
                    for os, oe in occupancy[ridx]
                )
                if ok:
                    assignments[ai]["room_idx"] = ridx
                    occupancy[ridx].append((a["start"], a["end"]))
                    reassigned += 1
                    assigned = True
                    break
            if not assigned:
                failed += 1

        print(f"  Reassigned: {reassigned}, Failed: {failed}")
        if failed > 0:
            print("  WARNING: Some room conflicts remain!")

    # Build final schedule
    qts = store.qts
    instructor_ids = list(store.instructors.keys())

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
            "room_name": (store.rooms[room_ids[ridx]].name
                          if hasattr(store.rooms[room_ids[ridx]], 'name')
                          else room_ids[ridx]),
            "start_quanta": start_q,
            "duration": a["duration"],
            "day": day_str,
            "time": time_str,
        })

    return schedule


# ────────────────────────────────────────────────────────────────────
# Schedule summary & constraint verification
# ────────────────────────────────────────────────────────────────────

def print_schedule_summary(schedule: list[dict], store: DataStore) -> None:
    """Print a human-readable summary with full constraint verification."""
    print("\n" + "=" * 72)
    print("  SCHEDULE SUMMARY")
    print("=" * 72)
    print(f"  Total entries: {len(schedule)}")

    # Per-day breakdown
    day_sessions: dict[str, list[dict]] = defaultdict(list)
    for entry in schedule:
        day_sessions[entry["day"]].append(entry)

    for day in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        entries = day_sessions.get(day, [])
        total_q = sum(e["duration"] for e in entries)
        print(f"  {day:10s}: {len(entries):3d} sessions, {total_q:4d} quanta")

    # Instructor utilization (top 10)
    inst_load: dict[str, int] = defaultdict(int)
    for entry in schedule:
        inst_load[entry["instructor_name"]] += entry["duration"]

    print(f"\n  Instructor utilization (top 10 / {len(inst_load)} total):")
    for name, load in sorted(inst_load.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * load
        print(f"    {name:30s}  {load:3d}q  {bar}")

    # Room utilization (top 10)
    room_load: dict[str, int] = defaultdict(int)
    for entry in schedule:
        room_load[entry["room_id"]] += entry["duration"]

    print(f"\n  Room utilization (top 10 / {len(room_load)} used):")
    for rid, load in sorted(room_load.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * (load // 2)
        print(f"    {rid:10s}  {load:3d}q  {bar}")

    # ── Constraint Verification ──
    print(f"\n  CONSTRAINT VERIFICATION:")

    # CTE: group no-overlap
    group_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for entry in schedule:
        for gid in entry["group_ids"]:
            group_intervals[gid].append((
                entry["start_quanta"],
                entry["start_quanta"] + entry["duration"],
                entry["course_id"],
            ))

    cte_violations = 0
    for gid, intervals in group_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                cte_violations += 1

    # FTE: instructor no-overlap
    inst_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for entry in schedule:
        inst_intervals[entry["instructor_id"]].append((
            entry["start_quanta"],
            entry["start_quanta"] + entry["duration"],
            entry["course_id"],
        ))

    fte_violations = 0
    for iid, intervals in inst_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                fte_violations += 1

    # SRE: room no-overlap
    room_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for entry in schedule:
        room_intervals[entry["room_id"]].append((
            entry["start_quanta"],
            entry["start_quanta"] + entry["duration"],
            entry["course_id"],
        ))

    sre_violations = 0
    sre_details: list[str] = []
    for rid, intervals in room_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                sre_violations += 1
                if len(sre_details) < 10:
                    sre_details.append(
                        f"    {rid}: [{intervals[i][2]} q{intervals[i][0]}-"
                        f"{intervals[i][1]}] vs [{intervals[i + 1][2]} "
                        f"q{intervals[i + 1][0]}-{intervals[i + 1][1]}]"
                    )

    print(f"    CTE (group no-overlap):      {cte_violations} violations")
    print(f"    FTE (instructor no-overlap): {fte_violations} violations")
    print(f"    SRE (room no-overlap):       {sre_violations} violations")

    if sre_details:
        print(f"    SRE details (first {len(sre_details)}):")
        for d in sre_details:
            print(d)

    total_v = cte_violations + fte_violations + sre_violations
    if total_v == 0:
        print(f"\n  >>> ZERO hard-constraint violations — SCHEDULE IS FEASIBLE! <<<")
    else:
        print(f"\n  {total_v} total violations — schedule needs work")

    print("=" * 72)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Selective-SRE Schedule Solver")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Solver time limit (seconds)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export schedule to JSON file")
    parser.add_argument("--relax-fca", action="store_true",
                        help="Relax instructor availability constraint")
    parser.add_argument("--relax-ictd", action="store_true",
                        help="Relax same-day dispersion constraint")
    parser.add_argument("--pool-threshold", type=int, default=10,
                        help="Pool size threshold for tight-room SRE (default: 10)")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    # Build sessions
    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions, {len(instructor_ids)} instructors, "
          f"{len(room_ids)} rooms")

    # Phase 1: Solve with selective SRE
    solution = phase1_solve(
        sessions, store, instructor_ids, room_ids,
        time_limit=args.time_limit,
        relax_fca=args.relax_fca,
        relax_ictd=args.relax_ictd,
        pool_threshold=args.pool_threshold,
    )

    if solution is None:
        print("\n  PHASE 1 FAILED — no solution found.")
        print("  Try: --time-limit 600, --relax-fca, --relax-ictd")
        sys.exit(1)

    # Phase 2: Room fix-up
    schedule = phase2_room_fixup(solution, sessions, room_ids, store)

    # Summary with constraint verification
    print_schedule_summary(schedule, store)

    # Export
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
        print(f"\n  Schedule exported to: {args.export}")
        print(f"  Total entries: {len(schedule)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
