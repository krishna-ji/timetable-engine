#!/usr/bin/env python3
"""Two-Phase Schedule Solver v3: Cumulative-Bounded Time + CP-SAT Room Assignment.

Phase 1: CP-SAT solves time+instructor with cumulative capacity bounds
         (no per-room NoOverlap — fast, ~1-10s).
Phase 2: CP-SAT assigns rooms with fixed time slots and per-room NoOverlap
         (room-only model — fast because time is fixed).

Usage:
    python solve_schedule.py --data-dir data_fixed --export schedule.json
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
# Phase 1: Time + Instructor (with cumulative room capacity bounds)
# ────────────────────────────────────────────────────────────────────

def phase1_time_instructor(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 120,
    relax_fca: bool = False,
    relax_ictd: bool = False,
) -> dict | None:
    """Solve time+instructor assignment with cumulative room capacity bounds.

    Does NOT enforce per-room NoOverlap. Instead, adds cumulative constraints:
    - At most K sessions sharing any specific room pool can overlap.
    This prevents over-packing while keeping the model fast.
    """
    print("\n" + "=" * 72)
    print("  PHASE 1: Time + Instructor (cumulative-bounded)")
    print("=" * 72)

    model, vars_dict = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_sre=True,
        relax_fca=relax_fca,
        relax_ictd=relax_ictd,
    )

    model_indices = vars_dict["model_indices"]
    interval_vars = vars_dict["interval"]

    # ── Add cumulative constraints per distinct room pool ──
    # Group sessions by their room pool. Sessions with the same pool compete
    # for exactly those rooms. At most |pool| can overlap at any quantum.
    pool_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        pool_key = tuple(sorted(s.compatible_room_idxs))
        pool_groups[pool_key].append(mi)

    cum_count = 0
    for pool_key, mi_list in pool_groups.items():
        capacity = len(pool_key)
        if len(mi_list) <= capacity:
            continue  # Can never exceed capacity
        if capacity <= 2:
            continue  # Skip single/dual room pools (too tight with CTE/FTE)
        intervals = [interval_vars[mi] for mi in mi_list]
        demands = [1] * len(intervals)
        model.add_cumulative(intervals, demands, capacity)
        cum_count += 1

    print(f"  Added {cum_count} per-pool cumulative constraints (pool>=3 only)")

    # ── Also add cumulative for room-type categories ──
    # Theory rooms: lecture + both
    theory_capacity = sum(
        1 for rid in room_ids
        if store.rooms[rid].room_features in ("lecture", "both")
    )
    theory_mis = [
        mi for mi, orig_i in enumerate(model_indices)
        if sessions[orig_i].course_type == "theory"
    ]
    if len(theory_mis) > theory_capacity:
        model.add_cumulative(
            [interval_vars[mi] for mi in theory_mis],
            [1] * len(theory_mis),
            theory_capacity,
        )
        print(f"  Theory cumulative: ≤{theory_capacity} concurrent "
              f"({len(theory_mis)} sessions)")

    # Practical rooms: practical + both
    prac_capacity = sum(
        1 for rid in room_ids
        if store.rooms[rid].room_features in ("practical", "both")
    )
    prac_mis = [
        mi for mi, orig_i in enumerate(model_indices)
        if sessions[orig_i].course_type == "practical"
    ]
    if len(prac_mis) > prac_capacity:
        model.add_cumulative(
            [interval_vars[mi] for mi in prac_mis],
            [1] * len(prac_mis),
            prac_capacity,
        )
        print(f"  Practical cumulative: ≤{prac_capacity} concurrent "
              f"({len(prac_mis)} sessions)")

    # Global cumulative
    all_ivs = [interval_vars[mi] for mi in range(len(model_indices))]
    model.add_cumulative(all_ivs, [1] * len(all_ivs), len(room_ids))
    print(f"  Global cumulative: ≤{len(room_ids)} concurrent")

    # NOTE: Per-room cumulative(≤1) is NOT correct here because
    # sessions only MIGHT use a given room. The per-pool cumulative
    # above is the correct relaxation. Phase 2 handles exact room assignment.

    # Solve
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

    solution = {
        "model_indices": model_indices,
        "impossible": vars_dict["impossible_sessions"],
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
    return solution


# ────────────────────────────────────────────────────────────────────
# Phase 2: Room assignment (CP-SAT with fixed time, per-room NoOverlap)
# ────────────────────────────────────────────────────────────────────

def phase2_room_assignment(
    solution: dict,
    sessions: list[Session],
    room_ids: list[str],
    store: DataStore,
    time_limit: int = 120,
) -> list[dict] | None:
    """Assign rooms with fixed time slots.

    Since time+instructor are fixed from Phase 1, this is a pure
    room-coloring problem: assign rooms from compatible pools such that
    no two sessions in the same room overlap in time.
    """
    print("\n" + "=" * 72)
    print("  PHASE 2: Room Assignment (CP-SAT, fixed time)")
    print("=" * 72)

    assignments = solution["assignments"]
    n = len(assignments)

    model = cp_model.CpModel()

    # Room variable per session
    room_vars = []
    for ai, a in enumerate(assignments):
        compat = a["compatible_rooms"]
        if not compat:
            print(f"  ERROR: Session {ai} has no compatible rooms!")
            return None
        rv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(compat), f"room_{ai}"
        )
        # Hint: use Phase 1's room assignment
        model.add_hint(rv, a["room_idx"])
        room_vars.append(rv)

    # Build per-room NoOverlap with fixed-size intervals
    room_possible: dict[int, list[int]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        for ridx in a["compatible_rooms"]:
            room_possible[ridx].append(ai)

    n_intervals = 0
    n_constraints = 0
    for ridx, ai_list in room_possible.items():
        if len(ai_list) <= 1:
            continue
        opt_ivs = []
        for ai in ai_list:
            a = assignments[ai]
            pres = model.new_bool_var(f"rp_{ai}_{ridx}")
            model.add(room_vars[ai] == ridx).only_enforce_if(pres)
            model.add(room_vars[ai] != ridx).only_enforce_if(~pres)
            # Fixed interval: start and duration are constants from Phase 1
            opt_iv = model.new_optional_fixed_size_interval_var(
                a["start"], a["duration"], pres, f"riv_{ai}_{ridx}"
            )
            opt_ivs.append(opt_iv)
            n_intervals += 1
        model.add_no_overlap(opt_ivs)
        n_constraints += 1

    print(f"  {n_constraints} NoOverlap constraints, {n_intervals} opt intervals")

    proto = model.proto
    print(f"  Variables: {len(proto.variables)}, "
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

    print(f"\n  Phase 2 result: {STATUS_NAMES.get(status, 'UNKNOWN')}")
    print(f"  Elapsed: {elapsed:.2f}s | Branches: {solver.num_branches:,} "
          f"| Conflicts: {solver.num_conflicts:,}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fall back to Phase 1 room assignments (may have conflicts)
        print("  WARNING: Falling back to Phase 1 room assignments "
              "(may have room conflicts)")
        return _build_schedule(assignments, sessions, room_ids, store,
                               room_override=None)

    # Extract room assignments
    room_override = {}
    for ai in range(n):
        room_override[ai] = solver.value(room_vars[ai])

    return _build_schedule(assignments, sessions, room_ids, store,
                           room_override=room_override)


def _build_schedule(
    assignments: list[dict],
    sessions: list[Session],
    room_ids: list[str],
    store: DataStore,
    room_override: dict[int, int] | None = None,
) -> list[dict]:
    """Build schedule list from assignments."""
    qts = store.qts
    instructor_ids = list(store.instructors.keys())

    schedule = []
    for ai, a in enumerate(assignments):
        s = sessions[a["session_idx"]]
        ridx = room_override[ai] if room_override and ai in room_override else a["room_idx"]
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
# Constraint verification
# ────────────────────────────────────────────────────────────────────

def print_schedule_summary(schedule: list[dict], store: DataStore) -> None:
    """Print summary with full constraint verification."""
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
    print(f"\n  Instructor utilization (top 10 / {len(inst_load)} instructors):")
    for name, load in sorted(inst_load.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:30s}  {load:3d}q  {'█' * load}")

    # Room utilization (top 10)
    room_load: dict[str, int] = defaultdict(int)
    for entry in schedule:
        room_load[entry["room_id"]] += entry["duration"]
    print(f"\n  Room utilization (top 10 / {len(room_load)} rooms used):")
    for rid, load in sorted(room_load.items(), key=lambda x: -x[1])[:10]:
        print(f"    {rid:10s}  {load:3d}q  {'█' * (load // 2)}")

    # ── Constraint Verification ──
    print(f"\n  CONSTRAINT VERIFICATION:")

    # CTE: group no-overlap
    group_iv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        for gid in e["group_ids"]:
            group_iv[gid].append((e["start_quanta"], e["start_quanta"] + e["duration"]))
    cte_v = _count_overlaps(group_iv)

    # FTE: instructor no-overlap
    inst_iv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        inst_iv[e["instructor_id"]].append(
            (e["start_quanta"], e["start_quanta"] + e["duration"]))
    fte_v = _count_overlaps(inst_iv)

    # SRE: room no-overlap
    room_iv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        room_iv[e["room_id"]].append(
            (e["start_quanta"], e["start_quanta"] + e["duration"]))
    sre_v = _count_overlaps(room_iv)

    print(f"    CTE (group no-overlap):      {cte_v} violations")
    print(f"    FTE (instructor no-overlap): {fte_v} violations")
    print(f"    SRE (room no-overlap):       {sre_v} violations")

    total_v = cte_v + fte_v + sre_v
    if total_v == 0:
        print(f"\n  >>> ZERO VIOLATIONS — SCHEDULE IS FEASIBLE! <<<")
    else:
        print(f"\n  {total_v} total violations")
    print("=" * 72)


def _count_overlaps(intervals: dict[str, list[tuple[int, int]]]) -> int:
    count = 0
    for key, ivs in intervals.items():
        ivs.sort()
        for i in range(len(ivs) - 1):
            if ivs[i][1] > ivs[i + 1][0]:
                count += 1
    return count


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Phase Schedule Solver v3")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--time-limit", type=int, default=120,
                        help="Per-phase time limit (seconds)")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--relax-fca", action="store_true")
    parser.add_argument("--relax-ictd", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions, {len(instructor_ids)} instructors, "
          f"{len(room_ids)} rooms")

    # Phase 1
    solution = phase1_time_instructor(
        sessions, store, instructor_ids, room_ids,
        time_limit=args.time_limit,
        relax_fca=args.relax_fca,
        relax_ictd=args.relax_ictd,
    )
    if solution is None:
        print("\n  PHASE 1 FAILED.")
        sys.exit(1)

    # Phase 2
    schedule = phase2_room_assignment(
        solution, sessions, room_ids, store,
        time_limit=args.time_limit,
    )
    if schedule is None:
        print("\n  PHASE 2 FAILED.")
        sys.exit(1)

    # Summary
    print_schedule_summary(schedule, store)

    # Export
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
        print(f"\n  Exported: {args.export} ({len(schedule)} entries)")

    print("\nDone!")


if __name__ == "__main__":
    main()
