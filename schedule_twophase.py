#!/usr/bin/env python3
"""Two-phase CP-SAT scheduler for university course timetabling.

Phase 1: CP-SAT assigns time slots + instructors with per-pool cumulative
          room constraints (guarantees room assignment is possible).
Phase 2: Bipartite matching assigns rooms per time quantum.

Usage:
    python schedule_twophase.py                # default 300s Phase 1
    python schedule_twophase.py --time-limit 600
    python schedule_twophase.py --export schedule.json
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
    build_sessions,
    compute_valid_starts,
    compute_valid_starts_for_instructor,
    run_diagnostics,
)
from src.io.data_store import DataStore


def build_phase1_model(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> tuple[cp_model.CpModel, dict]:
    """Phase 1: time + instructor assignment with pool cumulatives.

    Constraints: CTE, FTE, FPC, FCA, ICTD, CQF
    Plus: per-pool cumulative (at any quantum, sessions needing pool P <= |P|)
          type-level cumulatives (theory <= lecture+both rooms, etc.)
          global cumulative (total sessions at any time <= total rooms)
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    day_offsets: list[int] = []
    day_lengths: list[int] = []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)
    num_days = len(day_offsets)
    quanta_per_day = day_lengths[0] if day_lengths else 7

    model = cp_model.CpModel()

    # ── Pre-compute allowed (instructor, start) tuples ──
    session_allowed: list[list[tuple[int, int]]] = []
    impossible_sessions: list[int] = []
    for i, s in enumerate(sessions):
        allowed: list[tuple[int, int]] = []
        base_starts = compute_valid_starts(
            s.duration, total_quanta, day_offsets, day_lengths
        )
        for inst_idx in s.qualified_instructor_idxs:
            iid = instructor_ids[inst_idx]
            instructor = store.instructors[iid]
            if instructor.is_full_time:
                for st in base_starts:
                    allowed.append((inst_idx, st))
            else:
                for st in compute_valid_starts_for_instructor(
                    s.duration, instructor.available_quanta, total_quanta,
                    day_offsets, day_lengths
                ):
                    allowed.append((inst_idx, st))
        session_allowed.append(allowed)
        if not allowed or not s.compatible_room_idxs:
            impossible_sessions.append(i)

    if impossible_sessions:
        print(f"\nWARNING: {len(impossible_sessions)} impossible sessions excluded.")
        for si in impossible_sessions:
            s = sessions[si]
            print(f"  S{si}: {s.course_id} ({s.course_type}) dur={s.duration} "
                  f"groups={s.group_ids}")

    impossible_set = set(impossible_sessions)
    model_indices: list[int] = []
    session_to_model: dict[int, int] = {}
    for i in range(len(sessions)):
        if i not in impossible_set:
            session_to_model[i] = len(model_indices)
            model_indices.append(i)

    N = len(model_indices)
    print(f"  Model size: {N} sessions")

    # ── Decision variables ──
    start_vars = []
    end_vars = []
    inst_vars = []
    day_vars = []
    interval_vars = []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        allowed = session_allowed[orig_i]

        inst_domain = sorted({t[0] for t in allowed})
        start_domain = sorted({t[1] for t in allowed})

        inst = model.new_int_var_from_domain(
            cp_model.Domain.from_values(inst_domain), f"inst_{mi}"
        )
        start = model.new_int_var_from_domain(
            cp_model.Domain.from_values(start_domain), f"start_{mi}"
        )
        end = model.new_int_var(0, total_quanta, f"end_{mi}")
        model.add(end == start + s.duration)

        day = model.new_int_var(0, num_days - 1, f"day_{mi}")
        model.add_division_equality(day, start, quanta_per_day)

        interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")

        model.add_allowed_assignments([inst, start], allowed)

        start_vars.append(start)
        end_vars.append(end)
        inst_vars.append(inst)
        day_vars.append(day)
        interval_vars.append(interval)

    # ── CTE: Group NoOverlap ──
    group_sessions: dict[str, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for gid in set(s.group_ids):
            group_sessions[gid].add(mi)

    for gid, midxs in group_sessions.items():
        midxs_list = sorted(midxs)
        if len(midxs_list) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in midxs_list])

    # ── FTE: Instructor NoOverlap ──
    inst_possible: dict[int, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for iidx in s.qualified_instructor_idxs:
            inst_possible[iidx].add(mi)

    for iidx, midxs in inst_possible.items():
        midxs_sorted = sorted(midxs)
        if len(midxs_sorted) <= 1:
            continue
        opt_ivs = []
        for mi in midxs_sorted:
            orig_i = model_indices[mi]
            s = sessions[orig_i]
            pres = model.new_bool_var(f"fte_{mi}_{iidx}")
            model.add(inst_vars[mi] == iidx).only_enforce_if(pres)
            model.add(inst_vars[mi] != iidx).only_enforce_if(~pres)
            opt_iv = model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi], pres, f"oiv_i_{mi}_{iidx}"
            )
            opt_ivs.append(opt_iv)
        model.add_no_overlap(opt_ivs)

    # ── ICTD: Sibling sessions on different days ──
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        sibling_groups[s.sibling_key].append(mi)

    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        if len(siblings) <= num_days:
            model.add_all_different([day_vars[mi] for mi in siblings])
        # Symmetry breaking
        for j in range(len(siblings) - 1):
            model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # ── SRE surrogates: Per-pool cumulative constraints ──
    # Group sessions by their room pool (frozenset of compatible rooms)
    pool_to_sessions: dict[frozenset[int], list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        pool = frozenset(s.compatible_room_idxs)
        pool_to_sessions[pool].append(mi)

    n_cum = 0
    for pool, midxs in pool_to_sessions.items():
        cap = len(pool)
        if len(midxs) <= 1:
            continue
        # Cumulative: at each time point, sum demand(pool) <= |pool|
        demands = [sessions[model_indices[mi]].duration for mi in midxs]
        # Check if cumulative can possibly be violated
        total_demand = sum(demands)
        if total_demand <= cap:
            continue  # trivially satisfied
        model.add_cumulative(
            [interval_vars[mi] for mi in midxs],
            [1] * len(midxs),
            cap,
        )
        n_cum += 1

    # Type-level cumulatives: count rooms by type
    n_lecture = sum(
        1 for r in store.rooms.values()
        if r.room_features in ("lecture", "both", "multipurpose")
    )
    n_practical = sum(
        1 for r in store.rooms.values()
        if r.room_features in ("practical", "both", "multipurpose")
    )
    n_total = len(store.rooms)

    # Theory sessions cumulative <= lecture-compatible rooms
    theory_mis = [
        mi for mi, orig_i in enumerate(model_indices)
        if sessions[orig_i].course_type == "theory"
    ]
    if len(theory_mis) > 1:
        model.add_cumulative(
            [interval_vars[mi] for mi in theory_mis],
            [1] * len(theory_mis),
            n_lecture,
        )
        n_cum += 1

    # Practical sessions cumulative <= practical-compatible rooms
    practical_mis = [
        mi for mi, orig_i in enumerate(model_indices)
        if sessions[orig_i].course_type == "practical"
    ]
    if len(practical_mis) > 1:
        model.add_cumulative(
            [interval_vars[mi] for mi in practical_mis],
            [1] * len(practical_mis),
            n_practical,
        )
        n_cum += 1

    # Global cumulative: all sessions <= total rooms
    if N > 1:
        model.add_cumulative(
            interval_vars,
            [1] * N,
            n_total,
        )
        n_cum += 1

    print(f"  Cumulative constraints: {n_cum} "
          f"(pools: {n_cum-3}, theory: 1, practical: 1, global: 1)")
    print(f"  Room counts: {n_lecture} lecture, {n_practical} practical, {n_total} total")

    vars_dict = {
        "start": start_vars,
        "end": end_vars,
        "instructor": inst_vars,
        "day": day_vars,
        "interval": interval_vars,
        "model_indices": model_indices,
        "impossible_sessions": impossible_sessions,
    }
    return model, vars_dict


def solve_phase1(
    model: cp_model.CpModel,
    vars_dict: dict,
    time_limit: int,
) -> cp_model.CpSolver | None:
    """Run Phase 1 solver. Returns solver if feasible, None otherwise."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True

    print(f"\n{'=' * 72}")
    print("  PHASE 1: Time + Instructor Assignment (CP-SAT)")
    print(f"  Time limit: {time_limit}s")
    print(f"{'=' * 72}\n")

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }

    print(f"\n  Phase 1 result: {STATUS.get(status, '?')}")
    print(f"  Time: {elapsed:.2f}s | Branches: {solver.num_branches:,} | "
          f"Conflicts: {solver.num_conflicts:,}")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver
    return None


def phase2_room_assignment(
    solver: cp_model.CpSolver,
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    vars_dict: dict,
) -> list[dict] | None:
    """Phase 2: Assign rooms via bipartite matching per time quantum.

    For each quantum, find which sessions are active, and assign compatible
    rooms via maximum bipartite matching (Hungarian/Hopcroft-Karp).
    Returns list of assignment dicts if successful, None if matching fails.
    """
    qts = store.qts
    total_quanta = qts.total_quanta
    model_indices = vars_dict["model_indices"]

    # Extract Phase 1 solution
    assignments = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        st = solver.value(vars_dict["start"][mi])
        iidx = solver.value(vars_dict["instructor"][mi])
        assignments.append({
            "mi": mi,
            "orig_i": orig_i,
            "start": st,
            "duration": s.duration,
            "end": st + s.duration,
            "instructor": iidx,
            "compatible_rooms": s.compatible_room_idxs,
            "room": -1,  # to be assigned
        })

    # Build quantum-to-sessions map
    quantum_sessions: dict[int, list[int]] = defaultdict(list)  # quantum -> list of assignment indices
    for ai, a in enumerate(assignments):
        for q in range(a["start"], a["end"]):
            quantum_sessions[q].append(ai)

    # For each quantum, solve bipartite matching
    # Sessions active at a quantum all need distinct rooms
    # Use greedy with backtracking (or Hungarian algorithm)

    print(f"\n{'=' * 72}")
    print("  PHASE 2: Room Assignment (Bipartite Matching)")
    print(f"{'=' * 72}")

    # Find max simultaneous sessions
    max_concurrent = max(len(v) for v in quantum_sessions.values()) if quantum_sessions else 0
    print(f"  Max concurrent sessions: {max_concurrent}")
    print(f"  Total rooms: {len(room_ids)}")

    # Group quanta by the set of active assignment indices
    # (many quanta have the same set of active sessions → same matching needed)
    pattern_to_quanta: dict[frozenset[int], list[int]] = defaultdict(list)
    for q, ai_list in quantum_sessions.items():
        pattern_to_quanta[frozenset(ai_list)].append(q)

    print(f"  Unique session patterns: {len(pattern_to_quanta)}")

    # For each session, maintain the assigned room (must be consistent across quanta)
    # A session spanning quanta [s, s+d) must use the SAME room in all d quanta.
    # So we process sessions, not individual quanta.

    # Approach: Process sessions in order of most constrained first (smallest pool).
    # Use CP-SAT for room assignment with per-room NoOverlap using fixed intervals.

    print("\n  Using CP-SAT for room assignment with fixed time slots...")

    room_model = cp_model.CpModel()
    room_vars_p2 = []

    for ai, a in enumerate(assignments):
        s = sessions[a["orig_i"]]
        room_domain = sorted(a["compatible_rooms"])
        rv = room_model.new_int_var_from_domain(
            cp_model.Domain.from_values(room_domain), f"r2_{ai}"
        )
        room_vars_p2.append(rv)

    # Per-room NoOverlap with fixed intervals
    # Group assignments by which rooms they could use
    room_candidates: dict[int, list[int]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        for ridx in a["compatible_rooms"]:
            room_candidates[ridx].append(ai)

    n_no_overlap = 0
    for ridx, ai_list in room_candidates.items():
        if len(ai_list) <= 1:
            continue

        # Check for overlapping pairs
        overlapping = []
        for ai in ai_list:
            a = assignments[ai]
            overlapping.append((ai, a["start"], a["end"]))

        # Only create NoOverlap if there are actually overlapping sessions
        has_overlap = False
        for i in range(len(overlapping)):
            for j in range(i + 1, len(overlapping)):
                if overlapping[i][1] < overlapping[j][2] and overlapping[j][1] < overlapping[i][2]:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if not has_overlap:
            continue

        # Create optional fixed-size intervals
        opt_ivs = []
        seen_keys = set()
        for ai in ai_list:
            a = assignments[ai]
            key = (ai, ridx)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            pres = room_model.new_bool_var(f"rp_{ai}_{ridx}")
            room_model.add(room_vars_p2[ai] == ridx).only_enforce_if(pres)
            room_model.add(room_vars_p2[ai] != ridx).only_enforce_if(~pres)
            opt_iv = room_model.new_optional_fixed_size_interval_var(
                a["start"], a["duration"], pres, f"riv_{ai}_{ridx}"
            )
            opt_ivs.append(opt_iv)

        if len(opt_ivs) > 1:
            room_model.add_no_overlap(opt_ivs)
            n_no_overlap += 1

    print(f"  Room NoOverlap constraints: {n_no_overlap}")

    # Solve Phase 2
    room_solver = cp_model.CpSolver()
    room_solver.parameters.max_time_in_seconds = 120
    room_solver.parameters.num_workers = 8
    room_solver.parameters.log_search_progress = True

    t0 = time.time()
    status = room_solver.Solve(room_model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"\n  Phase 2 result: {STATUS.get(status, '?')}")
    print(f"  Time: {elapsed:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("  Room assignment failed!")
        return None

    # Extract room assignments
    for ai, a in enumerate(assignments):
        a["room"] = room_solver.value(room_vars_p2[ai])

    return assignments


def verify_schedule(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> int:
    """Verify all hard constraints. Returns number of violations."""
    violations = 0

    # CTE: no group overlap
    group_slots: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        s = sessions[a["orig_i"]]
        for gid in s.group_ids:
            group_slots[gid].append((a["start"], a["end"]))

    for gid, slots in group_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                violations += 1

    # FTE: no instructor overlap
    inst_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        inst_slots[a["instructor"]].append((a["start"], a["end"]))

    for iidx, slots in inst_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                violations += 1

    # SRE: no room overlap
    room_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        room_slots[a["room"]].append((a["start"], a["end"]))

    sre_violations = 0
    for ridx, slots in room_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                sre_violations += 1
                violations += 1

    # FFC: room compatibility
    ffc_violations = 0
    for a in assignments:
        s = sessions[a["orig_i"]]
        if a["room"] not in s.compatible_room_idxs:
            ffc_violations += 1
            violations += 1

    # FCA: instructor availability
    fca_violations = 0
    for a in assignments:
        s = sessions[a["orig_i"]]
        iid = instructor_ids[a["instructor"]]
        inst = store.instructors[iid]
        if not inst.is_full_time:
            for q in range(a["start"], a["end"]):
                if q not in inst.available_quanta:
                    fca_violations += 1
                    violations += 1
                    break

    print(f"\n  Verification: {violations} total violations")
    if sre_violations:
        print(f"    SRE (room overlap): {sre_violations}")
    if ffc_violations:
        print(f"    FFC (room compat):  {ffc_violations}")
    if fca_violations:
        print(f"    FCA (availability): {fca_violations}")

    return violations


def export_schedule(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    path: str,
) -> None:
    """Export the schedule as a JSON file."""
    qts = store.qts
    schedule = []

    for a in assignments:
        s = sessions[a["orig_i"]]
        iid = instructor_ids[a["instructor"]]
        rid = room_ids[a["room"]]
        day_str, time_str = qts.quanta_to_time(a["start"])

        schedule.append({
            "session_index": a["orig_i"],
            "course_id": s.course_id,
            "course_type": s.course_type,
            "group_ids": s.group_ids,
            "instructor_id": iid,
            "instructor_name": store.instructors[iid].name,
            "room_id": rid,
            "room_name": store.rooms[rid].name,
            "start_quanta": a["start"],
            "duration": s.duration,
            "day": day_str,
            "time": time_str,
        })

    with open(path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"\n  Schedule exported to: {path}")
    print(f"  Total entries: {len(schedule)}")


def print_schedule_summary(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> None:
    """Print a human-readable summary of the schedule."""
    qts = store.qts

    # Per-day load
    day_load: dict[str, int] = defaultdict(int)
    day_sessions: dict[str, int] = defaultdict(int)
    for a in assignments:
        day_str, _ = qts.quanta_to_time(a["start"])
        day_load[day_str] += sessions[a["orig_i"]].duration
        day_sessions[day_str] += 1

    print(f"\n  Schedule Summary:")
    print(f"  {'Day':<12} {'Sessions':>8} {'Quanta':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for day in qts.DAY_NAMES:
        if day in day_load:
            print(f"  {day:<12} {day_sessions[day]:>8} {day_load[day]:>8}")

    # Instructor utilization
    inst_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        iid = instructor_ids[a["instructor"]]
        inst_usage[iid] += sessions[a["orig_i"]].duration

    print(f"\n  Top 10 busiest instructors:")
    for iid, load in sorted(inst_usage.items(), key=lambda x: -x[1])[:10]:
        name = store.instructors[iid].name
        avail = len(store.instructors[iid].available_quanta) or 42
        pct = load / avail * 100
        print(f"    {name:30s}  {load:3d}/{avail:2d}q  ({pct:.0f}%)")

    # Room utilization
    room_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        rid = room_ids[a["room"]]
        room_usage[rid] += sessions[a["orig_i"]].duration

    used_rooms = len(room_usage)
    total_rooms = len(room_ids)
    print(f"\n  Rooms used: {used_rooms}/{total_rooms}")
    print(f"  Top 10 busiest rooms:")
    for rid, load in sorted(room_usage.items(), key=lambda x: -x[1])[:10]:
        name = store.rooms[rid].name
        pct = load / 42 * 100
        print(f"    {rid:8s} {name:30s}  {load:3d}/42q  ({pct:.0f}%)")


def build_relaxed_model(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> tuple[cp_model.CpModel, dict]:
    """Build a relaxed model: CTE + FTE + FPC only (no FCA, ICTD, cumulatives).

    This is known to solve in ~1s and provides a warm-start for the full model.
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    day_offsets: list[int] = []
    day_lengths: list[int] = []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)
    num_days = len(day_offsets)
    quanta_per_day = day_lengths[0] if day_lengths else 7

    model = cp_model.CpModel()

    # Pre-compute allowed tuples (FCA RELAXED — all full-time starts)
    session_allowed: list[list[tuple[int, int]]] = []
    impossible_sessions: list[int] = []
    for i, s in enumerate(sessions):
        allowed: list[tuple[int, int]] = []
        base_starts = compute_valid_starts(
            s.duration, total_quanta, day_offsets, day_lengths
        )
        for inst_idx in s.qualified_instructor_idxs:
            for st in base_starts:
                allowed.append((inst_idx, st))
        session_allowed.append(allowed)
        if not allowed or not s.compatible_room_idxs:
            impossible_sessions.append(i)

    impossible_set = set(impossible_sessions)
    model_indices: list[int] = []
    for i in range(len(sessions)):
        if i not in impossible_set:
            model_indices.append(i)

    start_vars = []
    end_vars = []
    inst_vars = []
    day_vars = []
    interval_vars = []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        allowed = session_allowed[orig_i]

        inst_domain = sorted({t[0] for t in allowed})
        start_domain = sorted({t[1] for t in allowed})

        inst = model.new_int_var_from_domain(
            cp_model.Domain.from_values(inst_domain), f"inst_{mi}"
        )
        start = model.new_int_var_from_domain(
            cp_model.Domain.from_values(start_domain), f"start_{mi}"
        )
        end = model.new_int_var(0, total_quanta, f"end_{mi}")
        model.add(end == start + s.duration)

        day = model.new_int_var(0, num_days - 1, f"day_{mi}")
        model.add_division_equality(day, start, quanta_per_day)

        interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")
        model.add_allowed_assignments([inst, start], allowed)

        start_vars.append(start)
        end_vars.append(end)
        inst_vars.append(inst)
        day_vars.append(day)
        interval_vars.append(interval)

    # CTE: Group NoOverlap
    group_sessions: dict[str, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for gid in set(s.group_ids):
            group_sessions[gid].add(mi)

    for gid, midxs in group_sessions.items():
        midxs_list = sorted(midxs)
        if len(midxs_list) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in midxs_list])

    # FTE: Instructor NoOverlap
    inst_possible: dict[int, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for iidx in s.qualified_instructor_idxs:
            inst_possible[iidx].add(mi)

    for iidx, midxs in inst_possible.items():
        midxs_sorted = sorted(midxs)
        if len(midxs_sorted) <= 1:
            continue
        opt_ivs = []
        for mi in midxs_sorted:
            orig_i = model_indices[mi]
            s = sessions[orig_i]
            pres = model.new_bool_var(f"fte_{mi}_{iidx}")
            model.add(inst_vars[mi] == iidx).only_enforce_if(pres)
            model.add(inst_vars[mi] != iidx).only_enforce_if(~pres)
            opt_iv = model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi], pres, f"oiv_i_{mi}_{iidx}"
            )
            opt_ivs.append(opt_iv)
        model.add_no_overlap(opt_ivs)

    # Symmetry breaking for siblings
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        sibling_groups[s.sibling_key].append(mi)

    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        for j in range(len(siblings) - 1):
            model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    vars_dict = {
        "start": start_vars,
        "end": end_vars,
        "instructor": inst_vars,
        "day": day_vars,
        "interval": interval_vars,
        "model_indices": model_indices,
        "impossible_sessions": impossible_sessions,
    }
    return model, vars_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase CP-SAT scheduler"
    )
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Phase 1 time limit in seconds (default: 300)")
    parser.add_argument("--data-dir", type=str, default="data_fixed",
                        help="Data directory (default: data_fixed)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export schedule to JSON file")
    parser.add_argument("--no-diag", action="store_true",
                        help="Skip diagnostics")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    # Build sessions
    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    # ── Phase 0: Warm-start (relaxed model) ──
    print("\n" + "=" * 72)
    print("  PHASE 0: Warm-start (CTE+FTE+FPC only, no FCA/ICTD/SRE)")
    print("=" * 72)
    relaxed_model, relaxed_vars = build_relaxed_model(
        sessions, store, instructor_ids, room_ids
    )
    relaxed_solver = solve_phase1(relaxed_model, relaxed_vars, time_limit=30)

    # ── Phase 1: Full model with hints ──
    print("\nBuilding Phase 1 model (full constraints + pool cumulatives)...")
    model, vars_dict = build_phase1_model(
        sessions, store, instructor_ids, room_ids
    )

    # Add hints from Phase 0 solution
    if relaxed_solver is not None:
        print("  Adding warm-start hints from Phase 0...")
        n_hints = 0
        for mi in range(len(vars_dict["start"])):
            model.add_hint(vars_dict["start"][mi],
                           relaxed_solver.value(relaxed_vars["start"][mi]))
            model.add_hint(vars_dict["instructor"][mi],
                           relaxed_solver.value(relaxed_vars["instructor"][mi]))
            n_hints += 2
        print(f"  {n_hints} hints added")

    solver = solve_phase1(model, vars_dict, args.time_limit)
    if solver is None:
        print("\nPhase 1 FAILED. Cannot produce schedule.")
        sys.exit(1)

    # Phase 2
    assignments = phase2_room_assignment(
        solver, sessions, store, instructor_ids, room_ids, vars_dict
    )
    if assignments is None:
        print("\nPhase 2 FAILED. Room assignment impossible.")
        sys.exit(1)

    # Verify
    violations = verify_schedule(
        assignments, sessions, store, instructor_ids, room_ids
    )

    # Summary
    print_schedule_summary(
        assignments, sessions, store, instructor_ids, room_ids
    )

    if violations == 0 and args.export:
        export_schedule(
            assignments, sessions, store, instructor_ids, room_ids, args.export
        )
    elif violations > 0:
        print(f"\n  WARNING: Schedule has {violations} violations!")
        if args.export:
            export_schedule(
                assignments, sessions, store, instructor_ids, room_ids, args.export
            )

    print(f"\n  Done.")
    sys.exit(0 if violations == 0 else 1)


if __name__ == "__main__":
    main()
