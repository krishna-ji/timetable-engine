#!/usr/bin/env python3
"""CP-SAT Feasibility Oracle for University Course Timetabling.

Uses Google OR-Tools CP-SAT solver to definitively answer:
  1. Does a feasible schedule exist for the current data?
  2. If infeasible, which constraint relaxation unlocks feasibility?

Usage:
    python cpsat_oracle.py                          # basic feasibility check
    python cpsat_oracle.py --time-limit 300         # give solver 5 minutes
    python cpsat_oracle.py --relax-ictd             # drop same-day constraint
    python cpsat_oracle.py --relax-fca              # drop availability constraint
    python cpsat_oracle.py --relax-ictd --relax-fca # drop both
    python cpsat_oracle.py --export schedule.json   # export solution
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ortools.sat.python import cp_model

from src.ga.core.population import (
    analyze_group_hierarchy,
    generate_course_group_pairs,
    get_subsession_durations,
)
from src.io.data_store import DataStore
from src.utils.room_compatibility import is_room_suitable_for_course


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────


@dataclass
class Session:
    """A single scheduling unit (one subsession of a course-group pair)."""

    idx: int
    course_id: str
    course_type: str  # "theory" or "practical"
    group_ids: list[str]
    duration: int  # quanta
    qualified_instructor_idxs: list[int]  # indices into instructor_ids
    compatible_room_idxs: list[int]  # indices into room_ids
    sibling_key: tuple  # (course_id, course_type, tuple(sorted(group_ids)))


# ──────────────────────────────────────────────────────────────────
# Session generation — mirrors GA population.py logic
# ──────────────────────────────────────────────────────────────────


def build_sessions(store: DataStore) -> tuple[list[Session], list[str], list[str]]:
    """Generate all sessions from the loaded data.

    Returns (sessions, instructor_ids_list, room_ids_list).
    """
    hierarchy = analyze_group_hierarchy(store.groups)
    pairs = generate_course_group_pairs(
        store.courses, store.groups, hierarchy, silent=True
    )

    instructor_ids = list(store.instructors.keys())
    inst_to_idx = {iid: i for i, iid in enumerate(instructor_ids)}
    room_ids = list(store.rooms.keys())
    room_to_idx = {rid: i for i, rid in enumerate(room_ids)}

    # Pre-compute compatible rooms per course key
    room_compat_cache: dict[tuple[str, str], list[int]] = {}

    sessions: list[Session] = []
    idx = 0

    for course_key, group_ids, _session_type, _num_quanta in pairs:
        course = store.courses.get(course_key)
        if course is None:
            continue

        # Qualified instructors (indices, deduplicated)
        q_inst = sorted(
            set(
                inst_to_idx[iid]
                for iid in course.qualified_instructor_ids
                if iid in inst_to_idx
            )
        )

        # Compatible rooms (indices) — cached per course key
        if course_key not in room_compat_cache:
            compat: list[int] = []
            for rid, room in store.rooms.items():
                if is_room_suitable_for_course(
                    course.required_room_features,
                    room.room_features,
                    course.specific_lab_features or None,
                    room.specific_features or None,
                ):
                    compat.append(room_to_idx[rid])
            room_compat_cache[course_key] = sorted(set(compat))
        c_rooms = room_compat_cache[course_key]

        # Subsession durations
        durations = get_subsession_durations(
            course.quanta_per_week, course.course_type
        )
        sibling_key = (course.course_id, course.course_type, tuple(sorted(group_ids)))

        for dur in durations:
            sessions.append(
                Session(
                    idx=idx,
                    course_id=course.course_id,
                    course_type=course.course_type,
                    group_ids=group_ids,
                    duration=dur,
                    qualified_instructor_idxs=q_inst,
                    compatible_room_idxs=c_rooms,
                    sibling_key=sibling_key,
                )
            )
            idx += 1

    return sessions, instructor_ids, room_ids


# ──────────────────────────────────────────────────────────────────
# Valid-start helpers
# ──────────────────────────────────────────────────────────────────


def compute_valid_starts(
    duration: int, total_quanta: int, day_offsets: list[int], day_lengths: list[int]
) -> list[int]:
    """Start quanta where the session fits.

    Sessions that fit within a single day must not cross day boundaries.
    Sessions longer than a single day can span multiple days (continuous quanta).
    """
    min_day_len = min(day_lengths) if day_lengths else 7

    if duration <= min_day_len:
        # Single-day: must fit entirely within one day
        valid: list[int] = []
        for offset, length in zip(day_offsets, day_lengths):
            for s in range(offset, offset + length - duration + 1):
                valid.append(s)
        return valid
    else:
        # Multi-day: can start anywhere that fits within total quanta
        return list(range(0, total_quanta - duration + 1))


def compute_valid_starts_for_instructor(
    duration: int,
    instructor_available: set[int],
    total_quanta: int,
    day_offsets: list[int],
    day_lengths: list[int],
) -> list[int]:
    """Starts where ALL quanta of the session are in instructor availability AND valid."""
    base = compute_valid_starts(duration, total_quanta, day_offsets, day_lengths)
    return [
        s
        for s in base
        if all(q in instructor_available for q in range(s, s + duration))
    ]


# ──────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────


def build_model(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    *,
    relax_ictd: bool = False,
    relax_fca: bool = False,
    relax_sre: bool = False,
    relax_fte: bool = False,
    relax_ffc: bool = False,
) -> tuple[cp_model.CpModel, dict]:
    """Build the CP-SAT model encoding all hard constraints.

    Returns (model, vars_dict).
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    # Day boundaries (from QTS)
    day_offsets: list[int] = []
    day_lengths: list[int] = []
    day_names_op: list[str] = []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)
            day_names_op.append(day)
    num_days = len(day_offsets)
    quanta_per_day = day_lengths[0] if day_lengths else 7

    model = cp_model.CpModel()

    # ── Pre-compute per-session allowed (instructor, start) tuples ──
    # This single table constraint per session handles:
    #   FPC (qualification), FCA (availability), day-boundary
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
            if instructor.is_full_time or relax_fca:
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

    # ── Report impossible sessions (not included in model) ──────────
    if impossible_sessions:
        print(
            f"\nWARNING: {len(impossible_sessions)} sessions are IMPOSSIBLE to "
            f"schedule (no valid instructor×start or no compatible room)."
        )
        print("  These sessions are EXCLUDED from the model (FCA/FPC data issue):")
        for si in impossible_sessions:
            s = sessions[si]
            print(
                f"  S{si}: {s.course_id} ({s.course_type}) dur={s.duration} "
                f"groups={s.group_ids} "
                f"qual_inst={len(s.qualified_instructor_idxs)} "
                f"compat_rooms={len(s.compatible_room_idxs)} "
                f"tuples={len(session_allowed[si])}"
            )
        print(
            "  → These need data fixes (instructor availability or room features)."
        )
        print(
            f"  → Run with --relax-fca to check feasibility ignoring availability.\n"
        )

    impossible_set = set(impossible_sessions)

    # ── Build index mapping: original session idx → model position ──
    # Only include sessions that have valid assignments
    model_indices: list[int] = []  # model position → original session idx
    session_to_model: dict[int, int] = {}  # original → model position
    for i in range(len(sessions)):
        if i not in impossible_set:
            session_to_model[i] = len(model_indices)
            model_indices.append(i)

    # ── Decision variables ──────────────────────────────────────────
    start_vars: list[cp_model.IntVar] = []
    end_vars: list[cp_model.IntVar] = []
    inst_vars: list[cp_model.IntVar] = []
    room_vars: list[cp_model.IntVar] = []
    day_vars: list[cp_model.IntVar] = []
    interval_vars: list[cp_model.IntervalVar] = []

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

        if relax_ffc:
            room_domain = list(range(len(room_ids)))
        else:
            room_domain = sorted(s.compatible_room_idxs)
        room = model.new_int_var_from_domain(
            cp_model.Domain.from_values(room_domain), f"room_{mi}"
        )

        day = model.new_int_var(0, num_days - 1, f"day_{mi}")
        model.add_division_equality(day, start, quanta_per_day)

        interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")

        model.add_allowed_assignments([inst, start], allowed)

        start_vars.append(start)
        end_vars.append(end)
        inst_vars.append(inst)
        room_vars.append(room)
        day_vars.append(day)
        interval_vars.append(interval)

    # ── C1: CTE — Group NoOverlap ──────────────────────────────────
    group_sessions: dict[str, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for gid in set(s.group_ids):
            group_sessions[gid].add(mi)

    for gid, midxs in group_sessions.items():
        midxs_list = sorted(midxs)
        if len(midxs_list) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in midxs_list])

    # ── C2: FTE — Instructor NoOverlap (channeling) ────────────────
    if not relax_fte:
        inst_possible: dict[int, set[int]] = defaultdict(set)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for iidx in s.qualified_instructor_idxs:
                inst_possible[iidx].add(mi)

        for iidx, midxs in inst_possible.items():
            midxs_sorted = sorted(midxs)
            if len(midxs_sorted) <= 1:
                continue
            opt_ivs: list[cp_model.IntervalVar] = []
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

    # ── C3: SRE — Room NoOverlap (channeling) ──────────────────────
    if not relax_sre:
        room_possible: dict[int, set[int]] = defaultdict(set)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            r_idxs = range(len(room_ids)) if relax_ffc else s.compatible_room_idxs
            for ridx in r_idxs:
                room_possible[ridx].add(mi)

        for ridx, midxs in room_possible.items():
            midxs_sorted = sorted(midxs)
            if len(midxs_sorted) <= 1:
                continue
            opt_ivs = []
            for mi in midxs_sorted:
                orig_i = model_indices[mi]
                s = sessions[orig_i]
                pres = model.new_bool_var(f"sre_{mi}_{ridx}")
                model.add(room_vars[mi] == ridx).only_enforce_if(pres)
                model.add(room_vars[mi] != ridx).only_enforce_if(~pres)
                opt_iv = model.new_optional_interval_var(
                    start_vars[mi], s.duration, end_vars[mi], pres, f"oiv_r_{mi}_{ridx}"
                )
                opt_ivs.append(opt_iv)
            model.add_no_overlap(opt_ivs)

    # ── C4+C5: FPC + FFC — handled by domain restriction + table ───

    # ── C6: FCA — handled by table constraint ──────────────────────

    # ── C7: CQF — structurally guaranteed by session generation ────

    # ── C8: ICTD — Sibling sessions on different days ──────────────
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        sibling_groups[s.sibling_key].append(mi)

    if not relax_ictd:
        ictd_infeasible = 0
        for key, siblings in sibling_groups.items():
            if len(siblings) <= 1:
                continue
            if len(siblings) > num_days:
                ictd_infeasible += 1
            model.add_all_different([day_vars[mi] for mi in siblings])

        if ictd_infeasible:
            print(
                f"\nWARNING: {ictd_infeasible} course offerings have more "
                f"sibling sessions ({max(len(v) for v in sibling_groups.values())}) "
                f"than available days ({num_days}) — ICTD forces infeasibility!"
            )

    # ── Symmetry breaking for sibling sessions ─────────────────────
    # Sibling sessions (same course, type, groups) are interchangeable.
    # Fix ascending start order to break symmetry AND prevent CP-SAT
    # presolve from merging identical-domain interval variables.
    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        for j in range(len(siblings) - 1):
            model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # ── C9: PMI — skipped (co-instructor assignment not modeled) ───

    vars_dict = {
        "start": start_vars,
        "end": end_vars,
        "instructor": inst_vars,
        "room": room_vars,
        "day": day_vars,
        "interval": interval_vars,
        "model_indices": model_indices,  # model pos → original session idx
        "impossible_sessions": impossible_sessions,
    }
    return model, vars_dict


# ──────────────────────────────────────────────────────────────────
# Solver + reporter
# ──────────────────────────────────────────────────────────────────


def solve_and_report(
    model: cp_model.CpModel,
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    vars_dict: dict,
    time_limit: int = 120,
    export_path: str | None = None,
) -> bool | None:
    """Solve the model and print a human-readable report.

    Returns True (feasible), False (infeasible), or None (unknown/timeout).
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True

    n_vars = len(vars_dict["start"]) * 4  # approximate variable count
    print("\n" + "=" * 72)
    print("  CP-SAT FEASIBILITY ORACLE")
    print("=" * 72)
    print(f"  Sessions:      {len(sessions)}")
    print(f"  Variables:     ~{n_vars}")
    print(f"  Time limit:    {time_limit}s")
    print(f"  Workers:       {solver.parameters.num_workers}")
    print("=" * 72 + "\n")

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

    print("\n" + "=" * 72)
    print(f"  RESULT: {STATUS_NAMES.get(status, 'UNKNOWN')}")
    print(f"  Elapsed:    {elapsed:.2f}s")
    print(f"  Branches:   {solver.num_branches:,}")
    print(f"  Conflicts:  {solver.num_conflicts:,}")
    print(f"  Wall time:  {solver.wall_time:.2f}s")
    print("=" * 72)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        model_indices = vars_dict["model_indices"]
        impossible = vars_dict["impossible_sessions"]

        if impossible:
            print(f"\n  A FEASIBLE SCHEDULE EXISTS (excluding {len(impossible)} "
                  f"impossible sessions)!\n")
        else:
            print("\n  A FEASIBLE SCHEDULE EXISTS!\n")

        # Print summary statistics
        inst_usage: dict[str, int] = defaultdict(int)
        room_usage: dict[str, int] = defaultdict(int)
        day_usage: dict[int, int] = defaultdict(int)

        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            iidx = solver.value(vars_dict["instructor"][mi])
            ridx = solver.value(vars_dict["room"][mi])
            d = solver.value(vars_dict["day"][mi])
            inst_usage[instructor_ids[iidx]] += s.duration
            room_usage[room_ids[ridx]] += s.duration
            day_usage[d] += s.duration

        print(f"  Instructor utilization (top 10):")
        for iid, load in sorted(inst_usage.items(), key=lambda x: -x[1])[:10]:
            name = store.instructors[iid].name
            print(f"    {iid:6s} {name:30s}  {load:3d} quanta")

        day_names_op = [
            d
            for d in store.qts.DAY_NAMES
            if store.qts.day_quanta_offset.get(d) is not None
        ]
        print(f"\n  Sessions per day:")
        for d_idx in sorted(day_usage):
            dname = day_names_op[d_idx] if d_idx < len(day_names_op) else f"Day{d_idx}"
            print(f"    {dname:12s}  {day_usage[d_idx]:3d} quanta")

        # Export solution if requested
        if export_path:
            _export_solution(
                solver, sessions, store, instructor_ids, room_ids, vars_dict, export_path
            )

        return True

    elif status == cp_model.INFEASIBLE:
        print("\n  NO FEASIBLE SCHEDULE EXISTS with current constraints!\n")
        print("  The problem is STRUCTURALLY INFEASIBLE.")
        print("  Possible causes:")
        print("    1. Groups need more quanta than available (pigeonhole)")
        print("    2. Part-time instructor availability too restrictive (FCA)")
        print("    3. Too many sibling sessions for available days (ICTD)")
        print("    4. Not enough compatible rooms for simultaneous sessions")
        print()
        print("  Recommended next steps:")
        print("    python cpsat_oracle.py --relax-ictd")
        print("    python cpsat_oracle.py --relax-fca")
        print("    python cpsat_oracle.py --relax-ictd --relax-fca")
        return False

    else:
        print("\n  Could not determine feasibility within time limit.")
        print("  Try: python cpsat_oracle.py --time-limit 600")
        return None


def _export_solution(
    solver: cp_model.CpSolver,
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    vars_dict: dict,
    path: str,
) -> None:
    """Export the CP-SAT solution as a JSON schedule."""
    qts = store.qts
    model_indices = vars_dict["model_indices"]
    schedule: list[dict] = []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        start_q = solver.value(vars_dict["start"][mi])
        iidx = solver.value(vars_dict["instructor"][mi])
        ridx = solver.value(vars_dict["room"][mi])

        day_str, time_str = qts.quanta_to_time(start_q)
        schedule.append(
            {
                "session_index": orig_i,
                "course_id": s.course_id,
                "course_type": s.course_type,
                "group_ids": s.group_ids,
                "instructor_id": instructor_ids[iidx],
                "instructor_name": store.instructors[instructor_ids[iidx]].name,
                "room_id": room_ids[ridx],
                "start_quanta": start_q,
                "duration": s.duration,
                "day": day_str,
                "time": time_str,
            }
        )

    with open(path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"\n  Solution exported to: {path}")


# ──────────────────────────────────────────────────────────────────
# Diagnostics — run before solving to surface obvious issues
# ──────────────────────────────────────────────────────────────────


def run_diagnostics(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> None:
    """Print pre-solve diagnostics about the problem instance."""
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

    print("\n" + "=" * 72)
    print("  PROBLEM DIAGNOSTICS")
    print("=" * 72)
    print(f"  Total sessions:     {len(sessions)}")
    print(f"  Total quanta/week:  {total_quanta}")
    print(f"  Operational days:   {num_days}")
    print(f"  Quanta/day:         {day_lengths}")
    print(f"  Instructors:        {len(instructor_ids)}")
    print(f"  Rooms:              {len(room_ids)}")

    # Group load analysis
    group_load: dict[str, int] = defaultdict(int)
    for s in sessions:
        for gid in s.group_ids:
            group_load[gid] += s.duration

    print(f"\n  Group load (quanta needed vs {total_quanta} available):")
    overloaded = []
    for gid, load in sorted(group_load.items(), key=lambda x: -x[1]):
        pct = load / total_quanta * 100
        flag = " *** OVERLOADED ***" if load > total_quanta else ""
        if pct > 80 or flag:
            print(f"    {gid:10s}  {load:3d}/{total_quanta}  ({pct:5.1f}%){flag}")
        if load > total_quanta:
            overloaded.append(gid)

    if overloaded:
        print(f"\n  CRITICAL: {len(overloaded)} groups need MORE quanta than exist!")
        print(f"  These groups: {overloaded}")
        print("  → The problem is INFEASIBLE regardless of other constraints.")

    # Instructor load analysis
    inst_demand: dict[int, int] = defaultdict(int)
    for s in sessions:
        for iidx in s.qualified_instructor_idxs:
            inst_demand[iidx] += s.duration

    # Sibling analysis (ICTD)
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(sessions):
        sibling_groups[s.sibling_key].append(i)

    max_siblings = max((len(v) for v in sibling_groups.values()), default=0)
    overday = sum(1 for v in sibling_groups.values() if len(v) > num_days)
    print(f"\n  Sibling analysis (ICTD):")
    print(f"    Max siblings per offering: {max_siblings}")
    print(f"    Offerings with >  {num_days} siblings: {overday}")
    if overday:
        print(f"    → ICTD constraint IMPOSSIBLE for {overday} offerings")

    # Sessions without qualified instructors or compatible rooms
    no_inst = sum(1 for s in sessions if not s.qualified_instructor_idxs)
    no_room = sum(1 for s in sessions if not s.compatible_room_idxs)
    if no_inst:
        print(f"\n  WARNING: {no_inst} sessions have NO qualified instructor")
    if no_room:
        print(f"\n  WARNING: {no_room} sessions have NO compatible room")

    print("=" * 72)


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CP-SAT Feasibility Oracle for University Course Timetabling"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="Solver time limit in seconds (default: 120)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--relax-ictd",
        action="store_true",
        help="Relax ICTD (same-day dispersion) constraint",
    )
    parser.add_argument(
        "--relax-fca",
        action="store_true",
        help="Relax FCA (instructor availability) constraint",
    )
    parser.add_argument(
        "--relax-sre",
        action="store_true",
        help="Relax SRE (room no-overlap) — allow double-booking rooms",
    )
    parser.add_argument(
        "--relax-fte",
        action="store_true",
        help="Relax FTE (instructor no-overlap) — allow instructor double-booking",
    )
    parser.add_argument(
        "--relax-ffc",
        action="store_true",
        help="Relax FFC (room compatibility) — any room can host any session",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export feasible solution to JSON file",
    )
    parser.add_argument(
        "--no-diag",
        action="store_true",
        help="Skip pre-solve diagnostics",
    )
    args = parser.parse_args()

    # ── Load data ──
    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    # ── Build sessions ──
    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    # ── Diagnostics ──
    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    # ── Build model ──
    label_parts = ["ALL hard constraints"]
    if args.relax_ictd:
        label_parts.append("ICTD relaxed")
    if args.relax_fca:
        label_parts.append("FCA relaxed")
    if args.relax_sre:
        label_parts.append("SRE relaxed")
    if args.relax_fte:
        label_parts.append("FTE relaxed")
    if args.relax_ffc:
        label_parts.append("FFC relaxed")
    print(f"\nBuilding CP-SAT model ({', '.join(label_parts)})...")

    model, vars_dict = build_model(
        sessions,
        store,
        instructor_ids,
        room_ids,
        relax_ictd=args.relax_ictd,
        relax_fca=args.relax_fca,
        relax_sre=args.relax_sre,
        relax_fte=args.relax_fte,
        relax_ffc=args.relax_ffc,
    )

    proto = model.proto
    print(
        f"  Variables:    {len(proto.variables)}"
    )
    print(
        f"  Constraints:  {len(proto.constraints)}"
    )

    # ── Solve ──
    result = solve_and_report(
        model,
        sessions,
        store,
        instructor_ids,
        room_ids,
        vars_dict,
        time_limit=args.time_limit,
        export_path=args.export,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
