#!/usr/bin/env python3
"""CP-SAT Phase 1: "No Availability" Core Solver.

Drops InstructorMustBeAvailable to massively expand the search space
and prove the core model works. Enforces:

  Hard Constraints:
    NoStudentDoubleBooking    — A student group cannot be in two places at once
    NoInstructorDoubleBooking — An instructor cannot teach two classes at the same time
    NoRoomDoubleBooking       — A room can only host one session at a time
    InstructorMustBeQualified — Instructors can only teach courses they are approved for
    RoomMustHaveFeatures      — Theory goes to lecture halls; practicals go to labs
    ExactWeeklyHours          — Every course must hit its exact required quanta per week
    SpreadAcrossDays          — Sub-sessions of the same course must happen on different days
    RequiresTwoInstructors    — Practical sessions need exactly 2 instructors (Sum == 2)

  Not enforced (deferred to Phase 2/3):
    InstructorMustBeAvailable — Dropped intentionally for Phase 1

  Soft Constraints (disabled — Phase 2):
    MinimizeStudentGaps       — Keep student schedules compact to avoid long idle times
    MinimizeInstructorGaps    — Keep instructor schedules compact
    EnsureLunchBreak          — Everyone gets a free block for lunch
    AvoidIsolatedSessions     — Penalize a single 1-quanta slot stranded by itself
    AlignCohortPracticals     — Paired student groups should ideally have their lab times synced up
    RespectBreakWindows       — Force scheduled breaks to happen during normal break hours

Usage:
    python cpsat_phase1.py                         # solve with defaults
    python cpsat_phase1.py --time-limit 300        # 5 minute timeout
    python cpsat_phase1.py --data-dir data_fixed   # use fixed data
    python cpsat_phase1.py --export solution.json  # export solution
    python cpsat_phase1.py --relax-ictd            # drop same-day rule
    python cpsat_phase1.py --relax-sre             # drop room exclusivity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
    """A single scheduling unit (one subsession of a course-group pair).

    Decision variables the solver must assign:
      - start_time  (which quantum does this session begin?)
      - instructor  (which qualified instructor teaches it?)
      - room        (which compatible room hosts it?)
    """

    idx: int
    course_id: str
    course_type: str  # "theory" or "practical"
    group_ids: list[str]
    duration: int  # in quanta (contiguous)
    qualified_instructor_idxs: list[int]  # indices into instructor_ids
    compatible_room_idxs: list[int]  # indices into room_ids
    sibling_key: tuple  # (course_id, course_type, tuple(sorted(group_ids)))


# ──────────────────────────────────────────────────────────────────
# Session generation — mirrors GA population.py logic
# ──────────────────────────────────────────────────────────────────


def build_sessions(store: DataStore) -> tuple[list[Session], list[str], list[str]]:
    """Generate all scheduling sessions from loaded data.

    Each (course, group) pair is split into subsessions based on L/T/P rules:
      - Theory → 2-quanta blocks (with 1-quanta remainder if odd)
      - Practical → single monolithic block

    ExactWeeklyHours is guaranteed structurally: the sum of
    subsession durations == course.quanta_per_week for every (course, group).

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

    # Cache compatible rooms per course key to avoid redundant checks
    room_compat_cache: dict[tuple[str, str], list[int]] = {}

    sessions: list[Session] = []
    idx = 0

    for course_key, group_ids, _session_type, _num_quanta in pairs:
        course = store.courses.get(course_key)
        if course is None:
            continue

        # InstructorMustBeQualified — restrict to qualified instructors only
        q_inst = sorted(
            set(
                inst_to_idx[iid]
                for iid in course.qualified_instructor_ids
                if iid in inst_to_idx
            )
        )

        # RoomMustHaveFeatures — restrict to rooms with matching features
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

        # L/T/P blocking: split into subsessions
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
# Valid-start computation (no availability filtering!)
# ──────────────────────────────────────────────────────────────────


def compute_valid_starts(
    duration: int,
    total_quanta: int,
    day_offsets: list[int],
    day_lengths: list[int],
) -> list[int]:
    """Compute start quanta where a session of given duration fits.

    Single-day sessions must not cross day boundaries.
    Multi-day sessions (duration > min day length) can span days.

    NOTE: No availability filtering — this is Phase 1.
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
        # Multi-day: can start anywhere that still fits
        return list(range(0, total_quanta - duration + 1))


# ──────────────────────────────────────────────────────────────────
# Phase 1 Model Builder
# ──────────────────────────────────────────────────────────────────


def build_phase1_model(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    *,
    relax_ictd: bool = False,
    relax_sre: bool = False,
    relax_fte: bool = False,
    relax_ffc: bool = False,
    relax_pmi: bool = False,
) -> tuple[cp_model.CpModel, dict]:
    """Build CP-SAT model with InstructorMustBeAvailable dropped (Phase 1 core).

    Variable design (4-step architecture):

      Step 1 — Core time variables per session i:
        start_i  ∈ {valid start quanta for duration_i}
        end_i    = start_i + duration_i
        iv_i     = IntervalVar(start_i, duration_i, end_i)  [MANDATORY]
        day_i    = start_i // quanta_per_day

      Step 2 — Resource assignment booleans:
        room_bool[(i, r)]  ∈ {0,1}  for each compatible room r
        instr_bool[(i, j)] ∈ {0,1}  for each qualified instructor j
        AddExactlyOne(room_bools per session)
        AddExactlyOne(instr_bools per session)  — theory
        Add(sum(instr_bools) == 2)              — practical (RequiresTwoInstructors)

      Step 3 — Optional intervals (the secret sauce):
        For each room_bool[(i, r)]:  optional interval active iff assigned
        For each instr_bool[(i, j)]: optional interval active iff assigned

      Step 4 — Hard constraints via NoOverlap:
        NoStudentDoubleBooking:    NoOverlap(mandatory intervals) per student group
        NoInstructorDoubleBooking: NoOverlap(optional instr intervals) per instructor
        NoRoomDoubleBooking:       NoOverlap(optional room intervals) per room
        SpreadAcrossDays:          AllDifferent(day) for sibling sessions

    Returns (model, vars_dict).
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    # ── Extract day geometry from QTS ──
    day_offsets: list[int] = []
    day_lengths: list[int] = []
    day_names: list[str] = []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)
            day_names.append(day)
    num_days = len(day_offsets)
    quanta_per_day = day_lengths[0] if day_lengths else 7

    # ── Filter impossible sessions ──
    impossible_sessions: list[int] = []
    for i, s in enumerate(sessions):
        has_instructors = len(s.qualified_instructor_idxs) > 0
        has_rooms = len(s.compatible_room_idxs) > 0
        # Practicals need at least 2 qualified instructors for RequiresTwoInstructors (unless relaxed)
        min_inst = 1 if relax_pmi else 2
        if s.course_type == "practical" and len(s.qualified_instructor_idxs) < min_inst:
            impossible_sessions.append(i)
        elif not has_instructors or not has_rooms:
            impossible_sessions.append(i)

    if impossible_sessions:
        print(
            f"\n⚠  {len(impossible_sessions)} sessions are IMPOSSIBLE "
            f"(no qualified instructor, < 2 for practical, or no compatible room):"
        )
        for si in impossible_sessions:
            s = sessions[si]
            print(
                f"   S{si}: {s.course_id} ({s.course_type}) dur={s.duration} "
                f"groups={s.group_ids} "
                f"qual_inst={len(s.qualified_instructor_idxs)} "
                f"compat_rooms={len(s.compatible_room_idxs)}"
            )
        print("   → These need data fixes. Excluded from model.\n")

    impossible_set = set(impossible_sessions)

    # ── Map original session idx → model position ──
    model_indices: list[int] = []
    for i in range(len(sessions)):
        if i not in impossible_set:
            model_indices.append(i)

    n_modeled = len(model_indices)
    print(f"  Modeled sessions: {n_modeled} (excluded: {len(impossible_sessions)})")

    model = cp_model.CpModel()

    # ================================================================
    # STEP 1: Core Time Variables (mandatory intervals)
    # ================================================================
    start_vars: list[cp_model.IntVar] = []
    end_vars: list[cp_model.IntVar] = []
    day_vars: list[cp_model.IntVar] = []
    interval_vars: list[cp_model.IntervalVar] = []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]

        # Start variable: domain = valid start positions for this duration
        valid_starts = compute_valid_starts(
            s.duration, total_quanta, day_offsets, day_lengths
        )
        start = model.new_int_var_from_domain(
            cp_model.Domain.from_values(valid_starts), f"start_{mi}"
        )

        # End variable: derived
        end = model.new_int_var(0, total_quanta, f"end_{mi}")
        model.add(end == start + s.duration)

        # Day variable: derived from start via integer division
        day = model.new_int_var(0, num_days - 1, f"day_{mi}")
        model.add_division_equality(day, start, quanta_per_day)

        # Mandatory interval — this session MUST be scheduled
        interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")

        start_vars.append(start)
        end_vars.append(end)
        day_vars.append(day)
        interval_vars.append(interval)

    # ================================================================
    # STEP 2: Resource Assignment Booleans
    # ================================================================

    # Room booleans: room_bool[(mi, ridx)] = "session mi is in room ridx"
    room_bool: dict[tuple[int, int], cp_model.IntVar] = {}
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        r_idxs = list(range(len(room_ids))) if relax_ffc else s.compatible_room_idxs
        session_room_bools = []
        for ridx in r_idxs:
            b = model.new_bool_var(f"room_{mi}_{ridx}")
            room_bool[(mi, ridx)] = b
            session_room_bools.append(b)
        # Each session goes to exactly one room (RoomMustHaveFeatures via domain)
        model.add_exactly_one(session_room_bools)

    # Instructor booleans: instr_bool[(mi, iidx)] = "instructor iidx teaches session mi"
    instr_bool: dict[tuple[int, int], cp_model.IntVar] = {}
    pmi_count = 0
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        session_instr_bools = []
        for iidx in s.qualified_instructor_idxs:
            b = model.new_bool_var(f"instr_{mi}_{iidx}")
            instr_bool[(mi, iidx)] = b
            session_instr_bools.append(b)

        if s.course_type == "practical" and not relax_pmi:
            # RequiresTwoInstructors — practicals need exactly 2 instructors
            model.add(sum(session_instr_bools) == 2)
            pmi_count += 1
        else:
            # Theory: exactly 1 instructor (InstructorMustBeQualified via domain)
            model.add_exactly_one(session_instr_bools)

    # ================================================================
    # STEP 3: Optional Intervals (The Secret Sauce)
    # ================================================================

    # Room optional intervals: active only when session is assigned to that room
    room_opt_intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
    for (mi, ridx), b in room_bool.items():
        orig_i = model_indices[mi]
        s = sessions[orig_i]
        opt_iv = model.new_optional_interval_var(
            start_vars[mi], s.duration, end_vars[mi], b, f"oiv_r_{mi}_{ridx}"
        )
        room_opt_intervals[(mi, ridx)] = opt_iv

    # Instructor optional intervals: active only when instructor is assigned
    instr_opt_intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
    for (mi, iidx), b in instr_bool.items():
        orig_i = model_indices[mi]
        s = sessions[orig_i]
        opt_iv = model.new_optional_interval_var(
            start_vars[mi], s.duration, end_vars[mi], b, f"oiv_i_{mi}_{iidx}"
        )
        instr_opt_intervals[(mi, iidx)] = opt_iv

    # ================================================================
    # STEP 4: Hard Constraints
    # ================================================================

    # ── NoStudentDoubleBooking ──
    # Group all MANDATORY intervals by student group → NoOverlap
    group_sessions: dict[str, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for gid in set(s.group_ids):
            group_sessions[gid].append(mi)

    cte_count = 0
    for gid, midxs in group_sessions.items():
        if len(midxs) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in midxs])
            cte_count += 1

    # ── NoInstructorDoubleBooking ──
    # Group all OPTIONAL instructor intervals by instructor → NoOverlap
    fte_count = 0
    if not relax_fte:
        instr_to_opts: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)
        for (mi, iidx), opt_iv in instr_opt_intervals.items():
            instr_to_opts[iidx].append(opt_iv)

        for iidx, opt_ivs in instr_to_opts.items():
            if len(opt_ivs) > 1:
                model.add_no_overlap(opt_ivs)
                fte_count += 1

    # ── NoRoomDoubleBooking ──
    # Group all OPTIONAL room intervals by room → NoOverlap
    sre_count = 0
    if not relax_sre:
        room_to_opts: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)
        for (mi, ridx), opt_iv in room_opt_intervals.items():
            room_to_opts[ridx].append(opt_iv)

        for ridx, opt_ivs in room_to_opts.items():
            if len(opt_ivs) > 1:
                model.add_no_overlap(opt_ivs)
                sre_count += 1

    # ── InstructorMustBeQualified + RoomMustHaveFeatures ──
    # Already enforced: instructor booleans only exist for qualified instructors,
    # room booleans only exist for compatible rooms.

    # ── InstructorMustBeAvailable — INTENTIONALLY DROPPED (Phase 1) ──

    # ── ExactWeeklyHours — structurally guaranteed by session generation ──

    # ── SpreadAcrossDays ──
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        sibling_groups[s.sibling_key].append(mi)

    ictd_count = 0
    ictd_infeasible = 0
    if not relax_ictd:
        for key, siblings in sibling_groups.items():
            if len(siblings) <= 1:
                continue
            if len(siblings) > num_days:
                ictd_infeasible += 1
            model.add_all_different([day_vars[mi] for mi in siblings])
            ictd_count += 1

        if ictd_infeasible:
            print(
                f"\n⚠  {ictd_infeasible} offerings have more siblings than "
                f"days ({num_days}) — SpreadAcrossDays forces infeasibility for those!"
            )

    # ── Symmetry breaking: sibling ordering ──
    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        for j in range(len(siblings) - 1):
            model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # ── Summary ──
    n_room_bools = len(room_bool)
    n_instr_bools = len(instr_bool)
    n_opt_intervals = len(room_opt_intervals) + len(instr_opt_intervals)
    print(f"\n  Variable counts:")
    print(f"    Mandatory intervals: {n_modeled}")
    print(f"    Room booleans:       {n_room_bools}")
    print(f"    Instructor booleans: {n_instr_bools}")
    print(f"    Optional intervals:  {n_opt_intervals}")

    print(f"\n  Constraints applied:")
    print(f"    NoStudentDoubleBooking:    {cte_count} groups")
    print(f"    NoInstructorDoubleBooking: {fte_count} instructors")
    print(f"    NoRoomDoubleBooking:       {sre_count} rooms")
    print(f"    InstructorMustBeQualified: via boolean domain")
    print(f"    RoomMustHaveFeatures:      via boolean domain")
    print(f"    InstructorMustBeAvailable: DROPPED (Phase 1)")
    print(f"    ExactWeeklyHours:          structural")
    print(f"    SpreadAcrossDays:          {ictd_count} sibling groups")
    print(f"    RequiresTwoInstructors:    {pmi_count} sessions")

    vars_dict = {
        "start": start_vars,
        "end": end_vars,
        "day": day_vars,
        "interval": interval_vars,
        "room_bool": room_bool,
        "instr_bool": instr_bool,
        "model_indices": model_indices,
        "impossible_sessions": impossible_sessions,
        "day_names": day_names,
    }
    return model, vars_dict


# ──────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────


def run_diagnostics(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> None:
    """Pre-solve diagnostics to surface obvious issues."""
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
    print("  PHASE 1 DIAGNOSTICS (No Availability)")
    print("=" * 72)
    print(f"  Total sessions:     {len(sessions)}")
    print(f"  Total quanta/week:  {total_quanta}")
    print(f"  Operational days:   {num_days}")
    print(f"  Quanta/day:         {day_lengths}")
    print(f"  Instructors:        {len(instructor_ids)}")
    print(f"  Rooms:              {len(room_ids)}")

    # Theory vs practical breakdown
    theory = [s for s in sessions if s.course_type == "theory"]
    prac = [s for s in sessions if s.course_type == "practical"]
    print(f"\n  Session breakdown:")
    print(f"    Theory:    {len(theory)} sessions, {sum(s.duration for s in theory)} total quanta")
    print(f"    Practical: {len(prac)} sessions, {sum(s.duration for s in prac)} total quanta")

    # Group load analysis
    group_load: dict[str, int] = defaultdict(int)
    for s in sessions:
        for gid in s.group_ids:
            group_load[gid] += s.duration

    overloaded = [(gid, load) for gid, load in group_load.items() if load > total_quanta]
    high_util = [(gid, load) for gid, load in group_load.items()
                 if load > total_quanta * 0.8 and load <= total_quanta]

    if overloaded:
        print(f"\n  CRITICAL: {len(overloaded)} groups OVERLOADED (> {total_quanta}q):")
        for gid, load in sorted(overloaded, key=lambda x: -x[1]):
            print(f"    {gid:10s}  {load}/{total_quanta} ({load/total_quanta*100:.0f}%)")
    if high_util:
        print(f"\n  High utilization (>80%):")
        for gid, load in sorted(high_util, key=lambda x: -x[1]):
            print(f"    {gid:10s}  {load}/{total_quanta} ({load/total_quanta*100:.0f}%)")

    # Sibling analysis (SpreadAcrossDays pressure)
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(sessions):
        sibling_groups[s.sibling_key].append(i)

    max_siblings = max((len(v) for v in sibling_groups.values()), default=0)
    over_day = sum(1 for v in sibling_groups.values() if len(v) > num_days)
    print(f"\n  Sibling analysis (SpreadAcrossDays):")
    print(f"    Max siblings per offering: {max_siblings}")
    print(f"    Offerings > {num_days} siblings: {over_day}")

    # Sessions without qualified instructors or compatible rooms
    no_inst = sum(1 for s in sessions if not s.qualified_instructor_idxs)
    no_room = sum(1 for s in sessions if not s.compatible_room_idxs)
    if no_inst:
        print(f"\n  WARNING: {no_inst} sessions have NO qualified instructor")
    if no_room:
        print(f"\n  WARNING: {no_room} sessions have NO compatible room")

    # Instructor demand (how many sessions each instructor could teach)
    print(f"\n  Instructor demand (sessions eligible to teach):")
    inst_eligible: dict[int, int] = defaultdict(int)
    for s in sessions:
        for iidx in s.qualified_instructor_idxs:
            inst_eligible[iidx] += 1
    top_demand = sorted(inst_eligible.items(), key=lambda x: -x[1])[:5]
    for iidx, count in top_demand:
        iid = instructor_ids[iidx]
        name = store.instructors[iid].name
        print(f"    {iid:6s} {name:30s}  eligible for {count} sessions")

    print("=" * 72)


# ──────────────────────────────────────────────────────────────────
# Solver + Reporter
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
    """Solve the model and print results.

    Returns True (feasible), False (infeasible), or None (timeout/unknown).
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True

    n_sessions = len(vars_dict["start"])
    print("\n" + "=" * 72)
    print("  CP-SAT PHASE 1: NO-AVAILABILITY CORE")
    print("=" * 72)
    print(f"  Sessions (modeled): {n_sessions}")
    print(f"  Time limit:         {time_limit}s")
    print(f"  Workers:            {solver.parameters.num_workers}")
    print(f"  InstructorMustBeAvailable: DROPPED")
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
        day_names = vars_dict["day_names"]

        if impossible:
            print(
                f"\n  FEASIBLE (excluding {len(impossible)} impossible sessions)\n"
            )
        else:
            print("\n  FEASIBLE SCHEDULE FOUND!\n")

        # ── Usage statistics ──
        inst_usage: dict[str, int] = defaultdict(int)
        room_usage: dict[str, int] = defaultdict(int)
        day_usage: dict[int, int] = defaultdict(int)

        room_bool = vars_dict["room_bool"]
        instr_bool = vars_dict["instr_bool"]

        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            d = solver.value(vars_dict["day"][mi])
            day_usage[d] += s.duration

            # Find assigned instructor(s) from booleans
            for iidx in s.qualified_instructor_idxs:
                if solver.value(instr_bool[(mi, iidx)]):
                    inst_usage[instructor_ids[iidx]] += s.duration

            # Find assigned room from booleans
            r_idxs = s.compatible_room_idxs
            for ridx in r_idxs:
                if (mi, ridx) in room_bool and solver.value(room_bool[(mi, ridx)]):
                    room_usage[room_ids[ridx]] += s.duration
                    break

        print(f"  Instructor utilization (top 10):")
        for iid, load in sorted(inst_usage.items(), key=lambda x: -x[1])[:10]:
            name = store.instructors[iid].name
            print(f"    {iid:6s} {name:30s}  {load:3d} quanta")

        print(f"\n  Room utilization (top 10):")
        for rid, load in sorted(room_usage.items(), key=lambda x: -x[1])[:10]:
            print(f"    {rid:10s}  {load:3d} quanta")

        print(f"\n  Sessions per day:")
        for d_idx in sorted(day_usage):
            dname = day_names[d_idx] if d_idx < len(day_names) else f"Day{d_idx}"
            print(f"    {dname:12s}  {day_usage[d_idx]:3d} quanta")

        # ── Export solution ──
        if export_path:
            _export_solution(
                solver, sessions, store, instructor_ids, room_ids, vars_dict, export_path
            )

        return True

    elif status == cp_model.INFEASIBLE:
        print("\n  NO FEASIBLE SCHEDULE EXISTS with Phase 1 constraints!\n")
        print("  Even without availability constraints, the problem is infeasible.")
        print("  This means the bottleneck is in the CORE constraints:")
        print("    - Too many sessions for available rooms (NoRoomDoubleBooking)")
        print("    - Too many sessions for available days (SpreadAcrossDays)")
        print("    - Too few qualified instructors (InstructorMustBeQualified)")
        print("    - Group overload (NoStudentDoubleBooking pigeonhole)")
        print()
        print("  Try relaxing constraints to find the bottleneck:")
        print("    python cpsat_phase1.py --relax-ictd")
        print("    python cpsat_phase1.py --relax-sre")
        print("    python cpsat_phase1.py --relax-ictd --relax-sre")
        return False

    else:
        print("\n  Could not determine feasibility within time limit.")
        print(f"  Try: python cpsat_phase1.py --time-limit {time_limit * 3}")
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
    room_bool = vars_dict["room_bool"]
    instr_bool = vars_dict["instr_bool"]
    schedule: list[dict] = []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        start_q = solver.value(vars_dict["start"][mi])

        # Resolve assigned instructor(s) from booleans
        assigned_instructors = []
        for iidx in s.qualified_instructor_idxs:
            if solver.value(instr_bool[(mi, iidx)]):
                assigned_instructors.append(instructor_ids[iidx])

        # Resolve assigned room from booleans
        assigned_room = None
        for ridx in s.compatible_room_idxs:
            if (mi, ridx) in room_bool and solver.value(room_bool[(mi, ridx)]):
                assigned_room = room_ids[ridx]
                break

        day_str, time_str = qts.quanta_to_time(start_q)

        entry = {
            "session_index": orig_i,
            "course_id": s.course_id,
            "course_type": s.course_type,
            "group_ids": s.group_ids,
            "instructor_id": assigned_instructors[0] if assigned_instructors else None,
            "instructor_name": (
                store.instructors[assigned_instructors[0]].name
                if assigned_instructors else None
            ),
            "room_id": assigned_room,
            "start_quanta": start_q,
            "duration": s.duration,
            "day": day_str,
            "time": time_str,
        }
        # For practicals with 2 instructors (RequiresTwoInstructors), include co-instructor
        if len(assigned_instructors) > 1:
            entry["co_instructor_ids"] = assigned_instructors[1:]
            entry["co_instructor_names"] = [
                store.instructors[iid].name for iid in assigned_instructors[1:]
            ]

        schedule.append(entry)

    with open(path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"\n  Solution exported to: {path}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CP-SAT Phase 1: No-Availability Core Solver"
    )
    parser.add_argument(
        "--time-limit", type=int, default=120,
        help="Solver time limit in seconds (default: 120)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--relax-ictd", action="store_true",
        help="Relax SpreadAcrossDays — allow sibling sessions on same day",
    )
    parser.add_argument(
        "--relax-sre", action="store_true",
        help="Relax NoRoomDoubleBooking — allow room double-booking",
    )
    parser.add_argument(
        "--relax-fte", action="store_true",
        help="Relax NoInstructorDoubleBooking — allow instructor double-booking",
    )
    parser.add_argument(
        "--relax-ffc", action="store_true",
        help="Relax RoomMustHaveFeatures — any room can host any session",
    )
    parser.add_argument(
        "--relax-pmi", action="store_true",
        help="Relax RequiresTwoInstructors — allow 1 instructor for practicals",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export solution to JSON file",
    )
    parser.add_argument(
        "--no-diag", action="store_true",
        help="Skip pre-solve diagnostics",
    )
    args = parser.parse_args()

    # ── Load data ──
    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    # ── Build sessions ──
    print("\nBuilding sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    # ── Diagnostics ──
    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    # ── Build model ──
    label_parts = ["Phase 1 (InstructorMustBeAvailable dropped)"]
    if args.relax_ictd:
        label_parts.append("SpreadAcrossDays relaxed")
    if args.relax_sre:
        label_parts.append("NoRoomDoubleBooking relaxed")
    if args.relax_fte:
        label_parts.append("NoInstructorDoubleBooking relaxed")
    if args.relax_ffc:
        label_parts.append("RoomMustHaveFeatures relaxed")
    if args.relax_pmi:
        label_parts.append("RequiresTwoInstructors relaxed")
    print(f"\nBuilding CP-SAT model ({', '.join(label_parts)})...")

    model, vars_dict = build_phase1_model(
        sessions, store, instructor_ids, room_ids,
        relax_ictd=args.relax_ictd,
        relax_sre=args.relax_sre,
        relax_fte=args.relax_fte,
        relax_ffc=args.relax_ffc,
        relax_pmi=args.relax_pmi,
    )

    proto = model.proto
    print(f"\n  Model size:")
    print(f"    Variables:   {len(proto.variables)}")
    print(f"    Constraints: {len(proto.constraints)}")

    # ── Solve ──
    result = solve_and_report(
        model, sessions, store, instructor_ids, room_ids,
        vars_dict, time_limit=args.time_limit, export_path=args.export,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
