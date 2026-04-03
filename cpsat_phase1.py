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


def cross_qualify_practicals(store: DataStore) -> dict[tuple[str, str], set[str]]:
    """For each practical course, find the theory variant and merge instructor pools.

    Returns dict mapping course_key → additional instructor IDs to add.
    This fixes the PMI bottleneck: practicals with only 1-2 qualified instructors
    gain the theory instructors for the same course code.
    """
    additions: dict[tuple[str, str], set[str]] = {}
    # Group courses by course_code
    code_to_keys: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for key, course in store.courses.items():
        code, ctype = key
        code_to_keys[code][ctype] = key

    for code, type_map in code_to_keys.items():
        if "practical" not in type_map or "theory" not in type_map:
            continue
        prac_key = type_map["practical"]
        theory_key = type_map["theory"]
        prac_course = store.courses[prac_key]
        theory_course = store.courses[theory_key]
        prac_insts = set(prac_course.qualified_instructor_ids)
        theory_insts = set(theory_course.qualified_instructor_ids)
        new_insts = theory_insts - prac_insts
        if new_insts:
            additions[prac_key] = new_insts

    return additions


def build_sessions(
    store: DataStore,
    *,
    cross_qualify: bool = False,
) -> tuple[list[Session], list[str], list[str]]:
    """Generate all scheduling sessions from loaded data.

    Each (course, group) pair is split into subsessions based on L/T/P rules:
      - Theory → 2-quanta blocks (with 1-quanta remainder if odd)
      - Practical → single monolithic block

    If cross_qualify=True, theory instructors are added to practical pools
    for the same course code (fixes PMI bottleneck).

    ExactWeeklyHours is guaranteed structurally: the sum of
    subsession durations == course.quanta_per_week for every (course, group).

    Returns (sessions, instructor_ids_list, room_ids_list).
    """
    # Cross-qualification: merge theory → practical instructor pools
    xq_additions: dict[tuple[str, str], set[str]] = {}
    if cross_qualify:
        xq_additions = cross_qualify_practicals(store)
        if xq_additions:
            total_new = sum(len(v) for v in xq_additions.values())
            print(f"  Cross-qualification: {len(xq_additions)} practicals gain {total_new} instructors from theory")
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
        base_ids = set(course.qualified_instructor_ids)
        # Add cross-qualified theory instructors for practicals
        if cross_qualify and course_key in xq_additions:
            base_ids |= xq_additions[course_key]
        q_inst = sorted(
            set(
                inst_to_idx[iid]
                for iid in base_ids
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
    no_rooms: bool = False,
    room_pool_limit: int = 0,
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

    When no_rooms=True, room modeling behavior depends on room_pool_limit:
      room_pool_limit=0  → skip ALL room variables (pure Phase A)
      room_pool_limit=N  → model rooms for pools with ≤ N rooms (hybrid)
    Remaining large-pool rooms are assigned in Phase B.

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
    # When no_rooms + room_pool_limit > 0: use soft overlap penalties instead
    # When no_rooms + room_pool_limit = 0: skip all room handling
    room_bool: dict[tuple[int, int], cp_model.IntVar] = {}
    room_modeled_mis: set[int] = set()

    # Precompute pool mapping for all sessions
    mi_pool: dict[int, tuple[int, ...]] = {}
    pool_to_mis: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        pk = tuple(sorted(s.compatible_room_idxs))
        mi_pool[mi] = pk
        pool_to_mis[pk].append(mi)

    if not no_rooms:
        # Full monolithic: all sessions get room booleans
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            r_idxs = list(range(len(room_ids))) if relax_ffc else s.compatible_room_idxs
            session_room_bools = []
            for ridx in r_idxs:
                b = model.new_bool_var(f"room_{mi}_{ridx}")
                room_bool[(mi, ridx)] = b
                session_room_bools.append(b)
            model.add_exactly_one(session_room_bools)
            room_modeled_mis.add(mi)

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
    # Created for all sessions that have room booleans (full mode or small pools)
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
    # Active for sessions with room modeling (full mode or small-pool hybrid)
    sre_count = 0
    if not relax_sre and room_opt_intervals:
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

    # ── Room Capacity Awareness ──
    # Deferred to Phase A' (cpsat_solver.py): after Phase A solves fast,
    # pin day assignments and add cumulative constraints for small pools.
    # Adding cumulative directly here makes the model intractable when
    # combined with ICTD (SpreadAcrossDays).

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

    # ── MaxLoadPermitted — cap instructor weekly teaching hours ──
    # For each instructor with maxLoad data, enforce:
    #   sum(duration * instr_bool) for lecture sessions ≤ max_load_lecture
    #   sum(duration * instr_bool) for practical sessions ≤ max_load_practical
    mlp_count = 0
    instr_to_mi_by_type: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for iidx in s.qualified_instructor_idxs:
            if (mi, iidx) in instr_bool:
                instr_to_mi_by_type[iidx][s.course_type].append(mi)

    for iidx, type_mis in instr_to_mi_by_type.items():
        iid = instructor_ids[iidx]
        inst = store.instructors.get(iid)
        if inst is None:
            continue

        # Lecture (theory) cap
        if inst.max_load_lecture is not None and "theory" in type_mis:
            theory_mis = type_mis["theory"]
            load_expr = sum(
                sessions[model_indices[mi]].duration * instr_bool[(mi, iidx)]
                for mi in theory_mis
            )
            model.add(load_expr <= inst.max_load_lecture)
            mlp_count += 1

        # Practical cap
        if inst.max_load_practical is not None and "practical" in type_mis:
            prac_mis = type_mis["practical"]
            load_expr = sum(
                sessions[model_indices[mi]].duration * instr_bool[(mi, iidx)]
                for mi in prac_mis
            )
            model.add(load_expr <= inst.max_load_practical)
            mlp_count += 1

    # ── Pool capacity for small room pools (in no_rooms mode) ──
    # When rooms are deferred, Phase A is blind to room pools. Add:
    # - NoOverlap for pools with 1 room (very cheap, like group NoOverlap)
    # - Daily session-count limits for pools with 2-5 rooms (linear bounds)
    # Note: Cumulative for pools with 2+ rooms causes TIMEOUT in full model.
    pool_cap_count = 0
    pool_daily_count = 0
    if no_rooms:
        for pool_key, mis in pool_to_mis.items():
            pool_size = len(pool_key)
            if len(mis) <= pool_size:
                continue

            if pool_size == 1:
                model.add_no_overlap([interval_vars[mi] for mi in mis])
                pool_cap_count += 1
            elif pool_size <= 5:
                # Daily session-count limits by duration group.
                # For duration D in a pool of C rooms with Q quanta/day:
                #   max sessions per day = floor(Q / D) * C
                dur_groups: dict[int, list[int]] = defaultdict(list)
                for mi in mis:
                    dur_groups[sessions[model_indices[mi]].duration].append(mi)

                for dur, dur_mis in dur_groups.items():
                    max_per_day = (quanta_per_day // dur) * pool_size + 1  # +1 slack
                    if len(dur_mis) <= max_per_day:
                        continue  # Can't exceed limit even if all on same day
                    for d in range(num_days):
                        on_day = [
                            model.new_bool_var(f"pd_{pool_key}_{mi}_{d}")
                            for mi in dur_mis
                        ]
                        for j, mi in enumerate(dur_mis):
                            model.add(day_vars[mi] == d).only_enforce_if(on_day[j])
                            model.add(day_vars[mi] != d).only_enforce_if(on_day[j].negated())
                        model.add(sum(on_day) <= max_per_day)
                        pool_daily_count += 1

    # ── Summary ──
    n_room_bools = len(room_bool)
    n_instr_bools = len(instr_bool)
    n_opt_intervals = len(room_opt_intervals) + len(instr_opt_intervals)
    print(f"\n  Variable counts:")
    print(f"    Mandatory intervals: {n_modeled}")
    if no_rooms and room_pool_limit > 0:
        print(f"    Room booleans:       {n_room_bools} (pools ≤ {room_pool_limit}: {len(room_modeled_mis)} sessions)")
    elif no_rooms:
        print(f"    Room booleans:       {n_room_bools} (SKIPPED — Phase A)")
    else:
        print(f"    Room booleans:       {n_room_bools}")
    print(f"    Instructor booleans: {n_instr_bools}")
    print(f"    Optional intervals:  {n_opt_intervals}")

    print(f"\n  Constraints applied:")
    print(f"    NoStudentDoubleBooking:    {cte_count} groups")
    print(f"    NoInstructorDoubleBooking: {fte_count} instructors")
    if no_rooms:
        print(f"    NoRoomDoubleBooking:       deferred to Phase A' + Phase B")
    else:
        print(f"    NoRoomDoubleBooking:       {sre_count} rooms")
    print(f"    InstructorMustBeQualified: via boolean domain")
    print(f"    RoomMustHaveFeatures:      {'deferred to Phase B' if no_rooms else 'via boolean domain'}")
    print(f"    InstructorMustBeAvailable: DROPPED (Phase 1)")
    print(f"    ExactWeeklyHours:          structural")
    print(f"    SpreadAcrossDays:          {ictd_count} sibling groups")
    print(f"    RequiresTwoInstructors:    {pmi_count} sessions")
    print(f"    MaxLoadPermitted:          {mlp_count} constraints")
    if no_rooms and pool_cap_count:
        print(f"    PoolCapacity (1-room):     {pool_cap_count} pools")
    if no_rooms and pool_daily_count:
        print(f"    PoolDailyLimits (2-5):     {pool_daily_count} constraints")

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
        "room_modeled_mis": room_modeled_mis,
        "pool_to_mis": dict(pool_to_mis),
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
    random_seed: int | None = None,
) -> tuple[bool | None, cp_model.CpSolver | None]:
    """Solve the model and print results.

    Returns (success, solver) where success is True/False/None and solver
    is the CpSolver with solution values (if feasible).
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False
    if random_seed is not None:
        solver.parameters.random_seed = random_seed

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

        return True, solver

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
        return False, None

    else:
        print("\n  Could not determine feasibility within time limit.")
        print(f"  Try: python cpsat_phase1.py --time-limit {time_limit * 3}")
        return None, None


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
# Phase B: Greedy Room Assignment with Stealing (post Phase A)
# ──────────────────────────────────────────────────────────────────


def add_warm_start_hints(
    model: cp_model.CpModel,
    vars_dict: dict,
    phase_a_solver: cp_model.CpSolver,
    phase_a_vars: dict,
    assignments: list[dict] | None = None,
) -> int:
    """Add solution hints from a solved Phase A model to a hybrid model.

    Hints time + instructor variables from Phase A, and room booleans
    from greedy assignments (if available). Returns number of hints added.
    """
    n_hints = 0
    model_indices = vars_dict["model_indices"]

    # Hint time variables (start, end, day)
    for mi in range(len(model_indices)):
        start_val = phase_a_solver.value(phase_a_vars["start"][mi])
        model.AddHint(vars_dict["start"][mi], start_val)
        model.AddHint(vars_dict["end"][mi], start_val + phase_a_solver.value(phase_a_vars["end"][mi]) - phase_a_solver.value(phase_a_vars["start"][mi]))
        model.AddHint(vars_dict["day"][mi], phase_a_solver.value(phase_a_vars["day"][mi]))
        n_hints += 3

    # Hint instructor booleans
    for (mi, iidx), var in vars_dict["instr_bool"].items():
        if (mi, iidx) in phase_a_vars["instr_bool"]:
            val = phase_a_solver.value(phase_a_vars["instr_bool"][(mi, iidx)])
            model.AddHint(var, val)
            n_hints += 1

    # Hint room booleans from greedy assignments
    if assignments and vars_dict["room_bool"]:
        for (mi, ridx), var in vars_dict["room_bool"].items():
            a = assignments[mi]
            model.AddHint(var, 1 if a.get("room") == ridx else 0)
            n_hints += 1

    return n_hints


def extract_assignments(
    solver: cp_model.CpSolver,
    sessions: list[Session],
    instructor_ids: list[str],
    vars_dict: dict,
) -> list[dict]:
    """Extract time+instructor assignments from a solved Phase A model."""
    model_indices = vars_dict["model_indices"]
    instr_bool = vars_dict["instr_bool"]
    room_bool = vars_dict["room_bool"]
    room_modeled_mis = vars_dict.get("room_modeled_mis", set())

    assignments: list[dict] = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        start_q = solver.value(vars_dict["start"][mi])

        assigned_instructors = []
        for iidx in s.qualified_instructor_idxs:
            if solver.value(instr_bool[(mi, iidx)]):
                assigned_instructors.append(instructor_ids[iidx])

        phase_a_room = None
        if mi in room_modeled_mis:
            for ridx in s.compatible_room_idxs:
                if (mi, ridx) in room_bool and solver.value(room_bool[(mi, ridx)]):
                    phase_a_room = ridx
                    break

        assignments.append({
            "mi": mi,
            "orig_i": orig_i,
            "start": start_q,
            "duration": s.duration,
            "end": start_q + s.duration,
            "instructors": assigned_instructors,
            "compatible_rooms": list(s.compatible_room_idxs),
            "phase_a_room": phase_a_room,
            "room": phase_a_room if phase_a_room is not None else -1,
        })

    return assignments


def phase_b_room_assignment(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    room_ids: list[str],
) -> tuple[int, list[tuple[int, int]]]:
    """Phase B: Assign rooms using greedy + room-stealing (depth-2 chains).

    Strategy (from schedule_threephase.py):
    1. Sort by pool size ascending (hardest-to-assign first)
    2. Try direct assignment to a free compatible room
    3. Try stealing: move a blocker to an alternative room
    4. Try chain stealing (depth 2): move blocker's blocker

    Returns (n_failed, conflict_pairs).
    """
    print(f"\n{'=' * 72}")
    print("  PHASE B: Greedy Room Assignment with Stealing")
    print(f"{'=' * 72}")

    n = len(assignments)

    # Track room → list of (start, end, ai) assignments
    room_schedule: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    # Pre-populate rooms assigned in Phase A
    for ai, a in enumerate(assignments):
        if a["room"] >= 0:
            room_schedule[a["room"]].append((a["start"], a["end"], ai))

    def is_room_free(ridx: int, start: int, end: int, exclude_ai: int = -1) -> bool:
        for (rs, re, ai2) in room_schedule[ridx]:
            if ai2 == exclude_ai:
                continue
            if start < re and rs < end:
                return False
        return True

    def assign_room(ai: int, ridx: int) -> None:
        a = assignments[ai]
        a["room"] = ridx
        room_schedule[ridx].append((a["start"], a["end"], ai))

    def unassign_room(ai: int) -> None:
        a = assignments[ai]
        ridx = a["room"]
        room_schedule[ridx] = [(rs, re, ai2) for (rs, re, ai2) in room_schedule[ridx] if ai2 != ai]
        a["room"] = -1

    # Sort: pool size ascending, then start time (hardest first)
    needs_room = [ai for ai in range(n) if assignments[ai]["room"] < 0]
    order = sorted(needs_room, key=lambda ai: (
        len(assignments[ai]["compatible_rooms"]),
        assignments[ai]["start"],
    ))

    n_steals = 0
    failed: list[int] = []

    for ai in order:
        a = assignments[ai]
        start, end = a["start"], a["end"]

        # 1. Direct assignment
        assigned = False
        for ridx in sorted(a["compatible_rooms"]):
            if is_room_free(ridx, start, end):
                assign_room(ai, ridx)
                assigned = True
                break
        if assigned:
            continue

        # 2. Steal: move a blocker to a different room
        stolen = False
        for ridx in sorted(a["compatible_rooms"]):
            blockers = [(rs, re, ai2) for (rs, re, ai2) in room_schedule[ridx]
                        if start < re and rs < end]
            for (_, _, ai_blocker) in blockers:
                b_rooms = assignments[ai_blocker]["compatible_rooms"]
                b_start = assignments[ai_blocker]["start"]
                b_end = assignments[ai_blocker]["end"]
                for alt_ridx in sorted(b_rooms):
                    if alt_ridx == ridx:
                        continue
                    if is_room_free(alt_ridx, b_start, b_end):
                        unassign_room(ai_blocker)
                        assign_room(ai_blocker, alt_ridx)
                        # Check room is actually free now (may have other blockers)
                        if is_room_free(ridx, start, end):
                            assign_room(ai, ridx)
                            stolen = True
                            n_steals += 1
                        else:
                            # Revert — other blockers remain
                            unassign_room(ai_blocker)
                            assign_room(ai_blocker, ridx)
                        break
                if stolen:
                    break
            if stolen:
                break
        if stolen:
            continue

        # 3. Chain steal (depth 2)
        chain_done = False
        for ridx in sorted(a["compatible_rooms"]):
            blockers = [(rs, re, ai2) for (rs, re, ai2) in room_schedule[ridx]
                        if start < re and rs < end]
            for (_, _, ai_b1) in blockers:
                b1_rooms = assignments[ai_b1]["compatible_rooms"]
                b1_start = assignments[ai_b1]["start"]
                b1_end = assignments[ai_b1]["end"]
                for alt_ridx in sorted(b1_rooms):
                    if alt_ridx == ridx:
                        continue
                    blockers2 = [(rs, re, ai3) for (rs, re, ai3) in room_schedule[alt_ridx]
                                 if b1_start < re and rs < b1_end]
                    for (_, _, ai_b2) in blockers2:
                        b2_rooms = assignments[ai_b2]["compatible_rooms"]
                        b2_start = assignments[ai_b2]["start"]
                        b2_end = assignments[ai_b2]["end"]
                        for alt2_ridx in sorted(b2_rooms):
                            if alt2_ridx == alt_ridx:
                                continue
                            if is_room_free(alt2_ridx, b2_start, b2_end):
                                unassign_room(ai_b2)
                                assign_room(ai_b2, alt2_ridx)
                                # Check alt_ridx is free for b1 after moving b2 out
                                if not is_room_free(alt_ridx, b1_start, b1_end):
                                    # Revert b2
                                    unassign_room(ai_b2)
                                    assign_room(ai_b2, alt_ridx)
                                    break
                                unassign_room(ai_b1)
                                assign_room(ai_b1, alt_ridx)
                                # Check ridx is free for ai after moving b1 out
                                if is_room_free(ridx, start, end):
                                    assign_room(ai, ridx)
                                    chain_done = True
                                    n_steals += 1
                                else:
                                    # Revert full chain
                                    unassign_room(ai_b1)
                                    assign_room(ai_b1, ridx)
                                    unassign_room(ai_b2)
                                    assign_room(ai_b2, alt_ridx)
                                break
                        if chain_done:
                            break
                    if chain_done:
                        break
                if chain_done:
                    break
            if chain_done:
                break

        if not chain_done:
            failed.append(ai)

    # Build conflict list
    conflicts: list[tuple[int, int]] = []
    if failed:
        for ai in failed:
            a = assignments[ai]
            start, end = a["start"], a["end"]
            pool = set(a["compatible_rooms"])
            pool_size = len(pool)
            for ai2 in range(n):
                if ai2 == ai or assignments[ai2]["room"] < 0:
                    continue
                if assignments[ai2]["room"] not in pool:
                    continue
                if assignments[ai2]["start"] < end and start < assignments[ai2]["end"]:
                    conflicts.append((ai, ai2))

    n_assigned = sum(1 for a in assignments if a["room"] >= 0)
    if not failed:
        print(f"\n  ✓ All {n} sessions assigned rooms! ({n_steals} steals)")
    else:
        print(f"\n  ✗ {len(failed)}/{n} sessions could not get a room "
              f"({n_assigned} assigned, {n_steals} steals)")
        for ai in failed[:10]:
            a = assignments[ai]
            s = sessions[a["orig_i"]]
            print(f"    S{a['orig_i']}: {s.course_id} ({s.course_type}) "
                  f"start={a['start']} dur={a['duration']} pool={len(a['compatible_rooms'])}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    return len(failed), failed


def phase_c_repair(
    assignments: list[dict],
    failed_ais: list[int],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 60,
) -> int:
    """Phase C: Re-schedule failed sessions + room-pool neighbors via CP-SAT.

    Frees failed sessions and their neighborhood (same room pool or
    time-overlapping on same pool). Builds a small CP-SAT model using
    table constraints (allowed start, instructor, room tuples).

    Returns number of sessions still without rooms (0 = success).
    Modifies assignments in-place.
    """
    print(f"\n{'=' * 72}")
    print("  PHASE C: CP-SAT Repair for Remaining Failures")
    print(f"{'=' * 72}")

    qts = store.qts
    total_quanta = qts.total_quanta

    # Compute valid start quanta per day
    day_offsets, day_lengths = [], []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)

    def compute_valid_starts(duration: int) -> list[int]:
        starts = []
        for off, length in zip(day_offsets, day_lengths):
            for s in range(off, off + length - duration + 1):
                starts.append(s)
        return starts

    # Expand neighborhood
    failed_set = set(failed_ais)
    neighbor_set: set[int] = set()

    # Pools of failed sessions
    failed_pools: set[frozenset[int]] = set()
    for ai in failed_ais:
        s = sessions[assignments[ai]["orig_i"]]
        failed_pools.add(frozenset(s.compatible_room_idxs))

    # Time-overlapping same-pool neighbors
    for ai in failed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        pool = set(s.compatible_room_idxs)
        start, end = a["start"], a["end"]
        for ai2 in range(len(assignments)):
            if ai2 in failed_set or ai2 in neighbor_set:
                continue
            a2 = assignments[ai2]
            if a2["room"] < 0:
                continue
            if a2["room"] not in pool:
                continue
            if a2["start"] < end and start < a2["end"]:
                neighbor_set.add(ai2)

    # Also free ALL sessions in same pool (even non-overlapping)
    pool_sibling_count = 0
    for ai2 in range(len(assignments)):
        if ai2 in failed_set or ai2 in neighbor_set:
            continue
        s2 = sessions[assignments[ai2]["orig_i"]]
        if frozenset(s2.compatible_room_idxs) in failed_pools:
            neighbor_set.add(ai2)
            pool_sibling_count += 1

    free_ais = sorted(failed_set | neighbor_set)
    fixed_ais = [ai for ai in range(len(assignments)) if ai not in set(free_ais)]
    F = len(free_ais)

    print(f"  {len(failed_ais)} failed + {len(neighbor_set)} neighbors "
          f"({pool_sibling_count} pool siblings) = {F} free sessions")

    # Build occupancy maps for fixed sessions
    group_at_q: dict[int, set[str]] = defaultdict(set)
    inst_at_q: dict[int, set[str]] = defaultdict(set)
    room_at_q: dict[int, set[int]] = defaultdict(set)

    for ai in fixed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        for q in range(a["start"], a["end"]):
            for gid in s.group_ids:
                group_at_q[q].add(gid)
            for iid in a["instructors"]:
                inst_at_q[q].add(iid)
            if a["room"] >= 0:
                room_at_q[q].add(a["room"])

    # Build repair model
    model = cp_model.CpModel()
    start_vars: list[cp_model.IntVar] = []
    inst_vars_r: list[cp_model.IntVar] = []
    room_vars_r: list[cp_model.IntVar] = []

    # Map instructor IDs to indices for the model
    iid_to_idx: dict[str, int] = {iid: idx for idx, iid in enumerate(instructor_ids)}

    for fi, ai in enumerate(free_ais):
        s = sessions[assignments[ai]["orig_i"]]
        base_starts = compute_valid_starts(s.duration)

        allowed: list[tuple[int, int, int]] = []
        for st in base_starts:
            quanta_range = range(st, st + s.duration)

            # Check group availability
            group_ok = True
            for q in quanta_range:
                for gid in s.group_ids:
                    if gid in group_at_q.get(q, set()):
                        group_ok = False
                        break
                if not group_ok:
                    break
            if not group_ok:
                continue

            for iidx in s.qualified_instructor_idxs:
                iid = instructor_ids[iidx]

                # Check instructor availability
                inst_ok = True
                for q in quanta_range:
                    if iid in inst_at_q.get(q, set()):
                        inst_ok = False
                        break
                if not inst_ok:
                    continue

                for ridx in s.compatible_room_idxs:
                    room_ok = True
                    for q in quanta_range:
                        if ridx in room_at_q.get(q, set()):
                            room_ok = False
                            break
                    if room_ok:
                        allowed.append((st, iidx, ridx))

        if not allowed:
            print(f"    No feasible (time, inst, room) for S{assignments[ai]['orig_i']} "
                  f"({s.course_id} dur={s.duration})")
            # Skip this session entirely — it can't be placed
            start_vars.append(None)
            inst_vars_r.append(None)
            room_vars_r.append(None)
            continue

        sv = model.new_int_var(0, total_quanta, f"rs_{fi}")
        iv = model.new_int_var(0, len(instructor_ids), f"ri_{fi}")
        rv = model.new_int_var(0, len(room_ids), f"rr_{fi}")
        model.add_allowed_assignments([sv, iv, rv], allowed)

        start_vars.append(sv)
        inst_vars_r.append(iv)
        room_vars_r.append(rv)

    # Mutual constraints among free sessions (skip infeasible ones)
    for i in range(F):
        if start_vars[i] is None:
            continue
        si = sessions[assignments[free_ais[i]]["orig_i"]]
        for j in range(i + 1, F):
            if start_vars[j] is None:
                continue
            sj = sessions[assignments[free_ais[j]]["orig_i"]]

            # Student group overlap
            if set(si.group_ids) & set(sj.group_ids):
                ivi = model.new_fixed_size_interval_var(
                    start_vars[i], si.duration, f"cte_{i}_{j}_i")
                ivj = model.new_fixed_size_interval_var(
                    start_vars[j], sj.duration, f"cte_{i}_{j}_j")
                model.add_no_overlap([ivi, ivj])

            # Instructor overlap
            b_same_inst = model.new_bool_var(f"si_{i}_{j}")
            model.add(inst_vars_r[i] == inst_vars_r[j]).only_enforce_if(b_same_inst)
            model.add(inst_vars_r[i] != inst_vars_r[j]).only_enforce_if(~b_same_inst)
            b_order = model.new_bool_var(f"so_{i}_{j}")
            model.add(
                start_vars[i] + si.duration <= start_vars[j]
            ).only_enforce_if(b_same_inst, b_order)
            model.add(
                start_vars[j] + sj.duration <= start_vars[i]
            ).only_enforce_if(b_same_inst, ~b_order)

            # Room overlap
            b_same_room = model.new_bool_var(f"sr_{i}_{j}")
            model.add(room_vars_r[i] == room_vars_r[j]).only_enforce_if(b_same_room)
            model.add(room_vars_r[i] != room_vars_r[j]).only_enforce_if(~b_same_room)
            b_order2 = model.new_bool_var(f"sro_{i}_{j}")
            model.add(
                start_vars[i] + si.duration <= start_vars[j]
            ).only_enforce_if(b_same_room, b_order2)
            model.add(
                start_vars[j] + sj.duration <= start_vars[i]
            ).only_enforce_if(b_same_room, ~b_order2)

    proto = model.proto
    print(f"  Repair model: {F} free sessions, "
          f"{len(proto.variables)} vars, {len(proto.constraints)} constraints")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"  Repair: {STATUS.get(status, '?')} in {elapsed:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("  ✗ Repair failed — could not find feasible rescheduling")
        return len(failed_ais)

    # Update assignments in-place
    for fi, ai in enumerate(free_ais):
        if start_vars[fi] is None:
            continue  # This session couldn't be placed
        s = sessions[assignments[ai]["orig_i"]]
        new_start = solver.value(start_vars[fi])
        new_iidx = solver.value(inst_vars_r[fi])
        new_ridx = solver.value(room_vars_r[fi])
        assignments[ai]["start"] = new_start
        assignments[ai]["end"] = new_start + s.duration
        assignments[ai]["instructors"] = [instructor_ids[new_iidx]]
        assignments[ai]["room"] = new_ridx

    still_failed = sum(1 for a in assignments if a["room"] < 0)
    if still_failed == 0:
        print(f"  ✓ Repair successful! All sessions now have rooms.")
    else:
        print(f"  ✗ {still_failed} sessions still without rooms after repair")

    return still_failed


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────


def build_schedule_json(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    room_ids: list[str],
) -> list[dict]:
    """Convert assignments (with rooms) into the standard schedule JSON."""
    qts = store.qts
    schedule: list[dict] = []
    for a in assignments:
        s = sessions[a["orig_i"]]
        rid = room_ids[a["room"]] if a["room"] >= 0 else None
        day_str, time_str = qts.quanta_to_time(a["start"])
        entry = {
            "session_index": a["orig_i"],
            "course_id": s.course_id,
            "course_type": s.course_type,
            "group_ids": s.group_ids,
            "instructor_id": a["instructors"][0] if a["instructors"] else None,
            "instructor_name": (
                store.instructors[a["instructors"][0]].name
                if a["instructors"] else None
            ),
            "room_id": rid,
            "start_quanta": a["start"],
            "duration": s.duration,
            "day": day_str,
            "time": time_str,
        }
        if len(a["instructors"]) > 1:
            entry["co_instructor_ids"] = a["instructors"][1:]
            entry["co_instructor_names"] = [
                store.instructors[iid].name for iid in a["instructors"][1:]
            ]
        schedule.append(entry)
    return schedule


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
        "--cross-qualify", action="store_true",
        help="Add theory instructors to practical pools (fix PMI bottleneck)",
    )
    parser.add_argument(
        "--no-rooms", action="store_true",
        help="Phase A mode: skip room variables entirely, assign rooms in Phase B",
    )
    parser.add_argument(
        "--room-pool-limit", type=int, default=0,
        help="When --no-rooms: model rooms for pools with ≤ N rooms (default: 0). "
             "Recommended: 5 (handles tight pools in Phase A, defers easy large pools to Phase B)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds to try (multi-start). Each seed produces a "
             "different Phase A solution (~4s each). Best room assignment wins.",
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
    sessions, instructor_ids, room_ids = build_sessions(
        store, cross_qualify=args.cross_qualify,
    )
    print(f"  {len(sessions)} sessions generated")

    # ── Diagnostics ──
    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    # ── Build label ──
    label_parts = ["Phase 1 (InstructorMustBeAvailable dropped)"]
    if args.cross_qualify:
        label_parts.append("cross-qualification ON")
    if args.no_rooms:
        if args.room_pool_limit > 0:
            label_parts.append(f"rooms for pools ≤ {args.room_pool_limit}")
        else:
            label_parts.append("Phase A: no rooms")
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

    # ── Non-decomposed mode (rooms in Phase A) ──
    if not args.no_rooms:
        print(f"\nBuilding CP-SAT model ({', '.join(label_parts)})...")
        model, vars_dict = build_phase1_model(
            sessions, store, instructor_ids, room_ids,
            relax_ictd=args.relax_ictd,
            relax_sre=args.relax_sre,
            relax_fte=args.relax_fte,
            relax_ffc=args.relax_ffc,
            relax_pmi=args.relax_pmi,
            no_rooms=False,
            room_pool_limit=args.room_pool_limit,
        )
        proto = model.proto
        print(f"\n  Model size:")
        print(f"    Variables:   {len(proto.variables)}")
        print(f"    Constraints: {len(proto.constraints)}")

        result, solver = solve_and_report(
            model, sessions, store, instructor_ids, room_ids,
            vars_dict, time_limit=args.time_limit, export_path=args.export,
        )
        sys.exit(0 if result else 1)

    # ── Decomposed mode: Phase A (time+inst) → Phase B (greedy rooms) ──
    n_seeds = max(1, args.seeds)
    best_failed = len(sessions) + 1
    best_assignments: list[dict] | None = None
    best_failed_ais: list[int] = []
    total_t0 = time.time()

    print(f"\n{'=' * 72}")
    if args.room_pool_limit > 0:
        print(f"  Warm-Start: Phase A (no rooms) → hint Phase A+ (rooms ≤ {args.room_pool_limit}) → greedy")
    else:
        print(f"  Multi-Start: {n_seeds} seed(s), Phase A + greedy rooms")
    print(f"{'=' * 72}")

    for seed_i in range(n_seeds):
        seed = seed_i + 1
        print(f"\n{'─' * 72}")
        print(f"  Seed {seed}/{n_seeds}")
        print(f"{'─' * 72}")

        # Step 1: Fast Phase A solve (no rooms, ~5s)
        model_a, vars_a = build_phase1_model(
            sessions, store, instructor_ids, room_ids,
            relax_ictd=args.relax_ictd,
            relax_sre=args.relax_sre,
            relax_fte=args.relax_fte,
            relax_ffc=args.relax_ffc,
            relax_pmi=args.relax_pmi,
            no_rooms=True,
            room_pool_limit=0,  # Pure no-rooms for fast solve
        )

        if seed_i == 0:
            proto = model_a.proto
            print(f"  Phase A model: {len(proto.variables)} vars, {len(proto.constraints)} constraints")

        # Quick solve: 10s should be plenty for the no-rooms model
        phase_a_limit = min(15, args.time_limit)
        result, solver_a = solve_and_report(
            model_a, sessions, store, instructor_ids, room_ids,
            vars_a, time_limit=phase_a_limit,
            random_seed=seed,
        )

        if not result or solver_a is None:
            print(f"  Seed {seed}: Phase A failed")
            continue

        # Extract Phase A solution and run greedy rooms
        assignments = extract_assignments(solver_a, sessions, instructor_ids, vars_a)
        n_failed, failed_ais = phase_b_room_assignment(
            assignments, sessions, store, room_ids,
        )

        # Step 2: If room_pool_limit > 0 and still failures, warm-start
        if args.room_pool_limit > 0 and n_failed > 0:
            remaining_time = max(10, args.time_limit - phase_a_limit)
            print(f"\n  Phase A+: Warm-starting hybrid model (rooms ≤ {args.room_pool_limit}), {remaining_time}s...")

            model_b, vars_b = build_phase1_model(
                sessions, store, instructor_ids, room_ids,
                relax_ictd=args.relax_ictd,
                relax_sre=args.relax_sre,
                relax_fte=args.relax_fte,
                relax_ffc=args.relax_ffc,
                relax_pmi=args.relax_pmi,
                no_rooms=True,
                room_pool_limit=args.room_pool_limit,
            )

            n_hints = add_warm_start_hints(
                model_b, vars_b, solver_a, vars_a, assignments
            )
            print(f"  Added {n_hints} solution hints from Phase A + greedy rooms")

            result_b, solver_b = solve_and_report(
                model_b, sessions, store, instructor_ids, room_ids,
                vars_b, time_limit=remaining_time,
                random_seed=seed,
            )

            if result_b and solver_b is not None:
                assignments = extract_assignments(solver_b, sessions, instructor_ids, vars_b)
                n_failed, failed_ais = phase_b_room_assignment(
                    assignments, sessions, store, room_ids,
                )

        print(f"  Seed {seed} result: {n_failed} failed room assignments")

        if n_failed < best_failed:
            best_failed = n_failed
            best_assignments = [a.copy() for a in assignments]
            best_failed_ais = list(failed_ais)

        if n_failed == 0:
            print(f"\n  ✓ Perfect solution found at seed {seed}!")
            break

    total_elapsed = time.time() - total_t0

    # ── Phase C: Repair remaining failures ──
    if best_assignments is not None and best_failed > 0:
        print(f"\n  Best seed had {best_failed} room failures. Running Phase C repair...")
        still_failed = phase_c_repair(
            best_assignments, best_failed_ais,
            sessions, store, instructor_ids, room_ids,
            time_limit=min(60, args.time_limit),
        )
        best_failed = still_failed
        total_elapsed = time.time() - total_t0

    # ── Final report ──
    print(f"\n{'=' * 72}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 72}")
    print(f"  Seeds tried:          {min(seed_i + 1, n_seeds)}")
    print(f"  Total time:           {total_elapsed:.1f}s")
    print(f"  Best room failures:   {best_failed}")

    if best_assignments is not None and best_failed == 0:
        schedule = build_schedule_json(best_assignments, sessions, store, room_ids)

        # Room utilization stats
        room_usage: dict[str, int] = defaultdict(int)
        for a in best_assignments:
            if a["room"] >= 0:
                room_usage[room_ids[a["room"]]] += a["duration"]

        print(f"  Sessions scheduled:   {len(schedule)}")
        print(f"  Rooms used:           {len(room_usage)}/{len(room_ids)}")
        print(f"\n  Room utilization (top 10):")
        for rid, load in sorted(room_usage.items(), key=lambda x: -x[1])[:10]:
            print(f"    {rid:10s}  {load:3d} quanta")

        if args.export:
            with open(args.export, "w") as f:
                json.dump(schedule, f, indent=2)
            print(f"\n  Schedule exported to: {args.export}")

        sys.exit(0)
    else:
        if best_assignments is not None:
            print(f"\n  Best attempt: {best_failed} sessions without rooms")
            for a in best_assignments:
                if a["room"] < 0:
                    s = sessions[a["orig_i"]]
                    print(f"    S{a['orig_i']}: {s.course_id} ({s.course_type}) "
                          f"pool={len(a['compatible_rooms'])}")
        else:
            print("\n  No feasible Phase A solution found!")

        sys.exit(1)


if __name__ == "__main__":
    main()
