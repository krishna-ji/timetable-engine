#!/usr/bin/env python3
"""Three-phase CP-SAT scheduler.

Phase 0: CTE+FTE+FPC+ICTD (relaxing FCA+SRE) → fast time assignment (~seconds)
Phase 1: Fix times, assign instructors (FCA+FTE matching)
Phase 2: Fix times, assign rooms (SRE matching)

This decomposition works because:
- Phase 0 handles the hard combinatorial core (group/instructor NoOverlap + day dispersion)
- Phases 1&2 reduce to bipartite matching problems with fixed times
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


# ══════════════════════════════════════════════════════════════════
# PHASE 0: Time Assignment (CTE+FTE+FPC+ICTD, FCA+SRE relaxed)
# ══════════════════════════════════════════════════════════════════

def phase0_time_assignment(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 60,
    extra_pool_nooverlap: set[tuple[int, int]] | None = None,
    random_seed: int | None = None,
) -> list[dict] | None:
    """Assign time slots + instructors: CTE+FTE+FPC+FCA+ICTD (relax SRE only).

    Returns list of {orig_i, start, duration, end, instructor} or None.
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    day_offsets, day_lengths = [], []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)
    num_days = len(day_offsets)
    quanta_per_day = day_lengths[0] if day_lengths else 7

    model = cp_model.CpModel()

    # Table constraints WITH FCA (part-time availability enforced)
    session_allowed: list[list[tuple[int, int]]] = []
    impossible_sessions: list[int] = []
    for i, s in enumerate(sessions):
        allowed = []
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
        print(f"  {len(impossible_sessions)} impossible sessions excluded")

    impossible_set = set(impossible_sessions)
    model_indices = [i for i in range(len(sessions)) if i not in impossible_set]
    N = len(model_indices)

    start_vars, end_vars, inst_vars, day_vars, interval_vars = [], [], [], [], []

    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        allowed = session_allowed[orig_i]
        inst_domain = sorted({t[0] for t in allowed})
        start_domain = sorted({t[1] for t in allowed})

        inst = model.new_int_var_from_domain(
            cp_model.Domain.from_values(inst_domain), f"inst_{mi}")
        start = model.new_int_var_from_domain(
            cp_model.Domain.from_values(start_domain), f"start_{mi}")
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
        for gid in set(sessions[orig_i].group_ids):
            group_sessions[gid].add(mi)
    for gid, midxs in group_sessions.items():
        ml = sorted(midxs)
        if len(ml) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in ml])

    # FTE: Instructor NoOverlap (with channeling)
    inst_possible: dict[int, set[int]] = defaultdict(set)
    for mi, orig_i in enumerate(model_indices):
        for iidx in sessions[orig_i].qualified_instructor_idxs:
            inst_possible[iidx].add(mi)
    for iidx, midxs in inst_possible.items():
        ms = sorted(midxs)
        if len(ms) <= 1:
            continue
        opt_ivs = []
        for mi in ms:
            s = sessions[model_indices[mi]]
            pres = model.new_bool_var(f"fte_{mi}_{iidx}")
            model.add(inst_vars[mi] == iidx).only_enforce_if(pres)
            model.add(inst_vars[mi] != iidx).only_enforce_if(~pres)
            opt_ivs.append(model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi], pres, f"oiv_i_{mi}_{iidx}"))
        model.add_no_overlap(opt_ivs)

    # ICTD: Sibling sessions — symmetry breaking only (no day dispersion constraint)
    # ICTD is very hard to enforce jointly with FCA+cumulatives.
    # We enforce ordering to break symmetry and aid solver efficiency.
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        sibling_groups[sessions[orig_i].sibling_key].append(mi)

    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        for j in range(len(siblings) - 1):
            model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # Pool-size-1 NoOverlap: sessions that can ONLY use one specific room
    pool1_rooms: dict[int, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        if len(s.compatible_room_idxs) == 1:
            pool1_rooms[list(s.compatible_room_idxs)[0]].append(mi)

    n_pool1 = 0
    for ridx, mis in pool1_rooms.items():
        if len(mis) <= 1:
            continue
        ivs = [model.new_fixed_size_interval_var(
            start_vars[mi], sessions[model_indices[mi]].duration,
            f"p1iv_{mi}_{ridx}") for mi in mis]
        model.add_no_overlap(ivs)
        n_pool1 += 1
    print(f"  Pool-size-1 NoOverlap: {n_pool1} constraints")

    # Soft penalty for pool-size 2-5: penalize overlapping sessions
    MAX_POOL_SIZE_FOR_PENALTY = 5
    pool_groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        pool = frozenset(s.compatible_room_idxs)
        if 2 <= len(pool) <= MAX_POOL_SIZE_FOR_PENALTY:
            pool_groups[pool].append(mi)

    penalty_vars: list = []
    n_penalty_pairs = 0
    for pool, mis in pool_groups.items():
        if len(mis) <= len(pool):
            continue
        pool_size = len(pool)
        weight = 10 * (MAX_POOL_SIZE_FOR_PENALTY + 1 - pool_size)
        for i in range(len(mis)):
            for j in range(i + 1, len(mis)):
                mi_a, mi_b = mis[i], mis[j]
                b_overlap = model.new_bool_var(f"olap_{mi_a}_{mi_b}")
                b_a_before_b = model.new_bool_var(f"ab_{mi_a}_{mi_b}")
                b_b_before_a = model.new_bool_var(f"ba_{mi_a}_{mi_b}")
                model.add(end_vars[mi_a] <= start_vars[mi_b]).only_enforce_if(b_a_before_b)
                model.add(end_vars[mi_b] <= start_vars[mi_a]).only_enforce_if(b_b_before_a)
                model.add(b_a_before_b + b_b_before_a + b_overlap >= 1)
                model.add(b_a_before_b + b_overlap <= 1)
                model.add(b_b_before_a + b_overlap <= 1)
                penalty_vars.append((b_overlap, weight))
                n_penalty_pairs += 1

    if penalty_vars:
        model.minimize(sum(w * v for v, w in penalty_vars))
        print(f"  Pool overlap penalty: {n_penalty_pairs} pairs across "
              f"{sum(1 for p, mis in pool_groups.items() if len(mis) > len(p))} pools")

    # Extra pool constraints from iterative refinement: pairwise separation
    # (much lighter than cumulatives — just force specific conflicting pairs apart)
    n_extra = 0
    if extra_pool_nooverlap:
        orig_to_mi = {orig_i: mi for mi, orig_i in enumerate(model_indices)}
        for oi_a, oi_b in extra_pool_nooverlap:
            if oi_a not in orig_to_mi or oi_b not in orig_to_mi:
                continue
            mi_a, mi_b = orig_to_mi[oi_a], orig_to_mi[oi_b]
            # Force A and B not to overlap: A.end <= B.start OR B.end <= A.start
            b = model.new_bool_var(f"sep_{mi_a}_{mi_b}")
            model.add(end_vars[mi_a] <= start_vars[mi_b]).only_enforce_if(b)
            model.add(end_vars[mi_b] <= start_vars[mi_a]).only_enforce_if(~b)
            n_extra += 1
        print(f"  Pairwise separations: {n_extra} (from iterative refinement)")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False
    if random_seed is not None:
        solver.parameters.random_seed = random_seed

    print(f"\n  Solving Phase 0 ({N} sessions, {time_limit}s"
          f"{f', seed={random_seed}' if random_seed else ''})...")
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
              cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}
    print(f"  Phase 0: {STATUS.get(status, '?')} in {elapsed:.2f}s "
          f"({solver.num_branches:,} branches, {solver.num_conflicts:,} conflicts)")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Solution found!")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    result = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        st = solver.value(start_vars[mi])
        iidx = solver.value(inst_vars[mi])
        result.append({
            "mi": mi,
            "orig_i": orig_i,
            "start": st,
            "duration": s.duration,
            "end": st + s.duration,
            "instructor": iidx,
            "room": -1,
        })
    return result


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Instructor Assignment (fixed times, matching with FCA)
# ══════════════════════════════════════════════════════════════════

def phase1_instructor_assignment(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    time_limit: int = 120,
) -> bool:
    """Assign instructors given fixed times.

    Constraints:
    - FPC: instructor must be qualified for the course
    - FCA: part-time instructor must be available at session time
    - FTE: no instructor teaches two overlapping sessions

    Returns True if successful (modifies assignments in-place).
    """
    total_quanta = store.qts.total_quanta
    model = cp_model.CpModel()

    # For each assignment, create instructor variable
    inst_vars = []
    for ai, a in enumerate(assignments):
        s = sessions[a["orig_i"]]
        # Filter qualified instructors by FCA
        valid_insts = []
        for iidx in s.qualified_instructor_idxs:
            iid = instructor_ids[iidx]
            inst = store.instructors[iid]
            if inst.is_full_time:
                valid_insts.append(iidx)
            else:
                # Check all quanta of this session are available
                if all(q in inst.available_quanta
                       for q in range(a["start"], a["end"])):
                    valid_insts.append(iidx)

        if not valid_insts:
            print(f"  WARNING: No valid instructor for S{a['orig_i']} "
                  f"({s.course_id}) at start={a['start']}")
            # Fall back to any qualified instructor (relax FCA for this session)
            valid_insts = list(s.qualified_instructor_idxs)

        iv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(valid_insts)), f"inst_{ai}")
        inst_vars.append(iv)

    # FTE: No instructor overlap
    # Find overlapping session pairs
    n = len(assignments)
    for iidx in range(len(instructor_ids)):
        # Sessions that could use this instructor
        candidates = []
        for ai, a in enumerate(assignments):
            s = sessions[a["orig_i"]]
            if iidx in s.qualified_instructor_idxs:
                candidates.append(ai)

        if len(candidates) <= 1:
            continue

        # For each pair of overlapping candidates, add constraint
        for i in range(len(candidates)):
            ai = candidates[i]
            for j in range(i + 1, len(candidates)):
                aj = candidates[j]
                a_i = assignments[ai]
                a_j = assignments[aj]
                # Check overlap
                if a_i["start"] < a_j["end"] and a_j["start"] < a_i["end"]:
                    # Cannot both use instructor iidx
                    b_i = model.new_bool_var(f"uses_{ai}_{iidx}")
                    b_j = model.new_bool_var(f"uses_{aj}_{iidx}")
                    model.add(inst_vars[ai] == iidx).only_enforce_if(b_i)
                    model.add(inst_vars[ai] != iidx).only_enforce_if(~b_i)
                    model.add(inst_vars[aj] == iidx).only_enforce_if(b_j)
                    model.add(inst_vars[aj] != iidx).only_enforce_if(~b_j)
                    model.add(b_i + b_j <= 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True

    print(f"\n  Solving Phase 1: Instructor Assignment ({n} sessions, {time_limit}s)...")
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
              cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}
    print(f"  Phase 1: {STATUS.get(status, '?')} in {elapsed:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False

    for ai in range(n):
        assignments[ai]["instructor"] = solver.value(inst_vars[ai])

    return True


# ══════════════════════════════════════════════════════════════════
# PHASE 1b: Repair failed sessions (small CP-SAT model)
# ══════════════════════════════════════════════════════════════════

def repair_failed_sessions(
    assignments: list[dict],
    failed_ais: list[int],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 60,
) -> bool:
    """Re-schedule failed sessions AND their room-pool neighbors.

    Expands the repair neighborhood: for each failed session, also frees any
    successfully-assigned session that shares its room pool AND overlaps in time.
    This gives the solver freedom to move competing sessions.

    Returns True if successful (modifies assignments in-place).
    """
    qts = store.qts
    total_quanta = qts.total_quanta

    day_offsets, day_lengths = [], []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
            day_lengths.append(cnt)

    # Expand neighborhood:
    # 1. Room-pool neighbors: sessions using same rooms and overlapping in time
    # 2. Same-pool siblings: ALL sessions sharing the exact same room pool
    #    as a failed session (they might be placed at non-overlapping times but
    #    collectively constrain the pool capacity)
    failed_set = set(failed_ais)
    neighbor_set: set[int] = set()

    # Collect pools of failed sessions
    failed_pools: set[frozenset[int]] = set()
    for ai in failed_ais:
        s = sessions[assignments[ai]["orig_i"]]
        failed_pools.add(frozenset(s.compatible_room_idxs))

    for ai in failed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        pool = s.compatible_room_idxs
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

    # Also free ALL sessions in the same pool as failed sessions
    # (even those at non-overlapping times — they constrain pool capacity)
    pool_sibling_count = 0
    for ai2 in range(len(assignments)):
        if ai2 in failed_set or ai2 in neighbor_set:
            continue
        s2 = sessions[assignments[ai2]["orig_i"]]
        if frozenset(s2.compatible_room_idxs) in failed_pools:
            neighbor_set.add(ai2)
            pool_sibling_count += 1

    # All "free" sessions = failed + neighbors
    free_ais = sorted(failed_set | neighbor_set)
    fixed_ais = [ai for ai in range(len(assignments)) if ai not in set(free_ais)]
    F = len(free_ais)

    print(f"  Repair neighborhood: {len(failed_ais)} failed + "
          f"{len(neighbor_set)} neighbors ({pool_sibling_count} pool siblings) = {F} free sessions")

    # Build occupancy maps for truly fixed sessions
    group_at_q: dict[int, set[str]] = defaultdict(set)
    inst_at_q: dict[int, set[int]] = defaultdict(set)
    room_at_q: dict[int, set[int]] = defaultdict(set)

    for ai in fixed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        for q in range(a["start"], a["end"]):
            for gid in s.group_ids:
                group_at_q[q].add(gid)
            inst_at_q[q].add(a["instructor"])
            room_at_q[q].add(a["room"])

    model = cp_model.CpModel()
    start_vars, inst_vars_r, room_vars_r = [], [], []
    infeasible_sessions: list[int] = []

    for fi, ai in enumerate(free_ais):
        s = sessions[assignments[ai]["orig_i"]]

        allowed: list[tuple[int, int, int]] = []
        base_starts = compute_valid_starts(s.duration, total_quanta, day_offsets, day_lengths)

        for st in base_starts:
            quanta_range = range(st, st + s.duration)

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
                inst = store.instructors[iid]

                if not inst.is_full_time:
                    avail = set(inst.available_quanta)
                    if not all(q in avail for q in quanta_range):
                        continue

                inst_ok = True
                for q in quanta_range:
                    if iidx in inst_at_q.get(q, set()):
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
            print(f"  Repair: NO feasible slot for session {ai} "
                  f"({s.course_id} dur={s.duration}) even with neighbors freed")
            infeasible_sessions.append(fi)
            # Add dummy variables so indexing stays consistent
            sv = model.new_int_var(0, 0, f"rs_{fi}")
            iv = model.new_int_var(0, 0, f"ri_{fi}")
            rv = model.new_int_var(0, 0, f"rr_{fi}")
            start_vars.append(sv)
            inst_vars_r.append(iv)
            room_vars_r.append(rv)
            continue

        sv = model.new_int_var(0, total_quanta, f"rs_{fi}")
        iv = model.new_int_var(0, len(instructor_ids), f"ri_{fi}")
        rv = model.new_int_var(0, len(room_ids), f"rr_{fi}")
        model.add_allowed_assignments([sv, iv, rv], allowed)
        start_vars.append(sv)
        inst_vars_r.append(iv)
        room_vars_r.append(rv)

    if infeasible_sessions:
        print(f"  {len(infeasible_sessions)} sessions have no feasible slot.")
        return False

    # Mutual constraints among free sessions
    for i in range(F):
        si = sessions[assignments[free_ais[i]]["orig_i"]]
        for j in range(i + 1, F):
            sj = sessions[assignments[free_ais[j]]["orig_i"]]

            if set(si.group_ids) & set(sj.group_ids):
                ivi = model.new_fixed_size_interval_var(start_vars[i], si.duration, f"cte_{i}_{j}_i")
                ivj = model.new_fixed_size_interval_var(start_vars[j], sj.duration, f"cte_{i}_{j}_j")
                model.add_no_overlap([ivi, ivj])

            b_same_inst = model.new_bool_var(f"si_{i}_{j}")
            model.add(inst_vars_r[i] == inst_vars_r[j]).only_enforce_if(b_same_inst)
            model.add(inst_vars_r[i] != inst_vars_r[j]).only_enforce_if(~b_same_inst)
            b_order = model.new_bool_var(f"so_{i}_{j}")
            model.add(start_vars[i] + si.duration <= start_vars[j]).only_enforce_if(b_same_inst, b_order)
            model.add(start_vars[j] + sj.duration <= start_vars[i]).only_enforce_if(b_same_inst, ~b_order)

            b_same_room = model.new_bool_var(f"sr_{i}_{j}")
            model.add(room_vars_r[i] == room_vars_r[j]).only_enforce_if(b_same_room)
            model.add(room_vars_r[i] != room_vars_r[j]).only_enforce_if(~b_same_room)
            b_order2 = model.new_bool_var(f"sro_{i}_{j}")
            model.add(start_vars[i] + si.duration <= start_vars[j]).only_enforce_if(b_same_room, b_order2)
            model.add(start_vars[j] + sj.duration <= start_vars[i]).only_enforce_if(b_same_room, ~b_order2)

    print(f"\n  Repair model: {F} free sessions, solving ({time_limit}s)...")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
              cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}
    print(f"  Repair: {STATUS.get(status, '?')} in {elapsed:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False

    for fi, ai in enumerate(free_ais):
        assignments[ai]["start"] = solver.value(start_vars[fi])
        s = sessions[assignments[ai]["orig_i"]]
        assignments[ai]["end"] = assignments[ai]["start"] + s.duration
        assignments[ai]["instructor"] = solver.value(inst_vars_r[fi])
        assignments[ai]["room"] = solver.value(room_vars_r[fi])

    return True


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Room Assignment (fixed times, matching)
# ══════════════════════════════════════════════════════════════════

def phase2_room_assignment(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    room_ids: list[str],
    time_limit: int = 120,
) -> tuple[bool, list[tuple[int, int]]]:
    """Assign rooms given fixed times using greedy with room-stealing.

    Strategy:
    1. Sort sessions by pool size ascending (hardest first)
    2. For each session, try to assign a free compatible room
    3. If no room free, try to "steal" a room from a session with a larger pool
       by moving that session to an alternative room

    Returns (success, conflicts) where conflicts is a list of (ai1, ai2)
    pairs that couldn't be separated.
    """
    n = len(assignments)

    # Track room → list of (start, end, ai) assignments
    room_schedule: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

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

    # Sort: pool size ascending, then start time
    order = sorted(range(n), key=lambda ai: (
        len(sessions[assignments[ai]["orig_i"]].compatible_room_idxs),
        assignments[ai]["start"]))

    failed: list[int] = []
    n_steals = 0

    for ai in order:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        start, end = a["start"], a["end"]

        # Try direct assignment
        assigned = False
        for ridx in sorted(s.compatible_room_idxs):
            if is_room_free(ridx, start, end):
                assign_room(ai, ridx)
                assigned = True
                break

        if assigned:
            continue

        # Try stealing: find a session using our pool's room that has alternatives
        stolen = False
        for ridx in sorted(s.compatible_room_idxs):
            # Find sessions blocking this room in our time window
            blockers = [(rs, re, ai2) for (rs, re, ai2) in room_schedule[ridx]
                        if start < re and rs < end]
            for (_, _, ai_blocker) in blockers:
                s_blocker = sessions[assignments[ai_blocker]["orig_i"]]
                b_start = assignments[ai_blocker]["start"]
                b_end = assignments[ai_blocker]["end"]
                # Can the blocker move to a different room?
                for alt_ridx in sorted(s_blocker.compatible_room_idxs):
                    if alt_ridx == ridx:
                        continue
                    if is_room_free(alt_ridx, b_start, b_end):
                        # Move blocker to alt_ridx, take ridx for our session
                        unassign_room(ai_blocker)
                        assign_room(ai_blocker, alt_ridx)
                        assign_room(ai, ridx)
                        stolen = True
                        n_steals += 1
                        break
                if stolen:
                    break
            if stolen:
                break

        if stolen:
            continue

        # Try chain stealing (depth 2): steal from blocker → move blocker's blocker
        chain_done = False
        for ridx in sorted(s.compatible_room_idxs):
            blockers = [(rs, re, ai2) for (rs, re, ai2) in room_schedule[ridx]
                        if start < re and rs < end]
            for (_, _, ai_b1) in blockers:
                s_b1 = sessions[assignments[ai_b1]["orig_i"]]
                b1_start = assignments[ai_b1]["start"]
                b1_end = assignments[ai_b1]["end"]
                for alt_ridx in sorted(s_b1.compatible_room_idxs):
                    if alt_ridx == ridx:
                        continue
                    # Is someone blocking alt_ridx for b1?
                    blockers2 = [(rs, re, ai3) for (rs, re, ai3) in room_schedule[alt_ridx]
                                 if b1_start < re and rs < b1_end]
                    for (_, _, ai_b2) in blockers2:
                        s_b2 = sessions[assignments[ai_b2]["orig_i"]]
                        b2_start = assignments[ai_b2]["start"]
                        b2_end = assignments[ai_b2]["end"]
                        for alt2_ridx in sorted(s_b2.compatible_room_idxs):
                            if alt2_ridx == alt_ridx:
                                continue
                            if is_room_free(alt2_ridx, b2_start, b2_end):
                                unassign_room(ai_b2)
                                assign_room(ai_b2, alt2_ridx)
                                unassign_room(ai_b1)
                                assign_room(ai_b1, alt_ridx)
                                assign_room(ai, ridx)
                                chain_done = True
                                n_steals += 1
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

    if not failed:
        print(f"\n  Phase 1 (Room Assignment): SUCCESS — all {n} sessions assigned "
              f"({n_steals} steals)")
        return True, []

    # Build conflict list for iterative refinement
    conflicts: list[tuple[int, int]] = []
    for ai in failed:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        start, end = a["start"], a["end"]
        pool = s.compatible_room_idxs
        pool_size = len(pool)
        overlapping = []
        for ai2 in range(n):
            if ai2 == ai or assignments[ai2]["room"] < 0:
                continue
            if assignments[ai2]["room"] not in pool:
                continue
            if assignments[ai2]["start"] < end and start < assignments[ai2]["end"]:
                overlapping.append(ai2)
        if len(overlapping) >= pool_size:
            day_str, _ = store.qts.quanta_to_time(start)
            print(f"  CONFLICT: session {ai} ({s.course_id} dur={s.duration} "
                  f"pool_sz={pool_size} {day_str} q{start}-{end}) "
                  f"blocked by {len(overlapping)} overlapping sessions")
            for ai2 in overlapping:
                conflicts.append((ai, ai2))

    print(f"\n  Phase 1 (Room Assignment): FAILED — "
          f"{len(failed)}/{n} sessions unassigned ({n_steals} steals), "
          f"{len(conflicts)} conflict pairs")
    return False, conflicts


# ══════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════

def verify_schedule(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> int:
    """Verify all hard constraints. Returns number of violations."""
    violations = 0
    qts = store.qts
    day_offsets = []
    for day in qts.DAY_NAMES:
        off = qts.day_quanta_offset.get(day)
        cnt = qts.day_quanta_count.get(day, 0)
        if off is not None and cnt > 0:
            day_offsets.append(off)
    quanta_per_day = 7

    details = defaultdict(int)

    # CTE: group no-overlap
    group_slots: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for a in assignments:
        s = sessions[a["orig_i"]]
        for gid in s.group_ids:
            group_slots[gid].append((a["start"], a["end"], a["orig_i"]))
    for gid, slots in group_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                details["CTE"] += 1
                violations += 1

    # FTE: instructor no-overlap
    inst_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        inst_slots[a["instructor"]].append((a["start"], a["end"]))
    for iidx, slots in inst_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                details["FTE"] += 1
                violations += 1

    # SRE: room no-overlap
    room_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        room_slots[a["room"]].append((a["start"], a["end"]))
    for ridx, slots in room_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i][1] > slots[i + 1][0]:
                details["SRE"] += 1
                violations += 1

    # FFC: room compatibility
    for a in assignments:
        s = sessions[a["orig_i"]]
        if a["room"] not in s.compatible_room_idxs:
            details["FFC"] += 1
            violations += 1

    # FCA: instructor availability
    for a in assignments:
        iid = instructor_ids[a["instructor"]]
        inst = store.instructors[iid]
        if not inst.is_full_time:
            for q in range(a["start"], a["end"]):
                if q not in inst.available_quanta:
                    details["FCA"] += 1
                    violations += 1
                    break

    # ICTD: sibling sessions on different days
    sibling_groups: dict[tuple, list[dict]] = defaultdict(list)
    for a in assignments:
        s = sessions[a["orig_i"]]
        sibling_groups[s.sibling_key].append(a)
    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        days = set()
        for a in siblings:
            days.add(a["start"] // quanta_per_day)
        if len(days) < len(siblings):
            details["ICTD"] += len(siblings) - len(days)
            violations += len(siblings) - len(days)

    print(f"\n  Verification: {violations} total violations")
    for k, v in sorted(details.items()):
        print(f"    {k}: {v}")

    return violations


# ══════════════════════════════════════════════════════════════════
# Export & Summary
# ══════════════════════════════════════════════════════════════════

def export_schedule(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    path: str,
) -> None:
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
    print(f"\n  Schedule exported to: {path}  ({len(schedule)} entries)")


def print_schedule_summary(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> None:
    qts = store.qts

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

    inst_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        inst_usage[instructor_ids[a["instructor"]]] += sessions[a["orig_i"]].duration
    print(f"\n  Top 10 busiest instructors:")
    for iid, load in sorted(inst_usage.items(), key=lambda x: -x[1])[:10]:
        name = store.instructors[iid].name
        avail = len(store.instructors[iid].available_quanta) or 42
        print(f"    {name:30s}  {load:3d}/{avail:2d}q ({load/avail*100:.0f}%)")

    room_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        room_usage[room_ids[a["room"]]] += sessions[a["orig_i"]].duration
    print(f"\n  Rooms used: {len(room_usage)}/{len(room_ids)}")
    print(f"  Top 10 busiest rooms:")
    for rid, load in sorted(room_usage.items(), key=lambda x: -x[1])[:10]:
        name = store.rooms[rid].name
        print(f"    {rid:8s} {name:30s}  {load:3d}/42q ({load/42*100:.0f}%)")


def force_assign_rooms(
    assignments: list[dict],
    failed_ais: list[int],
    sessions: list[Session],
    store: DataStore,
    room_ids: list[str],
) -> None:
    """Force-assign rooms to failed sessions, picking the least-conflicting room.

    For each failed session, finds the compatible room with the fewest
    time conflicts. Accepts SRE violations to ensure every session has a room.
    """
    # Build room occupancy map
    room_at_q: dict[int, set[int]] = defaultdict(set)  # q → set of room_idxs
    for ai, a in enumerate(assignments):
        if a["room"] >= 0:
            for q in range(a["start"], a["end"]):
                room_at_q[q].add(a["room"])

    for ai in failed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        quanta = range(a["start"], a["end"])

        # Count conflicts for each compatible room
        best_room = -1
        best_conflicts = 999999
        for ridx in sorted(s.compatible_room_idxs):
            conflicts = sum(1 for q in quanta if ridx in room_at_q[q])
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_room = ridx

        if best_room >= 0:
            a["room"] = best_room
            for q in quanta:
                room_at_q[q].add(best_room)
            rid = room_ids[best_room]
            day_str, time_str = store.qts.quanta_to_time(a["start"])
            print(f"    Session {ai} ({s.course_id}): → {rid} "
                  f"({day_str} {time_str}, {best_conflicts} SRE conflicts)")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Three-phase CP-SAT scheduler")
    parser.add_argument("--time-limit", type=int, default=120,
                        help="Phase 0 time limit (default: 120)")
    parser.add_argument("--data-dir", type=str, default="data_fixed")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--no-diag", action="store_true")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds to try (default: 10)")
    parser.add_argument("--repair-time", type=int, default=120,
                        help="Repair time limit (default: 120)")
    args = parser.parse_args()

    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    # ── Multi-start: try multiple random seeds ──
    best_assignments: list[dict] | None = None
    best_failures: int = 999999
    best_seed: int = 0

    N_SEEDS = args.seeds
    print(f"\n{'='*72}")
    print(f"  MULTI-START: trying {N_SEEDS} seeds for Phase 0 + greedy room assignment")
    print(f"{'='*72}")

    for seed in range(N_SEEDS):
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}/{N_SEEDS-1}")
        print(f"{'─'*60}")

        assignments = phase0_time_assignment(
            sessions, store, instructor_ids, room_ids,
            time_limit=args.time_limit, random_seed=seed
        )
        if assignments is None:
            print(f"  Seed {seed}: Phase 0 FAILED")
            continue

        # Run greedy room assignment (non-destructive: we can retry)
        ok, conflicts = phase2_room_assignment(
            assignments, sessions, store, room_ids, time_limit=120
        )

        if ok:
            print(f"\n  *** Seed {seed}: PERFECT — 0 failures! ***")
            best_assignments = assignments
            best_failures = 0
            best_seed = seed
            break

        n_failed = sum(1 for a in assignments if a["room"] < 0)
        print(f"  Seed {seed}: {n_failed} failures")

        if n_failed < best_failures:
            # Deep copy assignments
            best_assignments = [dict(a) for a in assignments]
            best_failures = n_failed
            best_seed = seed

        # If we already found very few failures, try repair immediately
        if n_failed <= 5:
            failed_ais = [ai for ai in range(len(assignments)) if assignments[ai]["room"] < 0]
            print(f"\n  Attempting repair for seed {seed} ({n_failed} failures)...")
            repaired = repair_failed_sessions(
                assignments, failed_ais, sessions, store,
                instructor_ids, room_ids, time_limit=args.repair_time
            )
            if repaired:
                print(f"\n  *** Seed {seed}: REPAIRED — 0 failures! ***")
                best_assignments = assignments
                best_failures = 0
                best_seed = seed
                break

    if best_assignments is None:
        print("\nAll seeds FAILED Phase 0.")
        sys.exit(1)

    assignments = best_assignments
    print(f"\n{'='*72}")
    print(f"  Best seed: {best_seed} with {best_failures} failures")
    print(f"{'='*72}")

    # If best still has failures, attempt repair on the best solution
    if best_failures > 0:
        failed_ais = [ai for ai in range(len(assignments)) if assignments[ai]["room"] < 0]
        print(f"\n  Final repair attempt: {len(failed_ais)} sessions...")
        repaired = repair_failed_sessions(
            assignments, failed_ais, sessions, store,
            instructor_ids, room_ids, time_limit=args.repair_time
        )
        if not repaired:
            still_failed = [ai for ai in range(len(assignments)) if assignments[ai]["room"] < 0]
            if still_failed:
                print(f"\n  Force-assigning rooms to {len(still_failed)} remaining sessions...")
                force_assign_rooms(assignments, still_failed, sessions, store, room_ids)

    # ── Verify ──
    violations = verify_schedule(
        assignments, sessions, store, instructor_ids, room_ids
    )

    print_schedule_summary(
        assignments, sessions, store, instructor_ids, room_ids
    )

    if args.export:
        export_schedule(
            assignments, sessions, store, instructor_ids, room_ids, args.export
        )

    unassigned = sum(1 for a in assignments if a["room"] < 0)
    if unassigned > 0:
        print(f"\n  WARNING: {unassigned} sessions have no room!")
    if violations > 0:
        print(f"\n  WARNING: {violations} constraint violations remain!")

    success = violations == 0 and unassigned == 0
    print(f"\n  Done. {'SUCCESS' if success else 'HAS ISSUES'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
