#!/usr/bin/env python3
"""LNS CP-SAT solver for university course timetabling.

Two-phase approach with LNS fallback:
  Phase 0: Time + instructor assignment (CP-SAT, soft room-pool penalties)
  Phase 1: Room assignment (CP-SAT, fixed times, full SRE)
  LNS:     If Phase 1 fails, iteratively repair via Large Neighborhood Search

Usage:
    python schedule_lns.py                              # default run
    python schedule_lns.py --export schedule.json       # export result
    python schedule_lns.py --seeds 5 --time-limit 120   # multi-seed
"""

from __future__ import annotations

import argparse
import json
import random
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
    compute_valid_starts,
    compute_valid_starts_for_instructor,
    run_diagnostics,
)
from src.io.data_store import DataStore


# ══════════════════════════════════════════════════════════════════
# PHASE 0: Time + Instructor Assignment
# ══════════════════════════════════════════════════════════════════


def phase0_time_instructor(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 120,
    random_seed: int | None = None,
) -> list[dict] | None:
    """Assign time slots and instructors via CP-SAT.

    Hard constraints: CTE, FTE, FPC, FCA, pool-1 SRE.
    Soft objective: minimize overlapping sessions in small room pools.
    Returns list of assignment dicts or None if failed.
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

    # Pre-compute allowed (instructor, start) tuples per session
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
                    day_offsets, day_lengths,
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

    start_vars, end_vars, inst_vars, interval_vars = [], [], [], []

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

        interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")
        model.add_allowed_assignments([inst, start], allowed)

        start_vars.append(start)
        end_vars.append(end)
        inst_vars.append(inst)
        interval_vars.append(interval)

    # CTE: Group NoOverlap
    group_sessions: dict[str, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        for gid in set(sessions[orig_i].group_ids):
            group_sessions[gid].append(mi)
    for gid, mis in group_sessions.items():
        if len(mis) > 1:
            model.add_no_overlap([interval_vars[mi] for mi in mis])

    # FTE: Instructor NoOverlap (channeling)
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
            opt_ivs.append(
                model.new_optional_interval_var(
                    start_vars[mi], s.duration, end_vars[mi], pres,
                    f"oiv_i_{mi}_{iidx}",
                )
            )
        model.add_no_overlap(opt_ivs)

    # Sibling symmetry breaking (ascending start order)
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        sibling_groups[sessions[orig_i].sibling_key].append(mi)
    for key, siblings in sibling_groups.items():
        if len(siblings) > 1:
            for j in range(len(siblings) - 1):
                model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # Pool-1 hard NoOverlap: sessions that can only use one room
    pool1_rooms: dict[int, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        if len(s.compatible_room_idxs) == 1:
            pool1_rooms[s.compatible_room_idxs[0]].append(mi)
    n_pool1 = 0
    for ridx, mis in pool1_rooms.items():
        if len(mis) > 1:
            ivs = [
                model.new_fixed_size_interval_var(
                    start_vars[mi], sessions[model_indices[mi]].duration,
                    f"p1iv_{mi}_{ridx}",
                )
                for mi in mis
            ]
            model.add_no_overlap(ivs)
            n_pool1 += 1
    print(f"  Pool-1 NoOverlap: {n_pool1} constraints")

    # Soft penalty for pool-size 2-5: discourage simultaneous sessions
    MAX_POOL = 5
    pool_groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        pool = frozenset(s.compatible_room_idxs)
        if 2 <= len(pool) <= MAX_POOL:
            pool_groups[pool].append(mi)

    penalty_vars: list[tuple] = []
    n_penalty = 0
    for pool, mis in pool_groups.items():
        if len(mis) <= len(pool):
            continue
        pool_size = len(pool)
        weight = 10 * (MAX_POOL + 1 - pool_size)
        for i in range(len(mis)):
            for j in range(i + 1, len(mis)):
                mi_a, mi_b = mis[i], mis[j]
                b_overlap = model.new_bool_var(f"olap_{mi_a}_{mi_b}")
                b_ab = model.new_bool_var(f"ab_{mi_a}_{mi_b}")
                b_ba = model.new_bool_var(f"ba_{mi_a}_{mi_b}")
                model.add(end_vars[mi_a] <= start_vars[mi_b]).only_enforce_if(b_ab)
                model.add(end_vars[mi_b] <= start_vars[mi_a]).only_enforce_if(b_ba)
                model.add(b_ab + b_ba + b_overlap >= 1)
                model.add(b_ab + b_overlap <= 1)
                model.add(b_ba + b_overlap <= 1)
                penalty_vars.append((b_overlap, weight))
                n_penalty += 1

    if penalty_vars:
        model.minimize(sum(w * v for v, w in penalty_vars))
        print(f"  Soft pool penalty: {n_penalty} pairs")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False
    if random_seed is not None:
        solver.parameters.random_seed = random_seed

    print(
        f"\n  Phase 0: solving {N} sessions ({time_limit}s"
        f"{f', seed={random_seed}' if random_seed is not None else ''})..."
    )
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(
        f"  Phase 0: {STATUS.get(status, '?')} in {elapsed:.1f}s "
        f"({solver.num_branches:,} branches)"
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    result = []
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        st = solver.value(start_vars[mi])
        result.append({
            "orig_i": orig_i,
            "start": st,
            "duration": s.duration,
            "end": st + s.duration,
            "instructor": solver.value(inst_vars[mi]),
            "room": -1,
        })
    return result


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Room Assignment (fixed times, full SRE)
# ══════════════════════════════════════════════════════════════════


def phase1_room_assignment(
    assignments: list[dict],
    sessions: list[Session],
    room_ids: list[str],
    time_limit: int = 120,
) -> bool:
    """Assign rooms with all times fixed. Full SRE via per-room NoOverlap.

    Returns True if feasible (updates assignments in-place).
    """
    N = len(assignments)
    model = cp_model.CpModel()

    # Room variable per session
    room_vars = []
    for i in range(N):
        s = sessions[assignments[i]["orig_i"]]
        rv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(s.compatible_room_idxs)),
            f"room_{i}",
        )
        room_vars.append(rv)

    # SRE: per-room NoOverlap with optional fixed-size intervals
    room_candidates: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        s = sessions[assignments[i]["orig_i"]]
        for ridx in s.compatible_room_idxs:
            room_candidates[ridx].append(i)

    n_constraints = 0
    n_optionals = 0
    for ridx, cands in room_candidates.items():
        if len(cands) <= 1:
            continue
        opt_ivs = []
        for i in cands:
            a = assignments[i]
            pres = model.new_bool_var(f"rp_{i}_{ridx}")
            model.add(room_vars[i] == ridx).only_enforce_if(pres)
            model.add(room_vars[i] != ridx).only_enforce_if(~pres)
            opt_ivs.append(
                model.new_optional_fixed_size_interval_var(
                    a["start"], a["duration"], pres, f"riv_{i}_{ridx}"
                )
            )
            n_optionals += 1
        model.add_no_overlap(opt_ivs)
        n_constraints += 1

    print(
        f"  Phase 1 model: {N} sessions, {n_constraints} room NoOverlap, "
        f"{n_optionals} optional intervals"
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"  Phase 1: {STATUS.get(status, '?')} in {elapsed:.1f}s")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            assignments[i]["room"] = solver.value(room_vars[i])
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# Greedy Room Assignment (fallback for LNS initialization)
# ══════════════════════════════════════════════════════════════════


def greedy_room_assignment(
    assignments: list[dict],
    sessions: list[Session],
) -> int:
    """Greedy room assignment with stealing + force-assign for remainder.

    Accepts SRE violations to produce a complete initial solution for LNS.
    Returns number of SRE violations.
    """
    N = len(assignments)
    room_schedule: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    def is_free(ridx: int, start: int, end: int) -> bool:
        for rs, re, _ in room_schedule[ridx]:
            if start < re and rs < end:
                return False
        return True

    def assign(ai: int, ridx: int) -> None:
        a = assignments[ai]
        a["room"] = ridx
        room_schedule[ridx].append((a["start"], a["end"], ai))

    def unassign(ai: int) -> None:
        a = assignments[ai]
        ridx = a["room"]
        room_schedule[ridx] = [
            (rs, re, a2) for rs, re, a2 in room_schedule[ridx] if a2 != ai
        ]
        a["room"] = -1

    # Sort: pool size ascending (hardest first), then start time
    order = sorted(
        range(N),
        key=lambda i: (
            len(sessions[assignments[i]["orig_i"]].compatible_room_idxs),
            assignments[i]["start"],
        ),
    )

    n_steals = 0
    forced = 0

    for ai in order:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        start, end = a["start"], a["end"]

        # Try direct assignment
        assigned = False
        for ridx in sorted(s.compatible_room_idxs):
            if is_free(ridx, start, end):
                assign(ai, ridx)
                assigned = True
                break
        if assigned:
            continue

        # Try stealing: find blockers that can move to alternative rooms
        stolen = False
        for ridx in sorted(s.compatible_room_idxs):
            blockers = [
                (rs, re, ai2) for rs, re, ai2 in room_schedule[ridx]
                if start < re and rs < end
            ]
            for _, _, ai_b in blockers:
                s_b = sessions[assignments[ai_b]["orig_i"]]
                b_start, b_end = assignments[ai_b]["start"], assignments[ai_b]["end"]
                for alt in sorted(s_b.compatible_room_idxs):
                    if alt == ridx:
                        continue
                    if is_free(alt, b_start, b_end):
                        unassign(ai_b)
                        assign(ai_b, alt)
                        assign(ai, ridx)
                        stolen = True
                        n_steals += 1
                        break
                if stolen:
                    break
            if stolen:
                break
        if stolen:
            continue

        # Force-assign: pick least-conflicting compatible room
        best_ridx, best_conf = s.compatible_room_idxs[0], 999
        for ridx in s.compatible_room_idxs:
            conf = sum(
                1 for rs, re, _ in room_schedule[ridx] if start < re and rs < end
            )
            if conf < best_conf:
                best_conf = conf
                best_ridx = ridx
        assign(ai, best_ridx)
        forced += 1

    violations = len(find_sre_violations(assignments, sessions))
    print(
        f"  Greedy rooms: {N - forced} clean + {forced} force-assigned, "
        f"{n_steals} steals, {violations} SRE violations"
    )
    return violations


# ══════════════════════════════════════════════════════════════════
# SRE Violation Detection
# ══════════════════════════════════════════════════════════════════


def find_sre_violations(
    assignments: list[dict],
    sessions: list[Session],
) -> list[tuple[int, int]]:
    """Find all SRE-violating pairs (same room, overlapping time)."""
    room_slots: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        if a["room"] >= 0:
            room_slots[a["room"]].append((a["start"], a["end"], ai))

    violations = []
    for ridx, slots in room_slots.items():
        slots.sort()
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                if slots[i][1] > slots[j][0]:
                    violations.append((slots[i][2], slots[j][2]))
                else:
                    break
    return violations


# ══════════════════════════════════════════════════════════════════
# LNS: Large Neighborhood Search
# ══════════════════════════════════════════════════════════════════


def select_neighborhood(
    assignments: list[dict],
    sessions: list[Session],
    violations: list[tuple[int, int]],
    max_size: int = 80,
    rng: random.Random | None = None,
    wide: bool = False,
) -> list[int]:
    """Select sessions to free in LNS iteration.

    Strategy:
    1. Include violating sessions
    2. ALL sessions in the same room pools (not just time-overlapping)
    3. Sessions sharing groups with violated sessions
    4. Random padding for diversity
    """
    if not violations:
        return []

    # Start with all violating sessions
    violation_ais: set[int] = set()
    for ai1, ai2 in violations:
        violation_ais.add(ai1)
        violation_ais.add(ai2)

    free_set = set(violation_ais)

    # Collect room pools of violating sessions
    violation_pools: set[frozenset[int]] = set()
    violation_groups: set[str] = set()
    for ai in violation_ais:
        s = sessions[assignments[ai]["orig_i"]]
        violation_pools.add(frozenset(s.compatible_room_idxs))
        for gid in s.group_ids:
            violation_groups.add(gid)

    # Ring 1: ALL sessions in the same room pools (regardless of time)
    for ai2 in range(len(assignments)):
        if ai2 in free_set:
            continue
        s2 = sessions[assignments[ai2]["orig_i"]]
        if frozenset(s2.compatible_room_idxs) in violation_pools:
            free_set.add(ai2)

    # Ring 2: sessions sharing groups with violation sessions
    if wide:
        for ai2 in range(len(assignments)):
            if ai2 in free_set:
                continue
            s2 = sessions[assignments[ai2]["orig_i"]]
            if set(s2.group_ids) & violation_groups:
                free_set.add(ai2)

    # Ring 3: random padding for diversity
    if len(free_set) < max_size and rng:
        remaining = [ai for ai in range(len(assignments)) if ai not in free_set]
        rng.shuffle(remaining)
        free_set.update(remaining[: max_size - len(free_set)])

    # Trim if too large
    if len(free_set) > max_size:
        extra = sorted(free_set - violation_ais)
        if rng:
            rng.shuffle(extra)
        free_set = violation_ais | set(extra[: max_size - len(violation_ais)])

    return sorted(free_set)


def lns_solve_neighborhood(
    assignments: list[dict],
    free_ais: list[int],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 60,
) -> bool:
    """Solve LNS subproblem: re-assign time+instructor+room for freed sessions.

    Fixed sessions constrain the freed sessions via CTE/FTE/SRE NoOverlap.
    Returns True if feasible (updates assignments in-place).
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

    free_set = set(free_ais)
    fixed_ais = [ai for ai in range(len(assignments)) if ai not in free_set]
    F = len(free_ais)

    model = cp_model.CpModel()

    # ── Decision variables for freed sessions ──
    start_vars, end_vars, inst_vars, room_vars, interval_vars = [], [], [], [], []

    for fi, ai in enumerate(free_ais):
        s = sessions[assignments[ai]["orig_i"]]

        # Compute allowed (instructor, start) pairs — FPC + FCA
        allowed: list[tuple[int, int]] = []
        base_starts = compute_valid_starts(
            s.duration, total_quanta, day_offsets, day_lengths
        )
        for iidx in s.qualified_instructor_idxs:
            iid = instructor_ids[iidx]
            inst = store.instructors[iid]
            if inst.is_full_time:
                for st in base_starts:
                    allowed.append((iidx, st))
            else:
                for st in compute_valid_starts_for_instructor(
                    s.duration, inst.available_quanta, total_quanta,
                    day_offsets, day_lengths,
                ):
                    allowed.append((iidx, st))

        if not allowed:
            return False

        inst_domain = sorted({t[0] for t in allowed})
        start_domain = sorted({t[1] for t in allowed})

        iv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(inst_domain), f"li_{fi}"
        )
        sv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(start_domain), f"ls_{fi}"
        )
        ev = model.new_int_var(0, total_quanta, f"le_{fi}")
        model.add(ev == sv + s.duration)

        rv = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(s.compatible_room_idxs)), f"lr_{fi}"
        )
        interval = model.new_interval_var(sv, s.duration, ev, f"liv_{fi}")
        model.add_allowed_assignments([iv, sv], allowed)

        # Hint from current assignment (warm start)
        a = assignments[ai]
        model.add_hint(sv, a["start"])
        model.add_hint(iv, a["instructor"])
        if a["room"] >= 0:
            model.add_hint(rv, a["room"])

        start_vars.append(sv)
        end_vars.append(ev)
        inst_vars.append(iv)
        room_vars.append(rv)
        interval_vars.append(interval)

    # ── CTE: group NoOverlap (freed + fixed) ──
    group_freed: dict[str, list[int]] = defaultdict(list)
    for fi, ai in enumerate(free_ais):
        for gid in sessions[assignments[ai]["orig_i"]].group_ids:
            group_freed[gid].append(fi)

    for gid, fis in group_freed.items():
        all_ivs = [interval_vars[fi] for fi in fis]
        for ai in fixed_ais:
            if gid in sessions[assignments[ai]["orig_i"]].group_ids:
                a = assignments[ai]
                all_ivs.append(
                    model.new_fixed_size_interval_var(
                        a["start"], a["duration"], f"fcte_{gid}_{ai}"
                    )
                )
        if len(all_ivs) > 1:
            model.add_no_overlap(all_ivs)

    # ── FTE: instructor NoOverlap (freed optional + fixed) ──
    inst_freed: dict[int, list[int]] = defaultdict(list)
    for fi, ai in enumerate(free_ais):
        for iidx in sessions[assignments[ai]["orig_i"]].qualified_instructor_idxs:
            inst_freed[iidx].append(fi)

    for iidx, fis in inst_freed.items():
        all_ivs: list[cp_model.IntervalVar] = []
        # Freed sessions — optional (present iff assigned to this instructor)
        for fi in fis:
            s = sessions[assignments[free_ais[fi]]["orig_i"]]
            pres = model.new_bool_var(f"lfte_{fi}_{iidx}")
            model.add(inst_vars[fi] == iidx).only_enforce_if(pres)
            model.add(inst_vars[fi] != iidx).only_enforce_if(~pres)
            all_ivs.append(
                model.new_optional_interval_var(
                    start_vars[fi], s.duration, end_vars[fi], pres,
                    f"loivi_{fi}_{iidx}",
                )
            )
        # Fixed sessions assigned to this instructor
        for ai in fixed_ais:
            if assignments[ai]["instructor"] == iidx:
                a = assignments[ai]
                all_ivs.append(
                    model.new_fixed_size_interval_var(
                        a["start"], a["duration"], f"ffte_{iidx}_{ai}"
                    )
                )
        if len(all_ivs) > 1:
            model.add_no_overlap(all_ivs)

    # ── SRE: room NoOverlap (freed optional + fixed) ──
    room_freed: dict[int, list[int]] = defaultdict(list)
    for fi, ai in enumerate(free_ais):
        for ridx in sessions[assignments[ai]["orig_i"]].compatible_room_idxs:
            room_freed[ridx].append(fi)

    for ridx, fis in room_freed.items():
        all_ivs = []
        # Freed sessions — optional (present iff assigned to this room)
        for fi in fis:
            s = sessions[assignments[free_ais[fi]]["orig_i"]]
            pres = model.new_bool_var(f"lsre_{fi}_{ridx}")
            model.add(room_vars[fi] == ridx).only_enforce_if(pres)
            model.add(room_vars[fi] != ridx).only_enforce_if(~pres)
            all_ivs.append(
                model.new_optional_interval_var(
                    start_vars[fi], s.duration, end_vars[fi], pres,
                    f"loivr_{fi}_{ridx}",
                )
            )
        # Fixed sessions assigned to this room
        for ai in fixed_ais:
            if assignments[ai]["room"] == ridx:
                a = assignments[ai]
                all_ivs.append(
                    model.new_fixed_size_interval_var(
                        a["start"], a["duration"], f"fsre_{ridx}_{ai}"
                    )
                )
        if len(all_ivs) > 1:
            model.add_no_overlap(all_ivs)

    # ── Sibling symmetry breaking (among freed) ──
    sibling_freed: dict[tuple, list[int]] = defaultdict(list)
    for fi, ai in enumerate(free_ais):
        sibling_freed[sessions[assignments[ai]["orig_i"]].sibling_key].append(fi)
    for key, siblings in sibling_freed.items():
        if len(siblings) > 1:
            for j in range(len(siblings) - 1):
                model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # ── Solve ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"    LNS ({F} freed): {STATUS.get(status, '?')} in {elapsed:.1f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False

    for fi, ai in enumerate(free_ais):
        s = sessions[assignments[ai]["orig_i"]]
        assignments[ai]["start"] = solver.value(start_vars[fi])
        assignments[ai]["end"] = solver.value(start_vars[fi]) + s.duration
        assignments[ai]["instructor"] = solver.value(inst_vars[fi])
        assignments[ai]["room"] = solver.value(room_vars[fi])
    return True


def lns_repair(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    max_iterations: int = 50,
    time_limit: int = 60,
    max_neighborhood: int = 100,
) -> int:
    """Run LNS iterations to eliminate SRE violations.

    Strategy:
    1. Focused neighborhood (pool siblings + random padding)
    2. If stuck, switch to wide mode (add group co-members)
    3. If still stuck, try large random neighborhoods

    Returns final SRE violation count.
    """
    rng = random.Random(42)
    stale_count = 0
    wide_mode = False
    best_v = 999
    best_state: list[dict] | None = None

    for iteration in range(max_iterations):
        violations = find_sre_violations(assignments, sessions)
        n_v = len(violations)
        if n_v == 0:
            print(f"  LNS: converged to 0 SRE violations at iteration {iteration}")
            return 0

        # Track best solution
        if n_v < best_v:
            best_v = n_v
            best_state = [dict(a) for a in assignments]

        print(f"\n  LNS iteration {iteration}: {n_v} SRE violation pairs"
              f" (best={best_v}, wide={wide_mode}, hood={max_neighborhood})")

        free_ais = select_neighborhood(
            assignments, sessions, violations,
            max_size=max_neighborhood, rng=rng, wide=wide_mode,
        )
        if not free_ais:
            break

        # Save state for rollback
        backup = [{k: v for k, v in assignments[ai].items()} for ai in free_ais]

        success = lns_solve_neighborhood(
            assignments, free_ais, sessions, store,
            instructor_ids, room_ids, time_limit,
        )

        if not success:
            # Rollback
            for i, ai in enumerate(free_ais):
                assignments[ai] = backup[i]
            stale_count += 1

            if stale_count == 3 and not wide_mode:
                wide_mode = True
                max_neighborhood = 150
                print(f"    Switching to wide mode (neighborhood={max_neighborhood})")
            elif stale_count == 6:
                max_neighborhood = min(max_neighborhood + 50, 300)
                print(f"    Expanding neighborhood to {max_neighborhood}")
            elif stale_count >= 10:
                print("  LNS: stuck — aborting")
                break
            continue

        new_violations = find_sre_violations(assignments, sessions)
        n_new = len(new_violations)
        print(f"    SRE violations: {n_v} → {n_new}")

        if n_new < n_v:
            stale_count = 0
            wide_mode = False
            max_neighborhood = 100
        else:
            stale_count += 1
            if stale_count >= 15:
                print("  LNS: no progress — aborting")
                break

    # Restore best solution if current is worse
    final_v = len(find_sre_violations(assignments, sessions))
    if best_state is not None and best_v < final_v:
        for i in range(len(assignments)):
            assignments[i] = best_state[i]
        final_v = best_v

    print(f"  LNS: finished with {final_v} SRE violations")
    return final_v


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
    details: dict[str, int] = defaultdict(int)
    quanta_per_day = 7

    # CTE: group no-overlap
    group_slots: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for a in assignments:
        for gid in sessions[a["orig_i"]].group_ids:
            group_slots[gid].append((a["start"], a["end"]))
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

    # FPC: instructor qualification
    for a in assignments:
        s = sessions[a["orig_i"]]
        if a["instructor"] not in s.qualified_instructor_idxs:
            details["FPC"] += 1
            violations += 1

    # ICTD: sibling sessions on different days (soft — counted but not blocking)
    sibling_groups: dict[tuple, list[dict]] = defaultdict(list)
    for a in assignments:
        sibling_groups[sessions[a["orig_i"]].sibling_key].append(a)
    ictd_violations = 0
    for key, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        days = {a["start"] // quanta_per_day for a in siblings}
        if len(days) < len(siblings):
            ictd_violations += len(siblings) - len(days)

    print(f"\n  Verification: {violations} hard violations, {ictd_violations} ICTD (soft)")
    for k, v in sorted(details.items()):
        print(f"    {k}: {v}")
    if ictd_violations:
        print(f"    ICTD (soft): {ictd_violations}")

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
    """Export schedule as JSON."""
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
    print(f"\n  Exported to: {path}  ({len(schedule)} entries)")


def print_summary(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
) -> None:
    """Print schedule summary."""
    qts = store.qts

    day_load: dict[str, int] = defaultdict(int)
    day_count: dict[str, int] = defaultdict(int)
    for a in assignments:
        day_str, _ = qts.quanta_to_time(a["start"])
        day_load[day_str] += a["duration"]
        day_count[day_str] += 1

    print(f"\n  {'Day':<12} {'Sessions':>8} {'Quanta':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for day in qts.DAY_NAMES:
        if day in day_load:
            print(f"  {day:<12} {day_count[day]:>8} {day_load[day]:>8}")

    inst_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        inst_usage[instructor_ids[a["instructor"]]] += a["duration"]
    print(f"\n  Top 10 instructors:")
    for iid, load in sorted(inst_usage.items(), key=lambda x: -x[1])[:10]:
        name = store.instructors[iid].name
        avail = len(store.instructors[iid].available_quanta) or 42
        print(f"    {name:30s}  {load:3d}/{avail:2d}q ({load / avail * 100:.0f}%)")

    room_usage: dict[str, int] = defaultdict(int)
    for a in assignments:
        room_usage[room_ids[a["room"]]] += a["duration"]
    print(f"\n  Rooms used: {len(room_usage)}/{len(room_ids)}")


# ══════════════════════════════════════════════════════════════════
# Hinted Full Solve (monolithic model with warm-start from Phase 0)
# ══════════════════════════════════════════════════════════════════


def hinted_full_solve(
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    time_limit: int = 300,
    phase0_time_limit: int = 120,
    phase0_seed: int = 0,
) -> list[dict] | None:
    """Iterative lazy-SRE solve with warm-start hints.

    Strategy: Build full model WITHOUT SRE (solves fast), then iteratively
    add NoOverlap constraints for rooms that have violations and re-solve.
    Each round is warm-started from the previous solution.

    Returns list of assignment dicts or None if failed.
    """
    print(f"\n  Iterative Lazy-SRE Solve (seed={phase0_seed}, "
          f"phase0={phase0_time_limit}s, round_limit={time_limit}s)")

    # Step 1: Phase 0 — get initial hints for time + instructor
    p0_assignments = phase0_time_instructor(
        sessions, store, instructor_ids, room_ids,
        time_limit=phase0_time_limit, random_seed=phase0_seed,
    )

    # Step 2: Build full model WITHOUT SRE (ICTD also relaxed)
    print("\n  Building full model (SRE relaxed, ICTD relaxed)...")
    model, vars_dict = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_ictd=True,
        relax_sre=True,
    )

    model_indices = vars_dict["model_indices"]
    start_vars = vars_dict["start"]
    end_vars = vars_dict["end"]
    inst_vars = vars_dict["instructor"]
    room_vars = vars_dict["room"]

    proto = model.proto
    print(f"    Variables:   {len(proto.variables)}")
    print(f"    Constraints: {len(proto.constraints)}")

    # Step 2b: Add room CAPACITY constraints to force even spreading
    # For each room, sum of durations of assigned sessions ≤ total_quanta
    total_quanta = store.qts.total_quanta
    room_candidates: dict[int, list[int]] = defaultdict(list)
    for mi, orig_i in enumerate(model_indices):
        s = sessions[orig_i]
        for ridx in s.compatible_room_idxs:
            room_candidates[ridx].append(mi)

    # Create assigned-to-room booleans (reused for SRE later)
    room_assigned: dict[int, dict[int, cp_model.IntVar]] = {}
    n_cap = 0
    for ridx, cands in room_candidates.items():
        if len(cands) <= 1:
            continue
        total_demand = sum(sessions[model_indices[mi]].duration for mi in cands)
        if total_demand <= total_quanta:
            continue  # Room can fit everything — no constraint needed
        assigned_bools: dict[int, cp_model.IntVar] = {}
        for mi in cands:
            b = model.new_bool_var(f"ra_{mi}_{ridx}")
            model.add(room_vars[mi] == ridx).only_enforce_if(b)
            model.add(room_vars[mi] != ridx).only_enforce_if(~b)
            assigned_bools[mi] = b
        room_assigned[ridx] = assigned_bools
        model.add(
            sum(
                sessions[model_indices[mi]].duration * assigned_bools[mi]
                for mi in cands
            )
            <= total_quanta
        )
        n_cap += 1
    print(f"    Room capacity constraints: {n_cap} rooms")
    print(f"    Variables:   {len(model.proto.variables)} (after capacity)")
    print(f"    Constraints: {len(model.proto.constraints)} (after capacity)")

    # Step 3: Add initial hints from Phase 0 + greedy (if available)
    if p0_assignments is not None:
        greedy_room_assignment(p0_assignments, sessions)
        hint_map: dict[int, dict] = {a["orig_i"]: a for a in p0_assignments}
        n_hinted = 0
        for mi, orig_i in enumerate(model_indices):
            if orig_i in hint_map:
                h = hint_map[orig_i]
                model.add_hint(start_vars[mi], h["start"])
                model.add_hint(inst_vars[mi], h["instructor"])
                if h["room"] >= 0:
                    model.add_hint(room_vars[mi], h["room"])
                n_hinted += 1
        print(f"    Hints added: {n_hinted}/{len(model_indices)} sessions")

    # Build mi → orig_i fast lookup and reverse
    mi_for_orig: dict[int, int] = {}
    for mi, orig_i in enumerate(model_indices):
        mi_for_orig[orig_i] = mi

    STATUS = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }

    constrained_rooms: set[int] = set()
    best_result: list[dict] | None = None
    best_sre_count = 999
    batch_size = 8
    unknown_streak = 0
    total_unknown = 0

    for round_num in range(30):
        # Solve current model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = 8
        solver.parameters.log_search_progress = False

        print(f"\n  Round {round_num}: solving ({len(constrained_rooms)} SRE-constrained rooms)...")
        t0 = time.time()
        status = solver.Solve(model)
        elapsed = time.time() - t0
        print(f"    {STATUS.get(status, f'UNKNOWN({status})')} in {elapsed:.1f}s"
              f" ({solver.num_branches:,} branches)")

        if status == cp_model.MODEL_INVALID:
            validation = model.validate()
            print(f"    Validation: {validation}")
            break

        if status == cp_model.INFEASIBLE:
            print(f"    INFEASIBLE — stopping")
            break

        if status == cp_model.UNKNOWN:
            # Increase time limit and retry with best solution hints
            unknown_streak += 1
            total_unknown += 1
            time_limit = min(time_limit * 2, 600)
            if unknown_streak >= 2:
                batch_size = max(batch_size // 2, 1)
                unknown_streak = 0
            print(f"    UNKNOWN — time→{time_limit}s, batch→{batch_size}"
                  f" (total unknowns: {total_unknown})")
            if total_unknown >= 5:
                print(f"    Too many UNKNOWNs — stopping iterative solve")
                break
            if best_result is not None:
                model.clear_hints()
                for a in best_result:
                    mi2 = mi_for_orig.get(a["orig_i"])
                    if mi2 is not None:
                        model.add_hint(start_vars[mi2], a["start"])
                        model.add_hint(inst_vars[mi2], a["instructor"])
                        model.add_hint(room_vars[mi2], a["room"])
            continue

        # Extract solution — reset unknown streak on success
        unknown_streak = 0
        solution: dict[int, dict] = {}
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            st = solver.value(start_vars[mi])
            solution[mi] = {
                "start": st,
                "end": st + s.duration,
                "instructor": solver.value(inst_vars[mi]),
                "room": solver.value(room_vars[mi]),
            }

        # Check SRE violations
        room_usage: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for mi, sol in solution.items():
            room_usage[sol["room"]].append((sol["start"], sol["end"], mi))

        violated_rooms: set[int] = set()
        sre_pairs = 0
        for ridx, slots in room_usage.items():
            slots.sort()
            for i in range(len(slots)):
                for j in range(i + 1, len(slots)):
                    if slots[i][1] > slots[j][0]:
                        violated_rooms.add(ridx)
                        sre_pairs += 1
                    else:
                        break

        print(f"    SRE: {sre_pairs} violation pairs in {len(violated_rooms)} rooms"
              f" (constrained: {len(constrained_rooms)})")

        # Build result for tracking best
        result = []
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            sol = solution[mi]
            result.append({
                "orig_i": orig_i,
                "start": sol["start"],
                "duration": s.duration,
                "end": sol["end"],
                "instructor": sol["instructor"],
                "room": sol["room"],
            })

        if sre_pairs < best_sre_count:
            best_sre_count = sre_pairs
            best_result = [dict(a) for a in result]

        if sre_pairs == 0:
            print(f"\n  *** 0 SRE violations — PERFECT SCHEDULE! ***")
            return result

        # Add NoOverlap only for NEWLY violated rooms (batched by severity)
        new_rooms = violated_rooms - constrained_rooms
        if not new_rooms:
            # All violated rooms already constrained — solver couldn't avoid them
            print(f"    All {len(violated_rooms)} violated rooms already constrained")
            print(f"    Increasing time limit and retrying...")
            time_limit = min(time_limit * 2, 600)
            # Add hints from current solution
            model.clear_hints()
            for mi, sol in solution.items():
                model.add_hint(start_vars[mi], sol["start"])
                model.add_hint(inst_vars[mi], sol["instructor"])
                model.add_hint(room_vars[mi], sol["room"])
            continue

        # Sort new rooms by violation count (most violations first)
        room_violation_count: dict[int, int] = defaultdict(int)
        for ridx, slots in room_usage.items():
            if ridx not in new_rooms:
                continue
            slots_sorted = sorted(slots)
            for i in range(len(slots_sorted)):
                for j in range(i + 1, len(slots_sorted)):
                    if slots_sorted[i][1] > slots_sorted[j][0]:
                        room_violation_count[ridx] += 1
                    else:
                        break

        # Add NoOverlap for top batch_size rooms by violation count
        sorted_rooms = sorted(new_rooms, key=lambda r: -room_violation_count.get(r, 0))
        batch = sorted_rooms[:batch_size]

        for ridx in sorted(batch):
            # Find ALL model sessions that could use this room
            candidates = room_candidates.get(ridx, [])

            if len(candidates) <= 1:
                constrained_rooms.add(ridx)
                continue

            # Add NoOverlap with optional intervals (channeled on room assignment)
            # Reuse existing assigned-to-room booleans from capacity constraints
            existing_bools = room_assigned.get(ridx, {})
            opt_ivs: list[cp_model.IntervalVar] = []
            for mi in candidates:
                s = sessions[model_indices[mi]]
                if mi in existing_bools:
                    pres = existing_bools[mi]
                else:
                    pres = model.new_bool_var(f"sre_{mi}_{ridx}")
                    model.add(room_vars[mi] == ridx).only_enforce_if(pres)
                    model.add(room_vars[mi] != ridx).only_enforce_if(~pres)
                opt_ivs.append(
                    model.new_optional_interval_var(
                        start_vars[mi], s.duration, end_vars[mi], pres,
                        f"oiv_r_{mi}_{ridx}",
                    )
                )
            model.add_no_overlap(opt_ivs)
            constrained_rooms.add(ridx)

        print(f"    Added NoOverlap for {len(batch)} rooms (batch): "
              f"{sorted(batch)} ({len(new_rooms) - len(batch)} deferred)")

        # Hint from current solution for warm start
        model.clear_hints()
        for mi, sol in solution.items():
            model.add_hint(start_vars[mi], sol["start"])
            model.add_hint(inst_vars[mi], sol["instructor"])
            model.add_hint(room_vars[mi], sol["room"])

    # ── Endgame: try clean full model with best hints ──
    if best_result is not None and best_sre_count > 0:
        print(f"\n  Endgame: clean full model with {best_sre_count}-pair hints...")
        full_model, full_vars = build_model(
            sessions, store, instructor_ids, room_ids,
            relax_ictd=True,
        )
        full_mi = full_vars["model_indices"]
        full_s = full_vars["start"]
        full_i = full_vars["instructor"]
        full_r = full_vars["room"]

        # Map best result by orig_i
        best_map = {a["orig_i"]: a for a in best_result}
        n_h = 0
        for mi2, oi in enumerate(full_mi):
            if oi in best_map:
                h = best_map[oi]
                full_model.add_hint(full_s[mi2], h["start"])
                full_model.add_hint(full_i[mi2], h["instructor"])
                full_model.add_hint(full_r[mi2], h["room"])
                n_h += 1
        print(f"    Hints: {n_h}/{len(full_mi)} sessions")

        solver2 = cp_model.CpSolver()
        solver2.parameters.max_time_in_seconds = 600
        solver2.parameters.num_workers = 8
        solver2.parameters.log_search_progress = False

        t0 = time.time()
        st2 = solver2.Solve(full_model)
        el2 = time.time() - t0
        print(f"    Full model: {STATUS.get(st2, '?')} in {el2:.1f}s"
              f" ({solver2.num_branches:,} branches)")

        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            full_result = []
            for mi2, oi in enumerate(full_mi):
                s = sessions[oi]
                sv = solver2.value(full_s[mi2])
                full_result.append({
                    "orig_i": oi,
                    "start": sv,
                    "duration": s.duration,
                    "end": sv + s.duration,
                    "instructor": solver2.value(full_i[mi2]),
                    "room": solver2.value(full_r[mi2]),
                })
            return full_result

    return best_result


# ══════════════════════════════════════════════════════════════════
# Micro-LNS: One-violation-at-a-time repair
# ══════════════════════════════════════════════════════════════════


def micro_lns_repair(
    assignments: list[dict],
    sessions: list[Session],
    store: DataStore,
    instructor_ids: list[str],
    room_ids: list[str],
    max_iterations: int = 100,
    time_limit: int = 30,
) -> int:
    """Repair SRE violations one at a time with tiny neighborhoods.

    For each violation pair, free ONLY those sessions + room co-occupants.
    This keeps neighborhoods small (5-20 sessions) for fast solves.
    """
    rng = random.Random(42)
    best_v = len(find_sre_violations(assignments, sessions))
    best_state = [dict(a) for a in assignments]
    stale = 0

    for iteration in range(max_iterations):
        violations = find_sre_violations(assignments, sessions)
        n_v = len(violations)
        if n_v == 0:
            print(f"    Micro-LNS: 0 violations at iteration {iteration}")
            return 0

        if n_v < best_v:
            best_v = n_v
            best_state = [dict(a) for a in assignments]
            stale = 0
        else:
            stale += 1

        if stale >= 20:
            print(f"    Micro-LNS: stale for {stale} iterations, stopping")
            break

        # Pick a random violation
        ai1, ai2 = violations[rng.randint(0, len(violations) - 1)]

        # Build tiny neighborhood: the violation pair + room co-occupants
        free_set: set[int] = {ai1, ai2}

        # Add sessions in the SAME ROOM that are near in time
        for target_ai in [ai1, ai2]:
            a = assignments[target_ai]
            r = a["room"]
            for ai3, a3 in enumerate(assignments):
                if ai3 in free_set:
                    continue
                if a3["room"] == r and a3["start"] < a["end"] + 7 and a3["end"] > a["start"] - 7:
                    free_set.add(ai3)

        # Add a few random sessions from the same room pools
        for target_ai in [ai1, ai2]:
            s = sessions[assignments[target_ai]["orig_i"]]
            for ai3, a3 in enumerate(assignments):
                if ai3 in free_set or len(free_set) >= 25:
                    break
                s3 = sessions[a3["orig_i"]]
                if frozenset(s3.compatible_room_idxs) == frozenset(s.compatible_room_idxs):
                    free_set.add(ai3)

        free_ais = sorted(free_set)

        # Backup
        backup = [{k: v for k, v in assignments[ai].items()} for ai in free_ais]

        success = lns_solve_neighborhood(
            assignments, free_ais, sessions, store,
            instructor_ids, room_ids, time_limit,
        )

        if not success:
            for i, ai in enumerate(free_ais):
                assignments[ai] = backup[i]
            continue

        new_v = len(find_sre_violations(assignments, sessions))
        if new_v > n_v:
            # Rollback — made things worse
            for i, ai in enumerate(free_ais):
                assignments[ai] = backup[i]

    # Restore best
    final_v = len(find_sre_violations(assignments, sessions))
    if best_v < final_v:
        for i in range(len(assignments)):
            assignments[i] = best_state[i]
        final_v = best_v

    print(f"    Micro-LNS: finished with {final_v} SRE violations (best={best_v})")
    return final_v


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LNS CP-SAT scheduler for university timetabling"
    )
    parser.add_argument("--time-limit", type=int, default=120,
                        help="Phase 0 time limit in seconds (default: 120)")
    parser.add_argument("--data-dir", type=str, default="data_fixed")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--no-diag", action="store_true")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of Phase 0 seeds to try")
    parser.add_argument("--lns-iters", type=int, default=50,
                        help="Max LNS iterations per seed")
    parser.add_argument("--lns-time", type=int, default=60,
                        help="LNS subproblem time limit")
    parser.add_argument("--full-time", type=int, default=300,
                        help="Full monolithic solve time limit (default: 300)")
    parser.add_argument("--mode", type=str, default="hinted",
                        choices=["hinted", "decompose", "both"],
                        help="Solver mode: hinted (full model with hints), "
                             "decompose (Phase0+1+LNS), both")
    args = parser.parse_args()

    print("Loading data...")
    store = DataStore.from_json(args.data_dir, run_preflight=False)
    print(f"  {store.summary()}")

    print("Building sessions...")
    sessions, instructor_ids, room_ids = build_sessions(store)
    print(f"  {len(sessions)} sessions generated")

    if not args.no_diag:
        run_diagnostics(sessions, store, instructor_ids, room_ids)

    best_assignments: list[dict] | None = None
    best_violations = 999

    # ── Strategy 1: Hinted full solve ──
    if args.mode in ("hinted", "both"):
        for seed in range(args.seeds):
            print(f"\n{'='*60}")
            print(f"  Hinted Full Solve — seed {seed}")
            print(f"{'='*60}")

            result = hinted_full_solve(
                sessions, store, instructor_ids, room_ids,
                time_limit=args.full_time,
                phase0_time_limit=args.time_limit,
                phase0_seed=seed,
            )
            if result is None:
                print(f"  Seed {seed}: hinted full solve returned no solution")
                continue

            v = verify_schedule(
                result, sessions, store, instructor_ids, room_ids
            )
            if v < best_violations:
                best_assignments = result
                best_violations = v
            if v == 0:
                print(f"\n  *** PERFECT SCHEDULE at seed {seed} (hinted full)! ***")
                break

        # Micro-LNS repair on best hinted result if violations remain
        if best_assignments is not None and best_violations > 0:
            print(f"\n{'='*60}")
            print(f"  Micro-LNS Repair on best hinted result ({best_violations} violations)")
            print(f"{'='*60}")
            lns_assignments = [dict(a) for a in best_assignments]
            micro_lns_repair(
                lns_assignments, sessions, store, instructor_ids, room_ids,
                max_iterations=args.lns_iters * 5, time_limit=30,
            )
            v = verify_schedule(
                lns_assignments, sessions, store, instructor_ids, room_ids
            )
            if v < best_violations:
                best_assignments = lns_assignments
                best_violations = v

    # ── Strategy 2: Decompose (Phase0 + Phase1/LNS) ──
    if args.mode in ("decompose", "both") and best_violations > 0:
        for seed in range(args.seeds):
            print(f"\n{'='*60}")
            print(f"  Decomposition — seed {seed}")
            print(f"{'='*60}")

            assignments = phase0_time_instructor(
                sessions, store, instructor_ids, room_ids,
                time_limit=args.time_limit, random_seed=seed,
            )
            if assignments is None:
                print(f"  Seed {seed}: Phase 0 failed")
                continue

            print(f"\n  Phase 1: Room assignment ({len(assignments)} sessions)...")
            ok = phase1_room_assignment(assignments, sessions, room_ids, time_limit=120)

            if ok:
                v = verify_schedule(
                    assignments, sessions, store, instructor_ids, room_ids
                )
                if v < best_violations:
                    best_assignments = [dict(a) for a in assignments]
                    best_violations = v
                if v == 0:
                    print(f"\n  *** PERFECT SCHEDULE at seed {seed}! ***")
                    break
                continue

            print(f"\n  Phase 1 failed — initializing LNS...")
            greedy_room_assignment(assignments, sessions)

            lns_repair(
                assignments, sessions, store, instructor_ids, room_ids,
                max_iterations=args.lns_iters, time_limit=args.lns_time,
            )

            v = verify_schedule(
                assignments, sessions, store, instructor_ids, room_ids
            )
            if v < best_violations:
                best_assignments = [dict(a) for a in assignments]
                best_violations = v
            if v == 0:
                print(f"\n  *** PERFECT SCHEDULE at seed {seed} (via LNS)! ***")
                break

    if best_assignments is None:
        print("\nFailed to produce any schedule.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT: {best_violations} hard violations")
    print(f"{'='*60}")

    verify_schedule(best_assignments, sessions, store, instructor_ids, room_ids)
    print_summary(best_assignments, sessions, store, instructor_ids, room_ids)

    if args.export:
        export_schedule(
            best_assignments, sessions, store, instructor_ids, room_ids, args.export
        )

    sys.exit(0 if best_violations == 0 else 1)


if __name__ == "__main__":
    main()
