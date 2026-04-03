#!/usr/bin/env python3
"""Deep diagnosis of Phase B room assignment failures."""
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpsat_phase1 import (
    build_sessions, build_phase1_model, solve_and_report,
    extract_assignments, phase_b_room_assignment,
)
from src.io.data_store import DataStore


def main():
    store = DataStore.from_json("data_fixed", run_preflight=False)
    sessions, iids, rids = build_sessions(store, cross_qualify=True)

    # Phase A: solve (no room pool penalties)
    model_a, vars_a = build_phase1_model(
        sessions, store, iids, rids,
        relax_pmi=True, no_rooms=True, room_pool_limit=0,
    )
    result_a, solver_a = solve_and_report(
        model_a, sessions, store, iids, rids, vars_a,
        time_limit=60, random_seed=1,
    )
    if not result_a:
        print("Phase A failed")
        return
    assignments = extract_assignments(solver_a, sessions, iids, vars_a)

    # Phase B
    n_failed, failed_ais = phase_b_room_assignment(assignments, sessions, store, rids)

    if not failed_ais:
        print("\n  No failures!")
        return

    # Build room schedule from assignments
    room_schedule: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for ai, a in enumerate(assignments):
        if a["room"] >= 0:
            room_schedule[a["room"]].append((a["start"], a["end"], ai))

    print()
    print("=" * 80)
    print("DETAILED FAILURE ANALYSIS")
    print("=" * 80)

    for ai in failed_ais:
        a = assignments[ai]
        s = sessions[a["orig_i"]]
        start, end = a["start"], a["end"]
        pool = s.compatible_room_idxs
        cid = s.course_id
        ctype = s.course_type
        print(f"\n--- FAILED S{a['orig_i']}: {cid} ({ctype}) ---")
        print(f"    Time: q{start}-q{end} (dur={s.duration})")
        print(f"    Groups: {s.group_ids}")
        print(f"    Room pool size: {len(pool)}")
        print(f"    Room pool: {[rids[r] for r in pool]}")

        free_rooms = []
        for ridx in pool:
            conflicts = [
                (rs, re, ai2)
                for (rs, re, ai2) in room_schedule[ridx]
                if start < re and rs < end
            ]
            if conflicts:
                for rs, re, ai2 in conflicts:
                    s2 = sessions[assignments[ai2]["orig_i"]]
                    print(
                        f"    Room {rids[ridx]}: BLOCKED by S{assignments[ai2]['orig_i']} "
                        f"{s2.course_id}({s2.course_type}) q{rs}-q{re}"
                    )
            else:
                free_rooms.append(rids[ridx])

        if free_rooms:
            print(f"    *** BUG: These rooms are FREE: {free_rooms}")

    # Pool utilization analysis
    print()
    print("=" * 80)
    print("POOL UTILIZATION ANALYSIS")
    print("=" * 80)

    pool_sessions: dict[tuple, list[int]] = defaultdict(list)
    for ai in range(len(assignments)):
        s = sessions[assignments[ai]["orig_i"]]
        pool_key = tuple(sorted(s.compatible_room_idxs))
        pool_sessions[pool_key].append(ai)

    failed_pools = set()
    for ai in failed_ais:
        s = sessions[assignments[ai]["orig_i"]]
        failed_pools.add(tuple(sorted(s.compatible_room_idxs)))

    qts = store.qts
    for pool_key in sorted(failed_pools):
        pool_names = [rids[r] for r in pool_key]
        ais_in_pool = pool_sessions[pool_key]
        n_total = len(ais_in_pool)
        n_rooms = len(pool_key)

        # Compute max concurrent demand per quanta
        demand_at_q: dict[int, int] = defaultdict(int)
        for ai in ais_in_pool:
            a = assignments[ai]
            for q in range(a["start"], a["end"]):
                demand_at_q[q] += 1

        max_concurrent = max(demand_at_q.values()) if demand_at_q else 0

        # Total quanta demanded
        total_demand = sum(
            assignments[ai]["end"] - assignments[ai]["start"]
            for ai in ais_in_pool
        )
        capacity = n_rooms * 42  # 42 quanta per room per week

        n_failed_in_pool = sum(
            1 for ai in ais_in_pool if assignments[ai]["room"] < 0
        )

        print(
            f"\n  Pool {pool_names}: {n_rooms} rooms, {n_total} sessions "
            f"({n_failed_in_pool} failed)"
        )
        print(f"    Max concurrent demand: {max_concurrent} (capacity: {n_rooms})")
        print(
            f"    Total quanta demand: {total_demand} / {capacity} capacity "
            f"({100 * total_demand / capacity:.1f}%)"
        )
        if max_concurrent > n_rooms:
            print(
                f"    *** OVERLOAD: {max_concurrent} sessions need "
                f"{n_rooms} rooms simultaneously!"
            )

            # Show the overloaded quanta
            overloaded_quanta = [q for q, d in sorted(demand_at_q.items()) if d > n_rooms]
            for q in overloaded_quanta[:10]:
                day_str, time_str = qts.quanta_to_time(q)
                sessions_at_q = [
                    ai for ai in ais_in_pool
                    if assignments[ai]["start"] <= q < assignments[ai]["end"]
                ]
                print(
                    f"      q{q} ({day_str} {time_str}): {demand_at_q[q]} sessions "
                    f"need {n_rooms} rooms"
                )
                for ai in sessions_at_q:
                    a = assignments[ai]
                    s = sessions[a["orig_i"]]
                    room_str = rids[a["room"]] if a["room"] >= 0 else "NONE"
                    print(
                        f"        S{a['orig_i']}: {s.course_id}({s.course_type}) "
                        f"q{a['start']}-q{a['end']} room={room_str}"
                    )

    # Also check: could the solver have placed them on different times?
    print()
    print("=" * 80)
    print("TIME FLEXIBILITY ANALYSIS")
    print("=" * 80)
    print("  Phase A assigns time BEFORE rooms — if it creates concurrent demand")
    print("  exceeding room pool size, no room assignment is possible.")
    print()

    for pool_key in sorted(failed_pools):
        pool_names = [rids[r] for r in pool_key]
        ais_in_pool = pool_sessions[pool_key]
        n_rooms = len(pool_key)

        # Build day→quanta demand profile
        day_profiles: dict[str, dict[int, int]] = {}
        for ai in ais_in_pool:
            a = assignments[ai]
            for q in range(a["start"], a["end"]):
                day_str, _ = qts.quanta_to_time(q)
                if day_str not in day_profiles:
                    day_profiles[day_str] = defaultdict(int)
                day_profiles[day_str][q] += 1

        print(f"  Pool {pool_names} ({n_rooms} rooms):")
        for day in qts.DAY_NAMES:
            if day not in day_profiles:
                continue
            profile = day_profiles[day]
            max_d = max(profile.values())
            slots = sorted(profile.keys())
            bar = ""
            for q in slots:
                d = profile[q]
                mark = str(d) if d <= 9 else "+"
                if d > n_rooms:
                    mark = f"[{mark}]"
                bar += mark
            print(f"    {day:10s}: {bar}  (max={max_d}, cap={n_rooms})")


if __name__ == "__main__":
    main()
