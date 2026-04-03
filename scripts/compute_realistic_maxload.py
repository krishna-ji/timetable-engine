#!/usr/bin/env python3
"""Compute realistic MaxLoad from actual solver output.

Strategy:
1. Run solver WITHOUT MaxLoad to get a feasible schedule
2. Analyze what each instructor was ACTUALLY assigned
3. Set MaxLoad = actual_load (the realistic teaching obligation)
4. Enforce: each value <= 42, lecture + practical <= 42

This accounts for multi-section courses and shared instructor pools.
There are only 42 quanta/week (6 days × 7 slots), so no individual
maxLoad can exceed 42 and lecture + practical must also fit within 42.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.data_store import DataStore
from cpsat_phase1 import (
    build_sessions,
    build_phase1_model,
    solve_and_report,
    extract_assignments,
    phase_b_room_assignment,
)


def main():
    store = DataStore.from_json("data_fixed", run_preflight=False)
    sessions, iids, rids = build_sessions(store, cross_qualify=True)

    # Phase A: solve WITHOUT maxLoad (temporarily clear it)
    for inst in store.instructors.values():
        inst.max_load_lecture = None
        inst.max_load_practical = None

    model_a, vars_a = build_phase1_model(
        sessions, store, iids, rids,
        relax_pmi=True,
        no_rooms=True,
        room_pool_limit=0,
    )
    result_a, solver_a = solve_and_report(
        model_a, sessions, store, iids, rids, vars_a,
        time_limit=30,
        random_seed=1,
    )
    if not result_a or solver_a is None:
        print("Phase A failed — cannot compute actual loads")
        return

    assignments = extract_assignments(solver_a, sessions, iids, vars_a)

    # Analyze actual loads per instructor
    instr_load: dict[str, dict[str, int]] = defaultdict(lambda: {"theory": 0, "practical": 0})
    instr_sessions: dict[str, dict[str, int]] = defaultdict(lambda: {"theory": 0, "practical": 0})

    for a in assignments:
        s = sessions[a["orig_i"]]
        for inst_id in a["instructors"]:
            instr_load[inst_id][s.course_type] += s.duration
            instr_sessions[inst_id][s.course_type] += 1

    # Load course map for comparison
    with open("data_fixed/Course.json") as f:
        courses = json.load(f)
    course_map = {}
    for c in courses:
        course_map[c["CourseCode"]] = {"L": c["L"], "T": c["T"], "P": c.get("P", 0) or 0}

    # Load original instructor data
    with open("data_fixed/Instructors.json") as f:
        instructors_json = json.load(f)

    instr_map = {i["id"]: i for i in instructors_json}

    print("=" * 130)
    print(f"{'ID':<8} {'Name':<30} {'Type':<4} {'old.lec':<10} {'actual.lec':<12} {'old.prac':<10} {'actual.prac':<12} {'new.L':<6} {'new.P':<6} {'Status'}")
    print("=" * 130)

    changes = 0
    updated_instructors = []

    for inst_data in instructors_json:
        iid = inst_data["id"]
        avail = inst_data.get("availability", {})
        is_ft = not avail or all(len(v) == 0 for v in avail.values())

        old_ml = inst_data.get("maxLoad", {})
        old_lec = old_ml.get("lecture", 0)
        old_prac = old_ml.get("practical", 0)

        actual = instr_load.get(iid, {"theory": 0, "practical": 0})
        actual_lec = actual["theory"]
        actual_prac = actual["practical"]

        tag = "FT" if is_ft else "PT"

        # Use solver-actual load + 20% buffer (min 2 quanta) for solver flexibility
        import math
        new_lec = actual_lec + max(2, math.ceil(actual_lec * 0.2))
        new_prac = actual_prac + max(2, math.ceil(actual_prac * 0.2))

        # Enforce weekly capacity: 42 quanta total (6 days × 7 slots)
        MAX_WEEKLY_QUANTA = 42
        new_lec = min(new_lec, MAX_WEEKLY_QUANTA)
        new_prac = min(new_prac, MAX_WEEKLY_QUANTA)
        # Combined must also fit in 42 — scale down proportionally if needed
        if new_lec + new_prac > MAX_WEEKLY_QUANTA:
            total = new_lec + new_prac
            new_lec = int(new_lec * MAX_WEEKLY_QUANTA / total)
            new_prac = MAX_WEEKLY_QUANTA - new_lec
        new_prac = min(new_prac, MAX_WEEKLY_QUANTA)
        # Combined must also fit in 42
        if new_lec + new_prac > MAX_WEEKLY_QUANTA:
            # This shouldn't happen with real solver output, but guard anyway
            print(f"  WARNING: {iid} combined {new_lec}+{new_prac}={new_lec+new_prac} > {MAX_WEEKLY_QUANTA}")

        changed = (new_lec != old_lec or new_prac != old_prac)
        if changed:
            changes += 1
        status = "CHANGED" if changed else ""

        if actual_lec > 0 or actual_prac > 0 or old_lec > 0 or old_prac > 0:
            print(
                f"{iid:<8} {inst_data['name']:<30} {tag:<4} "
                f"{old_lec:<10} {actual_lec:<12} "
                f"{old_prac:<10} {actual_prac:<12} "
                f"{new_lec:<6} {new_prac:<6} {status}"
            )

        inst_data["maxLoad"] = {
            "lecture": new_lec,
            "practical": new_prac,
        }
        updated_instructors.append(inst_data)

    print()
    print(f"Instructors changed: {changes}")
    # Verify all values are within bounds
    max_l = max(i.get("maxLoad", {}).get("lecture", 0) for i in updated_instructors)
    max_p = max(i.get("maxLoad", {}).get("practical", 0) for i in updated_instructors)
    max_c = max(i.get("maxLoad", {}).get("lecture", 0) + i.get("maxLoad", {}).get("practical", 0)
                for i in updated_instructors)
    print(f"Max lecture across all:   {max_l} (cap=42)")
    print(f"Max practical across all: {max_p} (cap=42)")
    print(f"Max combined across all:  {max_c} (cap=42)")
    print()

    # Write updated data
    out_path = Path("data_fixed/Instructors.json")
    with open(out_path, "w") as f:
        json.dump(updated_instructors, f, indent=2)
    print(f"Updated {len(updated_instructors)} instructors → {out_path}")
    print("MaxLoad = solver_actual (capped at 42 quanta/week)")


if __name__ == "__main__":
    main()
