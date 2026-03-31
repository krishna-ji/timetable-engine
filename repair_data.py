#!/usr/bin/env python3
"""
Data Repair Script for University Course Timetabling.

Creates a fixed copy of the scheduling data under data_fixed/
that resolves the structural infeasibility identified by the CP-SAT oracle.

Root cause: Instructor capacity (FTE + FPC interaction)
- Not enough qualified instructors for the session load
- 2 sole-instructor courses where demand exceeds availability
- 117/189 instructors have aggregate demand > their capacity

Fixes applied (in order):
  1. Immediate data fixes for overloaded sole-instructors
  2. Systematic broadening of instructor qualification pools
  3. Availability expansion for critical part-time instructors
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.data_store import DataStore
from src.ga.core.population import (
    analyze_group_hierarchy,
    generate_course_group_pairs,
    get_subsession_durations,
)


def load_raw_data(data_dir: str = "data"):
    """Load raw JSON files."""
    base = Path(data_dir)
    with open(base / "Instructors.json", encoding="utf-8") as f:
        instructors = json.load(f)
    with open(base / "Course.json", encoding="utf-8") as f:
        courses = json.load(f)
    with open(base / "Groups.json", encoding="utf-8") as f:
        groups = json.load(f)
    with open(base / "Rooms.json", encoding="utf-8") as f:
        rooms = json.load(f)
    return instructors, courses, groups, rooms


def build_indices(instructors, courses):
    """Build lookup dictionaries."""
    inst_by_name = {i["name"]: i for i in instructors}
    inst_by_id = {i["id"]: i for i in instructors}
    course_by_code = {c["CourseCode"]: c for c in courses}

    # course_key → set of instructor IDs
    course_instructors = defaultdict(set)
    for inst in instructors:
        for cq in inst.get("courses", []):
            key = (cq["coursecode"], cq["coursetype"])
            course_instructors[key].add(inst["id"])

    return inst_by_name, inst_by_id, course_by_code, course_instructors


def compute_session_data(data_dir: str = "data"):
    """Compute session-level demand using the engine's own logic."""
    ds = DataStore.from_json(data_dir, run_preflight=False)
    hierarchy = analyze_group_hierarchy(ds.groups)
    pairs = generate_course_group_pairs(ds.courses, ds.groups, hierarchy, silent=True)

    course_demand = defaultdict(int)
    course_sessions = defaultdict(int)
    for ck, gids, st, nq in pairs:
        for d in get_subsession_durations(nq, st):
            course_demand[ck] += d
            course_sessions[ck] += 1

    return course_demand, course_sessions, pairs, ds


def add_qualification(instructor: dict, coursecode: str, coursetype: str):
    """Add a qualification to an instructor if not already present."""
    existing = {(c["coursecode"], c["coursetype"]) for c in instructor.get("courses", [])}
    if (coursecode, coursetype) not in existing:
        instructor["courses"].append({
            "coursecode": coursecode,
            "coursetype": coursetype,
        })
        return True
    return False


def expand_availability(instructor: dict, day: str, start: str, end: str):
    """Add or extend availability on a specific day."""
    if "availability" not in instructor:
        instructor["availability"] = {}
    avail = instructor["availability"]
    if day not in avail:
        avail[day] = []
    # Check if this slot already exists or overlaps
    for slot in avail[day]:
        if slot["start"] == start and slot["end"] == end:
            return False  # already exists
    avail[day].append({"start": start, "end": end})
    return True


def make_full_time(instructor: dict):
    """Make an instructor full-time (clear availability = available everywhere)."""
    old = instructor.get("availability", {})
    instructor["availability"] = {}
    return bool(old)


def main():
    print("=" * 72)
    print("  DATA REPAIR SCRIPT")
    print("=" * 72)

    # Load data
    instructors_raw, courses_raw, groups_raw, rooms_raw = load_raw_data("data")
    instructors = deepcopy(instructors_raw)
    courses = deepcopy(courses_raw)

    inst_by_name, inst_by_id, course_by_code, course_instructors = build_indices(
        instructors, courses
    )

    # Compute actual session demands
    course_demand, course_sessions, pairs, ds = compute_session_data("data")

    fixes_applied = []

    # ── FIX 1: Saroj Shakya — CT 765 07 ─────────────────────────────
    # Problem: Sole instructor, 8q demand, 6q available (Sun+Mon 10-13)
    # Fix: Expand availability to cover full days on Sun+Mon (10-17)
    print("\n[Fix 1] Saroj Shakya — CT 765 07")
    saroj = inst_by_name.get("Saroj Shakya")
    if saroj:
        saroj["availability"] = {
            "Sunday": [{"start": "10:00", "end": "17:00"}],
            "Monday": [{"start": "10:00", "end": "17:00"}],
        }
        fixes_applied.append("Saroj Shakya: expanded availability Sun+Mon 10-17 (was 07-13)")
        print(f"  Expanded availability: Sun+Mon full day (14q, was 6q)")

    # ── FIX 2: Aayush Pudasaini — IE654 ─────────────────────────────
    # Problem: Sole instructor for IE654, 6q demand, ~5q available
    # Fix: Add one more day of availability
    print("\n[Fix 2] Aayush Pudasaini — IE654")
    aayush = inst_by_name.get("Aayush Pudasaini")
    if aayush:
        aayush["availability"] = {
            "Thursday": [{"start": "10:00", "end": "17:00"}],
            "Friday": [{"start": "10:00", "end": "17:00"}],
            "Tuesday": [{"start": "10:00", "end": "17:00"}],
        }
        fixes_applied.append("Aayush Pudasaini: expanded availability to full Tue+Thu+Fri (was partial)")
        print(f"  Expanded availability: Tue+Thu+Fri full day (21q, was ~5q)")

    # ── FIX 3: Systematically broaden instructor pools ───────────────
    # Strategy: For each course with ≤2 qualified instructors, find instructors
    # teaching similar courses (same department/semester) and cross-qualify them.
    print("\n[Fix 3] Broadening instructor qualification pools")

    # Build course metadata
    course_dept = {}
    course_sem = {}
    for c in courses:
        code = c["CourseCode"]
        course_dept[code] = set(d.strip() for d in c.get("Dept", "").split(","))
        course_sem[code] = c.get("Semester", 0)

    # Find courses with ≤2 qualified instructors
    thin_courses = []
    for ck, demand in sorted(course_demand.items(), key=lambda x: -x[1]):
        code, ctype = ck
        key = (code, ctype.capitalize() if ctype[0].islower() else ctype)
        # Map session_type to coursetype format
        ct_lookup = "Theory" if ctype == "theory" else "Practical"
        qualified = course_instructors.get((code, ct_lookup), set())
        if len(qualified) <= 2 and demand > 2:
            thin_courses.append((code, ct_lookup, demand, qualified))

    # For each thin course, find the best candidate instructors to add
    # Candidates: instructors who teach other courses in the same department+semester
    added_qualifications = 0
    for code, ct, demand, existing_qual in thin_courses:
        depts = course_dept.get(code, set())
        sem = course_sem.get(code, 0)
        if not depts or sem == 0:
            continue

        # Find instructors teaching other courses with overlapping dept and nearby semester
        candidates = []
        for inst in instructors:
            iid = inst["id"]
            if iid in existing_qual:
                continue  # already qualified

            # Check if this instructor teaches courses in the same dept/semester range
            inst_courses_in_scope = 0
            for cq in inst.get("courses", []):
                ic_depts = course_dept.get(cq["coursecode"], set())
                ic_sem = course_sem.get(cq["coursecode"], 0)
                if ic_depts & depts and abs(ic_sem - sem) <= 2:
                    inst_courses_in_scope += 1

            if inst_courses_in_scope >= 1:
                # Check instructor has capacity
                is_ft = not inst.get("availability") or inst["availability"] == {}
                capacity = 42 if is_ft else _estimate_quanta(inst)
                candidates.append((inst, capacity, inst_courses_in_scope))

        # Sort candidates: prefer full-time, then most relevant courses
        candidates.sort(key=lambda x: (-int(x[1] >= 42), -x[2], x[0]["id"]))

        # Add up to 2 more instructors per thin course
        added_for_course = 0
        target = max(0, 3 - len(existing_qual))
        for inst, cap, relevance in candidates[:target]:
            if add_qualification(inst, code, ct):
                added_qualifications += 1
                added_for_course += 1
                course_instructors[(code, ct)].add(inst["id"])

        if added_for_course > 0:
            fixes_applied.append(
                f"{code} ({ct}): added {added_for_course} instructors "
                f"(was {len(existing_qual)}, now {len(existing_qual) + added_for_course})"
            )

    print(f"  Added {added_qualifications} cross-qualifications across {len(thin_courses)} thin courses")

    # ── FIX 4: Expand availability of critical part-time instructors ─
    # For part-time instructors who are the sole instructor for any course,
    # ensure they have enough available quanta to cover their sole-course demand.
    print("\n[Fix 4] Expanding critical part-time instructor availability")

    # Rebuild course_instructors after fix 3
    course_instructors_updated = defaultdict(set)
    for inst in instructors:
        for cq in inst.get("courses", []):
            key = (cq["coursecode"], cq["coursetype"])
            course_instructors_updated[key].add(inst["id"])

    # Find sole instructors that are part-time
    sole_pt_fixes = 0
    for ck, demand in course_demand.items():
        code, ctype = ck
        ct_lookup = "Theory" if ctype == "theory" else "Practical"
        qualified = course_instructors_updated.get((code, ct_lookup), set())
        if len(qualified) == 1:
            iid = list(qualified)[0]
            inst = inst_by_id[iid]
            is_ft = not inst.get("availability") or inst["availability"] == {}
            if not is_ft:
                cap = _estimate_quanta(inst)
                if cap < demand:
                    # Make them available for more days
                    old_days = list(inst.get("availability", {}).keys())
                    all_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                    needed_extra = demand - cap
                    extra_quanta = 0
                    for day in all_days:
                        if day not in old_days and extra_quanta < needed_extra + 4:
                            expand_availability(inst, day, "10:00", "17:00")
                            extra_quanta += 7
                    if extra_quanta > 0:
                        sole_pt_fixes += 1
                        fixes_applied.append(
                            f"{inst['name']}: expanded availability by {extra_quanta}q "
                            f"for sole-instructor course {code}"
                        )

    print(f"  Expanded {sole_pt_fixes} part-time instructors")

    # ── FIX 5: For heavily-demanded courses (>20q demand, <5 instructors), ──
    # add 1-2 more full-time instructors
    print("\n[Fix 5] Reinforcing high-demand courses")
    reinforced = 0
    for ck, demand in sorted(course_demand.items(), key=lambda x: -x[1]):
        code, ctype = ck
        ct_lookup = "Theory" if ctype == "theory" else "Practical"
        qualified = course_instructors_updated.get((code, ct_lookup), set())
        if demand >= 16 and len(qualified) < 4:
            depts = course_dept.get(code, set())
            sem = course_sem.get(code, 0)

            # Find full-time instructors in the same department
            for inst in instructors:
                if inst["id"] in qualified:
                    continue
                is_ft = not inst.get("availability") or inst["availability"] == {}
                if not is_ft:
                    continue
                # Same department check
                for cq in inst.get("courses", []):
                    ic_depts = course_dept.get(cq["coursecode"], set())
                    if ic_depts & depts:
                        if add_qualification(inst, code, ct_lookup):
                            qualified.add(inst["id"])
                            course_instructors_updated[(code, ct_lookup)].add(inst["id"])
                            reinforced += 1
                        break
                if len(qualified) >= 4:
                    break

    print(f"  Reinforced {reinforced} high-demand course-instructor pairings")

    # ── Save fixed data ──────────────────────────────────────────────
    out_dir = Path("data_fixed")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "Instructors.json", "w", encoding="utf-8") as f:
        json.dump(instructors, f, indent=2, ensure_ascii=False)

    # Copy unchanged files
    for fname in ["Course.json", "Groups.json", "Rooms.json"]:
        shutil.copy2(Path("data") / fname, out_dir / fname)

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for fix in fixes_applied:
        print(f"  - {fix}")
    print(f"\n  Total fixes: {len(fixes_applied)}")
    print(f"  Output: {out_dir.resolve()}")
    print("=" * 72)

    # ── Verify with quick stats ──────────────────────────────────────
    print("\n  Verification: re-computing instructor pools...")
    new_course_inst = defaultdict(set)
    for inst in instructors:
        for cq in inst.get("courses", []):
            key = (cq["coursecode"], cq["coursetype"])
            new_course_inst[key].add(inst["id"])

    sole_count = sum(1 for k, v in new_course_inst.items() if len(v) == 1)
    duo_count = sum(1 for k, v in new_course_inst.items() if len(v) == 2)
    print(f"  Sole-instructor courses: {sole_count} (check demand vs capacity)")
    print(f"  Duo-instructor courses: {duo_count}")


def _estimate_quanta(instructor: dict) -> int:
    """Estimate available quanta from availability dict."""
    avail = instructor.get("availability", {})
    if not avail:
        return 42

    total = 0
    for day, slots in avail.items():
        for slot in slots:
            # Parse times
            sh, sm = map(int, slot["start"].split(":"))
            eh, em = map(int, slot["end"].split(":"))
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            # Clip to schedule window: 10:00-17:00
            start_min = max(start_min, 600)  # 10:00
            end_min = min(end_min, 1020)  # 17:00
            if end_min > start_min:
                hours = (end_min - start_min) / 60
                total += int(hours)  # 1 quantum = 1 hour
    return total


if __name__ == "__main__":
    main()
