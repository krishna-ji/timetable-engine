#!/usr/bin/env python3
"""Debug: Tek Bahadur load analysis — DELETE after use."""
import json
from collections import defaultdict

instructors = json.load(open("data/Instructors.json"))
courses = json.load(open("data/Course.json"))
groups = json.load(open("data/Groups.json"))
cd = {c["CourseCode"]: c for c in courses}

# Find Tek Bahadur
tek = [i for i in instructors if "Tek Bahadur" in i["name"]][0]
print("=" * 70)
print(f"TEK BAHADUR BUDHATHOKI  id={tek['id']}")
print(f"Availability: {tek.get('availability', 'FULL-TIME (no restriction)')}")
print(f"Courses assigned: {len(tek['courses'])}")
print("=" * 70)

total_sessions = 0
for c in tek["courses"]:
    code = c["coursecode"]
    ctype = c["coursetype"]
    cdata = cd.get(code, {})
    title = cdata.get("CourseTitle", "???")
    L = cdata.get("L", 0)
    P = cdata.get("P", 0)
    hrs = L if ctype == "Theory" else P

    # Which groups take this course?
    grps = [g["group_id"] for g in groups if code in g.get("courses", [])]
    sessions = hrs * len(grps) if ctype == "Theory" else hrs  # theory = all groups same event
    # Actually theory is ONE event for all groups, but the instructor needs
    # to be available for those quanta. For FTE, the issue is when multiple
    # theory events overlap.
    print(f"\n  {code} ({ctype}): \"{title}\"")
    print(f"    Hours: L={L} T={cdata.get('T',0)} P={P}")
    print(f"    Groups: {len(grps)} -> {grps}")
    print(f"    Quanta needed: {hrs}")
    total_sessions += hrs

    # Who ELSE is qualified for this exact (code, type)?
    others = []
    for inst in instructors:
        if inst["id"] == tek["id"]:
            continue
        for ic in inst.get("courses", []):
            if ic["coursecode"] == code and ic["coursetype"] == ctype:
                pt = "PT" if inst.get("availability") else "FT"
                others.append(f"{inst['name']} ({pt})")
    if others:
        print(f"    Also qualified: {others}")
    else:
        print(f"    Also qualified: *** NONE — SOLE INSTRUCTOR ***")

print(f"\n{'='*70}")
print(f"TOTAL quanta Tek Bahadur must teach: {total_sessions}")
print(f"Available quanta/week: ~76 (6 days x ~12.7 quanta)")
print(f"Overload: {total_sessions - 76} quanta" if total_sessions > 76 else "Fits")

# Now do the same for top-5 overloaded
print(f"\n{'='*70}")
print("TOP OVERLOADED INSTRUCTORS — SOLE vs SHARED")
print("=" * 70)

inst_load = {}
for inst in instructors:
    total = 0
    sole_count = 0
    shared_count = 0
    for c in inst.get("courses", []):
        code = c["coursecode"]
        ctype = c["coursetype"]
        cdata = cd.get(code, {})
        hrs = cdata.get("L", 0) if ctype == "Theory" else cdata.get("P", 0)
        total += hrs
        # Check if sole instructor
        qualified = sum(
            1 for other in instructors
            if any(ic["coursecode"] == code and ic["coursetype"] == ctype
                   for ic in other.get("courses", []))
        )
        if qualified == 1:
            sole_count += 1
        else:
            shared_count += 1
    inst_load[inst["name"]] = (total, sole_count, shared_count, inst.get("availability"))

for name, (total, sole, shared, avail) in sorted(inst_load.items(), key=lambda x: -x[1][0])[:10]:
    pt = "PT" if avail else "FT"
    print(f"  {name} ({pt}): {total} quanta/wk, {sole} sole courses, {shared} shared courses")
