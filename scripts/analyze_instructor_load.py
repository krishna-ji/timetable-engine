#!/usr/bin/env python3
"""Analyze instructor workloads from course assignments to derive MaxLoad."""

import json
from collections import defaultdict

# Load data
with open("data_fixed/Instructors.json") as f:
    instructors = json.load(f)
with open("data_fixed/Course.json") as f:
    courses = json.load(f)
with open("data_fixed/Groups.json") as f:
    groups = json.load(f)

# Build course lookup: coursecode -> {L, T, P}
course_map = {}
for c in courses:
    code = c["CourseCode"]
    course_map[code] = {
        "L": c["L"],
        "T": c["T"],
        "P": c.get("P", 0) or 0,
        "title": c["CourseTitle"],
        "dept": c.get("Dept", ""),
    }

# Count how many groups take each course (to understand teaching sections)
course_groups = defaultdict(list)
for g in groups:
    for code in g.get("courses", []):
        course_groups[code].append(g["group_id"])

# Classify instructors
ft_instructors = []
pt_instructors = []

for inst in instructors:
    avail = inst.get("availability", {})
    is_empty = not avail or all(len(v) == 0 for v in avail.values())
    if is_empty:
        ft_instructors.append(inst)
    else:
        pt_instructors.append(inst)

print(f"Total instructors: {len(instructors)}")
print(f"Full-time (empty availability): {len(ft_instructors)}")
print(f"Part-time (has availability):   {len(pt_instructors)}")
print()

# For each instructor, compute their total teaching load
hdr = f"{'ID':<8} {'Name':<35} {'Type':<5} {'#Crs':<6} {'L+T':<6} {'P':<6} {'Tot':<6} {'Courses'}"
print("=" * 120)
print(hdr)
print("=" * 120)

all_loads = []
for inst in instructors:
    avail = inst.get("availability", {})
    is_ft = not avail or all(len(v) == 0 for v in avail.values())

    lt_total = 0
    p_total = 0.0
    course_details = []
    missing = []

    for c in inst.get("courses", []):
        code = c["coursecode"]
        ctype = c["coursetype"]
        if code in course_map:
            cm = course_map[code]
            if ctype == "Theory":
                hours = cm["L"] + cm["T"]
                lt_total += hours
                course_details.append(f"{code}(T:{hours})")
            elif ctype == "Practical":
                hours = cm["P"]
                p_total += hours
                course_details.append(f"{code}(P:{int(hours)})")
        else:
            missing.append(code)
            course_details.append(f"{code}(?)")

    tag = "FT" if is_ft else "PT"
    total = lt_total + p_total
    print(
        f"{inst['id']:<8} {inst['name']:<35} {tag:<5} {len(inst.get('courses', [])):<6} "
        f"{lt_total:<6} {p_total:<6.0f} {total:<6.0f} {', '.join(course_details)}"
    )
    all_loads.append(
        {
            "id": inst["id"],
            "name": inst["name"],
            "ft": is_ft,
            "lt": lt_total,
            "p": p_total,
            "total": total,
            "courses": course_details,
            "n_courses": len(inst.get("courses", [])),
        }
    )

print()
print("=" * 60)
print("STATISTICS")
print("=" * 60)

ft_loads = [x for x in all_loads if x["ft"]]
pt_loads = [x for x in all_loads if not x["ft"]]

if ft_loads:
    ft_totals = [x["total"] for x in ft_loads]
    ft_lt = [x["lt"] for x in ft_loads]
    ft_p = [x["p"] for x in ft_loads]
    print(f"Full-time ({len(ft_loads)} instructors):")
    print(f"  Total weekly load: min={min(ft_totals):.0f}, max={max(ft_totals):.0f}, avg={sum(ft_totals)/len(ft_totals):.1f}")
    print(f"  L+T (lecture):     min={min(ft_lt)}, max={max(ft_lt)}, avg={sum(ft_lt)/len(ft_lt):.1f}")
    print(f"  P (practical):     min={min(ft_p):.0f}, max={max(ft_p):.0f}, avg={sum(ft_p)/len(ft_p):.1f}")

if pt_loads:
    pt_totals = [x["total"] for x in pt_loads]
    pt_lt = [x["lt"] for x in pt_loads]
    pt_p = [x["p"] for x in pt_loads]
    print(f"Part-time ({len(pt_loads)} instructors):")
    print(f"  Total weekly load: min={min(pt_totals):.0f}, max={max(pt_totals):.0f}, avg={sum(pt_totals)/len(pt_totals):.1f}")
    print(f"  L+T (lecture):     min={min(pt_lt)}, max={max(pt_lt)}, avg={sum(pt_lt)/len(pt_lt):.1f}")
    print(f"  P (practical):     min={min(pt_p):.0f}, max={max(ft_p):.0f}, avg={sum(pt_p)/len(pt_loads):.1f}")

# Distribution
print()
print("LOAD DISTRIBUTION (Full-time):")
buckets = defaultdict(int)
for x in ft_loads:
    bucket = int(x["total"] // 5) * 5
    buckets[bucket] += 1
for b in sorted(buckets):
    print(f"  {b:2d}-{b+4:2d} hrs/week: {buckets[b]} instructors")

print()
print("LOAD DISTRIBUTION (Part-time):")
buckets = defaultdict(int)
for x in pt_loads:
    bucket = int(x["total"] // 5) * 5
    buckets[bucket] += 1
for b in sorted(buckets):
    print(f"  {b:2d}-{b+4:2d} hrs/week: {buckets[b]} instructors")

# Show part-time availability slots for reference
print()
print("=" * 60)
print("PART-TIME AVAILABILITY SLOTS (hrs/week):")
print("=" * 60)
for inst in pt_instructors[:20]:
    avail = inst.get("availability", {})
    total_slots = 0
    day_detail = []
    for day, windows in avail.items():
        day_hrs = 0
        for w in windows:
            sh, sm = map(int, w["start"].split(":"))
            eh, em = map(int, w["end"].split(":"))
            hrs = (eh * 60 + em - sh * 60 - sm) / 60
            day_hrs += hrs
        total_slots += day_hrs
        day_detail.append(f"{day[:3]}:{day_hrs:.1f}h")
    print(f"  {inst['id']:<6} {inst['name']:<30} avail={total_slots:.1f}h/wk  [{', '.join(day_detail)}]")

# Top 20 heaviest full-time loads
print()
print("=" * 60)
print("TOP 20 HEAVIEST FULL-TIME LOADS:")
print("=" * 60)
for x in sorted(ft_loads, key=lambda x: -x["total"])[:20]:
    print(f"  {x['id']:<6} {x['name']:<30} L+T={x['lt']:<4} P={x['p']:<4.0f} Total={x['total']:.0f}")
