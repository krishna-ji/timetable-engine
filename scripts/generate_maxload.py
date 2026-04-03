#!/usr/bin/env python3
"""Generate updated Instructors.json with maxLoad field.

For each instructor, maxLoad = {
    "lecture": sum of (L+T) for all Theory courses they teach,
    "practical": sum of P for all Practical courses they teach
}

Full-time teachers (empty availability) → available all 6 days
Part-time teachers (has availability)   → capped by their time windows
"""

import json
from pathlib import Path

DATA_DIR = Path("data_fixed")

with open(DATA_DIR / "Instructors.json") as f:
    instructors = json.load(f)
with open(DATA_DIR / "Course.json") as f:
    courses = json.load(f)

# Build course lookup
course_map = {}
for c in courses:
    course_map[c["CourseCode"]] = {
        "L": c["L"],
        "T": c["T"],
        "P": c.get("P", 0) or 0,
    }

updated = []
for inst in instructors:
    avail = inst.get("availability", {})
    is_ft = not avail or all(len(v) == 0 for v in avail.values())

    lt_total = 0
    p_total = 0

    for c in inst.get("courses", []):
        code = c["coursecode"]
        ctype = c["coursetype"]
        if code not in course_map:
            continue
        cm = course_map[code]
        if ctype == "Theory":
            lt_total += cm["L"] + cm["T"]
        elif ctype == "Practical":
            p_total += int(cm["P"])

    # For part-time: also compute availability cap (total available hours)
    avail_cap = None
    if not is_ft:
        total_avail_minutes = 0
        for day, windows in avail.items():
            for w in windows:
                sh, sm = map(int, w["start"].split(":"))
                eh, em = map(int, w["end"].split(":"))
                total_avail_minutes += (eh * 60 + em) - (sh * 60 + sm)
        # Convert to quanta (1 quantum = ~45-50 min in this system)
        # But we store as hours for clarity
        avail_cap = round(total_avail_minutes / 60, 1)

    new_inst = dict(inst)
    new_inst["maxLoad"] = {
        "lecture": lt_total,
        "practical": p_total,
    }
    if avail_cap is not None:
        new_inst["availableHoursPerWeek"] = avail_cap

    updated.append(new_inst)

# Write output
out_path = DATA_DIR / "Instructors.json"
with open(out_path, "w") as f:
    json.dump(updated, f, indent=2)

print(f"Updated {len(updated)} instructors with maxLoad field")
print(f"Written to {out_path}")

# Summary
ft = [i for i in updated if not i.get("availability") or all(len(v) == 0 for v in i.get("availability", {}).values())]
pt = [i for i in updated if i not in ft]

print(f"\nFull-time ({len(ft)}):")
ft_lec = [i["maxLoad"]["lecture"] for i in ft]
ft_pra = [i["maxLoad"]["practical"] for i in ft]
print(f"  Lecture  maxLoad: min={min(ft_lec)}, max={max(ft_lec)}, avg={sum(ft_lec)/len(ft_lec):.1f}")
print(f"  Practical maxLoad: min={min(ft_pra)}, max={max(ft_pra)}, avg={sum(ft_pra)/len(ft_pra):.1f}")

print(f"\nPart-time ({len(pt)}):")
pt_lec = [i["maxLoad"]["lecture"] for i in pt]
pt_pra = [i["maxLoad"]["practical"] for i in pt]
print(f"  Lecture  maxLoad: min={min(pt_lec)}, max={max(pt_lec)}, avg={sum(pt_lec)/len(pt_lec):.1f}")
print(f"  Practical maxLoad: min={min(pt_pra)}, max={max(pt_pra)}, avg={sum(pt_pra)/len(pt_pra):.1f}")

# Show sample entries
print("\nSample entries:")
for inst in updated[:5]:
    print(f"  {inst['id']}: {inst['name']}")
    print(f"    maxLoad: {inst['maxLoad']}")
    if "availableHoursPerWeek" in inst:
        print(f"    availableHoursPerWeek: {inst['availableHoursPerWeek']}")
