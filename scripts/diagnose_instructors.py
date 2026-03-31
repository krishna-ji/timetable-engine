"""Find instructor bottleneck causing infeasibility."""
from src.io.data_store import DataStore
from src.ga.core.population import (
    generate_course_group_pairs,
    get_subsession_durations,
    analyze_group_hierarchy,
)
from collections import defaultdict

ds = DataStore.from_json("data")
hierarchy = analyze_group_hierarchy(ds.groups)
pairs = generate_course_group_pairs(ds.courses, ds.groups, hierarchy, silent=True)

# For each course, count total quanta needed
course_demand = defaultdict(int)
course_sessions = defaultdict(int)
course_groups = defaultdict(set)

for ck, gids, st, nq in pairs:
    for d in get_subsession_durations(nq, st):
        course_demand[ck] += d
        course_sessions[ck] += 1
    for g in gids:
        course_groups[ck].add(g)

# Find courses with few qualified instructors
print("=== Courses with 1 qualified instructor ===")
for ck in sorted(course_demand.keys()):
    c = ds.courses[ck]
    qual = [iid for iid, inst in ds.instructors.items() if ck in inst.qualified_courses]
    if len(qual) == 1:
        demand = course_demand[ck]
        n_sess = course_sessions[ck]
        inst = ds.instructors[qual[0]]
        flag = " *** OVER ***" if demand > 42 else ""
        print(
            f"  {c.course_code} ({ck[1]}): {n_sess} sess, {demand}q demand, "
            f"{len(course_groups[ck])} groups -> {inst.name}{flag}"
        )

# Instructor total load (from ALL qualified courses, not just sole ones)
# For sole-instructor courses, the load is fixed.
# For multi-instructor courses, the load is shared.
inst_fixed_load = defaultdict(int)  # load from sole-instructor courses
inst_fixed_sessions = defaultdict(list)
inst_total_possible = defaultdict(int)  # max load if they took everything

for ck in sorted(course_demand.keys()):
    qual = [iid for iid, inst in ds.instructors.items() if ck in inst.qualified_courses]
    demand = course_demand[ck]
    if len(qual) == 1:
        inst_fixed_load[qual[0]] += demand
        inst_fixed_sessions[qual[0]].append((ck, demand, course_sessions[ck]))
    for iid in qual:
        inst_total_possible[iid] += demand

print("\n=== Instructors overloaded from sole-instructor courses ===")
for iid in sorted(inst_fixed_load, key=lambda x: -inst_fixed_load[x]):
    load = inst_fixed_load[iid]
    if load <= 42:
        continue
    inst = ds.instructors[iid]
    print(f"\n  {inst.name} ({iid}): {load}/42 quanta *** OVER by {load-42}q ***")
    for ck, d, ns in inst_fixed_sessions[iid]:
        c = ds.courses[ck]
        print(f"    {c.course_code} ({ck[1]}): {ns} sessions, {d}q")

print("\n=== Top 20 instructors by sole-course load ===")
for iid in sorted(inst_fixed_load, key=lambda x: -inst_fixed_load[x])[:20]:
    load = inst_fixed_load[iid]
    inst = ds.instructors[iid]
    total = inst_total_possible.get(iid, 0)
    avail = len(inst.available_quanta) if inst.available_quanta else 42
    flag = " *** OVER ***" if load > 42 else ""
    print(f"  {inst.name}: sole={load}q possible={total}q avail={avail}q{flag}")

# Also check: for courses with 2 instructors, what's the combined constraint?
print("\n=== Instructor pairs under pressure (2-instructor courses) ===")
for ck in sorted(course_demand.keys()):
    qual = [iid for iid, inst in ds.instructors.items() if ck in inst.qualified_courses]
    if len(qual) == 2:
        demand = course_demand[ck]
        if demand > 42:  # Can't even fit with 2 instructors sharing
            c = ds.courses[ck]
            i1 = ds.instructors[qual[0]]
            i2 = ds.instructors[qual[1]]
            r1 = inst_fixed_load.get(qual[0], 0)
            r2 = inst_fixed_load.get(qual[1], 0)
            remaining1 = max(0, 42 - r1)
            remaining2 = max(0, 42 - r2)
            can_cover = remaining1 + remaining2
            flag = " *** CAN'T COVER ***" if can_cover < demand else ""
            print(
                f"  {c.course_code}: {demand}q demand, "
                f"{i1.name}(free={remaining1}q) + "
                f"{i2.name}(free={remaining2}q) = {can_cover}q capacity{flag}"
            )
