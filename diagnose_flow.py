"""Instructor capacity network flow analysis.

Checks whether there's enough aggregate instructor capacity to cover all sessions,
treating it as a bipartite matching / max-flow problem.
"""
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

# Build session-level data  
# Each sub-session is a "task" that needs exactly 1 instructor for `duration` quanta
sessions = []  # (course_key, session_type, duration, group_ids, qualified_instructors)
for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    qual = [iid for iid, inst in ds.instructors.items() if ck in inst.qualified_courses]
    for d in get_subsession_durations(nq, st):
        sessions.append((ck, st, d, gids, qual))

total_demand = sum(s[2] for s in sessions)
print(f"Total sessions: {len(sessions)}")
print(f"Total demand: {total_demand} quanta")

# Instructor capacities (available quanta)
inst_capacity = {}
for iid, inst in ds.instructors.items():
    if inst.is_full_time:
        inst_capacity[iid] = 42
    else:
        inst_capacity[iid] = len(inst.available_quanta)

total_supply = sum(inst_capacity.values())
print(f"Total instructor supply: {total_supply} quanta ({len(inst_capacity)} instructors)")
print(f"Utilization: {total_demand/total_supply*100:.1f}%")

# Group sessions by their set of qualified instructors (pool)
pool_data = defaultdict(lambda: {"demand": 0, "sessions": 0, "courses": set()})
for ck, st, d, gids, qual in sessions:
    pool_key = frozenset(qual)
    pool_data[pool_key]["demand"] += d
    pool_data[pool_key]["sessions"] += 1
    pool_data[pool_key]["courses"].add(ck)

print(f"\nUnique instructor pools: {len(pool_data)}")

# For each pool, check if the pool's total capacity covers the demand
print("\n=== Instructor pools where demand > pool capacity ===")
overloaded_pools = 0
for pool_key, info in sorted(pool_data.items(), key=lambda x: -x[1]["demand"]):
    pool_cap = sum(inst_capacity.get(iid, 0) for iid in pool_key)
    if info["demand"] > pool_cap:
        overloaded_pools += 1
        courses = info["courses"]
        sample = list(pool_key)[:3]
        names = [ds.instructors[iid].name for iid in sample]
        print(f"  Pool({len(pool_key)} inst, e.g. {names}): "
              f"{info['sessions']} sessions, {info['demand']}q > {pool_cap}q capacity")
        for ck in sorted(courses):
            c = ds.courses[ck]
            print(f"    {c.course_code} ({ck[1]})")

if overloaded_pools == 0:
    print("  No individual pool is overloaded.")
    print("  Infeasibility comes from OVERLAPPING instructor pools.")

# Check overlapping pools — instructors shared between pools
print("\n=== Instructor-level aggregate demand ===")
inst_demand = defaultdict(int)
inst_courses = defaultdict(set)
for ck, st, d, gids, qual in sessions:
    for iid in qual:
        inst_demand[iid] += d
        inst_courses[iid].add(ck)

print("Instructors where demand from all qualified courses > capacity:")
overloaded_instructors = 0
for iid in sorted(inst_demand, key=lambda x: -inst_demand[x]):
    cap = inst_capacity.get(iid, 42)
    demand = inst_demand[iid]
    if demand > cap:
        overloaded_instructors += 1
        inst = ds.instructors[iid]
        ratio = demand / cap
        n_courses = len(inst_courses[iid])
        print(f"  {inst.name}: demand={demand}q cap={cap}q ratio={ratio:.1f}x courses={n_courses}")

print(f"\nOverloaded instructors: {overloaded_instructors}")

# Check the tightest constraint: for theory sessions, if ALL sibling groups
# share the same session, only 1 instructor-slot is needed
# But for practical sessions, each group needs its own session
print("\n=== Checking by session type ===")
theory_demand = sum(d for ck, st, d, gids, qual in sessions if st == "theory")
practical_demand = sum(d for ck, st, d, gids, qual in sessions if st == "practical")
print(f"  Theory: {sum(1 for ck,st,d,g,q in sessions if st=='theory')} sessions, {theory_demand}q")
print(f"  Practical: {sum(1 for ck,st,d,g,q in sessions if st=='practical')} sessions, {practical_demand}q")

# Now check the REAL constraint: for groups sharing theory sessions,
# the instructor teaching a shared theory session is "locked" during those quanta
# for ALL groups in that session. Let me compute the effective constraint.

# Theory sessions that span many groups 
print("\n=== Theory sessions with most groups ===")
theory_multi = [(ck, st, d, gids, qual) for ck, st, d, gids, qual in sessions if st == "theory" and len(gids) > 2]
for ck, st, d, gids, qual in sorted(theory_multi, key=lambda x: -len(x[3]))[:10]:
    c = ds.courses[ck]
    print(f"  {c.course_code}: {len(gids)} groups, {d}q, {len(qual)} instructors")

# Check if groups with near-100% utilization share all sessions with common instructors
print("\n=== Tight groups and their instructor overlap ===")
bce3_groups = [g for g in ds.groups if g.startswith("BCE3")]
print(f"BCE3 groups: {bce3_groups}")

bce3_sessions = []
bce3_instructors = set()
for ck, st, d, gids, qual in sessions:
    if any(g in bce3_groups for g in gids):
        bce3_sessions.append((ck, st, d, gids, qual))
        bce3_instructors.update(qual)

bce3_demand = sum(d for ck, st, d, gids, qual in bce3_sessions)
print(f"BCE3 total sessions: {len(bce3_sessions)}, demand: {bce3_demand}q")
print(f"BCE3 unique qualified instructors: {len(bce3_instructors)}")

# Of those instructors, how much other load do they carry?
bce3_inst_other = {}
for iid in bce3_instructors:
    other_demand = 0
    for ck, st, d, gids, qual in sessions:
        if iid in qual and not any(g in bce3_groups for g in gids):
            other_demand += d
    bce3_inst_other[iid] = other_demand

# This is a rough analysis — the real constraint is much more complex
