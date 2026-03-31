"""Diagnose structural infeasibility."""
from src.io.data_store import DataStore
from src.ga.core.population import (
    generate_course_group_pairs,
    get_subsession_durations,
    analyze_group_hierarchy,
)
from src.utils.room_compatibility import is_room_suitable_for_course
from collections import defaultdict, Counter

ds = DataStore.from_json("data")
ctx = ds.to_context()
hierarchy = analyze_group_hierarchy(ds.groups)
pairs = generate_course_group_pairs(ds.courses, ds.groups, hierarchy, silent=True)
rooms_list = list(ds.rooms.values())

# ── Group loads ──────────────────────────────────────────────────────
group_load = defaultdict(int)
for ck, gids, st, nq in pairs:
    for d in get_subsession_durations(nq, st):
        for gid in gids:
            group_load[gid] += d

print("=== Top 20 group loads ===")
for gid, load in sorted(group_load.items(), key=lambda x: -x[1])[:20]:
    print(f"  {gid}: {load}/42 ({100*load/42:.0f}%)")
print(f"Overloaded: {sum(1 for l in group_load.values() if l > 42)} / {len(group_load)}")

# ── Room demand / supply ─────────────────────────────────────────────
room_features_cnt = Counter(r.room_features for r in rooms_list)
print("\n=== Room supply by room_features ===")
for rf, cnt in room_features_cnt.most_common():
    print(f"  {rf}: {cnt} rooms × 42q = {cnt*42}q")

# Track required room type per course
demand_q = Counter()
demand_sess = Counter()
for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    req_type = c.required_room_features
    for d in get_subsession_durations(nq, st):
        demand_q[req_type] += d
        demand_sess[req_type] += 1

print("\n=== Room-quanta demand by required_room_features ===")
for rt in sorted(set(list(demand_q.keys()) + list(room_features_cnt.keys()))):
    d = demand_q.get(rt, 0)
    rooms = room_features_cnt.get(rt, 0)
    s = rooms * 42
    pct = d / s * 100 if s > 0 else float("inf")
    flag = " *** OVER ***" if d > s else ""
    print(f"  {rt}: demand={d}q supply={s}q ({pct:.0f}% util) [{rooms} rooms, {demand_sess.get(rt,0)} sessions]{flag}")

# ── Room compatibility: how many rooms can each session actually use? ─
session_room_counts = []
for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    compat = sum(1 for r in rooms_list if is_room_suitable_for_course(
        c.required_room_features, r.room_features,
        c.specific_lab_features or None, r.specific_features or None))
    for d in get_subsession_durations(nq, st):
        session_room_counts.append((ck, st, d, compat, gids))

print("\n=== Sessions with fewest compatible rooms ===")
for ck, st, dur, compat, gids in sorted(session_room_counts, key=lambda x: x[3])[:20]:
    c = ds.courses[ck]
    print(f"  {c.course_code} ({st}, {dur}q, groups={gids}): {compat} rooms")

# ── Instructor bottleneck: courses with few qualified instructors ────
print("\n=== Courses with <=2 qualified instructors ===")
for ck in sorted(ds.courses.keys()):
    c = ds.courses[ck]
    qual = [i for i in ctx.instructors if ck in i.qualified_courses]
    if len(qual) <= 2:
        print(f"  {c.course_code} ({ck}): {len(qual)} instructors")

# ── Peak concurrent demand by compatible room pools ──────────────────
# Group sessions by their room pool (frozenset of compatible room IDs)
pool_groups = defaultdict(list)
for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    compat = frozenset(r.room_id for r in rooms_list if is_room_suitable_for_course(
        c.required_room_features, r.room_features,
        c.specific_lab_features or None, r.specific_features or None))
    for d in get_subsession_durations(nq, st):
        pool_groups[compat].append((ck, st, d, gids))

print("\n=== Room pools with highest contention ===")
for pool, sessions in sorted(pool_groups.items(), key=lambda x: -len(x[1]))[:10]:
    total_q = sum(s[2] for s in sessions)
    pool_size = len(pool)
    supply = pool_size * 42
    util = total_q / supply * 100 if supply > 0 else float("inf")
    flag = " *** OVER ***" if total_q > supply else ""
    sample_rooms = sorted(pool)[:3]
    print(f"  Pool({pool_size} rooms, e.g. {sample_rooms}): {len(sessions)} sessions, {total_q}q demand, {supply}q supply ({util:.0f}%){flag}")

# ── Check for groups that share ALL the same sessions (sibling groups) ──
# These groups must not overlap, constraining things further
print("\n=== Sibling group clusters (share all theory sessions) ===")
group_theory_courses = defaultdict(set)
for ck, gids, st, nq in pairs:
    if st == "theory" and len(gids) > 1:
        for gid in gids:
            group_theory_courses[gid].add(ck)

# Find groups that always appear together in theory
from itertools import combinations
group_list = sorted(group_load.keys())
co_occurrences = defaultdict(int)
total_sessions_per_group = defaultdict(int)
for ck, gids, st, nq in pairs:
    for g1 in gids:
        total_sessions_per_group[g1] += len(get_subsession_durations(nq, st))
        for g2 in gids:
            if g1 < g2:
                co_occurrences[(g1,g2)] += len(get_subsession_durations(nq, st))

# Groups that always co-occur: their combined load determines if they can fit
print("  Checking groups that share sessions and combined constraints...")
tight_groups = [(gid, load) for gid, load in group_load.items() if load >= 35]
tight_groups.sort(key=lambda x: -x[1])
for gid, load in tight_groups[:10]:
    print(f"  {gid}: {load}/42 q, {total_sessions_per_group[gid]} sub-sessions")
