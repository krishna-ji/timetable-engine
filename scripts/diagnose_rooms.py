"""Diagnose room bottlenecks causing structural infeasibility."""
from src.io.data_store import DataStore
from src.ga.core.population import (
    generate_course_group_pairs,
    get_subsession_durations,
    analyze_group_hierarchy,
)
from src.utils.room_compatibility import is_room_suitable_for_course
from collections import defaultdict

ds = DataStore.from_json("data")
hierarchy = analyze_group_hierarchy(ds.groups)
pairs = generate_course_group_pairs(ds.courses, ds.groups, hierarchy, silent=True)
rooms_list = list(ds.rooms.values())

# Map sessions to compatible rooms
room_exclusive_demand = defaultdict(int)  # room -> quanta from sessions with only 1 room
room_exclusive_sessions = defaultdict(list)

pool_demand = defaultdict(lambda: {"sessions": [], "total_q": 0})

for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    compat = tuple(sorted(
        r.room_id for r in rooms_list
        if is_room_suitable_for_course(
            c.required_room_features, r.room_features,
            c.specific_lab_features or None, r.specific_features or None
        )
    ))
    for d in get_subsession_durations(nq, st):
        pool_demand[compat]["sessions"].append((ck, st, d, gids))
        pool_demand[compat]["total_q"] += d
        if len(compat) == 1:
            room_exclusive_demand[compat[0]] += d
            room_exclusive_sessions[compat[0]].append((ck, st, d, gids))

print("=== EXCLUSIVE room assignments (only 1 compatible room) ===")
for rid in sorted(room_exclusive_demand, key=lambda x: -room_exclusive_demand[x]):
    total_q = room_exclusive_demand[rid]
    sessions = room_exclusive_sessions[rid]
    r = ds.rooms[rid]
    flag = " *** OVER ***" if total_q > 42 else ""
    print(f"  {r.name} ({rid}): {len(sessions)} sessions, {total_q}/42q{flag}")
    if total_q > 42:
        for ck, st, d, gids in sessions:
            c = ds.courses[ck]
            print(f"    {c.course_code} {st} {d}q groups={gids}")

print()
print("=== Room pools where demand > supply ===")
overloaded_pools = []
for pool, info in sorted(pool_demand.items(), key=lambda x: -x[1]["total_q"]):
    supply = len(pool) * 42
    if info["total_q"] > supply:
        overloaded_pools.append((pool, info))
        room_names = [ds.rooms[rid].name for rid in pool[:3]]
        extra = f"... +{len(pool)-3}" if len(pool) > 3 else ""
        tq = info["total_q"]
        ns = len(info["sessions"])
        print(f"  Pool({len(pool)} rooms: {room_names}{extra}): {ns} sessions, {tq}q > {supply}q supply")
        for ck, st, d, gids in info["sessions"][:5]:
            c = ds.courses[ck]
            print(f"    {c.course_code} {st} {d}q groups={gids}")
        if ns > 5:
            print(f"    ... and {ns - 5} more")

if not overloaded_pools:
    print("  No individual pool is overloaded!")
    print("  Infeasibility must come from OVERLAPPING room pools competing for shared rooms.")

# Check overlapping pools
print()
print("=== Overlapping pool analysis ===")
# For each room, sum demand from ALL pools that include it
room_total_demand = defaultdict(int)
room_pools = defaultdict(list)
for pool, info in pool_demand.items():
    for rid in pool:
        room_total_demand[rid] += info["total_q"]
        room_pools[rid].append((pool, info["total_q"], len(info["sessions"])))

print("Top 20 most demanded rooms:")
for rid in sorted(room_total_demand, key=lambda x: -room_total_demand[x])[:20]:
    r = ds.rooms[rid]
    td = room_total_demand[rid]
    np = len(room_pools[rid])
    flag = " *** OVER ***" if td > 42 else ""
    print(f"  {r.name} ({rid}, {r.room_features}, spec={r.specific_features}): "
          f"demand={td}q from {np} pools{flag}")

# Check: for practical rooms, how many sessions compete for rooms with specific features?
print()
print("=== Specific feature demand vs supply ===")
feat_rooms = defaultdict(set)
for r in rooms_list:
    for sf in r.specific_features:
        feat_rooms[sf].add(r.room_id)

feat_demand = defaultdict(int)
feat_sessions = defaultdict(int)
for ck, gids, st, nq in pairs:
    c = ds.courses[ck]
    if c.specific_lab_features:
        for d in get_subsession_durations(nq, st):
            for sf in c.specific_lab_features:
                feat_demand[sf] += d
                feat_sessions[sf] += 1

for feat in sorted(feat_demand, key=lambda x: -feat_demand[x]):
    d = feat_demand[feat]
    n_rooms = len(feat_rooms.get(feat, set()))
    supply = n_rooms * 42
    ns = feat_sessions[feat]
    flag = " *** OVER ***" if d > supply else ""
    print(f"  {feat}: {ns} sessions, {d}q demand, {n_rooms} rooms, {supply}q supply{flag}")
