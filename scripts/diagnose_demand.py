#!/usr/bin/env python3
"""Diagnose: which room pool's cumulative constraint causes infeasibility?"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cpsat_oracle import build_sessions, Session
from src.io.data_store import DataStore

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, instructor_ids, room_ids = build_sessions(store)

# Group sessions by their exact pool
pool_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
for i, s in enumerate(sessions):
    pool_key = tuple(sorted(s.compatible_room_idxs))
    pool_groups[pool_key].append(i)

print(f"Total sessions: {len(sessions)}")
print(f"Distinct pools: {len(pool_groups)}")
print()

# For each pool, compute demand vs capacity
print(f"{'Pool':60s} {'Cap':>4s} {'#Sess':>6s} {'TotQ':>6s} {'Avail':>6s} {'Status'}")
print("-" * 100)
for pool_key in sorted(pool_groups, key=lambda k: len(k)):
    ses_idxs = pool_groups[pool_key]
    capacity = len(pool_key)
    total_demand = sum(sessions[i].duration for i in ses_idxs)
    available = capacity * 42  # 42 quanta
    status = "OVER" if total_demand > available else "ok"
    
    # Show room names for small pools
    if len(pool_key) <= 5:
        room_names = [room_ids[ridx] for ridx in pool_key]
    else:
        room_names = [f"{len(pool_key)} rooms"]
    
    # Show course ids
    course_ids = sorted(set(sessions[i].course_id for i in ses_idxs))
    courses_str = ", ".join(course_ids[:5])
    if len(course_ids) > 5:
        courses_str += f"... (+{len(course_ids) - 5})"
    
    pool_str = f"{room_names} [{courses_str}]"
    if len(pool_str) > 60:
        pool_str = pool_str[:57] + "..."
    
    flag = "***" if status == "OVER" else ""
    print(f"{pool_str:60s} {capacity:4d} {len(ses_idxs):6d} {total_demand:6d} {available:6d} {status} {flag}")

# Also check: which sessions have the highest duration?
print("\nLongest sessions:")
by_dur = sorted(range(len(sessions)), key=lambda i: -sessions[i].duration)
for i in by_dur[:20]:
    s = sessions[i]
    print(f"  S{i}: {s.course_id} ({s.course_type}) dur={s.duration} "
          f"groups={s.group_ids} pool_size={len(s.compatible_room_idxs)}")
