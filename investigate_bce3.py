#!/usr/bin/env python3
"""Check BCE3 group load and Z999 scheduling pressure."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cpsat_oracle import build_sessions
from src.io.data_store import DataStore

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, instructor_ids, room_ids = build_sessions(store)

# Total quanta per BCE3 group
bce3_groups = ['BCE3A', 'BCE3B', 'BCE3C', 'BCE3D', 'BCE3E', 'BCE3F']
group_load = defaultdict(int)
group_sessions = defaultdict(list)

for i, s in enumerate(sessions):
    for gid in s.group_ids:
        if gid in bce3_groups:
            group_load[gid] += s.duration
            group_sessions[gid].append(i)

print("BCE3 group loads:")
for g in sorted(bce3_groups):
    print(f"  {g}: {group_load[g]}q / 42q  ({len(group_sessions[g])} sessions, "
          f"{group_load[g]/42*100:.0f}% utilization)")

# Check: are there sessions shared across multiple BCE3 groups?
print("\nMulti-BCE3-group sessions:")
for i, s in enumerate(sessions):
    bce3_in = [g for g in s.group_ids if g in bce3_groups]
    if len(bce3_in) > 1:
        print(f"  S{i}: {s.course_id} ({s.course_type}) dur={s.duration} groups={bce3_in}")

# Check Z999 instructor load
print("\nZ999 instructors' total teaching load:")
z999_instructors = set()
for i, s in enumerate(sessions):
    if len(s.compatible_room_idxs) == 1:
        rname = room_ids[s.compatible_room_idxs[0]]
        if rname == 'Z999':
            z999_instructors.update(s.qualified_instructor_idxs)

for iidx in sorted(z999_instructors):
    iid = instructor_ids[iidx]
    inst = store.instructors[iid]
    total_q = 0
    n_sess = 0
    for i, s in enumerate(sessions):
        if iidx in s.qualified_instructor_idxs:
            total_q += s.duration
            n_sess += 1
    avail = len(inst.available_quanta)
    print(f"  {inst.name} ({iid}): qualified for {n_sess} sessions, "
          f"{total_q}q total load, {avail}q available")

# What other sessions do BCE3 groups have that are at pool-size-1 rooms?
print("\nBCE3 sessions in pool-size-1 rooms:")
for i, s in enumerate(sessions):
    bce3_in = [g for g in s.group_ids if g in bce3_groups]
    if bce3_in and len(s.compatible_room_idxs) == 1:
        rname = room_ids[s.compatible_room_idxs[0]]
        print(f"  S{i}: {s.course_id} ({s.course_type}) dur={s.duration} "
              f"room={rname} groups={bce3_in}")
