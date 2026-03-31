#!/usr/bin/env python3
"""Investigate Z999 (surveying lab) infeasibility."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cpsat_oracle import build_sessions
from src.io.data_store import DataStore

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, instructor_ids, room_ids = build_sessions(store)

print("Z999 sessions:")
for i, s in enumerate(sessions):
    if 'Z999' in [room_ids[r] for r in s.compatible_room_idxs] and len(s.compatible_room_idxs) == 1:
        insts = [instructor_ids[j] for j in s.qualified_instructor_idxs]
        inst_names = [store.instructors[iid].name for iid in insts]
        print(f"  S{i}: {s.course_id} ({s.course_type}) dur={s.duration} "
              f"groups={s.group_ids} "
              f"instructors={inst_names}")

# Check Z999 room
z999_idx = room_ids.index('Z999')
z999_room = store.rooms['Z999']
print(f"\nRoom Z999: features={z999_room.room_features} "
      f"specific={getattr(z999_room, 'specific_features', [])}")

# Check ENCE 203 course
print("\nCourse ENCE 203 details:")
for ckey, course in store.courses.items():
    if course.course_id == 'ENCE 203':
        print(f"  key={ckey} type={course.course_type} "
              f"quanta={course.quanta_per_week} "
              f"req={course.required_room_features} "
              f"spec={course.specific_lab_features}")
        print(f"  enrolled_groups={course.enrolled_group_ids}")
        print(f"  qualified_instructors={course.qualified_instructor_ids}")
        for iid in course.qualified_instructor_ids:
            inst = store.instructors.get(iid)
            if inst:
                ft = "full-time" if inst.is_full_time else "part-time"
                avail = len(inst.available_quanta)
                print(f"    {iid}: {inst.name} ({ft}, {avail}q available)")

# Total demand for Z999
total_q = 0
for s in sessions:
    if len(s.compatible_room_idxs) == 1 and room_ids[s.compatible_room_idxs[0]] == 'Z999':
        total_q += s.duration
print(f"\nTotal Z999 demand: {total_q}q / 42q available")
print(f"Utilization: {total_q/42*100:.0f}%")
print(f"Each session must NOT overlap: needs {total_q}q spread across 42q")
