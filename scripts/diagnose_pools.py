#!/usr/bin/env python3
"""Diagnostic: Room pool sizes and tight-pool room analysis."""
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cpsat_oracle import build_sessions
from src.io.data_store import DataStore

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, instructor_ids, room_ids = build_sessions(store)

print(f"Sessions: {len(sessions)}, Rooms: {len(room_ids)}")

# Pool size distribution
pool_sizes = [len(s.compatible_room_idxs) for s in sessions]
dist = Counter(pool_sizes)
print("\nPool size distribution:")
for sz in sorted(dist.keys()):
    print(f"  pool={sz:3d}: {dist[sz]:4d} sessions")

# Tight-pool rooms (pool <= threshold)
for threshold in [5, 10, 15]:
    tight_rooms = set()
    tight_sessions = 0
    for s in sessions:
        if len(s.compatible_room_idxs) <= threshold:
            tight_sessions += 1
            tight_rooms.update(s.compatible_room_idxs)
    print(f"\nThreshold={threshold}: {tight_sessions} sessions use {len(tight_rooms)} tight rooms")
    # Show which rooms
    for ridx in sorted(tight_rooms):
        rid = room_ids[ridx]
        room = store.rooms[rid]
        n_sessions = sum(1 for s in sessions if ridx in s.compatible_room_idxs and len(s.compatible_room_idxs) <= threshold)
        print(f"  {rid:8s} features={room.room_features:10s} specific={getattr(room, 'specific_features', [])}  ({n_sessions} tight sessions)")

# Room type counts
types = Counter()
for rid in room_ids:
    types[store.rooms[rid].room_features] += 1
print(f"\nRoom types: {dict(types)}")

# Per course_type session counts
ct = Counter(s.course_type for s in sessions)
print(f"Session types: {dict(ct)}")

# Check: which practical sessions have pool > 23+7=30?
print("\nPractical sessions with pool > 30:")
for s in sessions:
    if s.course_type == "practical" and len(s.compatible_room_idxs) > 30:
        print(f"  {s.course_id} pool={len(s.compatible_room_idxs)}")

# Check: which theory sessions have pool < 45?
print("\nTheory sessions with pool < 40:")
for s in sessions:
    if s.course_type == "theory" and len(s.compatible_room_idxs) < 40:
        print(f"  {s.course_id} pool={len(s.compatible_room_idxs)}")
