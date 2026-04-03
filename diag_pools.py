"""Diagnose pool sizes and session counts for cumulative constraint design."""
from cpsat_solver import SolveConfig
from cpsat_phase1 import build_sessions
from src.io.data_store import DataStore
from collections import defaultdict

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, iids, rids = build_sessions(store, cross_qualify=True)

pool_to_sessions = defaultdict(list)
for i, s in enumerate(sessions):
    pk = tuple(sorted(s.compatible_room_idxs))
    pool_to_sessions[pk].append(i)

print(f"{'Pool Size':>10} {'Sessions':>10} {'Ratio':>8} {'Room IDs'}")
print("-" * 80)
total_sessions_in_tight = 0
for pk in sorted(pool_to_sessions.keys(), key=lambda k: len(pool_to_sessions[k]), reverse=True):
    pool_size = len(pk)
    n_sessions = len(pool_to_sessions[pk])
    ratio = n_sessions / pool_size
    room_names = [rids[r] for r in pk]
    tight = "*" if ratio > 2 else ""
    if ratio > 2:
        total_sessions_in_tight += n_sessions
    print(f"{pool_size:>10} {n_sessions:>10} {ratio:>8.1f} {tight:>2} {room_names}")

print(f"\nTotal sessions: {len(sessions)}")
print(f"Total pools: {len(pool_to_sessions)}")
print(f"Sessions in tight pools (ratio>2): {total_sessions_in_tight}")
