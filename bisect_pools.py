#!/usr/bin/env python3
"""Bisect: which pool-size-1 cumulative causes infeasibility?"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from ortools.sat.python import cp_model
from cpsat_oracle import build_sessions, build_model
from src.io.data_store import DataStore

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, instructor_ids, room_ids = build_sessions(store)

# Get pool grouping
pool_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
for i, s in enumerate(sessions):
    pool_key = tuple(sorted(s.compatible_room_idxs))
    pool_groups[pool_key].append(i)

# Get model-session mapping from a base model
model, vars_dict = build_model(
    sessions, store, instructor_ids, room_ids,
    relax_sre=True, relax_fca=True, relax_ictd=True,
)
mi = vars_dict["model_indices"]
mi_set = set(mi)
# Map: original session index → model index
orig_to_mi = {}
for model_idx, orig_i in enumerate(mi):
    orig_to_mi[orig_i] = model_idx

# Test each pool individually
print("Testing each pool with cumulative(<=capacity):")
print(f"{'Pool':50s} {'Cap':>4s} {'MSess':>6s} {'TotQ':>6s} {'Result'}")
print("-" * 80)

for pool_key in sorted(pool_groups, key=lambda k: (len(k), k)):
    capacity = len(pool_key)
    # Only test small pools
    if capacity > 5:
        break
    
    ses_idxs = pool_groups[pool_key]
    # Filter to model sessions only
    model_idxs = [orig_to_mi[i] for i in ses_idxs if i in orig_to_mi]
    if len(model_idxs) <= capacity:
        continue
    
    total_q = sum(sessions[mi[m]].duration for m in model_idxs)
    room_names = [room_ids[r] for r in pool_key]
    
    # Build a fresh model and add just this one cumulative
    m2, v2 = build_model(
        sessions, store, instructor_ids, room_ids,
        relax_sre=True, relax_fca=True, relax_ictd=True,
    )
    mi2 = v2["model_indices"]
    iv2 = v2["interval"]
    
    # Map model sessions
    orig_to_mi2 = {}
    for idx, orig_i in enumerate(mi2):
        orig_to_mi2[orig_i] = idx
    
    model_idxs2 = [orig_to_mi2[i] for i in ses_idxs if i in orig_to_mi2]
    intervals = [iv2[m] for m in model_idxs2]
    demands = [1] * len(intervals)
    m2.add_cumulative(intervals, demands, capacity)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_workers = 4
    status = solver.Solve(m2)
    
    status_str = {
        cp_model.OPTIMAL: "FEASIBLE",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE ***",
        cp_model.UNKNOWN: "UNKNOWN (timeout)",
    }.get(status, f"?{status}")
    
    print(f"{str(room_names):50s} {capacity:4d} {len(model_idxs2):6d} {total_q:6d} {status_str}")
