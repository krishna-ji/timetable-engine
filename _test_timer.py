#!/usr/bin/env python3
"""Quick test: does the solver respect time limits with room_pool_limit=5?"""
import sys, time
sys.path.insert(0, ".")
from ortools.sat.python import cp_model
from src.io.data_store import DataStore
import cpsat_phase1 as cp1

store = DataStore.from_json("data_fixed", run_preflight=False)
sessions, inst_ids, room_ids = cp1.build_sessions(store, cross_qualify=True)
model, vd = cp1.build_phase1_model(
    sessions, store, inst_ids, room_ids,
    relax_pmi=True, no_rooms=True, room_pool_limit=5,
)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_workers = 1
solver.parameters.log_search_progress = False

t0 = time.time()
status = solver.Solve(model)
elapsed = time.time() - t0

names = {0: "UNKNOWN", 1: "MODEL_INVALID", 2: "FEASIBLE", 3: "INFEASIBLE", 4: "OPTIMAL"}
print(f"Status: {names.get(status, status)}")
print(f"Wall time: {elapsed:.2f}s (limit: 30s)")
print(f"Solver walltime: {solver.wall_time:.2f}s")
print(f"Branches: {solver.num_branches}")
print(f"Conflicts: {solver.num_conflicts}")
