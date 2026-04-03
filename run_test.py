"""Quick test run of solver with cumulative constraints."""
from cpsat_solver import solve_timetable, SolveConfig

cfg = SolveConfig(
    seeds=5,
    cross_qualify=True,
    relax_pmi=True,
    time_limit_phase_a=120,
    time_limit_phase_c=60,
)
r = solve_timetable("data_fixed", config=cfg)

print()
print(f"Success: {r.success}")
print(f"Sessions: {r.total_sessions}")
sched = r.schedule or []
print(f"Rooms assigned: {r.rooms_assigned}")
print(f"Rooms failed: {r.rooms_failed}")
print(f"Violations: {r.violations}")
if r.violation_details:
    for k, v in r.violation_details.items():
        print(f"  {k}: {v}")
print(f"Error: {r.error}")
print(f"Elapsed: {r.elapsed_seconds:.2f}s")
