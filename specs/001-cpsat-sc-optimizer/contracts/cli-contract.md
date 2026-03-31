# CLI Contract: CP-SAT Soft Constraint Optimizer

**Feature**: 001-cpsat-sc-optimizer  
**Interface**: Command-line entry point

## Standalone CLI (`cpsat_sc_optimize.py`)

```
python cpsat_sc_optimize.py [OPTIONS]

Required:
  --input PATH           Path to HC-feasible schedule.json (or Timetable pickle)

Optional:
  --data-dir DIR         Data directory for domain loading (default: data_fixed/)
  --output PATH          Output path for optimized schedule.json (default: stdout)
  --time-limit SECONDS   Solver time budget (default: 60)
  --seed INT             Random seed (default: 42)
  --workers INT          CP-SAT parallel workers (default: 8)
  --target-sc SC[,SC]    Comma-separated SC names to target (default: all)
                         Valid: CSC,FSC,MIP,SessionContinuity,SSCP,BreakPlacementCompliance
  --weight KEY=VAL       Override weight for a specific SC (repeatable)
  --relax-ictd           Relax SpreadAcrossDays hard constraint
  --quiet                Suppress progress logging
  --report PATH          Write JSON improvement report to PATH
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — optimized solution written |
| 1 | Input validation failed (hard violations detected) |
| 2 | Solver returned INFEASIBLE — model error (should not happen with valid input) |
| 3 | No improvement found — original solution returned |

### Example Usage

```bash
# Optimize all SCs with 60s budget
python cpsat_sc_optimize.py --input output/ga_adaptive/latest/schedule.json

# Target only CSC and MIP with 30s budget
python cpsat_sc_optimize.py --input schedule.json --target-sc CSC,MIP --time-limit 30

# Full pipeline: GA + SC optimization
python solve.py --gens 300 --pop 100 && \
python cpsat_sc_optimize.py --input output/ga_adaptive/latest/schedule.json \
                            --output output/ga_adaptive/latest/schedule_optimized.json

# With improvement report
python cpsat_sc_optimize.py --input schedule.json --report improvement_report.json
```

## Pipeline Integration (`solve.py`)

```
python solve.py [EXISTING_FLAGS] [--optimize-sc] [--sc-time-limit SECONDS] [--sc-target SC[,SC]]

New flags:
  --optimize-sc          Enable SC optimization phase after GA (default: disabled)
  --sc-time-limit INT    Time budget for SC optimizer (default: 60)
  --sc-target SC[,SC]    SCs to target in post-optimization (default: all)
```

When `--optimize-sc` is provided:

1. GA runs normally, producing best HC-feasible solution
2. SC optimizer runs on that solution
3. Both pre- and post-optimization schedules are saved
4. Improvement report is logged to console and saved to output directory

## Python API

```python
from src.pipeline.cpsat_sc_optimizer import SCOptimizer, SCOptimizerConfig

config = SCOptimizerConfig(
    time_budget_seconds=60.0,
    target_constraints=["CSC", "MIP"],  # or None for all
    seed=42,
)

optimizer = SCOptimizer(data_store=store)
result = optimizer.optimize(timetable, config)

print(result.improvement_pct)          # e.g., -18.5 (18.5% reduction)
print(result.after_penalties)          # {"CSC": 12.0, "MIP": 3.0, ...}
optimized_tt = result.output_timetable # Timetable object
```
