# Data Model: CP-SAT Soft Constraint Optimizer

**Feature**: 001-cpsat-sc-optimizer  
**Date**: 2026-04-01

## Entities

### SCOptimizerConfig

Configuration for a single SC optimization run.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time_budget_seconds` | float | 60.0 | Max wall-clock time for solver |
| `target_constraints` | list[str] \| None | None (all 6) | SC names to optimize; None = all |
| `weight_overrides` | dict[str, float] \| None | None | Per-SC weight overrides; None = use defaults from constraint system |
| `seed` | int | 42 | Random seed for reproducibility |
| `num_workers` | int | 8 | CP-SAT parallel worker threads |
| `log_progress` | bool | True | Whether to log solver search progress |
| `relax_ictd` | bool | False | Whether to relax SpreadAcrossDays (mirrors cpsat_phase1 flag) |

### SCOptimizationResult

Output of a single optimizer run.

| Field | Type | Description |
|-------|------|-------------|
| `input_timetable` | Timetable | The original HC-feasible timetable |
| `output_timetable` | Timetable | The SC-optimized timetable |
| `before_penalties` | dict[str, float] | Per-SC penalty scores before optimization |
| `after_penalties` | dict[str, float] | Per-SC penalty scores after optimization |
| `total_before` | float | Weighted sum of all SC penalties before |
| `total_after` | float | Weighted sum of all SC penalties after |
| `improvement_pct` | float | Percentage improvement in total weighted penalty |
| `hard_violations_before` | int | Hard violations in input (expected: 0) |
| `hard_violations_after` | int | Hard violations in output (must be: 0) |
| `solver_status` | str | CP-SAT status: OPTIMAL, FEASIBLE, INFEASIBLE, etc. |
| `solve_time_seconds` | float | Actual wall-clock time spent solving |
| `solutions_found` | int | Number of improving solutions found during search |

### ImprovementReport

Human-readable report generated after optimization.

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | str | "CP-SAT SC Optimizer" |
| `timestamp` | str | ISO 8601 timestamp of the run |
| `config` | SCOptimizerConfig | Configuration used |
| `result` | SCOptimizationResult | Full result data |
| `per_constraint_summary` | list[ConstraintDelta] | Per-SC before/after/change |

### ConstraintDelta

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Constraint short name (e.g., "CSC") |
| `before` | float | Penalty before optimization |
| `after` | float | Penalty after optimization |
| `change_pct` | float | Percentage change (negative = improvement) |
| `weight` | float | Weight used in objective |

## Relationships

```
SCOptimizerConfig ──→ SCOptimizer.optimize(timetable, config) ──→ SCOptimizationResult
                                                                        │
                                                                        ▼
                                                                  ImprovementReport
                                                                        │
                                                                        ▼
                                                              list[ConstraintDelta]
```

## State Transitions

```
Input Timetable (HC-feasible, high SC penalty)
        │
        ▼
   [Validation: hard_violations == 0?]
        │ yes                │ no
        ▼                    ▼
   Build CP-SAT Model    REJECT with error
        │
        ▼
   Seed with hints from input
        │
        ▼
   Solve (optimize SC objective)
        │
        ├── OPTIMAL/FEASIBLE → Extract improved timetable
        │                           │
        │                           ▼
        │                    Validate output HC == 0
        │                           │
        │                           ▼
        │                    Return SCOptimizationResult
        │
        └── INFEASIBLE/TIMEOUT with no solution → Return input unchanged
```

## Validation Rules

- `SCOptimizerConfig.time_budget_seconds` must be > 0
- `SCOptimizerConfig.target_constraints` elements must be valid SC names from {CSC, FSC, MIP, SessionContinuity, SSCP, BreakPlacementCompliance}
- `SCOptimizerConfig.weight_overrides` keys must match valid SC names
- `SCOptimizerConfig.seed` must be non-negative
- `SCOptimizerConfig.num_workers` must be >= 1
- Input timetable must have zero hard constraint violations
- Output timetable must have zero hard constraint violations (enforced structurally by CP-SAT model + post-validation)
