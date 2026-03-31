# Quickstart: CP-SAT Soft Constraint Optimizer

**Feature**: 001-cpsat-sc-optimizer

## Prerequisites

- Python 3.12+ with the project's virtual environment activated
- `ortools` package (already in `pyproject.toml`)
- A valid HC-feasible `schedule.json` (produced by `python solve.py`)

## Quick Run

```bash
# 1. Generate an HC-feasible solution (if you don't have one)
python solve.py --gens 100 --pop 50 --seed 42

# 2. Optimize soft constraints (standalone)
python cpsat_sc_optimize.py \
  --input output/ga_adaptive/latest/schedule.json \
  --output schedule_optimized.json \
  --time-limit 60

# 3. Or run the full pipeline with integrated SC optimization
python solve.py --gens 300 --pop 100 --optimize-sc --sc-time-limit 60
```

## Verify Results

```bash
# Check the improvement report printed to console:
#   SC Optimization Report
#   ─────────────────────
#   Constraint    Before   After   Change
#   CSC           45.2     32.1    -29.0%
#   FSC           18.7     15.3    -18.2%
#   MIP            8.0      2.0    -75.0%
#   ...
#   Total         98.4     72.5    -26.3%
#   Hard violations: 0 → 0 ✓
```

## Key Files

| File | Purpose |
|------|---------|
| `src/pipeline/cpsat_sc_optimizer.py` | Core optimizer (SCOptimizer class) |
| `cpsat_sc_optimize.py` | CLI entry point |
| `solve.py` | Pipeline integration (--optimize-sc flag) |
| `tests/test_sc_optimizer.py` | Tests |

## Configuration

Default configuration optimizes all 6 soft constraints with a 60-second budget:

```python
from src.pipeline.cpsat_sc_optimizer import SCOptimizer, SCOptimizerConfig

# Default — all SCs, 60s
config = SCOptimizerConfig()

# Targeted — only student-facing SCs, 30s
config = SCOptimizerConfig(
    time_budget_seconds=30.0,
    target_constraints=["CSC", "MIP"],
)

# Custom weights
config = SCOptimizerConfig(
    weight_overrides={"CSC": 2.0, "MIP": 3.0},  # Prioritize lunch breaks
)
```
