# Implementation Plan: CP-SAT Soft Constraint Optimizer

**Branch**: `001-cpsat-sc-optimizer` | **Date**: 2026-04-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-cpsat-sc-optimizer/spec.md`

## Summary

Add a third algorithm — a CP-SAT-based post-processing optimizer that takes any HC-feasible timetable and minimizes the weighted sum of all 6 soft constraints (CSC, FSC, MIP, SessionContinuity, SSCP, BreakPlacementCompliance) while structurally preserving hard constraint feasibility. The approach builds a full CP-SAT model with all hard constraints encoded as structural constraints and all soft constraints as linearized penalty terms in the objective, warm-started via `model.add_hint()` from the input solution. This leverages CP-SAT's internal LNS for optimization without modifying the existing GA or Adaptive GA algorithms.

## Technical Context

**Language/Version**: Python 3.12+  
**Primary Dependencies**: ortools (CP-SAT solver), numpy (for vectorized evaluation), pydantic (config validation)  
**Storage**: JSON files (schedule.json input/output)  
**Testing**: pytest  
**Target Platform**: Linux server (Docker) + local development  
**Project Type**: CLI tool + library module integrated into existing pipeline  
**Performance Goals**: ≥15% SC penalty reduction within 60s time budget on benchmark dataset  
**Constraints**: Must complete within configured time budget + 5s grace; must preserve 100% HC feasibility  
**Scale/Scope**: 790 sessions, 42 quanta/week, ~120 groups, ~60 instructors, ~45 rooms

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Algorithm Agnosticism** | ✅ PASS | New algorithm is a pluggable post-processor accepting Timetable input. Uses shared domain models and constraint evaluator. No changes to existing algorithms. |
| **II. Dual-Mode Operation** | ✅ PASS | Standalone CLI (`cpsat_sc_optimize.py`) for research + pipeline integration (`--optimize-sc` flag) for end-to-end runs. No new server endpoints required (post-processing only). |
| **III. Constraint Correctness** | ✅ PASS | All 9 hard constraints encoded in CP-SAT model. Output validated via shared `Evaluator`. Input rejected if HC-infeasible. |
| **IV. Research Reproducibility** | ✅ PASS | Seed parameter, all config captured in result, improvement report saved alongside schedule. |
| **V. Clean Python & Strict Linting** | ✅ PASS | Type hints on all public APIs, Ruff-compliant, no print() in library code (uses logging). |
| **VI. Test Discipline** | ✅ PASS | Smoke test (small input → SC improvement), HC preservation test, regression tests for edge cases. Marked `@pytest.mark.slow` for solver tests. |
| **VII. Domain Purity** | ✅ PASS | No domain model changes. Optimizer takes Timetable, returns Timetable. CP-SAT encoding stays in pipeline layer. |

**Post-Design Re-check**: All gates still pass after Phase 1 design. No constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-cpsat-sc-optimizer/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: design decisions and rationale
├── data-model.md        # Phase 1: entities and relationships
├── quickstart.md        # Phase 1: how to run the optimizer
├── contracts/
│   └── cli-contract.md  # CLI interface and Python API contract
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
# New files (this feature)
cpsat_sc_optimize.py                    # CLI entry point (standalone)
src/pipeline/cpsat_sc_optimizer.py      # Core optimizer module
tests/test_sc_optimizer.py              # Tests

# Modified files (this feature)
solve.py                                # Add --optimize-sc, --sc-time-limit, --sc-target flags

# Existing files (unchanged, consumed by optimizer)
src/constraints/constraints.py          # 6 soft constraint classes (read-only)
src/constraints/evaluator.py            # Shared evaluator for validation (read-only)
src/domain/timetable.py                 # Timetable class (read-only)
src/domain/gene.py                      # SessionGene class (read-only)
src/io/data_store.py                    # DataStore loading (read-only)
src/io/time_system.py                   # QuantumTimeSystem (read-only)
src/ga/core/population.py               # Session generation utilities (read-only)
cpsat_phase1.py                         # Reference for CP-SAT model patterns (read-only)
```

**Structure Decision**: Single project layout, following existing conventions. The optimizer is a new module in `src/pipeline/` alongside `soft_evaluator_vectorized.py` and `scheduling_problem.py`. CLI entry point at repo root follows the existing pattern of `solve.py` and `cpsat_phase1.py`.

## Complexity Tracking

No constitution violations — this section intentionally empty.
