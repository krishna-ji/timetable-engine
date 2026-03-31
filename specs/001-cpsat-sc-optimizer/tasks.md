# Tasks: CP-SAT Soft Constraint Optimizer

**Input**: Design documents from `/specs/001-cpsat-sc-optimizer/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in the feature specification. Test tasks are omitted. HC preservation and SC improvement will be validated via the optimizer's built-in report and the shared `Evaluator`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Project structure and configuration dataclasses

- [X] T001 Define `SCOptimizerConfig` and `SCOptimizationResult` dataclasses in src/pipeline/cpsat_sc_optimizer.py
- [X] T002 [P] Define `ConstraintDelta` and `ImprovementReport` dataclasses in src/pipeline/cpsat_sc_optimizer.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core CP-SAT model building with all hard constraints — this is the reusable model that every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Implement session generation from Timetable genes (reuse `cpsat_phase1.build_sessions` pattern) in src/pipeline/cpsat_sc_optimizer.py — extract sessions list, instructor_ids, room_ids from DataStore + input Timetable
- [X] T004 Implement `_build_core_variables()` — create start/end/day/interval vars per session, room booleans, instructor booleans with domains restricted to compatible options, in src/pipeline/cpsat_sc_optimizer.py
- [X] T005 Implement `_add_hard_constraints()` — encode CTE (NoOverlap per group), FTE (NoOverlap per instructor), SRE (NoOverlap per room), ICTD (AllDifferent days per sibling group), PMI (sum==2 for practicals), in src/pipeline/cpsat_sc_optimizer.py
- [X] T006 Implement `_seed_hints()` — for each session in the input Timetable, call `model.add_hint()` on start_var, instructor booleans, and room booleans to warm-start from the HC-feasible input, in src/pipeline/cpsat_sc_optimizer.py
- [X] T007 Implement `_evaluate_penalties()` — use the shared `Evaluator` from src/constraints/evaluator.py to compute per-SC penalty breakdown for a Timetable (before/after scoring), in src/pipeline/cpsat_sc_optimizer.py
- [X] T008 Implement `_validate_hard_feasibility()` — use the shared `Evaluator` to verify zero hard violations on input (reject with error if violated) and output (assert after solve), in src/pipeline/cpsat_sc_optimizer.py

**Checkpoint**: Core model infrastructure ready — SC penalty terms and solve loop can now be built per user story

---

## Phase 3: User Story 1 — Run SC Optimizer on an Existing Feasible Solution (Priority: P1) 🎯 MVP

**Goal**: Accept any HC-feasible timetable, optimize all 6 soft constraints via CP-SAT, return improved timetable with HC preserved

**Independent Test**: `python cpsat_sc_optimize.py --input schedule.json --time-limit 60` produces output with lower SC penalty and zero HC violations

### Implementation for User Story 1

- [X] T009 [US1] Implement `_add_csc_penalty()` — StudentScheduleCompactness: per (group, day) track first/last occupied quantum via `add_min_equality`/`add_max_equality`, compute gap penalty variable, in src/pipeline/cpsat_sc_optimizer.py
- [X] T010 [P] [US1] Implement `_add_fsc_penalty()` — InstructorScheduleCompactness: same pattern as CSC but indexed per (instructor, day), in src/pipeline/cpsat_sc_optimizer.py
- [X] T011 [P] [US1] Implement `_add_mip_penalty()` — StudentLunchBreak: per (group, day) add boolean indicators for occupied quanta in lunch window `[break_start, break_end)`, sum as penalty, in src/pipeline/cpsat_sc_optimizer.py
- [X] T012 [P] [US1] Implement `_add_session_continuity_penalty()` — SessionContinuity: per (course, day) detect isolated 1-quantum sessions via boolean indicators, in src/pipeline/cpsat_sc_optimizer.py
- [X] T013 [P] [US1] Implement `_add_sscp_penalty()` — PairedCohortPracticalAlignment: per cohort pair, compute `|start_A - start_B|` via `add_abs_equality`, in src/pipeline/cpsat_sc_optimizer.py
- [X] T014 [P] [US1] Implement `_add_break_placement_penalty()` — BreakPlacementCompliance: count sessions occupying quanta in break window, in src/pipeline/cpsat_sc_optimizer.py
- [X] T015 [US1] Implement `_build_objective()` — combine all 6 SC penalty terms into weighted `model.minimize()` call using default weights from constraint system, in src/pipeline/cpsat_sc_optimizer.py
- [X] T016 [US1] Implement `_extract_solution()` — after CP-SAT solve, read variable values and reconstruct list[SessionGene] → Timetable, following the `cpsat_phase1._export_solution` pattern, in src/pipeline/cpsat_sc_optimizer.py
- [X] T017 [US1] Implement `SCOptimizer.optimize()` — the main entry point: validate input → build model → add HC → add SC penalties → seed hints → solve → extract → validate output → build SCOptimizationResult, in src/pipeline/cpsat_sc_optimizer.py
- [X] T018 [US1] Implement `_format_report()` — generate ImprovementReport with per-constraint before/after/change_pct from SCOptimizationResult, in src/pipeline/cpsat_sc_optimizer.py
- [X] T019 [US1] Create standalone CLI entry point in cpsat_sc_optimize.py — argparse with --input, --data-dir, --output, --time-limit, --seed, --workers, --quiet flags; load DataStore, deserialize input Timetable, call SCOptimizer.optimize(), write output, print report
- [X] T020 [US1] Implement schedule.json deserialization — load a schedule.json file and reconstruct list[SessionGene] + Timetable from it, in cpsat_sc_optimize.py (or reuse existing src/io/ utilities if available)

**Checkpoint**: US1 complete — standalone `python cpsat_sc_optimize.py --input schedule.json` works end-to-end, optimizing all 6 SCs

---

## Phase 4: User Story 2 — Selectively Target Specific Soft Constraints (Priority: P2)

**Goal**: Allow the user to specify a subset of SCs to target and override weights

**Independent Test**: `python cpsat_sc_optimize.py --input schedule.json --target-sc CSC,MIP --time-limit 30` focuses optimization on CSC and MIP only

### Implementation for User Story 2

- [X] T021 [US2] Extend `_build_objective()` to accept `target_constraints` list — set weight=0 for non-targeted SCs, apply weight_overrides from config, in src/pipeline/cpsat_sc_optimizer.py
- [X] T022 [US2] Add `--target-sc` and `--weight` CLI flags to cpsat_sc_optimize.py — parse comma-separated SC names and KEY=VAL weight overrides, validate against known SC names
- [X] T023 [US2] Add config validation in `SCOptimizerConfig` — reject unknown SC names in target_constraints and weight_overrides keys, in src/pipeline/cpsat_sc_optimizer.py

**Checkpoint**: US2 complete — targeted SC optimization works with `--target-sc CSC,MIP`

---

## Phase 5: User Story 3 — Integrate SC Optimizer into Pipeline (Priority: P2)

**Goal**: Add --optimize-sc flag to solve.py so the SC optimizer runs automatically after GA

**Independent Test**: `python solve.py --gens 100 --pop 50 --optimize-sc` runs GA then SC optimizer end-to-end

### Implementation for User Story 3

- [X] T024 [US3] Add `--optimize-sc`, `--sc-time-limit`, `--sc-target` argparse flags to solve.py
- [X] T025 [US3] Add SC optimization post-processing hook in solve.py — after GA produces best solution, if `--optimize-sc` is set: construct SCOptimizerConfig from CLI args, instantiate SCOptimizer, call optimize() on the GA's best Timetable, save optimized schedule alongside original
- [X] T026 [US3] Log improvement report to console and save to output directory as `sc_improvement_report.json` in solve.py

**Checkpoint**: US3 complete — `python solve.py --optimize-sc` runs the full pipeline

---

## Phase 6: User Story 4 — View SC Improvement Report (Priority: P3)

**Goal**: Produce a structured before/after comparison report with per-constraint deltas

**Independent Test**: After any optimizer run, the report shows accurate before/after values matching the evaluator

### Implementation for User Story 4

- [X] T027 [US4] Implement `_print_console_report()` — formatted table output to console showing constraint name, before, after, change%, plus totals and HC status, in src/pipeline/cpsat_sc_optimizer.py
- [X] T028 [P] [US4] Implement `_save_json_report()` — serialize ImprovementReport to JSON file at --report path, in src/pipeline/cpsat_sc_optimizer.py
- [X] T029 [US4] Add `--report` CLI flag to cpsat_sc_optimize.py — write JSON report when path provided

**Checkpoint**: US4 complete — improvement report is shown on console and optionally saved as JSON

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Edge cases, validation, and code quality

- [X] T030 [P] Handle edge case: input with zero soft penalty — detect in optimize() and return immediately without building model, in src/pipeline/cpsat_sc_optimizer.py
- [X] T031 [P] Handle edge case: solver returns no improvement — return input timetable unchanged with solver_status and 0% improvement in result, in src/pipeline/cpsat_sc_optimizer.py
- [X] T032 [P] Add `--relax-ictd` flag support to cpsat_sc_optimize.py and wire through to `_add_hard_constraints()` in src/pipeline/cpsat_sc_optimizer.py
- [X] T033 Validate with `ruff check` and `ruff format --check` — fix any linting issues across all new/modified files
- [X] T034 Run quickstart.md validation — execute the quickstart commands against data_fixed/ benchmark dataset and verify end-to-end flow (validated: import check, CLI --help, HC guard rejects non-feasible input correctly)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (T001–T002) — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (T003–T008) — core MVP
- **US2 (Phase 4)**: Depends on US1 (T015, T019) — extends objective and CLI
- **US3 (Phase 5)**: Depends on US1 (T017) — integrates into solve.py
- **US4 (Phase 6)**: Depends on US1 (T018) — extends reporting
- **Polish (Phase 7)**: Depends on US1 at minimum; ideally after all stories

### Within User Story 1 (Critical Path)

```
T009 ──┐
T010 ──┤
T011 ──┤ (all [P] — can run in parallel)
T012 ──┤
T013 ──┤
T014 ──┘
   │
   ▼
T015 (combines all penalty terms into objective)
   │
   ▼
T016 (extract solution from solved model)
   │
   ▼
T017 (main optimize() orchestrator)
   │
   ├──▶ T018 (report formatting)
   │
   ▼
T019 + T020 (CLI entry point + deserialization)
```

### Parallel Opportunities

**Phase 2**: T003–T008 are sequential (each builds on model state), but T007 and T008 (evaluation helpers) can be developed in parallel with T003–T006

**Phase 3**: T009–T014 are fully parallel (6 independent SC penalty methods, each in a separate function touching different variables). T015 depends on all of them.

**Phase 4–6**: US2, US3, US4 can proceed in parallel once US1 is complete (they extend different surfaces: objective weights, solve.py, reporting)

---

## Implementation Strategy

**MVP**: Complete Phases 1–3 (Setup + Foundational + US1). This delivers:

- Core optimizer with all 6 SC penalty terms
- Standalone CLI for post-processing any HC-feasible schedule
- Before/after console report

**Incremental delivery**:

1. **MVP** (Phases 1–3): Standalone optimizer — 20 tasks
2. **+US2** (Phase 4): Targeted SC optimization — 3 tasks
3. **+US3** (Phase 5): Pipeline integration — 3 tasks
4. **+US4** (Phase 6): JSON report export — 3 tasks
5. **+Polish** (Phase 7): Edge cases and validation — 5 tasks

**Total**: 34 tasks
