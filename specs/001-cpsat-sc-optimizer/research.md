# Research: CP-SAT Soft Constraint Optimizer

**Feature**: 001-cpsat-sc-optimizer  
**Date**: 2026-04-01

## Research Questions

### RQ1: Which approach best optimizes soft constraints from an HC-feasible solution?

**Decision**: **Two-phase approach — CP-SAT with hint-seeded warm start + application-level LNS**

**Rationale**:

The optimizer accepts an HC-feasible timetable (790 sessions, 42 quanta/week, 6 operational days × 7 quanta/day). Three approaches were evaluated:

1. **Embed all SCs into CP-SAT objective (monolithic re-solve)**: Model all 6 SCs as linear penalty terms alongside hard constraints. Use `model.add_hint()` to warm-start from the input solution. Pro: single-pass, leverages CP-SAT's internal LNS. Con: the gap computation (CSC/FSC) requires `min_equality`/`max_equality` per (group × day) — with ~120 groups × 6 days = 720 min/max auxiliary variables plus boolean gap indicators, plus MIP lunch indicators, the model becomes large (already ~790 sessions × 3 variable triples). Risk of timeout on the full model within a 60s budget.

2. **Application-level LNS with CP-SAT sub-solver**: Partition sessions into neighborhoods (e.g., one group's sessions, or one day's sessions), fix everything outside the neighborhood, and re-optimize the neighborhood with SC objective via CP-SAT. Each sub-problem is small (~20-50 sessions) and solves in 1-5s. Iterate until time budget exhausted. Pro: each sub-problem is tractable; can use the full CP-SAT model with SC penalties. Con: requires neighborhood selection strategy; purely local improvements.

3. **Simulated Annealing / Tabu Search post-processor**: Define neighborhood moves (slot swap, slot shift, instructor swap), accept SC-improving moves. Pro: very fast per iteration (no model building). Con: gets stuck in local optima; harder to enforce hard constraint preservation without a satisfaction model.

**Chosen approach: Hybrid (1) + (2)** — Build a full CP-SAT model with all hard constraints + SC penalty objective, seed it with the input solution via `add_hint()`, and run with LNS parameters tuned for optimization. If the full model is tractable within the time budget, we get the globally-optimized result. As a fallback, implement application-level LNS that decomposes by group-day neighborhoods for targeted improvement.

The key insight is that CP-SAT's internal LNS (enabled by default in optimization mode) already does neighborhood decomposition. By providing a good hint (the HC-feasible input), the solver starts from a feasible point and uses its internal LNS to explore SC-improving neighborhoods. The `model.add_hint()` mechanism makes this efficient — the solver doesn't need to find feasibility first.

**Alternatives considered**:

- Pure SA: Rejected because hard constraint enforcement requires expensive validation per move, and SA has no native constraint satisfaction mechanism.
- Separate Phase 2 CP-SAT (fixing HC, only optimizing time slots): Rejected because fixing instructors would prevent instructor-swap moves that can significantly improve FSC.
- Lexicographic optimization (feasibility first, then SC): This is effectively what we're doing — the input is already feasible, so the hint provides the feasibility guarantee and the solver optimizes SC.

### RQ2: How to model each soft constraint as CP-SAT penalty terms?

**Decision**: Linearized penalty terms using auxiliary variables and channeling constraints.

| SC | CP-SAT Modeling | Variables Added |
|----|-----------------|-----------------|
| **CSC** (StudentScheduleCompactness) | Per (group, day): track `first_q` = `model.new_int_var()` with `add_min_equality` over occupied quanta, `last_q` with `add_max_equality`. Gap = `last_q - first_q + 1 - count_occupied`. Penalty = `gap * density_scale`. | 2 IntVars + 1 gap var per (group, day) = ~720 vars |
| **FSC** (InstructorScheduleCompactness) | Same as CSC but per (instructor, day). | ~2 IntVars per (instructor, day) = ~600 vars |
| **MIP** (StudentLunchBreak) | Per (group, day): boolean `lunch_busy[g,d,q]` for each quantum in lunch window `[break_start, break_end)`. Penalty = count of busy quanta. | ~1-2 BoolVars per (group, day, lunch_quantum) |
| **SessionContinuity** | Per (course, day): count isolated 1-quantum sessions. Boolean indicator for each session-day combination. | Small — only relevant for 1-quantum theory remainders |
| **SSCP** (PairedCohortPracticalAlignment) | Per cohort pair sharing a practical: `abs_diff = |start_A - start_B|` via `add_abs_equality`. Penalty =`abs_diff`. | 1 IntVar per cohort pair |
| **BreakPlacementCompliance** | Same modeling as MIP — count occupied quanta in break window. | Reuses MIP indicators with different window |

**Total auxiliary variable overhead**: ~1500-2000 variables on top of the ~2400 core variables (790 sessions × 3). This is well within CP-SAT's capacity (it handles 100K+ variables routinely).

### RQ3: How to guarantee hard constraint preservation?

**Decision**: Encode all 9 hard constraints in the CP-SAT model identically to `cpsat_phase1.py`. The model is infeasible if any HC is violated, so the solver structurally cannot return a solution with HC violations. Additionally, validate the output solution using the shared `Evaluator` before returning.

The hard constraints are:

- CTE (NoStudentDoubleBooking): `add_no_overlap` per group — mandatory interval grouping
- FTE (NoInstructorDoubleBooking): `add_no_overlap` per instructor — optional intervals
- SRE (NoRoomDoubleBooking): `add_no_overlap` per room — optional intervals  
- FPC (InstructorMustBeQualified): Domain restriction on instructor booleans
- FFC (RoomMustHaveFeatures): Domain restriction on room booleans
- FCA (InstructorMustBeAvailable): Can be relaxed per existing Phase 1 pattern
- CQF (ExactWeeklyHours): Structural — session durations are fixed
- ICTD (SpreadAcrossDays): `add_all_different` on day variables per sibling group
- PMI (RequiresTwoInstructors): `sum(instr_bools) == 2` for practicals

### RQ4: How to handle the hint mechanism for warm-starting?

**Decision**: For each session in the input timetable, call `model.add_hint(start_vars[mi], gene.start_quanta)` and `model.add_hint(instr_bool[(mi, iidx)], 1 if iid == gene.instructor_id else 0)` and similarly for rooms. CP-SAT uses hints as a starting point for its internal LNS — the solver is free to move away from the hint to find better SC solutions.

### RQ5: How to integrate into the existing pipeline?

**Decision**:

- **Standalone module**: `src/pipeline/cpsat_sc_optimizer.py` — the core optimizer class.
- **CLI entry point**: `cpsat_sc_optimize.py` at repo root — accepts a `schedule.json` path and outputs an optimized version.
- **Pipeline integration**: Add an optional `--optimize-sc` flag to `solve.py` that runs the optimizer after the GA completes.
- **No modification to GA code**: The optimizer is purely additive — it reads a Timetable, optimizes it, and writes a new Timetable.

### RQ6: What time budget and solver parameters are optimal?

**Decision**: Default time budget = 60s. CP-SAT parameters:

- `solver.parameters.max_time_in_seconds = time_budget`
- `solver.parameters.num_workers = 8` (use available cores for parallel LNS)
- `solver.parameters.log_search_progress = True` (for progress reporting)
- `solver.parameters.random_seed = seed` (reproducibility per Constitution IV)

For targeted optimization (subset of SCs), set weight=0 for non-targeted SCs in the objective. This allows the solver to freely move non-targeted SC values while focusing on the targeted ones.
