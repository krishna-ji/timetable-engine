# Feature Specification: CP-SAT Soft Constraint Optimizer

**Feature Branch**: `001-cpsat-sc-optimizer`  
**Created**: 2026-04-01  
**Status**: Draft  
**Input**: User description: "Add a third algorithm — a CP-SAT-based soft constraint optimizer that takes HC-feasible solutions from existing algorithms and polishes soft constraint quality, without modifying the existing GA or Adaptive GA algorithms."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run SC Optimizer on an Existing Feasible Solution (Priority: P1)

A scheduler administrator has already generated a hard-constraint-feasible timetable using the existing Adaptive GA algorithm. The overall timetable is valid but students have large gaps between classes and some groups miss their lunch break. The administrator wants to improve soft constraint quality without risking the hard-constraint feasibility they already achieved.

**Why this priority**: This is the core value proposition — taking a valid but "rough" schedule and polishing it for student and instructor comfort. Without this, the feature has no purpose.

**Independent Test**: Run the optimizer on a known HC-feasible solution and verify that (a) hard constraint feasibility is preserved, (b) the aggregate soft penalty score decreases.

**Acceptance Scenarios**:

1. **Given** an HC-feasible timetable (zero hard violations) produced by the GA, **When** the SC optimizer is invoked on that solution, **Then** the output timetable has zero hard violations AND a lower or equal total soft penalty score.
2. **Given** an HC-feasible timetable, **When** the SC optimizer runs for the configured time budget, **Then** it returns the best-found solution even if the time budget is exhausted before full convergence.
3. **Given** an HC-feasible timetable, **When** the optimizer cannot find any soft improvement, **Then** it returns the original solution unchanged.

---

### User Story 2 — Selectively Target Specific Soft Constraints (Priority: P2)

A scheduler administrator notices that student schedule compactness (CSC) and lunch break compliance (MIP) are particularly poor, while other soft constraints are acceptable. They want to focus the optimizer's effort on just those two constraints to get faster, more targeted improvement.

**Why this priority**: Different institutions prioritize different quality aspects. Allowing targeted optimization lets administrators direct compute effort where it matters most, improving turnaround time and relevance of results.

**Independent Test**: Run the optimizer targeting only CSC and MIP on a solution that has poor values for those two constraints, and verify only those penalties decrease while the optimizer completes faster than a full-SC run.

**Acceptance Scenarios**:

1. **Given** an HC-feasible timetable and a configuration specifying only CSC and MIP as target constraints, **When** the SC optimizer runs, **Then** CSC and MIP penalties decrease or remain stable, and the other soft constraint penalties do not worsen significantly (within a configurable tolerance).
2. **Given** a configuration targeting a single soft constraint, **When** the optimizer runs, **Then** improvement effort is concentrated on that constraint.

---

### User Story 3 — Integrate SC Optimizer into the Pipeline as a Post-Processing Phase (Priority: P2)

A scheduler administrator wants to run the entire pipeline end-to-end: first the GA finds a feasible solution, then the SC optimizer automatically polishes it. They do not want to manually export/import intermediate results.

**Why this priority**: Seamless integration reduces operational friction and makes the optimizer accessible as part of the standard workflow rather than requiring manual intervention.

**Independent Test**: Invoke the pipeline with the SC-optimization phase enabled, and verify it produces a final solution that went through both GA and SC optimization stages automatically.

**Acceptance Scenarios**:

1. **Given** the pipeline is configured with SC optimization enabled, **When** the pipeline runs end-to-end, **Then** the GA phase produces an HC-feasible solution and the SC optimizer phase runs automatically on that result.
2. **Given** the pipeline is configured with SC optimization disabled (default), **When** the pipeline runs, **Then** behavior is identical to the current system — no SC optimization phase occurs.

---

### User Story 4 — View SC Improvement Report (Priority: P3)

After the optimizer runs, the administrator wants to see a before/after comparison showing how each soft constraint improved, so they can assess whether the optimization was worthwhile and communicate quality improvements to stakeholders.

**Why this priority**: Observability is important for trust and decision-making, but the optimizer delivers value even without a dedicated report (constraint scores are already visible in existing outputs).

**Independent Test**: Run the optimizer and verify the output includes a per-constraint before/after comparison with percentage changes.

**Acceptance Scenarios**:

1. **Given** the SC optimizer has completed a run, **When** the results are presented, **Then** a summary shows each soft constraint's before and after penalty values and the percentage change.
2. **Given** the optimizer made no improvement on a particular constraint, **When** the report is displayed, **Then** that constraint shows 0% change.

---

### Edge Cases

- What happens when the input solution already has zero soft penalty? The optimizer should detect this and return immediately without unnecessary computation.
- What happens when the input solution has hard constraint violations? The optimizer should reject the input with a clear error — it is not designed to fix hard violations.
- What happens when the time budget runs out before any improvement is found? The optimizer returns the original solution unchanged.
- What happens when the optimizer improves one soft constraint but worsens another? The optimizer should respect the weighted sum objective — individual constraints may shift as long as the total weighted penalty decreases (or the user-targeted subset improves within tolerance).
- What happens when room reassignment is needed to improve soft constraints? The optimizer may reassign rooms as part of its search, provided hard constraint feasibility is maintained.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept any HC-feasible timetable as input to the SC optimizer, regardless of which algorithm produced it (GA, Adaptive GA, or manual construction).
- **FR-002**: System MUST guarantee that the output timetable has zero additional hard constraint violations compared to the input — hard feasibility MUST be preserved.
- **FR-003**: System MUST optimize the weighted sum of all 6 soft constraints: StudentScheduleCompactness (CSC), InstructorScheduleCompactness (FSC), StudentLunchBreak (MIP), SessionContinuity, PairedCohortPracticalAlignment (SSCP), and BreakPlacementCompliance.
- **FR-004**: System MUST allow the user to configure a time budget (in seconds) for the optimization phase, defaulting to a reasonable duration.
- **FR-005**: System MUST return the best-found solution when the time budget expires, even if further improvement is theoretically possible.
- **FR-006**: System MUST allow the user to specify a subset of soft constraints to target, with all 6 targeted by default.
- **FR-007**: System MUST allow the user to configure relative weights for each targeted soft constraint, defaulting to the weights already defined in the constraint system.
- **FR-008**: System MUST be invocable as a standalone post-processing step (given a serialized timetable) and as an integrated pipeline phase after the GA.
- **FR-009**: System MUST produce a per-constraint before/after penalty report upon completion.
- **FR-010**: System MUST reject input solutions that have hard constraint violations, with a clear error message indicating the specific violations found.
- **FR-011**: System MUST NOT modify any code or behavior in the existing GA or Adaptive GA algorithms.
- **FR-012**: System MUST use the existing soft constraint evaluation mechanism (vectorized evaluator and/or OOP evaluator) for scoring, ensuring consistency with how the GA measures soft penalties.

### Key Entities

- **HC-Feasible Solution**: A complete timetable assignment (sessions → time slots, instructors, rooms) with zero hard constraint violations. This is the input to the optimizer.
- **Soft Constraint Penalty**: A numeric score produced by each of the 6 soft constraint evaluators. Lower is better. The optimizer minimizes the weighted sum.
- **Time Budget**: The maximum wall-clock time the optimizer is allowed to spend searching for improvements.
- **SC Optimization Configuration**: The set of parameters controlling the optimizer: time budget, targeted constraints, weight overrides, and whether the phase is enabled in the pipeline.
- **Improvement Report**: A structured before/after comparison of soft constraint penalties produced after each optimization run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The optimizer reduces the total weighted soft penalty by at least 15% on the benchmark dataset compared to the GA-only solution, given a 60-second time budget.
- **SC-002**: Hard constraint feasibility is preserved in 100% of optimizer runs — zero hard violations are introduced.
- **SC-003**: The optimizer completes and returns a result within the configured time budget plus a 5-second grace period for finalization.
- **SC-004**: When targeting only CSC and MIP, those two penalties decrease by at least 20% on the benchmark dataset within a 30-second time budget.
- **SC-005**: End-to-end pipeline runs (GA + SC optimizer) complete without manual intervention or intermediate file handling.
- **SC-006**: The per-constraint improvement report is generated for every optimizer run and shows accurate before/after values matching the evaluator's scoring.

## Assumptions

- The existing 6 soft constraints and their evaluation logic are stable and will not change during the implementation of this feature.
- The existing constraint weights represent reasonable relative priorities; the optimizer will use them as defaults but allow overrides.
- The CP-SAT solver (Google OR-Tools) is already available in the project's dependency set and does not need to be added.
- The benchmark dataset in `data_fixed/` is representative of real-world problem instances for measuring optimizer effectiveness.
- The existing Timetable data structure and serialization format can represent all information the optimizer needs without modification.
- The optimizer is a single-machine, single-process tool — distributed or parallel solving is out of scope for this feature.
- The existing pipeline entry point (`solve.py`) and experiment framework can accommodate an additional post-processing phase without structural overhaul.
