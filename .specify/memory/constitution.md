<!--
  Sync Impact Report
  ===================
  Version change: 0.0.0 → 1.0.0 (MAJOR — initial ratification)
  Added principles:
    - I. Algorithm Agnosticism
    - II. Dual-Mode Operation (Microservice + Direct CLI)
    - III. Constraint Correctness Is Non-Negotiable
    - IV. Research Reproducibility
    - V. Clean Python & Strict Linting
    - VI. Test Discipline
    - VII. Domain Purity
  Added sections:
    - Communication & Integration Architecture
    - Development Workflow
  Removed sections: none (initial version)
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ no changes needed (generic)
    - .specify/templates/spec-template.md ✅ no changes needed (generic)
    - .specify/templates/tasks-template.md ✅ no changes needed (generic)
  Follow-up TODOs: none
-->

# Timetable Engine Constitution

## Core Principles

### I. Algorithm Agnosticism

The engine MUST treat solver algorithms as **pluggable strategies**,
not hard-coded paths. The current algorithms (Adaptive NSGA-II GA,
CP-SAT via OR-Tools) are implementations of a common interface
pattern, not the only options.

- Every algorithm MUST accept the same domain inputs (`Course`,
  `Group`, `Instructor`, `Room` collections) and produce the same
  output shape (a `Timetable` of session assignments).
- Adding a new algorithm (e.g., simulated annealing, ILP, hybrid
  LNS, tabu search) MUST NOT require changes to domain models,
  constraint definitions, or the service layer.
- Algorithm-specific configuration MUST be isolated — either in
  dedicated config sections or in the algorithm module itself.
- Shared utilities (constraint evaluation, feasibility checking,
  domain parsing) MUST live in common modules (`src/constraints/`,
  `src/domain/`, `src/io/`) reusable by any algorithm.

**Rationale**: This is a research project exploring multiple
optimization approaches for UCTP. Coupling the architecture to one
solver kills experimentation velocity.

### II. Dual-Mode Operation (Microservice + Direct CLI)

The engine MUST support two execution modes with equal priority:

**Microservice mode** — the production path where the EdutableStudio
Next.js frontend communicates with the engine:

- **gRPC** (port 50051) is the primary inter-service protocol.
  It provides streaming progress updates via server-streaming RPCs,
  strong typing via protobuf, and efficient binary serialization
  suited for large schedule payloads.
- **HTTP/REST** (port 8100, FastAPI) is the secondary interface
  for simple requests (health checks, validation, one-shot solves)
  and for debugging/testing from browsers or curl.
- The protobuf schema (`proto/scheduler.proto`) is the contract.
  Both gRPC and HTTP endpoints MUST stay in sync with it.
- CORS is configured for the Next.js frontend origins.

**Direct CLI mode** — the development and research path:

- `python solve.py` runs the GA solver directly with CLI args.
- `python cpsat_solver.py` (or equivalent) runs CP-SAT directly.
- Each algorithm SHOULD expose its own CLI entry point for rapid
  iteration without spinning up servers.
- Environment variables (`SCH_GENS`, `SCH_POP`, `SCH_SEED`, etc.)
  MUST be supported alongside CLI args.

**When to use which**:

| Scenario | Use |
|----------|-----|
| Frontend integration / production | gRPC (streaming progress) |
| Quick debugging from terminal | HTTP REST (`curl`) |
| Algorithm research & benchmarking | Direct CLI (`python solve.py`) |
| Automated CI testing | Direct CLI or HTTP |

**Rationale**: The engine is both a research tool and a deployable
microservice. Forcing server startup for every experiment wastes
time; forcing CLI-only blocks frontend integration.

### III. Constraint Correctness Is Non-Negotiable

Every schedule the engine produces MUST be auditable against the
defined hard constraints. A schedule with **any** hard constraint
violation is considered invalid regardless of soft score.

- Hard constraints (HC1–HC8 as defined in the problem formulation)
  MUST be enforced by every algorithm, either structurally (domain
  encoding) or via repair/rejection.
- The constraint evaluator (`src/constraints/`) is the single
  source of truth. Algorithms MUST NOT implement their own
  constraint-checking logic — they use the shared evaluator.
- Soft constraints are optimization objectives, not correctness
  requirements. A feasible schedule with poor soft scores is valid;
  an infeasible schedule with perfect soft scores is rejected.
- Any new hard constraint MUST be added to the shared evaluator
  first, then enforced across all algorithms.

**Rationale**: UCTP is a constraint satisfaction problem at its
core. A "fast" algorithm that produces invalid timetables is
worthless.

### IV. Research Reproducibility

Every solver run MUST be reproducible given the same inputs and
configuration.

- Random seeds MUST be explicit and logged. Default seed is 42.
- All algorithm parameters MUST be captured in the output
  `results.json` alongside the schedule.
- Input data versions MUST be traceable — the `data/` directory
  is versioned; any data transformations are logged.
- Timestamped output directories (`output/<algorithm>/<timestamp>/`)
  preserve every run's artifacts.
- Experiment scripts (`experiments/`, `runs/`) MUST document their
  parameter sweeps and be runnable end-to-end.

**Rationale**: Research claims require evidence. "It works on my
machine with the right seed" is not evidence.

### V. Clean Python & Strict Linting

All code MUST pass the project's Ruff linter configuration with
zero warnings. The existing `pyproject.toml` linting rules are the
standard.

- **Python 3.12+** is the minimum version. Use modern syntax:
  `match` statements, `type` aliases, `X | Y` unions, f-strings.
- **Ruff** is the linter and formatter. Line length is 88 chars.
  The enabled rule sets (E, W, F, I, N, UP, B, C4, SIM, RUF, PTH,
  PERF, PL, TC, T20, PIE, RET, ICN, A, FLY, LOG) MUST NOT be
  weakened without constitution amendment.
- **Type hints** MUST be used on all public function signatures.
  Internal helpers MAY omit them where types are obvious.
- **Pydantic v2** for data validation at system boundaries (API
  requests/responses, config parsing). Domain models use plain
  dataclasses or named tuples for performance.
- **No stray `print()` in library code** — use the `logging`
  module. CLI entry points and scripts MAY use `print()`.
- **Imports**: `from __future__ import annotations` at the top of
  every module. isort via Ruff handles ordering.

**Rationale**: Consistent style eliminates code review bikeshedding
and prevents common bugs. The rules are already established in the
codebase.

### VI. Test Discipline

Tests live in `tests/` and run via `pytest`.

- **Hard constraint correctness tests** are mandatory for every
  constraint. If a constraint exists, a test MUST verify it catches
  violations and accepts valid schedules.
- **Algorithm smoke tests** MUST exist for each solver: given a
  small input, it produces a feasible schedule within a time bound.
- **Regression tests** MUST be added when a bug is found — the
  failing case becomes a permanent test.
- **Integration tests** cover the server endpoints (gRPC and HTTP)
  ensuring the API contract matches the protobuf schema.
- Tests MUST be fast. Unit tests SHOULD complete in <1 second each.
  Slow algorithm tests SHOULD be marked with `@pytest.mark.slow`.
- **Coverage**: aim for >80% on `src/constraints/` and
  `src/domain/`. Algorithm internals (GA operators, CP-SAT model
  building) are tested via smoke tests rather than line coverage.

**Rationale**: The constraint evaluator is the correctness backbone.
Undertested constraints produce silently wrong schedules.

### VII. Domain Purity

Domain models (`src/domain/`) MUST remain algorithm-agnostic and
serialization-agnostic.

- `Course`, `Group`, `Instructor`, `Room`, `Session`, `Timetable`
  represent the UCTP domain, not any solver's internal encoding.
- Algorithms transform domain models into their internal
  representation (e.g., GA genes, CP-SAT variables) at the
  algorithm boundary — never in the domain layer.
- I/O logic (`src/io/`) handles JSON ↔ domain conversion.
  Domain models MUST NOT import `json`, `grpc`, or `fastapi`.

**Rationale**: Domain models are the stable core shared by all
algorithms and all interfaces. Coupling them to a specific solver
or transport makes adding new algorithms expensive.

## Communication & Integration Architecture

The engine runs as a microservice inside Docker (or directly for
development). The frontend (EdutableStudio, Next.js) communicates
with it as follows:

```
┌─────────────────┐         gRPC :50051          ┌──────────────────┐
│                  │◄────── streaming progress ──►│                  │
│  EdutableStudio  │                              │  Timetable       │
│  (Next.js)       │         HTTP :8100           │  Engine          │
│                  │◄────── REST (JSON) ─────────►│  (Python)        │
└─────────────────┘                               └──────────────────┘
     Frontend                                       This project
```

- **gRPC** is preferred for `RunSchedule` because it streams
  generation-by-generation progress (hard/soft scores, ETA).
  The Next.js app uses `@grpc/grpc-js` or similar client.
- **HTTP** is used for stateless queries: ping, validate, engine
  status, and simple one-shot schedule requests.
- **Protobuf** (`proto/scheduler.proto`) is the single source of
  truth for message shapes. Both servers MUST conform to it.
- New RPC methods MUST be added to the `.proto` file first, then
  implemented in both `grpc_server.py` and `http_server.py`.
- The Docker entrypoint (`entrypoint.sh`) starts both servers.

## Development Workflow

### Running locally (development)

```bash
# Direct CLI — fastest for algorithm work
python solve.py --gens 100 --pop 50 --seed 42

# Start servers locally (for frontend integration)
python http_server.py &   # port 8100
python grpc_server.py &   # port 50051

# Or via Docker
docker build -t timetable-engine .
docker run -p 8100:8100 -p 50051:50051 timetable-engine
```

### Code quality gates

Every change MUST pass before merge:

1. `ruff check .` — zero lint errors
2. `ruff format --check .` — formatting matches
3. `pytest tests/ -x` — all tests pass
4. No new `# noqa` without justification in the commit message

### Dependency management

- **uv** is the package manager (`uv sync`, `uv add`).
- `pyproject.toml` is the single source of dependency truth.
- Pin major versions for scientific libs (numpy, scipy, ortools,
  pymoo) to avoid silent behavioral changes.
- Dev tools (pytest, ruff, mypy) go in `[project.optional-dependencies] dev`.

### Branch and feature workflow

- Use spec-kit feature branches: `###-feature-name`.
- Each feature gets a spec → plan → tasks → implement cycle.
- The `main` branch MUST always have passing CI.

## Governance

This constitution is the authoritative guide for all development
decisions in the timetable-engine project. It supersedes ad-hoc
conventions, chat suggestions, and individual preferences.

- **Amendments** require updating this file with a version bump,
  documenting the change rationale, and verifying all dependent
  templates remain consistent.
- **Version bumps** follow semantic versioning:
  - MAJOR: principle removed or redefined incompatibly
  - MINOR: new principle or section added
  - PATCH: wording clarifications, typo fixes
- **Compliance**: all PRs and code reviews MUST verify adherence.
  The spec-kit `/speckit.analyze` command SHOULD be run before
  implementation to catch drift.
- **Exceptions**: any deviation from a principle MUST be documented
  inline with a `CONSTITUTION-EXCEPTION: <rationale>` comment.

**Version**: 1.0.0 | **Ratified**: 2026-04-01 | **Last Amended**: 2026-04-01
