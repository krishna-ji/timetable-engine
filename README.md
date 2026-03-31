# Schedule Engine

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-grade university course scheduling engine using **Adaptive NSGA-II** — a multi-objective genetic algorithm with stagnation-aware mutation escalation and constraint-guided repair.

## Features

- **Multi-objective optimization**: Simultaneously minimizes hard constraint violations and soft preference penalties
- **Adaptive stagnation detection**: Automatically escalates mutation rate and activates elite repair when search stalls
- **9 hard constraints**: Time clashes, room conflicts, instructor overlap, capacity, qualification, etc.
- **6 soft constraints**: Instructor daily load, room utilization, schedule compactness, etc.
- **Parallel repair**: Multi-process elite repair using bitset-accelerated operators
- **Rich output**: JSON schedules, PDF calendars (student/instructor/room views), convergence plots, violation reports

## Quick Start

```bash
# Install dependencies
pip install -e .
# or with uv:
uv sync

# Run the scheduler
python solve.py

# Custom parameters
python solve.py --gens 300 --pop 100 --seed 42

# Skip PDF generation (faster)
python solve.py --no-pdf
```

## Output

Each run creates a timestamped directory under `output/ga_adaptive/` containing:

| File | Description |
|------|-------------|
| `results.json` | Full metrics, convergence data, configuration |
| `schedule.json` | Best timetable in machine-readable format |
| `*.pdf` | Schedule calendar views + convergence plots |
| `violation_report.txt` | Residual constraint violations |

## Input Data

Place JSON files in `data/`:

| File | Description |
|------|-------------|
| `Course.json` | Course definitions (theory/practical, credits, groups) |
| `Groups.json` | Student group definitions and sizes |
| `Instructors.json` | Instructor availability and qualifications |
| `Rooms.json` | Room capacity, type, and availability |

## Project Structure

```
schedule-engine/
├── solve.py              # CLI entry point
├── src/
│   ├── domain/           # Domain models (Course, Group, Instructor, Room)
│   ├── constraints/      # Hard & soft constraint definitions + evaluator
│   ├── ga/               # GA operators, repair, metrics
│   │   ├── core/         # Population, evaluator, encoding
│   │   ├── operators/    # Crossover, mutation, local search
│   │   ├── repair/       # Constraint repair (basic, heuristic, bitset)
│   │   └── metrics/      # Hypervolume, diversity, convergence
│   ├── pipeline/         # pymoo integration (NSGA-II problem, operators)
│   ├── experiments/      # Experiment runner (AdaptiveExperiment)
│   ├── io/               # Data loading, export (JSON, PDF, CSV)
│   ├── config/           # Configuration loader
│   └── utils/            # Logging, console, parallel workers
├── data/                 # Input JSON (courses, groups, instructors, rooms)
├── tests/                # Test suite
├── runs/                 # Run scripts
└── docs/                 # API reference, architecture guide
```

## Configuration

The adaptive GA uses these defaults (override via `AdaptiveExperiment` kwargs):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Population size |
| `ngen` | 300 | Number of generations |
| `crossover_prob` | 0.5 | Crossover probability |
| `mutation_event_prob` | 0.05 | Base mutation rate |
| `stagnation_window` | 15 | Generations before escalation |
| `mutation_hi` | 0.20 | Escalated mutation rate |
| `elite_pct` | 0.10 | Fraction of population to repair |
| `repair_iters` | 5 | Repair passes per elite individual |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Tech Stack

- **Python 3.12+**
- **GA**: pymoo 0.6.1.3, NumPy, SciPy
- **JIT**: Numba (vectorized constraint evaluation)
- **Visualization**: matplotlib, seaborn
- **Config**: Pydantic 2.x
- **CLI**: Rich (progress bars, tables)

## License

MIT
