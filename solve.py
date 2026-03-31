#!/usr/bin/env python3
"""University course scheduling solver.

Runs the Adaptive GA (NSGA-II with stagnation-aware mutation
escalation and elite repair) to produce optimized timetables.

Usage:
    python solve.py
    python solve.py --gens 300 --pop 100 --seed 42
    python solve.py --no-pdf          # skip PDF generation

Environment variables (CLI args override these):
    SCH_GENS   Number of generations (default: 300)
    SCH_POP    Population size (default: 100)
    SCH_SEED   Random seed (default: 42)

Outputs (written to ``output/ga_adaptive/<timestamp>/``):
    - results.json          Full metrics and convergence data
    - schedule.json         Best timetable in machine-readable format
    - Schedule PDFs         Calendar, instructor, and room views
    - Convergence plots     Hard/soft objective evolution
    - Violation report      Residual constraint violations
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def _env_int(key: str, default: int) -> int:
    """Read integer from environment variable, falling back to default."""
    val = os.environ.get(key)
    return int(val) if val is not None else default


def main():
    parser = argparse.ArgumentParser(
        description="University timetable scheduling solver (Adaptive GA)"
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=_env_int("SCH_GENS", 300),
        help="Number of generations (env: SCH_GENS, default: 300)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=_env_int("SCH_POP", 100),
        help="Population size (env: SCH_POP, default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("SCH_SEED", 42),
        help="Random seed (env: SCH_SEED, default: 42)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip schedule PDF generation (faster)",
    )
    parser.add_argument(
        "--force-pdf",
        action="store_true",
        help="Generate schedule PDFs even when no feasible solution exists",
    )
    args = parser.parse_args()

    from src.experiments import AdaptiveExperiment

    exp = AdaptiveExperiment(
        pop_size=args.pop,
        ngen=args.gens,
        seed=args.seed,
        export_pdf=not args.no_pdf,
        force_pdf=args.force_pdf,
        verbose=True,
    )
    exp.run()


if __name__ == "__main__":
    main()
