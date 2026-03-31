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
    parser.add_argument(
        "--optimize-sc",
        action="store_true",
        help="Run CP-SAT soft constraint optimization after GA",
    )
    parser.add_argument(
        "--sc-time-limit",
        type=float,
        default=60.0,
        help="Time budget for SC optimizer in seconds (default: 60)",
    )
    parser.add_argument(
        "--sc-target",
        default=None,
        help="Comma-separated SC names to target (default: all)",
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

    # ── SC Optimization Phase ──
    if args.optimize_sc:
        from src.domain.gene import set_time_system
        from src.io.data_store import DataStore
        from src.pipeline.cpsat_sc_optimizer import (
            SCOptimizer,
            SCOptimizerConfig,
            load_schedule_json,
        )

        # Find latest schedule.json from GA output
        output_dirs = sorted(Path("output/ga_adaptive").glob("*/schedule.json"))
        if not output_dirs:
            print("Warning: No schedule.json found — skipping SC optimization.")
        else:
            schedule_path = Path(output_dirs[-1])
            output_dir = schedule_path.parent

            store = DataStore.from_json("data_fixed")
            set_time_system(store.qts)

            target_scs = None
            if args.sc_target:
                target_scs = [s.strip() for s in args.sc_target.split(",")]

            config = SCOptimizerConfig(
                time_budget_seconds=args.sc_time_limit,
                target_constraints=target_scs,
                seed=args.seed,
            )

            genes = load_schedule_json(schedule_path, store)
            from src.domain.timetable import Timetable

            context = store.to_context()
            input_tt = Timetable(genes, context, store.qts)

            optimizer = SCOptimizer(data_store=store)
            try:
                opt_result = optimizer.optimize(input_tt, config)
                report = optimizer._format_report(opt_result, config)
                SCOptimizer.print_console_report(report)

                # Save optimized schedule alongside original
                import json

                optimized_path = output_dir / "schedule_optimized.json"
                qts = store.qts
                entries = []
                for gene in opt_result.output_timetable.genes:
                    decoded = qts.decode_schedule(set(gene.get_quanta_list()))
                    entries.append(
                        {
                            "course_id": gene.course_id,
                            "course_type": gene.course_type,
                            "instructor_id": gene.instructor_id,
                            "group_ids": gene.group_ids,
                            "room_id": gene.room_id,
                            "start_quanta": gene.start_quanta,
                            "num_quanta": gene.num_quanta,
                            "co_instructor_ids": gene.co_instructor_ids or [],
                            "time": decoded,
                        }
                    )
                optimized_path.write_text(
                    json.dumps(entries, indent=2, ensure_ascii=False)
                )
                print(f"Optimized schedule saved to {optimized_path}")

                # Save report
                report_path = output_dir / "sc_improvement_report.json"
                SCOptimizer.save_json_report(report, report_path)
                print(f"SC report saved to {report_path}")
            except ValueError as e:
                print(f"SC optimization failed: {e}")


if __name__ == "__main__":
    main()
