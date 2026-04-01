#!/usr/bin/env python3
"""CP-SAT Soft Constraint Optimizer — standalone CLI.

Takes an HC-feasible schedule.json and minimizes weighted soft constraint
penalties using CP-SAT.  Outputs the optimized schedule to stdout or file.

Usage:
    python cpsat_sc_optimize.py --input schedule.json
    python cpsat_sc_optimize.py --input schedule.json --target-sc CSC,MIP --time-limit 30
    python cpsat_sc_optimize.py --input schedule.json --output optimized.json --report report.json

Exit codes:
    0  Success — optimized solution written
    1  Input validation failed (hard violations detected)
    2  Solver returned INFEASIBLE
    3  No improvement found — original solution returned
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CP-SAT Soft Constraint Optimizer for HC-feasible timetables"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to HC-feasible schedule.json",
    )
    parser.add_argument(
        "--data-dir",
        default="data_fixed",
        help="Data directory for domain loading (default: data_fixed/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for optimized schedule.json (default: stdout)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="Solver time budget in seconds (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CP-SAT parallel workers (default: 8)",
    )
    parser.add_argument(
        "--target-sc",
        default=None,
        help="Comma-separated SC names to target (default: all). "
        "Valid: CSC,FSC,MIP,session_continuity,SSCP,break_placement_compliance",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override weight for a specific SC: KEY=VAL (repeatable)",
    )
    parser.add_argument(
        "--relax-ictd",
        action="store_true",
        help="Relax SpreadAcrossDays hard constraint",
    )
    parser.add_argument(
        "--relax-hc",
        default=None,
        help="Comma-separated HC names to ignore in feasibility check "
        "(e.g. FPC,FCA,CQF,PMI)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver progress logging",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write JSON improvement report to PATH",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.domain.gene import set_time_system
    from src.io.data_store import DataStore
    from src.pipeline.cpsat_sc_optimizer import (
        SCOptimizer,
        SCOptimizerConfig,
        load_schedule_json,
    )

    # Load data
    store = DataStore.from_json(args.data_dir)
    set_time_system(store.qts)

    # Parse target SCs
    target_scs = None
    if args.target_sc:
        target_scs = [s.strip() for s in args.target_sc.split(",")]

    # Parse weight overrides
    weight_overrides: dict[str, float] | None = None
    if args.weight:
        weight_overrides = {}
        for w in args.weight:
            if "=" not in w:
                print(
                    f"Error: Invalid weight format '{w}'. Use KEY=VAL.", file=sys.stderr
                )
                return 1
            key, val = w.split("=", 1)
            try:
                weight_overrides[key.strip()] = float(val)
            except ValueError:
                print(
                    f"Error: Invalid weight value '{val}' for '{key}'.", file=sys.stderr
                )
                return 1

    # Parse relaxed HC names
    relaxed_hc_names = None
    if args.relax_hc:
        relaxed_hc_names = {s.strip() for s in args.relax_hc.split(",")}

    # Build config
    try:
        config = SCOptimizerConfig(
            time_budget_seconds=args.time_limit,
            target_constraints=target_scs,
            weight_overrides=weight_overrides,
            seed=args.seed,
            num_workers=args.workers,
            log_progress=not args.quiet,
            relax_ictd=args.relax_ictd,
            relaxed_hc_names=relaxed_hc_names,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load input schedule
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    genes = load_schedule_json(input_path, store)
    if not genes:
        print("Error: No sessions loaded from input.", file=sys.stderr)
        return 1

    # Build timetable
    from src.domain.timetable import Timetable

    context = store.to_context()
    input_tt = Timetable(genes, context, store.qts)

    # Run optimizer
    optimizer = SCOptimizer(data_store=store)
    try:
        result = optimizer.optimize(input_tt, config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build report
    report = optimizer._format_report(result, config)

    # Print console report
    if not args.quiet:
        SCOptimizer.print_console_report(report)

    # Handle exit codes
    if result.solver_status == "INFEASIBLE":
        print("Error: Solver returned INFEASIBLE.", file=sys.stderr)
        return 2
    if result.solver_status in ("UNKNOWN", "MODEL_INVALID"):
        print(f"Warning: Solver returned {result.solver_status}.", file=sys.stderr)
        return 3
    if result.improvement_pct <= 0 and result.solver_status != "SKIPPED":
        print("No improvement found — returning original solution.", file=sys.stderr)
        # Still write output (original schedule)

    # Export optimized schedule
    qts = store.qts
    output_entries = []
    for gene in result.output_timetable.genes:
        decoded_time = qts.decode_schedule(set(gene.get_quanta_list()))
        entry = {
            "course_id": gene.course_id,
            "course_type": gene.course_type,
            "instructor_id": gene.instructor_id,
            "group_ids": gene.group_ids,
            "room_id": gene.room_id,
            "start_quanta": gene.start_quanta,
            "num_quanta": gene.num_quanta,
            "co_instructor_ids": gene.co_instructor_ids or [],
            "time": decoded_time,
        }
        output_entries.append(entry)

    output_json = json.dumps(output_entries, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        if not args.quiet:
            print(f"Optimized schedule written to {args.output}")
    else:
        print(output_json)

    # Save report
    if args.report:
        SCOptimizer.save_json_report(report, args.report)
        if not args.quiet:
            print(f"Improvement report saved to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
