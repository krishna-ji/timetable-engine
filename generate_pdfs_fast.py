#!/usr/bin/env python3
"""Fast schedule PDF generator using SA polishing.

Generates a high-quality schedule (SA-polished) and exports all PDFs
(calendar, instructor, room schedules) without running the full GA.

Takes ~2-3 minutes total vs 35+ minutes for a full GA run.

Usage:
    python generate_pdfs_fast.py
    python generate_pdfs_fast.py --output-dir output/fast_pdf_run
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fast PDF schedule generator")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/fast_pdf_<timestamp>)",
    )
    parser.add_argument(
        "--sa-iters-mult",
        type=int,
        default=400,
        help="SA iterations multiplier (iters = E * mult, default: 400)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / "fast_pdf" / ts
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # ── 1. Load pkl data ─────────────────────────────────────
    pkl_path = str(PROJECT_ROOT / ".cache" / "events_with_domains.pkl")
    if not os.path.exists(pkl_path):
        log.info("Building pkl file...")
        from src.pipeline.build_events import ensure_pkl
        ensure_pkl(pkl_path, data_dir=str(PROJECT_ROOT / "data"))

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    E = len(pkl_data["events"])
    ai = pkl_data["allowed_instructors"]
    ar = pkl_data["allowed_rooms"]
    ast = pkl_data["allowed_starts"]
    log.info("Loaded %d events", E)

    # ── 2. Generate initial chromosome using bitset repair ─────
    log.info("Generating initial chromosome (bitset construct_feasible + repair)...")
    t0 = time.time()

    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair
    bitset_rep = BitsetSchedulingRepair(pkl_path)

    # Try multiple seeds, pick best starting point
    best_init_X = None
    best_init_hard = float("inf")
    for trial_seed in range(20):
        trial_rng = np.random.default_rng(trial_seed)
        trial_X = bitset_rep.construct_feasible(rng=trial_rng)
        for _ in range(5):
            trial_X = bitset_rep.repair(trial_X, rng=trial_rng)

        from src.pipeline.fast_evaluator_vectorized import (
            fast_evaluate_hard_vectorized,
            prepare_vectorized_data,
        )
        from src.pipeline.scheduling_problem import _TOLERATED_HARD_COLS

        if best_init_X is None:
            vd = prepare_vectorized_data(pkl_data)
            _strict = [i for i in range(8) if i not in _TOLERATED_HARD_COLS]

        G_trial = fast_evaluate_hard_vectorized(trial_X.reshape(1, -1), vd)
        trial_hard = float(sum(G_trial[0, j] for j in _strict))
        if trial_hard < best_init_hard:
            best_init_hard = trial_hard
            best_init_X = trial_X.copy()
            log.info("  Trial seed=%d: hard=%d (new best)", trial_seed, int(trial_hard))

    X = best_init_X
    log.info("Best initial chromosome: hard=%d (%.1fs, 20 trials)", int(best_init_hard), time.time() - t0)

    # ── 3. SA polishing (multi-restart) ─────────────────────
    n_iters = E * args.sa_iters_mult
    n_restarts = 5
    log.info("Running SA polishing (%d iters x %d restarts = %d total)...",
             n_iters, n_restarts, n_iters * n_restarts)
    t1 = time.time()

    global_best_X = X.copy()
    global_best_hard = best_init_hard

    for restart in range(n_restarts):
        best_X = X.copy()  # always restart from the bitset-repaired starting point
        best_hard = best_init_hard
        current_X = X.copy()
        current_hard = best_init_hard

        rng_sa = np.random.default_rng(args.seed + restart * 1000)
        T_start, T_end = 5.0, 0.01

    for it in range(n_iters):
        t_frac = it / max(n_iters - 1, 1)
        temp = T_start * ((T_end / T_start) ** t_frac)

        e = int(rng_sa.integers(E))
        old_i = current_X[3 * e]
        old_r = current_X[3 * e + 1]
        old_t = current_X[3 * e + 2]

        coin = rng_sa.random()
        if coin < 0.4 and len(ai[e]) > 1:
            current_X[3 * e] = ai[e][rng_sa.integers(len(ai[e]))]
        elif coin < 0.7 and len(ast[e]) > 1:
            current_X[3 * e + 2] = ast[e][rng_sa.integers(len(ast[e]))]
        elif coin < 0.85 and len(ar[e]) > 1:
            current_X[3 * e + 1] = ar[e][rng_sa.integers(len(ar[e]))]
        else:
            if len(ai[e]) > 1:
                current_X[3 * e] = ai[e][rng_sa.integers(len(ai[e]))]
            if len(ar[e]) > 1:
                current_X[3 * e + 1] = ar[e][rng_sa.integers(len(ar[e]))]
            if len(ast[e]) > 1:
                current_X[3 * e + 2] = ast[e][rng_sa.integers(len(ast[e]))]

        G_new = fast_evaluate_hard_vectorized(current_X.reshape(1, -1), vd)
        new_hard = float(sum(G_new[0, j] for j in _strict))

        delta = new_hard - current_hard
        if delta <= 0:
            current_hard = new_hard
            if new_hard < best_hard:
                best_hard = new_hard
                best_X = current_X.copy()
        elif rng_sa.random() < np.exp(-delta / max(temp, 1e-10)):
            current_hard = new_hard
        else:
            current_X[3 * e] = old_i
            current_X[3 * e + 1] = old_r
            current_X[3 * e + 2] = old_t

        # Periodic restart from best within this SA run
        if it > 0 and it % (E * 10) == 0:
            current_X = best_X.copy()
            current_hard = best_hard
            log.info(
                "  SA restart @ iter %d/%d: best_hard=%d",
                it, n_iters, int(best_hard),
            )

        # Update global best across all restarts
        if best_hard < global_best_hard:
            global_best_hard = best_hard
            global_best_X = best_X.copy()

    log.info(
        "  Restart %d/%d done: best_hard=%d (global_best=%d)",
        restart + 1, n_restarts, int(best_hard), int(global_best_hard),
    )

    sa_time = time.time() - t1
    log.info(
        "SA polishing done: hard %d -> %d (%.1fs, %d total iters)",
        int(best_init_hard), int(global_best_hard), sa_time, n_iters * n_restarts,
    )
    best_X = global_best_X
    best_hard = global_best_hard

    # Final evaluation breakdown
    G_final = fast_evaluate_hard_vectorized(best_X.reshape(1, -1), vd)
    names = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD"]
    breakdown = {}
    for j in range(min(len(names), G_final.shape[1])):
        val = int(G_final[0, j])
        if val > 0:
            breakdown[names[j]] = val
    log.info("Breakdown: %s", breakdown)

    # ── 4. Convert chromosome to genes and export PDFs ───────
    log.info("Converting chromosome to genes...")
    from src.domain.gene import SessionGene
    from src.ga.core.population import _assign_practical_co_instructors
    from src.io.data_store import DataStore
    from src.io.decoder import decode_individual
    from src.io.export.exporter import export_everything
    from src.io.export.schedule_views import (
        generate_instructor_schedules_pdf,
        generate_room_schedules_pdf,
    )
    from src.io.export.violation_reporter import generate_violation_report
    from src.io.time_system import QuantumTimeSystem

    events = pkl_data["events"]
    idx_to_inst = pkl_data["idx_to_instructor"]
    idx_to_room = pkl_data["idx_to_room"]

    genes = []
    for e in range(E):
        ev = events[e]
        genes.append(
            SessionGene(
                course_id=ev["course_id"],
                course_type=ev["course_type"],
                instructor_id=idx_to_inst[int(best_X[3 * e])],
                group_ids=list(ev["group_ids"]),
                room_id=idx_to_room[int(best_X[3 * e + 1])],
                start_quanta=int(best_X[3 * e + 2]),
                num_quanta=ev["num_quanta"],
            )
        )
    log.info("Built %d SessionGene objects", len(genes))

    # Load context for decoding
    try:
        store = DataStore.from_json(str(PROJECT_ROOT / "data"), run_preflight=False)
    except Exception:
        store = DataStore.from_json(str(PROJECT_ROOT / "data"), run_preflight=False)
    ctx = store.to_context()
    qts = QuantumTimeSystem()

    # Assign co-instructors for practical sessions
    _assign_practical_co_instructors(genes, ctx)

    # Decode genes -> CourseSession objects
    log.info("Decoding to CourseSession objects...")
    sessions = decode_individual(
        genes, ctx.courses, ctx.instructors, ctx.groups, ctx.rooms
    )
    log.info("Decoded %d sessions", len(sessions))

    course_lookup = ctx.courses
    out = str(output_dir)

    # 4a. schedule.json + calendar.pdf
    log.info("Generating schedule.json + calendar.pdf...")
    try:
        export_everything(
            sessions, out, qts,
            course_lookup=course_lookup, parallel=False,
        )
        log.info("  [ok] calendar PDF + schedule.json")
    except Exception as exc:
        log.error("  [FAIL] calendar PDF: %s", exc, exc_info=True)

    # 4b. instructor_schedules.pdf
    log.info("Generating instructor_schedules.pdf...")
    try:
        generate_instructor_schedules_pdf(
            sessions, ctx.instructors, course_lookup, qts, out,
        )
        log.info("  [ok] instructor schedules PDF")
    except Exception as exc:
        log.error("  [FAIL] instructor PDF: %s", exc, exc_info=True)

    # 4c. room_schedules.pdf
    log.info("Generating room_schedules.pdf...")
    try:
        generate_room_schedules_pdf(
            sessions, ctx.rooms, course_lookup, qts, out,
            groups=ctx.groups,
        )
        log.info("  [ok] room schedules PDF")
    except Exception as exc:
        log.error("  [FAIL] room PDF: %s", exc, exc_info=True)

    # 4d. violation report
    log.info("Generating violation report...")
    try:
        generate_violation_report(sessions, course_lookup, qts, out)
        log.info("  [ok] violation report")
    except Exception as exc:
        log.error("  [FAIL] violation report: %s", exc, exc_info=True)

    # ── 5. Summary ───────────────────────────────────────────
    total_time = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.1fs", total_time)
    log.info("  hard violations = %d", int(best_hard))
    log.info("  breakdown: %s", breakdown)
    log.info("  output: %s", output_dir)
    log.info("=" * 60)

    # List generated files
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        log.info("  %s  (%.1f KB)", f.name, size_kb)


if __name__ == "__main__":
    main()
