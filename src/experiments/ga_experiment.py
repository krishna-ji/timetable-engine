"""GA experiment runners — pymoo-based NSGA-II modes.

Each subclass pre-fills its mode-specific defaults (callback kind,
mutation parameters, repair settings) so the run script only needs to
set user-facing knobs like ``pop_size``, ``ngen``, and ``seed``.

All modes share the same pipeline:
    build_events_with_domains()  →  SchedulingProblem  →  NSGA-II  →  result
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from .base import PROJECT_ROOT, BaseExperiment
from .callback_core import GACallbackBase

logger = logging.getLogger(__name__)

__version__ = "3.0.0"  # pymoo runner v3


# ── Module-level worker for ProcessPoolExecutor (must be picklable) ──


def _repair_single_elite(
    X: np.ndarray,
    pkl_path: str,
    repair_iters: int,
    gen: int,
    idx: int,
) -> np.ndarray:
    """Run the repair loop for one elite individual in a worker process.

    Instantiates its own ``BitsetSchedulingRepair`` (each process needs its
    own copy since the repairer is not fork-safe / not picklable).

    Parameters
    ----------
    X : ndarray, shape (3*E,)
        Chromosome to repair (already copied by the caller).
    pkl_path : str
        Path to events_with_domains.pkl.
    repair_iters : int
        Number of alternating stochastic/deterministic repair passes.
    gen : int
        Current generation (used to seed stochastic passes).
    idx : int
        Individual index (used to seed stochastic passes).

    Returns
    -------
    ndarray
        Repaired chromosome.
    """
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

    repairer = BitsetSchedulingRepair(pkl_path)
    for p in range(repair_iters):
        if p % 2 == 0:
            rng_p = np.random.default_rng([gen, idx, p])
        else:
            rng_p = None
        X_new = repairer.repair(X, rng=rng_p)
        if np.array_equal(X_new, X):
            break
        X = X_new
    return X


def _reeval_modified(algorithm: Any, modified_inds: list) -> None:
    """Clear stale F/G/CV on modified individuals and force re-evaluation.

    Must be called whenever a callback mutates ``pop[i].X`` so that
    pymoo's NSGA-II sorting sees the updated fitness, not the stale
    pre-repair scores.
    """
    if not modified_inds:
        return
    for ind in modified_inds:
        ind.set("F", None)
        ind.set("G", None)
        ind.set("CV", None)

        # Remove from the evaluated set (crucial for pymoo to trigger re-evaluation)
        if "F" in ind.evaluated:
            ind.evaluated.remove("F")
        if "G" in ind.evaluated:
            ind.evaluated.remove("G")
        if "CV" in ind.evaluated:
            ind.evaluated.remove("CV")
    eval_pop = Population.create(*modified_inds)
    Evaluator().eval(algorithm.problem, eval_pop)


# =====================================================================
#  GA Experiment (base for all GA modes)
# =====================================================================


class GAExperiment(BaseExperiment):
    """Base GA experiment using pymoo NSGA-II.

    Parameters
    ----------
    mode : str
        Mode name (``"baseline"``, ``"memetic"``, etc.).
    pop_size : int
        Population size.
    ngen : int
        Number of generations.
    crossover_prob : float
        Per-event crossover probability.
    mutation_event_prob : float
        Per-event mutation probability.
    n_offsprings_mult : float
        Offspring multiplier (1.0 = pop_size offspring per gen).
    log_interval : int | None
        Generations between detailed logs.  ``None`` → auto (ngen / 20).
    export_pdf : bool
        Generate schedule PDFs (calendar, instructor, room).
        Set ``False`` for fast dev iterations (saves ~55s per run).
    """

    def __init__(
        self,
        *,
        mode: str,
        pop_size: int = 100,
        ngen: int = 200,
        crossover_prob: float = 0.5,
        mutation_event_prob: float = 0.05,
        n_offsprings_mult: float = 1.0,
        log_interval: int | None = None,
        export_pdf: bool = True,
        force_pdf: bool = False,
        use_repair: bool = True,
        # BaseExperiment kwargs
        seed: int = 42,
        data_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        verbose: bool = True,
    ) -> None:
        tag = f"ga_{mode}"
        super().__init__(
            name=f"GA {mode.title()}",
            tag=tag,
            seed=seed,
            data_dir=data_dir,
            output_dir=output_dir,
            verbose=verbose,
        )
        self.mode = mode
        self.pop_size = pop_size
        self.ngen = ngen
        self.crossover_prob = crossover_prob
        self.mutation_event_prob = mutation_event_prob
        self.n_offsprings_mult = n_offsprings_mult
        self.log_interval = log_interval or max(1, ngen // 20)
        self.export_pdf = export_pdf
        self.force_pdf = force_pdf
        self.use_repair = use_repair

    # ── Pipeline helpers ──────────────────────────────────────────

    def _ensure_pkl(self) -> str:
        """Build ``events_with_domains.pkl`` if missing; return its path."""
        from src.pipeline.build_events import PKL_DEFAULT_PATH, ensure_pkl

        pkl_path = str(PROJECT_ROOT / PKL_DEFAULT_PATH)
        ensure_pkl(pkl_path, data_dir=str(self.data_dir))
        return pkl_path

    def _build_callback(self, pkl_path: str) -> Any:
        """Override in subclasses for mode-specific callbacks."""
        return GACallbackBase(self.log_interval)

    # ── Output generation (plots + PDFs) ─────────────────────────

    @staticmethod
    def _chromosome_to_genes(X: np.ndarray, pkl_data: dict) -> list | None:
        """Convert a flat pymoo chromosome back to a list of SessionGene."""
        try:
            from src.domain.gene import SessionGene

            events = pkl_data["events"]
            idx_to_inst = pkl_data["idx_to_instructor"]
            idx_to_room = pkl_data["idx_to_room"]
            E = len(events)
            genes = []
            skipped = 0
            for e in range(E):
                ev = events[e]
                inst_idx = int(X[3 * e])
                room_idx = int(X[3 * e + 1])
                # Skip events with out-of-range indices (from SA/repair artifacts)
                if inst_idx not in idx_to_inst or room_idx not in idx_to_room:
                    skipped += 1
                    continue
                genes.append(
                    SessionGene(
                        course_id=ev["course_id"],
                        course_type=ev["course_type"],
                        instructor_id=idx_to_inst[inst_idx],
                        group_ids=list(ev["group_ids"]),
                        room_id=idx_to_room[room_idx],
                        start_quanta=int(X[3 * e + 2]),
                        num_quanta=ev["num_quanta"],
                    )
                )
            if skipped:
                logger.warning(
                    "chromosome->genes: skipped %d/%d events with out-of-range indices", skipped, E)
            return genes
        except Exception as exc:
            logger.exception("chromosome -> genes bridge error: %s", exc)
            return None

    def _generate_outputs(
        self,
        *,
        res: Any,
        callback: Any,
        pkl_data: dict,
        ctx: Any,
        qts: Any,
        best_idx: int,
    ) -> None:
        """Generate all plots, schedule PDFs, and reports.

        Called automatically at the end of ``_execute()``.  Failures in
        individual export steps are logged but never propagate — the
        experiment result is always returned.
        """
        import matplotlib as mpl

        mpl.use("Agg")  # non-interactive backend for headless envs

        out = str(self.output_dir)
        F = res.pop.get("F")
        best_hards: list[float] = getattr(callback, "best_hards", [])
        best_softs: list[float] = getattr(callback, "best_softs", [])
        best_breakdowns: list[dict] = getattr(callback, "best_breakdowns", [])

        # ── 1. Convergence plots ───────────────────────────────────
        self._safe_call(
            "hard-violation plot",
            lambda: (
                __import__(
                    "src.io.export.plothard",
                    fromlist=["plot_hard_constraint_violation_over_generation"],
                ).plot_hard_constraint_violation_over_generation(best_hards, out)
            ),
        )
        self._safe_call(
            "soft-penalty plot",
            lambda: (
                __import__(
                    "src.io.export.plotsoft",
                    fromlist=["plot_soft_constraint_violation_over_generation"],
                ).plot_soft_constraint_violation_over_generation(best_softs, out)
            ),
        )

        # ── 2. Per-constraint trend plots ──────────────────────────
        if best_breakdowns:
            # Transpose list[dict] -> dict[str, list[int]]
            all_keys = best_breakdowns[0].keys()
            hard_trends: dict[str, list[int]] = {
                k: [bd.get(k, 0) for bd in best_breakdowns] for k in all_keys
            }
            self._safe_call(
                "individual constraint plots",
                lambda: (
                    __import__(
                        "src.io.export.plot_detailed_constraints",
                        fromlist=["plot_individual_hard_constraints"],
                    ).plot_individual_hard_constraints(hard_trends, out)
                ),
            )

        # ── 2b. Per-soft-constraint trend plots ───────────────────
        best_soft_breakdowns: list[dict] = getattr(
            callback, "best_soft_breakdowns", [])
        if best_soft_breakdowns and best_soft_breakdowns[0]:
            all_soft_keys = best_soft_breakdowns[0].keys()
            soft_trends: dict[str, list[int]] = {
                k: [bd.get(k, 0) for bd in best_soft_breakdowns] for k in all_soft_keys
            }
            self._safe_call(
                "individual soft constraint plots",
                lambda: (
                    __import__(
                        "src.io.export.plot_detailed_constraints",
                        fromlist=["plot_individual_soft_constraints"],
                    ).plot_individual_soft_constraints(soft_trends, out)
                ),
            )

        # ── 3. Convergence rate analysis ───────────────────────────
        if len(best_hards) >= 11:
            self._safe_call(
                "convergence rate plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_convergence_rate"],
                    ).plot_convergence_rate(best_hards, out, "Hard Violations")
                ),
            )

        # ── 4. Pareto front (pymoo F matrix) ──────────────────────
        self._safe_call(
            "Pareto front plot",
            lambda: (
                __import__(
                    "src.io.export.plotpareto",
                    fromlist=["plot_pareto_front_from_F"],
                ).plot_pareto_front_from_F(F, out)
            ),
        )

        # ── 4b. MOEA metric plots (HV, spacing, diversity, feasibility) ──
        hv_hist: list[float] = getattr(callback, "hypervolumes", [])
        sp_hist: list[float] = getattr(callback, "spacings", [])
        div_hist: list[float] = getattr(callback, "diversities", [])
        feas_hist: list[float] = getattr(callback, "feasibility_rates", [])
        igd_hist: list[float] = getattr(callback, "igds", [])

        # ── 4a-2. Pareto Evolution scatter (generation-coloured) ──
        f_history: list = getattr(callback, "f_history", [])
        if f_history:
            self._safe_call(
                "Pareto evolution plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_pareto_evolution",
                        fromlist=["plot_pareto_evolution"],
                    ).plot_pareto_evolution(f_history, out)
                ),
            )

        if hv_hist:
            self._safe_call(
                "hypervolume trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_hypervolume",
                        fromlist=["plot_hypervolume_trend"],
                    ).plot_hypervolume_trend(hv_hist, out)
                ),
            )
        if sp_hist:
            self._safe_call(
                "spacing trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_spacing",
                        fromlist=["plot_spacing_trend"],
                    ).plot_spacing_trend(sp_hist, out)
                ),
            )
        if div_hist:
            self._safe_call(
                "diversity trend",
                lambda: (
                    __import__(
                        "src.io.export.plotdiversity",
                        fromlist=["plot_diversity_trend"],
                    ).plot_diversity_trend(div_hist, out)
                ),
            )
        if feas_hist:
            self._safe_call(
                "feasibility rate trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_constraint_satisfaction_evolution"],
                    ).plot_constraint_satisfaction_evolution(feas_hist, out)
                ),
            )

        # ── 4c. IGD trend (only if reference front was available) ──
        # Filter out nan values to check if any real IGD values exist
        igd_real = [v for v in igd_hist if v == v]  # nan != nan
        if igd_real:
            self._safe_call(
                "IGD trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_igd",
                        fromlist=["plot_igd_trend"],
                    ).plot_igd_trend(igd_hist, out)
                ),
            )

        # Convergence dashboard (needs all 6 inputs)
        if hv_hist and sp_hist and div_hist and feas_hist:
            self._safe_call(
                "convergence dashboard",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_convergence_dashboard"],
                    ).plot_convergence_dashboard(
                        best_hards,
                        best_softs,
                        div_hist,
                        hv_hist,
                        sp_hist,
                        feas_hist,
                        out,
                    )
                ),
            )

        # ── 5. Generation-wise CSV log ─────────────────────────────
        self._safe_call(
            "convergence CSV",
            lambda: self._write_convergence_csv(
                out,
                best_hards,
                best_softs,
                best_breakdowns,
                best_soft_breakdowns,
            ),
        )

        # ── 6. Schedule export (schedule.json always; PDFs optional) ──────────
        # schedule.json is always written so the gRPC service can read it back.
        # PDFs are only generated when export_pdf=True (they add ~55s).
        X_best = res.pop[best_idx].get("X")
        genes = self._chromosome_to_genes(X_best, pkl_data)
        if genes is not None:
            # Assign co-instructors for practical genes (pymoo chromosomes
            # don't encode co_instructor_ids so we add them here).
            from src.ga.core.population import _assign_practical_co_instructors

            _assign_practical_co_instructors(genes, ctx)

            # Always write schedule.json (critical for gRPC callers).
            # Do NOT wrap in _safe_call so errors propagate visibly.
            try:
                self._export_schedule_json_only(genes, ctx, qts, out)
            except Exception as exc:
                self.logger.error(
                    "schedule.json export FAILED: %s", exc, exc_info=True)

            if self.export_pdf:
                self._safe_call(
                    "schedule decode + PDF export",
                    lambda: self._export_schedule_pdfs(genes, ctx, qts, out),
                )
        else:
            self.logger.error(
                "Skipping schedule export — chromosome-to-gene "
                "bridge returned None (see traceback above)"
            )

        # ── 7. Memetic / repair-specific thesis plots ─────────────
        repair_gens: list[int] = getattr(callback, "repair_gens", [])
        if repair_gens:
            self._safe_call(
                "repair intervention plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_memetic",
                        fromlist=["plot_repair_interventions"],
                    ).plot_repair_interventions(best_hards, repair_gens, out)
                ),
            )
            if f_history:
                self._safe_call(
                    "Pareto repair shift plot",
                    lambda: (
                        __import__(
                            "src.io.export.plot_memetic",
                            fromlist=["plot_pareto_repair_shift"],
                        ).plot_pareto_repair_shift(f_history, repair_gens, out)
                    ),
                )

        self.logger.info(f"Output artefacts written to {self.output_dir}")

    # ── CSV helpers ────────────────────────────────────────────

    @staticmethod
    def _write_convergence_csv(
        output_dir: str,
        best_hards: list[float],
        best_softs: list[float],
        best_breakdowns: list[dict],
        best_soft_breakdowns: list[dict],
    ) -> None:
        """Write ``convergence_history.csv`` with per-gen constraint data."""
        path = Path(output_dir) / "convergence_history.csv"

        # Hard constraint short names (Academic Nomenclature)
        hc_keys = ["CTE", "FTE", "SRE", "FPC",
                   "FFC", "FCA", "CQF", "ICTD", "sib"]
        # Soft constraint short names (Academic Nomenclature)
        sc_keys = ["CSC", "FSC", "MIP", "SSCP"]

        header = (
            ["Gen", "Best_Hard", "Best_Soft"]
            + [f"hc_{k}" for k in hc_keys]
            + [f"sc_{k}" for k in sc_keys]
        )

        n_gens = len(best_hards)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for g in range(n_gens):
                hc_vals = [
                    best_breakdowns[g].get(k, 0) if g < len(
                        best_breakdowns) else 0
                    for k in hc_keys
                ]
                sc_vals = [
                    (
                        best_soft_breakdowns[g].get(k, 0)
                        if g < len(best_soft_breakdowns) and best_soft_breakdowns[g]
                        else 0
                    )
                    for k in sc_keys
                ]
                writer.writerow(
                    [g + 1, int(best_hards[g]), int(best_softs[g])
                     ] + hc_vals + sc_vals
                )
        logger.info("  convergence CSV -> %s (%d rows)", path, n_gens)

    def _export_schedule_pdfs(
        self,
        genes: list,
        ctx: Any,
        qts: Any,
        output_dir: str,
    ) -> None:
        """Decode genes -> CourseSession list, then export all PDFs + reports."""
        from src.io.decoder import decode_individual
        from src.io.export.exporter import export_everything
        from src.io.export.schedule_views import (
            generate_instructor_schedules_pdf,
            generate_room_schedules_pdf,
        )
        from src.io.export.violation_reporter import generate_violation_report

        sessions = decode_individual(
            genes,
            ctx.courses,
            ctx.instructors,
            ctx.groups,
            ctx.rooms,
        )

        # Build course lookup {(course_id, course_type): Course}
        course_lookup: dict[tuple[str, str], Any] = ctx.courses

        # 5a. schedule.json + calendar.pdf (group-wise)
        self._safe_call(
            "calendar PDF",
            lambda: (
                export_everything(
                    sessions,
                    output_dir,
                    qts,
                    course_lookup=course_lookup,
                    parallel=False,
                )
            ),
        )

        # 5b. instructor_schedules.pdf
        self._safe_call(
            "instructor PDF",
            lambda: (
                generate_instructor_schedules_pdf(
                    sessions,
                    ctx.instructors,
                    course_lookup,
                    qts,
                    output_dir,
                )
            ),
        )

        # 5c. room_schedules.pdf
        self._safe_call(
            "room PDF",
            lambda: (
                generate_room_schedules_pdf(
                    sessions,
                    ctx.rooms,
                    course_lookup,
                    qts,
                    output_dir,
                    groups=ctx.groups,
                )
            ),
        )

        # 5d. log_violations.log
        self._safe_call(
            "violation report",
            lambda: (
                generate_violation_report(
                    sessions, course_lookup, qts, output_dir)
            ),
        )

    def _export_schedule_json_only(
        self,
        genes: list,
        ctx: Any,
        qts: Any,
        output_dir: str,
    ) -> None:
        """Decode genes → CourseSession list and write schedule.json only (no PDFs)."""
        from src.io.decoder import decode_individual
        from src.io.export.exporter import _save_schedule_as_json

        sessions = decode_individual(
            genes,
            ctx.courses,
            ctx.instructors,
            ctx.groups,
            ctx.rooms,
        )
        course_lookup: dict = ctx.courses
        _save_schedule_as_json(sessions, output_dir, qts,
                               course_lookup=course_lookup)
        self.logger.info("  schedule.json written to %s", output_dir)

    def _safe_call(self, label: str, fn: Any) -> None:
        """Execute *fn* and swallow any exception, logging it with full traceback."""
        try:
            fn()
            self.logger.info(f"  [ok] {label}")
        except Exception as exc:
            self.logger.error("  [FAIL] %s: %s", label, exc, exc_info=True)

    # ── Core execution ─────────────────────────────────────────────

    def _load_data(self) -> tuple[Any, Any, Any]:
        """Load scheduling data and save feasibility report to output dir.

        If the data has known infeasibilities (e.g. instructor qualification
        bottleneck), the GA still runs — it's designed to *optimise toward*
        feasibility.  The feasibility report is saved, but the exception
        is not propagated.

        Returns (store, ctx, qts).
        """
        from src.io.data_store import DataStore
        from src.io.feasibility import (
            InfeasibleProblemError,
            generate_feasibility_report_file,
        )
        from src.io.time_system import QuantumTimeSystem

        feasibility_report = None
        try:
            store = DataStore.from_json(str(self.data_dir))
            feasibility_report = store.feasibility_report
        except InfeasibleProblemError as exc:
            # Capture the report from the exception, then retry without preflight
            feasibility_report = exc.report
            self.logger.warning(
                "Feasibility check failed — running GA anyway "
                "(the optimizer will try to minimise violations)"
            )
            store = DataStore.from_json(
                str(self.data_dir), run_preflight=False)
        except Exception:
            self.logger.warning(
                "Data loading failed — retrying without preflight")
            store = DataStore.from_json(
                str(self.data_dir), run_preflight=False)

        ctx = store.to_context()
        qts = QuantumTimeSystem()

        # Always save feasibility report to the timestamped output folder
        if feasibility_report is not None:
            report_path = self.output_dir / "feasibility_report.md"
            generate_feasibility_report_file(
                feasibility_report, str(report_path))
            self.logger.info(f"Feasibility report -> {report_path}")

        return store, ctx, qts

    def _execute(self) -> dict[str, Any]:
        from pymoo.optimize import minimize

        from src.pipeline.pymoo_operators import create_algorithm
        from src.pipeline.scheduling_problem import create_problem

        pkl_path = self._ensure_pkl()
        _store, ctx, qts = self._load_data()

        # Apply tutorial-practical fix consistent with pkl build
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        if pkl_data.get("fix_tutorial_practicals", False):
            for course in ctx.courses.values():
                lab_feats = getattr(course, "specific_lab_features", None)
                if lab_feats:
                    feats_lower = [
                        (f if isinstance(f, str) else str(f)).lower().strip()
                        for f in lab_feats
                    ]
                    if any(f in ("lecture hall", "seminar room") for f in feats_lower):
                        course.specific_lab_features = []

        # ── Pre-feasibility report (runs before GA, survives crashes) ──
        self._safe_call(
            "pre-feasibility report",
            lambda: (
                __import__(
                    "src.io.export.feasibility_reporter",
                    fromlist=["generate_pre_feasibility_report"],
                ).generate_pre_feasibility_report(pkl_data, self.output_dir)
            ),
        )

        n_offsprings = int(self.pop_size * self.n_offsprings_mult)
        self.logger.info(
            f"Mode: {self.mode}  |  pop={self.pop_size}  "
            f"gens={self.ngen}  seed={self.seed}  "
            f"cx={self.crossover_prob}  mut={self.mutation_event_prob}"
        )

        prob = create_problem(pkl_path, ctx=ctx, qts=qts)
        algo = create_algorithm(
            pkl_path=pkl_path,
            pop_size=self.pop_size,
            n_offsprings=n_offsprings,
            crossover_prob=self.crossover_prob,
            mutation_event_prob=self.mutation_event_prob,
            algorithm="nsga2",
            seed=self.seed,
            use_repair=self.use_repair,
        )

        callback = self._build_callback(pkl_path)

        t0 = time.time()
        res = minimize(
            prob,
            algo,
            ("n_gen", self.ngen),
            seed=self.seed,
            verbose=False,
            callback=callback,
        )
        elapsed = time.time() - t0

        # Extract best
        F = res.pop.get("F")
        G = res.pop.get("G")
        # Exclude tolerated columns (iAvl col 5) from cv
        from src.pipeline.scheduling_problem import _TOLERATED_HARD_COLS

        _strict = [i for i in range(
            G.shape[1]) if i not in _TOLERATED_HARD_COLS]
        cv = G[:, _strict].sum(axis=1).clip(0)
        best_idx = int(np.argmin(cv))

        self.logger.info(
            f"Done in {elapsed:.1f}s  ({elapsed / self.ngen:.2f}s/gen)")
        self.logger.info(
            f"Best: hard={F[best_idx, 0]:.0f}  "
            f"soft={F[best_idx, 1]:.0f}  cv={cv[best_idx]:.0f}"
        )

        # ── Post-GA local-search polishing (conflict-guided SA) ────
        best_X = res.pop[best_idx].get("X").copy()
        best_hard_before = float(cv[best_idx])
        if best_hard_before > 0:
            try:
                from src.pipeline.fast_evaluator_vectorized import (
                    fast_evaluate_hard_vectorized,
                    prepare_vectorized_data,
                )
                from src.pipeline.repair_operator_vectorized import VectorizedRepair

                vd = prepare_vectorized_data(pkl_data)
                sa_repairer = VectorizedRepair(pkl_path)
                E = len(pkl_data["events"])
                ai = pkl_data["allowed_instructors"]
                ar = pkl_data["allowed_rooms"]
                ast = pkl_data["allowed_starts"]
                events = pkl_data["events"]

                # Pre-convert domains to numpy for faster indexing
                ai_arr = [np.array(a, dtype=np.int64) for a in ai]
                ar_arr = [np.array(a, dtype=np.int64) for a in ar]
                ast_arr = [np.array(a, dtype=np.int64) for a in ast]

                # Build group-to-events mapping for group-aware moves
                _group_to_events: dict[str, list[int]] = {}
                for e_idx in range(E):
                    for gid in events[e_idx]["group_ids"]:
                        _group_to_events.setdefault(gid, []).append(e_idx)

                # Event durations for occupancy queries
                _durations = np.array(
                    [ev["num_quanta"] for ev in events], dtype=np.int32
                )

                # Instructor-to-events mapping for fast occupancy in SA
                _inst_to_events: dict[int, list[int]] = {}
                for e_idx in range(E):
                    for c in ai[e_idx]:
                        _inst_to_events.setdefault(int(c), []).append(e_idx)

                best_X_sa = best_X.copy()
                best_hard_sa = best_hard_before
                current_X = best_X.copy()
                current_hard = best_hard_before

                rng_sa = np.random.default_rng(42)

                # ── Pre-SA repair: apply full CTE+conflict repair ──
                # The GA often degrades CTE during evolution; repair it back
                _pre = current_X.reshape(1, -1).copy()
                _pre = sa_repairer.repair_batch(_pre, passes=4)
                _pre_h = fast_evaluate_hard_vectorized(_pre, vd)
                _pre_hard = float(sum(_pre_h[0, j] for j in _strict))
                if _pre_hard < current_hard:
                    current_X = _pre[0].copy()
                    current_hard = _pre_hard
                    best_X_sa = current_X.copy()
                    best_hard_sa = _pre_hard
                    self.logger.info(
                        "Pre-SA repair: hard %d -> %d",
                        int(best_hard_before), int(_pre_hard),
                    )

                # Compute initial per-event conflict scores (decomposed)
                _room_sc, _inst_sc, _grp_sc = sa_repairer._score_decomposed_batch(
                    current_X.reshape(1, -1)
                )
                grp_scores_sa = _grp_sc[0]
                room_scores_sa = _room_sc[0]
                event_scores = (_room_sc[0] + _inst_sc[0] + _grp_sc[0])

                # ── Persistent occupancy tables (incrementally updated) ──
                from src.pipeline.bitset_time import T as _T_quanta
                _grp_to_idx = {}
                for ev in events:
                    for gid in ev["group_ids"]:
                        if gid not in _grp_to_idx:
                            _grp_to_idx[gid] = len(_grp_to_idx)
                _n_grp = len(_grp_to_idx)
                _n_inst = sa_repairer.n_instructors
                _n_room = sa_repairer.n_rooms
                _gocc = np.zeros((_n_grp, _T_quanta), dtype=np.int16)
                _iocc = np.zeros((_n_inst, _T_quanta), dtype=np.int16)
                _rocc = np.zeros((_n_room, _T_quanta), dtype=np.int16)
                for _e in range(E):
                    _t = int(current_X[3 * _e + 2])
                    _d = int(_durations[_e])
                    _ii = int(current_X[3 * _e])
                    _rr = min(int(current_X[3 * _e + 1]), _n_room - 1)
                    for _q in range(_t, min(_t + _d, _T_quanta)):
                        _iocc[min(_ii, _n_inst - 1), _q] += 1
                        _rocc[_rr, _q] += 1
                        for _gid in events[_e]["group_ids"]:
                            _gocc[_grp_to_idx[_gid], _q] += 1

                # Helper: add/remove event from occupancy
                def _occ_add(_e, _X):
                    _t = int(_X[3*_e+2])
                    _d = int(_durations[_e])
                    _ii = min(int(_X[3*_e]), _n_inst - 1)
                    _rr = min(int(_X[3*_e+1]), _n_room - 1)
                    for _q in range(_t, min(_t+_d, _T_quanta)):
                        _iocc[_ii, _q] += 1
                        _rocc[_rr, _q] += 1
                        for _gid in events[_e]["group_ids"]:
                            _gocc[_grp_to_idx[_gid], _q] += 1

                def _occ_remove(_e, _X):
                    _t = int(_X[3*_e+2])
                    _d = int(_durations[_e])
                    _ii = min(int(_X[3*_e]), _n_inst - 1)
                    _rr = min(int(_X[3*_e+1]), _n_room - 1)
                    for _q in range(_t, min(_t+_d, _T_quanta)):
                        _iocc[_ii, _q] -= 1
                        _rocc[_rr, _q] -= 1
                        for _gid in events[_e]["group_ids"]:
                            _gocc[_grp_to_idx[_gid], _q] -= 1

                # Fast hard-score from maintained occupancy tables.
                # CTE = sum(max(gocc-1, 0)), FTE = sum(max(iocc-1, 0)),
                # SRE = sum(max(rocc-1, 0)).  O(G*T + I*T + R*T) ≈ O(15K)
                # vs full eval which has numpy overhead for N=1.
                # Pre-allocate scratch buffers to avoid per-call allocation
                _gocc_buf = np.empty_like(_gocc, dtype=np.int32)
                _iocc_buf = np.empty_like(_iocc, dtype=np.int32)
                _rocc_buf = np.empty_like(_rocc, dtype=np.int32)

                def _hard_from_occ():
                    np.subtract(_gocc, 1, out=_gocc_buf, casting='unsafe')
                    np.clip(_gocc_buf, 0, None, out=_gocc_buf)
                    np.subtract(_iocc, 1, out=_iocc_buf, casting='unsafe')
                    np.clip(_iocc_buf, 0, None, out=_iocc_buf)
                    np.subtract(_rocc, 1, out=_rocc_buf, casting='unsafe')
                    np.clip(_rocc_buf, 0, None, out=_rocc_buf)
                    return int(_gocc_buf.sum() + _iocc_buf.sum() + _rocc_buf.sum())

                # SA configuration — fast occ-based scoring allows more iters
                n_phases = 12
                iters_per_phase = E * 120
                n_iters = n_phases * iters_per_phase
                no_improve_count = 0
                score_refresh_interval = max(E // 8, 25)

                # Calibrate: switch current_hard to occ-based scoring
                current_hard = _hard_from_occ()
                best_hard_sa = current_hard
                sa_iter_total = 0
                sa_time_limit = 180.0  # hard cap: 180 seconds
                sa_start_time = time.time()
                sa_stagnation_restarts = 0  # count restarts without global improvement
                _full_eval_interval = 5000  # periodic full-eval sync

                self.logger.info(
                    "SA polishing: start=%d, %d phases × %d iters = %d total",
                    int(best_hard_before), n_phases, iters_per_phase, n_iters,
                )

                for phase in range(n_phases):
                    # Time limit check
                    if time.time() - sa_start_time > sa_time_limit:
                        self.logger.info(
                            "SA polishing: time limit reached (%.0fs)", sa_time_limit)
                        break
                    # Reheat at each phase (geometric cooling)
                    T_start_ph = max(5.0 * (0.6 ** phase), 0.05)
                    T_end_ph = max(T_start_ph * 0.005, 0.002)
                    phase_best_before = best_hard_sa

                    for it in range(iters_per_phase):
                        # Check time limit every 500 iterations
                        if it % 500 == 0 and time.time() - sa_start_time > sa_time_limit:
                            break
                        sa_iter_total += 1
                        t_frac = it / max(iters_per_phase - 1, 1)
                        temp = T_start_ph * ((T_end_ph / T_start_ph) ** t_frac)

                        # Periodically refresh per-event conflict scores
                        # from occupancy tables (vectorized, avoids full batch eval)
                        if it % score_refresh_interval == 0:
                            _t_all = current_X[2::3].astype(np.int64)
                            _i_all = np.clip(current_X[0::3], 0, _n_inst - 1).astype(np.int64)
                            _r_all = np.clip(current_X[1::3], 0, _n_room - 1).astype(np.int64)

                            # FTE: check iocc at each event's (inst, quantum)
                            _s_exp = _t_all[sa_repairer.exp_event]
                            _q_exp = np.clip(_s_exp + sa_repairer.exp_offset, 0, _T_quanta - 1)
                            _fte_f = (_iocc[_i_all[sa_repairer.exp_event], _q_exp] > 1)
                            _ev_inst = np.bincount(
                                sa_repairer.exp_event, weights=_fte_f.astype(np.float64),
                                minlength=E).astype(np.int32)

                            # SRE: check rocc at each event's (room, quantum)
                            _sre_f = (_rocc[_r_all[sa_repairer.exp_event], _q_exp] > 1)
                            _ev_room = np.bincount(
                                sa_repairer.exp_event, weights=_sre_f.astype(np.float64),
                                minlength=E).astype(np.int32)

                            # CTE: check gocc at each event's (group, quantum)
                            _gs_exp = _t_all[sa_repairer.grp_exp_event]
                            _gq_exp = np.clip(
                                _gs_exp + sa_repairer.grp_exp_offset, 0, _T_quanta - 1)
                            _cte_f = (_gocc[sa_repairer.grp_exp_group, _gq_exp] > 1)
                            _ev_grp = np.bincount(
                                sa_repairer.grp_exp_event, weights=_cte_f.astype(np.float64),
                                minlength=E).astype(np.int32)

                            grp_scores_sa = _ev_grp
                            room_scores_sa = _ev_room
                            event_scores = _ev_grp + _ev_inst + _ev_room

                        saved_genes = []
                        _move_e_self_managed = False  # MOVE E manages occupancy itself

                        # --- Move type selection ---
                        move_coin = rng_sa.random()

                        if move_coin < 0.45 and event_scores.max() > 0:
                            # MOVE A: Conflict-guided single event (45%)
                            # Weighted sampling: higher-score events more likely
                            scores_f = event_scores.astype(np.float64)
                            scores_f = np.maximum(scores_f, 0)
                            total_s = scores_f.sum()
                            if total_s > 0:
                                probs = scores_f / total_s
                                e = int(rng_sa.choice(E, p=probs))
                            else:
                                e = int(rng_sa.integers(E))

                            old_i = current_X[3 * e]
                            old_r = current_X[3 * e + 1]
                            old_t = current_X[3 * e + 2]
                            saved_genes.append((e, old_i, old_r, old_t))

                            # Occupancy-aware move: pick time/inst with least conflicts
                            coin = rng_sa.random()
                            td_a = ast_arr[e]
                            d_e_a = int(_durations[e])
                            gids_a = events[e]["group_ids"]
                            if coin < 0.55 and len(td_a) > 1:
                                # Time move: use occupancy to pick best slot
                                q_off_a = np.arange(d_e_a, dtype=np.int64)
                                q_mat_a = np.clip(
                                    td_a[:, None] + q_off_a[None, :], 0, 41
                                )
                                _occ_remove(e, current_X)
                                ci_a = min(int(old_i), _n_inst - 1)
                                ri_a = min(int(old_r), _n_room - 1)
                                # Instructor conflicts
                                i_conf_a = (
                                    _iocc[ci_a][q_mat_a] > 0
                                ).sum(axis=1).astype(np.int32)
                                # Room conflicts
                                r_conf_a = (
                                    _rocc[ri_a][q_mat_a] > 0
                                ).sum(axis=1).astype(np.int32)
                                # Group conflicts
                                g_conf_a = np.zeros(len(td_a), dtype=np.int32)
                                for _ga in gids_a:
                                    _gidx = _grp_to_idx[_ga]
                                    g_conf_a += (
                                        _gocc[_gidx][q_mat_a] > 0
                                    ).sum(axis=1).astype(np.int32)
                                combined_a = g_conf_a * 10000 + i_conf_a * 100 + r_conf_a
                                min_ca = int(combined_a.min())
                                cands_a = np.nonzero(combined_a == min_ca)[0]
                                ch_a = int(
                                    cands_a[rng_sa.integers(len(cands_a))])
                                current_X[3 * e + 2] = int(td_a[ch_a])
                                _occ_add(e, current_X)
                                _move_e_self_managed = True
                            elif coin < 0.80 and len(ai_arr[e]) > 1:
                                current_X[3 * e] = ai_arr[e][
                                    rng_sa.integers(len(ai_arr[e]))
                                ]
                            elif len(ar_arr[e]) > 1:
                                current_X[3 * e + 1] = ar_arr[e][
                                    rng_sa.integers(len(ar_arr[e]))
                                ]
                            else:
                                if len(ast_arr[e]) > 1:
                                    current_X[3 * e + 2] = ast_arr[e][
                                        rng_sa.integers(len(ast_arr[e]))
                                    ]

                        elif move_coin < 0.55:
                            # MOVE B: Group-aware swap (10%)
                            # Pick a conflicting event, then move ALL events
                            # in one of its groups to a new timeslot
                            scores_f = event_scores.astype(np.float64)
                            scores_f = np.maximum(scores_f, 0)
                            total_s = scores_f.sum()
                            if total_s > 0:
                                probs = scores_f / total_s
                                anchor = int(rng_sa.choice(E, p=probs))
                            else:
                                anchor = int(rng_sa.integers(E))

                            gids = events[anchor]["group_ids"]
                            if gids:
                                gid = gids[int(rng_sa.integers(len(gids)))]
                                group_events = _group_to_events.get(
                                    gid, [anchor])
                                # Only move a subset (up to 4) to limit disruption
                                if len(group_events) > 4:
                                    ge_idx = rng_sa.choice(
                                        len(group_events), 4, replace=False
                                    )
                                    group_events = [group_events[i]
                                                    for i in ge_idx]
                            else:
                                group_events = [anchor]

                            for e in group_events:
                                old_i = current_X[3 * e]
                                old_r = current_X[3 * e + 1]
                                old_t = current_X[3 * e + 2]
                                saved_genes.append((e, old_i, old_r, old_t))
                                # Move time (primary), room (secondary)
                                if len(ast_arr[e]) > 1:
                                    current_X[3 * e + 2] = ast_arr[e][
                                        rng_sa.integers(len(ast_arr[e]))
                                    ]

                        elif move_coin < 0.80:
                            # MOVE E: CTE solver (25%)
                            # Uses persistent occupancy tables for O(d)
                            # group/inst lookups instead of O(E) scans
                            cte_evts = np.nonzero(grp_scores_sa > 0)[0]
                            if len(cte_evts) > 0:
                                cte_s = grp_scores_sa[cte_evts].astype(
                                    np.float64)
                                cte_probs = cte_s / cte_s.sum()
                                e = int(rng_sa.choice(cte_evts, p=cte_probs))
                            else:
                                e = int(rng_sa.integers(E))

                            old_i = current_X[3 * e]
                            old_r = current_X[3 * e + 1]
                            old_t = current_X[3 * e + 2]
                            saved_genes.append((e, old_i, old_r, old_t))

                            td = ast_arr[e]
                            K = len(td)
                            if K > 1:
                                d_e = int(_durations[e])
                                gids = events[e]["group_ids"]
                                q_off = np.arange(d_e, dtype=np.int64)
                                q_mat = np.clip(
                                    td[:, None] + q_off[None, :], 0, 41
                                )  # (K, d)

                                # Group conflicts from persistent _gocc
                                # (subtract self first)
                                _occ_remove(e, current_X)
                                g_conf = np.zeros(K, dtype=np.int32)
                                for gid in gids:
                                    gidx = _grp_to_idx[gid]
                                    g_conf += (
                                        _gocc[gidx][q_mat] > 0
                                    ).sum(axis=1).astype(np.int32)

                                min_gc = int(g_conf.min())
                                if min_gc == 0:
                                    # Time-only fix
                                    ci = min(int(old_i), _n_inst - 1)
                                    ri = min(int(old_r), _n_room - 1)
                                    i_conf = (
                                        _iocc[ci][q_mat] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    r_conf = (
                                        _rocc[ri][q_mat] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    combined = g_conf * 10000 + i_conf * 100 + r_conf
                                    min_sc = int(combined.min())
                                    cands = np.nonzero(combined == min_sc)[0]
                                    ch = int(
                                        cands[rng_sa.integers(len(cands))])
                                    current_X[3 * e + 2] = int(td[ch])
                                else:
                                    # Joint (instructor, time) search
                                    i_cands = ai_arr[e]
                                    best_sc = 10**9
                                    best_ci, best_ct = int(old_i), int(old_t)
                                    n_try = min(len(i_cands), 10)
                                    if n_try < len(i_cands):
                                        idx_try = rng_sa.choice(
                                            len(i_cands), n_try, replace=False
                                        )
                                        i_try = i_cands[idx_try]
                                    else:
                                        i_try = i_cands
                                    ri_e = min(int(old_r), _n_room - 1)
                                    r_conf_e = (
                                        _rocc[ri_e][q_mat] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    for ci in i_try:
                                        ci = min(int(ci), _n_inst - 1)
                                        i_c = (
                                            _iocc[ci][q_mat] > 0
                                        ).sum(axis=1).astype(np.int32)
                                        combined = g_conf * 10000 + i_c * 100 + r_conf_e
                                        mc = int(combined.min())
                                        if mc < best_sc:
                                            best_sc = mc
                                            eq = np.nonzero(combined == mc)[0]
                                            ch = int(
                                                eq[rng_sa.integers(len(eq))])
                                            best_ct = int(td[ch])
                                            best_ci = ci
                                            if mc == 0:
                                                break
                                    current_X[3 * e] = best_ci
                                    current_X[3 * e + 2] = best_ct
                                # Re-add self to occupancy
                                _occ_add(e, current_X)
                                _move_e_self_managed = True
                            else:
                                pass  # single-slot domain, can't move

                        elif move_coin < 0.92:
                            # MOVE C: FTE-targeted instructor+time swap (17%)
                            # Uses occupancy to find (instructor, time) with fewest conflicts
                            scores_f = event_scores.astype(np.float64)
                            scores_f = np.maximum(scores_f, 0)
                            total_s = scores_f.sum()
                            if total_s > 0:
                                probs = scores_f / total_s
                                e = int(rng_sa.choice(E, p=probs))
                            else:
                                e = int(rng_sa.integers(E))

                            old_i = current_X[3 * e]
                            old_r = current_X[3 * e + 1]
                            old_t = current_X[3 * e + 2]
                            saved_genes.append((e, old_i, old_r, old_t))

                            i_cands_c = ai_arr[e]
                            td_c = ast_arr[e]
                            d_e_c = int(_durations[e])
                            if len(i_cands_c) > 1 and len(td_c) > 1:
                                _occ_remove(e, current_X)
                                q_off_c = np.arange(d_e_c, dtype=np.int64)
                                q_mat_c = np.clip(
                                    td_c[:, None] + q_off_c[None, :], 0, 41
                                )
                                # Group conflicts at each time
                                gids_c = events[e]["group_ids"]
                                g_conf_c = np.zeros(len(td_c), dtype=np.int32)
                                for _gc in gids_c:
                                    _gidx = _grp_to_idx[_gc]
                                    g_conf_c += (
                                        _gocc[_gidx][q_mat_c] > 0
                                    ).sum(axis=1).astype(np.int32)
                                # Try instructors
                                n_try_c = min(len(i_cands_c), 6)
                                if n_try_c < len(i_cands_c):
                                    idx_try_c = rng_sa.choice(
                                        len(i_cands_c), n_try_c, replace=False
                                    )
                                    i_try_c = i_cands_c[idx_try_c]
                                else:
                                    i_try_c = i_cands_c
                                best_sc_c = 10**9
                                best_ci_c = min(int(old_i), _n_inst - 1)
                                best_ct_c = int(old_t)
                                ri_c = min(int(old_r), _n_room - 1)
                                r_conf_c = (
                                    _rocc[ri_c][q_mat_c] > 0
                                ).sum(axis=1).astype(np.int32)
                                for ci_c in i_try_c:
                                    ci_c = min(int(ci_c), _n_inst - 1)
                                    i_conf_c = (
                                        _iocc[ci_c][q_mat_c] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    combined_c = g_conf_c * 10000 + i_conf_c * 100 + r_conf_c
                                    mc_c = int(combined_c.min())
                                    if mc_c < best_sc_c:
                                        best_sc_c = mc_c
                                        eq_c = np.nonzero(
                                            combined_c == mc_c)[0]
                                        ch_c = int(
                                            eq_c[rng_sa.integers(len(eq_c))])
                                        best_ct_c = int(td_c[ch_c])
                                        best_ci_c = ci_c
                                        if mc_c == 0:
                                            break
                                current_X[3 * e] = best_ci_c
                                current_X[3 * e + 2] = best_ct_c
                                _occ_add(e, current_X)
                                _move_e_self_managed = True
                            elif len(i_cands_c) > 1:
                                current_X[3 * e] = i_cands_c[
                                    rng_sa.integers(len(i_cands_c))
                                ]
                            elif len(td_c) > 1:
                                current_X[3 * e + 2] = td_c[
                                    rng_sa.integers(len(td_c))
                                ]

                        else:
                            # MOVE D: SRE-targeted room/time swap (8%)
                            # Preferentially pick events with room conflicts
                            sre_evts = np.nonzero(room_scores_sa > 0)[0]
                            if len(sre_evts) > 0 and rng_sa.random() < 0.7:
                                sre_s = room_scores_sa[sre_evts].astype(
                                    np.float64)
                                sre_probs = sre_s / sre_s.sum()
                                e = int(rng_sa.choice(sre_evts, p=sre_probs))
                            else:
                                e = int(rng_sa.integers(E))

                            old_i = current_X[3 * e]
                            old_r = current_X[3 * e + 1]
                            old_t = current_X[3 * e + 2]
                            saved_genes.append((e, old_i, old_r, old_t))

                            # Try occupancy-aware room+time swap
                            rd_d = ar_arr[e]
                            td_d = ast_arr[e]
                            d_e_d = int(_durations[e])
                            if len(rd_d) > 1 or len(td_d) > 1:
                                _occ_remove(e, current_X)
                                q_off_d = np.arange(d_e_d, dtype=np.int64)

                                if len(td_d) > 1 and len(rd_d) > 1:
                                    # Try rooms at best time
                                    q_mat_d = np.clip(
                                        td_d[:, None] + q_off_d[None, :], 0, 41
                                    )
                                    # Score all times by room+inst+group
                                    ci_d = min(int(old_i), _n_inst - 1)
                                    i_conf_d = (
                                        _iocc[ci_d][q_mat_d] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    g_conf_d = np.zeros(
                                        len(td_d), dtype=np.int32)
                                    for _gd in events[e]["group_ids"]:
                                        _gidx = _grp_to_idx[_gd]
                                        g_conf_d += (
                                            _gocc[_gidx][q_mat_d] > 0
                                        ).sum(axis=1).astype(np.int32)
                                    # Find best time
                                    time_sc = g_conf_d * 10000 + i_conf_d
                                    min_ts = int(time_sc.min())
                                    t_cands = np.nonzero(time_sc == min_ts)[0]
                                    t_ch = int(
                                        t_cands[rng_sa.integers(len(t_cands))])
                                    best_t_d = int(td_d[t_ch])
                                    # Now find best room at that time
                                    q_slots = np.clip(
                                        np.arange(
                                            best_t_d, best_t_d + d_e_d), 0, 41
                                    )
                                    best_r_sc = 10**9
                                    best_r_d = min(int(old_r), _n_room - 1)
                                    n_r_try = min(len(rd_d), 8)
                                    if n_r_try < len(rd_d):
                                        r_idx = rng_sa.choice(
                                            len(rd_d), n_r_try, replace=False
                                        )
                                        r_try = rd_d[r_idx]
                                    else:
                                        r_try = rd_d
                                    for ri_d in r_try:
                                        ri_d = min(int(ri_d), _n_room - 1)
                                        r_sc = int(
                                            (_rocc[ri_d][q_slots] > 0).sum())
                                        if r_sc < best_r_sc:
                                            best_r_sc = r_sc
                                            best_r_d = ri_d
                                            if r_sc == 0:
                                                break
                                    current_X[3 * e + 1] = best_r_d
                                    current_X[3 * e + 2] = best_t_d
                                elif len(rd_d) > 1:
                                    # Room-only swap
                                    cur_t = int(old_t)
                                    q_slots = np.clip(
                                        np.arange(cur_t, cur_t + d_e_d), 0, 41
                                    )
                                    best_r_sc = 10**9
                                    best_r_d = min(int(old_r), _n_room - 1)
                                    for ri_d in rd_d:
                                        ri_d = min(int(ri_d), _n_room - 1)
                                        r_sc = int(
                                            (_rocc[ri_d][q_slots] > 0).sum())
                                        if r_sc < best_r_sc:
                                            best_r_sc = r_sc
                                            best_r_d = ri_d
                                            if r_sc == 0:
                                                break
                                    current_X[3 * e + 1] = best_r_d
                                else:
                                    # Time-only swap (occupancy-aware)
                                    q_mat_d = np.clip(
                                        td_d[:, None] + q_off_d[None, :], 0, 41
                                    )
                                    ri_d = min(int(old_r), _n_room - 1)
                                    r_conf_d = (
                                        _rocc[ri_d][q_mat_d] > 0
                                    ).sum(axis=1).astype(np.int32)
                                    min_rc = int(r_conf_d.min())
                                    r_cands = np.nonzero(r_conf_d == min_rc)[0]
                                    r_ch = int(
                                        r_cands[rng_sa.integers(len(r_cands))])
                                    current_X[3 * e + 2] = int(td_d[r_ch])
                                _occ_add(e, current_X)
                                _move_e_self_managed = True
                            else:
                                # Truly stuck — random perturbation
                                if len(ai_arr[e]) > 1:
                                    current_X[3 * e] = ai_arr[e][
                                        rng_sa.integers(len(ai_arr[e]))
                                    ]

                        # --- Evaluate and accept/reject ---
                        # Remove old, add new in occupancy (skip for MOVE E which self-manages)
                        if not _move_e_self_managed:
                            for e_sg, oi, orr, ot in saved_genes:
                                # Remove old placement
                                _d_sg = int(_durations[e_sg])
                                _oi = min(int(oi), _n_inst - 1)
                                _orr_c = min(int(orr), _n_room - 1)
                                for _q in range(int(ot), min(int(ot) + _d_sg, _T_quanta)):
                                    _iocc[_oi, _q] -= 1
                                    _rocc[_orr_c, _q] -= 1
                                    for _gid in events[e_sg]["group_ids"]:
                                        _gocc[_grp_to_idx[_gid], _q] -= 1
                                # Add new placement
                                _ni = min(int(current_X[3*e_sg]), _n_inst - 1)
                                _nr = min(
                                    int(current_X[3*e_sg+1]), _n_room - 1)
                                _nt = int(current_X[3*e_sg+2])
                                for _q in range(_nt, min(_nt + _d_sg, _T_quanta)):
                                    _iocc[_ni, _q] += 1
                                    _rocc[_nr, _q] += 1
                                    for _gid in events[e_sg]["group_ids"]:
                                        _gocc[_grp_to_idx[_gid], _q] += 1

                        # --- Fast scoring from occupancy tables ---
                        # FPC/FFC/CQF/ICTD are 0 since we pick from allowed
                        # domains; periodic full-eval sync catches any drift.
                        new_hard = _hard_from_occ()

                        # Periodic full-eval sync to correct any occ drift
                        if sa_iter_total % _full_eval_interval == 0:
                            G_sync = fast_evaluate_hard_vectorized(
                                current_X.reshape(1, -1), vd
                            )
                            new_hard = float(sum(G_sync[0, j] for j in _strict))
                            # Rebuild occ tables if drift detected
                            occ_hard = _hard_from_occ()
                            if abs(occ_hard - new_hard) > 1:
                                _gocc[:] = 0
                                _iocc[:] = 0
                                _rocc[:] = 0
                                for _e in range(E):
                                    _occ_add(_e, current_X)
                                current_hard = new_hard

                        delta = new_hard - current_hard
                        if delta <= 0:
                            current_hard = new_hard
                            no_improve_count = 0
                            if new_hard < best_hard_sa:
                                best_hard_sa = new_hard
                                best_X_sa = current_X.copy()
                                if best_hard_sa == 0:
                                    break
                        elif rng_sa.random() < np.exp(
                            -delta / max(temp, 1e-10)
                        ):
                            current_hard = new_hard
                            no_improve_count += 1
                        else:
                            # Reject: undo occupancy + restore genes
                            if _move_e_self_managed:
                                # MOVE E self-managed: undo its inline occ changes
                                for e_sg, oi, orr, ot in saved_genes:
                                    _occ_remove(e_sg, current_X)  # remove new
                                    current_X[3 * e_sg] = oi
                                    current_X[3 * e_sg + 1] = orr
                                    current_X[3 * e_sg + 2] = ot
                                    _occ_add(e_sg, current_X)  # restore old
                            else:
                                for e_sg, oi, orr, ot in saved_genes:
                                    _d_sg = int(_durations[e_sg])
                                    # Remove new placement
                                    _ni = min(
                                        int(current_X[3*e_sg]), _n_inst - 1)
                                    _nr = min(
                                        int(current_X[3*e_sg+1]), _n_room - 1)
                                    _nt = int(current_X[3*e_sg+2])
                                    for _q in range(_nt, min(_nt + _d_sg, _T_quanta)):
                                        _iocc[_ni, _q] -= 1
                                        _rocc[_nr, _q] -= 1
                                        for _gid in events[e_sg]["group_ids"]:
                                            _gocc[_grp_to_idx[_gid], _q] -= 1
                                    # Restore old placement
                                    current_X[3 * e_sg] = oi
                                    current_X[3 * e_sg + 1] = orr
                                    current_X[3 * e_sg + 2] = ot
                                    _oi = min(int(oi), _n_inst - 1)
                                    _orr_c = min(int(orr), _n_room - 1)
                                    for _q in range(int(ot), min(int(ot) + _d_sg, _T_quanta)):
                                        _iocc[_oi, _q] += 1
                                        _rocc[_orr_c, _q] += 1
                                        for _gid in events[e_sg]["group_ids"]:
                                            _gocc[_grp_to_idx[_gid], _q] += 1
                            no_improve_count += 1

                        # Restart from best when deeply stuck
                        if no_improve_count > E * 4:
                            current_X = best_X_sa.copy()
                            current_hard = best_hard_sa
                            no_improve_count = 0
                            sa_stagnation_restarts += 1
                            # Exit if too many restarts without improvement
                            if sa_stagnation_restarts >= 5:
                                self.logger.info(
                                    "SA polishing: stagnation exit after %d restarts", sa_stagnation_restarts)
                                break
                            # Apply random perturbation to ~5% of events
                            # to escape local minima
                            n_perturb = max(E // 20, 5)
                            perturb_evts = rng_sa.choice(
                                E, n_perturb, replace=False)
                            for _pe in perturb_evts:
                                if len(ast_arr[_pe]) > 1:
                                    current_X[3*_pe+2] = ast_arr[_pe][
                                        rng_sa.integers(len(ast_arr[_pe]))
                                    ]
                                if rng_sa.random() < 0.3 and len(ai_arr[_pe]) > 1:
                                    current_X[3*_pe] = ai_arr[_pe][
                                        rng_sa.integers(len(ai_arr[_pe]))
                                    ]
                            # Rebuild occupancy tables
                            _gocc[:] = 0
                            _iocc[:] = 0
                            _rocc[:] = 0
                            for _e in range(E):
                                _occ_add(_e, current_X)
                            # Re-evaluate from freshly built occ tables
                            current_hard = _hard_from_occ()

                    if best_hard_sa == 0:
                        break
                    # Reset stagnation counter if this phase improved
                    if best_hard_sa < phase_best_before:
                        sa_stagnation_restarts = 0
                    # Exit if stagnation limit hit (inner loop broke out)
                    if sa_stagnation_restarts >= 5:
                        break

                if best_hard_sa < best_hard_before:
                    # Log detailed breakdown
                    G_final_sa = fast_evaluate_hard_vectorized(
                        best_X_sa.reshape(1, -1), vd
                    )
                    _cte = float(G_final_sa[0, 0]
                                 ) if G_final_sa.shape[1] > 0 else 0
                    _fte = float(G_final_sa[0, 1]
                                 ) if G_final_sa.shape[1] > 1 else 0
                    _sre = float(G_final_sa[0, 2]
                                 ) if G_final_sa.shape[1] > 2 else 0
                    self.logger.info(
                        "SA polishing: hard %d -> %d (delta=%d, %d iters, %d phases)"
                        " [CTE=%d FTE=%d SRE=%d]",
                        int(best_hard_before),
                        int(best_hard_sa),
                        int(best_hard_before - best_hard_sa),
                        sa_iter_total,
                        n_phases,
                        int(_cte), int(_fte), int(_sre),
                    )
                    res.pop[best_idx].set("X", best_X_sa)
                    from src.pipeline.scheduling_problem import SchedulingProblem
                    prob_pol = SchedulingProblem(pkl_path)
                    out = {}
                    prob_pol._evaluate(best_X_sa.reshape(1, -1), out)
                    F[best_idx] = out["F"][0]
                    G[best_idx] = out["G"][0]
                    cv[best_idx] = best_hard_sa
                else:
                    self.logger.info(
                        "SA polishing: no improvement (best=%d, %d iters)",
                        int(best_hard_sa),
                        sa_iter_total,
                    )
            except Exception:
                self.logger.exception("Post-GA SA polishing failed")

        # ── Generate all output artefacts (plots + PDFs) ─────────
        self._generate_outputs(
            res=res,
            callback=callback,
            pkl_data=pkl_data,
            ctx=ctx,
            qts=qts,
            best_idx=best_idx,
        )

        # Per-gen timing summary
        gen_times: list[float] = getattr(callback, "gen_times", [])
        timing_summary: dict[str, Any] = {}
        if gen_times:
            gt = np.array(gen_times)
            timing_summary = {
                "mean_s": round(float(gt.mean()), 4),
                "std_s": round(float(gt.std()), 4),
                "min_s": round(float(gt.min()), 4),
                "max_s": round(float(gt.max()), 4),
                "p50_s": round(float(np.median(gt)), 4),
                "p95_s": round(float(np.percentile(gt, 95)), 4),
            }

        return {
            "solver": "pymoo",
            "mode": self.mode,
            "version": __version__,
            "experiment_class": type(self).__name__,
            "framework": "experiments_v3",
            "config": self._config_dict(),
            "best_hard": float(F[best_idx, 0]),
            "best_soft": float(F[best_idx, 1]),
            "best_cv": float(cv[best_idx]),
            "n_feasible": int((cv == 0).sum()),
            "elapsed_s": round(elapsed, 2),
            "sec_per_gen": round(elapsed / self.ngen, 3) if self.ngen else 0,
            "timing_per_gen": timing_summary,
            "convergence_hard": getattr(callback, "best_hards", []),
            "convergence_soft": getattr(callback, "best_softs", []),
            "convergence_constraints": getattr(callback, "best_breakdowns", []),
            "hypervolumes": getattr(callback, "hypervolumes", []),
            "spacings": getattr(callback, "spacings", []),
            "diversities": getattr(callback, "diversities", []),
            "feasibility_rates": getattr(callback, "feasibility_rates", []),
            "igds": getattr(callback, "igds", []),
            "final_F": F.tolist(),
            "final_G": G.tolist(),
        }


# =====================================================================
#  Concrete Modes
# =====================================================================


class AdaptiveExperiment(GAExperiment):
    """Stagnation-aware: ramps mutation + elite repair + diversity injection.

    Starts conservative (mut=0.05), escalates to *mutation_hi* when
    best_hard stalls for *stagnation_window* generations.  On deep
    stagnation (2× window), injects random individuals to escape
    local minima.

    Default config:
        pop_size=100, ngen=300, cx=0.5, mut=0.05→0.25
    """

    def __init__(
        self,
        *,
        stagnation_window: int = 12,
        mutation_hi: float = 0.25,
        elite_pct: float = 0.15,
        repair_iters: int = 4,
        **kwargs,
    ):
        kwargs.setdefault("mode", "adaptive")
        kwargs.setdefault("ngen", 300)
        kwargs.setdefault("crossover_prob", 0.5)
        kwargs.setdefault("mutation_event_prob", 0.05)
        super().__init__(**kwargs)
        self.stagnation_window = stagnation_window
        self.mutation_hi = mutation_hi
        self.elite_pct = elite_pct
        self.repair_iters = repair_iters

    def _build_callback(self, pkl_path: str):
        log_interval = self.log_interval
        stagnation_window = self.stagnation_window
        mutation_lo = self.mutation_event_prob
        mutation_hi = self.mutation_hi
        elite_pct = self.elite_pct
        repair_iters = self.repair_iters
        _pkl_path = pkl_path
        n_workers = max(1, (os.cpu_count() or 1) - 1)

        class CB(GACallbackBase):
            """Adaptive callback — escalates mutation + diversity on stagnation."""

            def __init__(self, _log_interval):
                super().__init__(_log_interval)
                self._stagnant = 0
                self._escalated = False
                self._deep_stagnant = 0  # tracks prolonged stagnation

            def _on_generation(self, algorithm, F, G, cv, best_idx):
                cur_hard = self.best_hards[-1]

                if len(self.best_hards) >= 2 and cur_hard >= self.best_hards[-2]:
                    self._stagnant += 1
                    self._deep_stagnant += 1
                else:
                    self._stagnant = 0
                    self._deep_stagnant = 0
                    if self._escalated:
                        self._set_mutation(algorithm, mutation_lo)
                        self._escalated = False

                # --- Stage 1: Mutation escalation + elite repair ---
                if self._stagnant >= stagnation_window and not self._escalated:
                    self._set_mutation(algorithm, mutation_hi)
                    self._escalated = True
                    logging.getLogger(__name__).info(
                        "stagnation @ gen %d — mutation -> %s, elite repair ON",
                        algorithm.n_gen,
                        mutation_hi,
                    )

                if self._escalated:
                    self.repair_gens.append(algorithm.n_gen or 0)
                    pop = algorithm.pop
                    gen = algorithm.n_gen or 0
                    n_elite = max(1, int(len(pop) * elite_pct))
                    elite_idxs = np.argsort(cv)[:n_elite]

                    X_copies = [pop[idx].get("X").copy() for idx in elite_idxs]
                    with ProcessPoolExecutor(max_workers=n_workers) as pool:
                        futures = [
                            pool.submit(
                                _repair_single_elite,
                                X_copies[j],
                                _pkl_path,
                                repair_iters,
                                gen,
                                int(elite_idxs[j]),
                            )
                            for j in range(len(elite_idxs))
                        ]
                        results = [f.result() for f in futures]

                    modified = []
                    for j, idx in enumerate(elite_idxs):
                        pop[idx].set("X", results[j])
                        modified.append(pop[idx])
                    _reeval_modified(algorithm, modified)

                # --- Stage 2: Deep stagnation → diversity injection ---
                if self._deep_stagnant >= stagnation_window * 3:
                    self._inject_diversity(algorithm, cv)
                    self._deep_stagnant = 0
                    logging.getLogger(__name__).info(
                        "deep stagnation @ gen %d — injecting diverse individuals",
                        algorithm.n_gen,
                    )

            def _inject_diversity(self, algorithm, cv):
                """Replace worst 20% of population with fresh random individuals."""
                pop = algorithm.pop
                n_pop = len(pop)
                n_replace = max(2, int(n_pop * 0.20))
                worst_idxs = np.argsort(cv)[-n_replace:]

                # Generate fresh random chromosomes from domains
                from src.pipeline.pymoo_operators import RandomDomainSampling
                sampler = RandomDomainSampling(_pkl_path)
                fresh_X = sampler._do(algorithm.problem, n_replace)

                modified = []
                for j, idx in enumerate(worst_idxs):
                    pop[idx].set("X", fresh_X[j])
                    modified.append(pop[idx])
                _reeval_modified(algorithm, modified)

            @staticmethod
            def _set_mutation(algorithm, prob):
                mut = algorithm.mating.mutation
                if hasattr(mut, "event_prob"):
                    mut.event_prob = prob

        return CB(log_interval)
