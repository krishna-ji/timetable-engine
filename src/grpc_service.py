"""gRPC service implementation for the Scheduling Engine.

Implements the SchedulerEngine service defined in proto/scheduler.proto.
Runs AdaptiveExperiment in-process and streams progress updates back to
the caller (Django backend).
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
import traceback
from pathlib import Path
from threading import Event
from typing import Iterator

import grpc

from src.proto import scheduler_pb2, scheduler_pb2_grpc

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SchedulerEngineServicer(scheduler_pb2_grpc.SchedulerEngineServicer):
    """gRPC service that wraps AdaptiveExperiment."""

    # ── RunSchedule (server streaming) ────────────────────────

    def RunSchedule(
        self,
        request: scheduler_pb2.ScheduleRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[scheduler_pb2.ScheduleUpdate]:
        """Run the scheduling algorithm and stream progress updates."""
        job_id = request.job_id
        logger.info("RunSchedule job_id=%s  gens=%d  pop=%d  seed=%d",
                     job_id, request.generations, request.population_size, request.seed)

        try:
            # Write incoming data to a temp directory that the engine can read
            work_dir = Path(tempfile.mkdtemp(prefix="sch_grpc_"))
            data_dir = work_dir / "data"
            data_dir.mkdir()

            (data_dir / "Course.json").write_text(request.courses_json or "[]")
            (data_dir / "Groups.json").write_text(request.groups_json or "[]")
            (data_dir / "Instructors.json").write_text(request.instructors_json or "[]")
            (data_dir / "Rooms.json").write_text(request.rooms_json or "[]")

            output_dir = work_dir / "output"
            output_dir.mkdir()

            # Import engine components (lazy so server starts fast)
            from src.experiments import AdaptiveExperiment

            ngen = request.generations or 300
            pop_size = request.population_size or 100
            seed = request.seed or 42

            # Progress broker: the callback puts updates here, the
            # streaming response reads them.
            progress_queue: list[scheduler_pb2.ScheduleUpdate] = []
            done_event = Event()

            # Build experiment
            exp = AdaptiveExperiment(
                pop_size=pop_size,
                ngen=ngen,
                seed=seed,
                export_pdf=False,
                verbose=False,
                data_dir=str(data_dir),
                output_dir=str(output_dir),
            )

            # Patch the callback's notify method to also push progress to the
            # queue. We do this by patching the notify method on the built
            # callback instance, so the callback retains its original type
            # (and __call__) from pymoo's Callback base class.
            _original_build = exp._build_callback

            def _patched_build(pkl_path):
                inner = _original_build(pkl_path)
                _original_notify = inner.notify

                def _notify(algorithm):
                    _original_notify(algorithm)
                    gen = algorithm.n_gen
                    best_hard = float(inner.best_hards[-1]) if inner.best_hards else 0
                    best_soft = float(inner.best_softs[-1]) if inner.best_softs else 0
                    pct = (gen / ngen) * 100.0
                    progress_queue.append(scheduler_pb2.ScheduleUpdate(
                        job_id=job_id,
                        type=scheduler_pb2.PROGRESS,
                        current_generation=gen,
                        total_generations=ngen,
                        progress_percentage=pct,
                        best_hard=best_hard,
                        best_soft=best_soft,
                        message=f"Gen {gen}/{ngen}: hard={best_hard:.0f} soft={best_soft:.1f}",
                    ))

                inner.notify = _notify
                return inner

            exp._build_callback = _patched_build

            # Run the experiment in the current thread.
            # We yield progress messages in a polling loop on a separate
            # mechanism: because pymoo's minimize() blocks, we run the
            # experiment in a helper thread and yield from the main one.
            import threading

            result_holder: dict = {}
            error_holder: list = []

            def _run():
                try:
                    result_holder["result"] = exp.run()
                except Exception as exc:
                    error_holder.append(exc)
                finally:
                    done_event.set()

            t = threading.Thread(target=_run, daemon=True)
            t0 = time.time()
            t.start()

            # Yield progress updates as they arrive
            sent = 0
            ga_done = False
            sa_poll_count = 0
            while not done_event.is_set():
                done_event.wait(timeout=1.0)
                while sent < len(progress_queue):
                    update = progress_queue[sent]
                    yield update
                    sent += 1
                    # Detect when GA generations are done
                    if (update.type == scheduler_pb2.PROGRESS
                            and update.current_generation >= ngen
                            and not ga_done):
                        ga_done = True

                # If GA is done but experiment still running → SA polishing
                if ga_done and not done_event.is_set():
                    sa_poll_count += 1
                    yield scheduler_pb2.ScheduleUpdate(
                        job_id=job_id,
                        type=scheduler_pb2.PROGRESS,
                        current_generation=ngen,
                        total_generations=ngen,
                        progress_percentage=90.0 + min(sa_poll_count * 0.1, 8.0),
                        best_hard=progress_queue[-1].best_hard if progress_queue else 0,
                        best_soft=progress_queue[-1].best_soft if progress_queue else 0,
                        message=f"SA polishing — refining conflicts ({sa_poll_count}s)...",
                    )

            # Drain remaining progress messages
            while sent < len(progress_queue):
                yield progress_queue[sent]
                sent += 1

            elapsed = time.time() - t0

            if error_holder:
                err = error_holder[0]
                logger.error("RunSchedule failed: %s", err)
                yield scheduler_pb2.ScheduleUpdate(
                    job_id=job_id,
                    type=scheduler_pb2.FAILED,
                    error_message=str(err),
                    elapsed_seconds=elapsed,
                )
                return

            # Read the output files
            results = result_holder.get("result", {})
            schedule_json_str = ""
            results_json_str = json.dumps(results)

            # List all files in output_dir for diagnostics
            try:
                all_files = list(output_dir.rglob("*"))
                logger.info(
                    "output_dir contents (%d files): %s",
                    len(all_files),
                    [str(f.relative_to(output_dir)) for f in all_files if f.is_file()],
                )
            except Exception:
                logger.warning("Could not list output_dir contents")

            schedule_path = _find_file(output_dir, "schedule.json")
            if schedule_path:
                schedule_json_str = schedule_path.read_text()
                logger.info("schedule.json found (%d bytes)", len(schedule_json_str))
            else:
                logger.warning("schedule.json not found in %s", output_dir)
                # Check if run.log has errors about schedule export
                run_log_path = _find_file(output_dir, "run.log")
                if run_log_path:
                    log_tail = run_log_path.read_text()[-2000:]
                    logger.warning("run.log tail:\n%s", log_tail)

            yield scheduler_pb2.ScheduleUpdate(
                job_id=job_id,
                type=scheduler_pb2.COMPLETED,
                current_generation=ngen,
                total_generations=ngen,
                progress_percentage=100.0,
                best_hard=float(results.get("best_hard", 0)),
                best_soft=float(results.get("best_soft", 0)),
                results_json=results_json_str,
                schedule_json=schedule_json_str,
                elapsed_seconds=elapsed,
                message="Completed",
            )

            # Cleanup
            import shutil
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

            logger.info("RunSchedule job_id=%s completed in %.1fs", job_id, elapsed)

        except Exception as exc:
            logger.exception("RunSchedule unexpected error")
            yield scheduler_pb2.ScheduleUpdate(
                job_id=job_id,
                type=scheduler_pb2.FAILED,
                error_message=traceback.format_exc(),
            )

    # ── ValidateData ──────────────────────────────────────────

    def ValidateData(
        self,
        request: scheduler_pb2.ValidateRequest,
        context: grpc.ServicerContext,
    ) -> scheduler_pb2.ValidateResponse:
        """Validate scheduling data without running the solver."""
        try:
            import tempfile as _tmpmod

            tmp = Path(_tmpmod.mkdtemp(prefix="sch_validate_"))
            (tmp / "Course.json").write_text(request.courses_json or "[]")
            (tmp / "Groups.json").write_text(request.groups_json or "[]")
            (tmp / "Instructors.json").write_text(request.instructors_json or "[]")
            (tmp / "Rooms.json").write_text(request.rooms_json or "[]")

            from src.io.time_system import QuantumTimeSystem
            from src.io.data_loader import (
                load_courses, load_groups, load_instructors, load_rooms,
                link_courses_and_groups, link_courses_and_instructors,
            )
            from src.domain.types import SchedulingContext
            from src.io.validator import InputValidator

            qts = QuantumTimeSystem()
            courses, _skipped = load_courses(str(tmp / "Course.json"))
            groups = load_groups(str(tmp / "Groups.json"), qts)
            instructors = load_instructors(str(tmp / "Instructors.json"), qts)
            rooms = load_rooms(str(tmp / "Rooms.json"), qts)

            link_courses_and_groups(courses, groups)
            link_courses_and_instructors(courses, instructors)

            ctx = SchedulingContext(
                courses=courses,
                instructors=instructors,
                groups=groups,
                rooms=rooms,
                available_quanta=list(range(qts.total_quanta)),
            )

            validator = InputValidator(ctx)
            errors = validator.validate(parallel=True)

            issues = []
            stats: dict[str, int] = {
                "critical_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "missing_instructors": 0,
                "missing_rooms": 0,
                "capacity_issues": 0,
                "availability_issues": 0,
            }
            for err in errors:
                sev = "critical" if err.severity == "ERROR" else (
                    "warning" if err.severity == "WARNING" else "info")
                issues.append(scheduler_pb2.ValidationIssue(
                    category=err.category.lower(),
                    severity=sev,
                    message=err.message,
                ))
                stats[f"{sev}_count"] = stats.get(f"{sev}_count", 0) + 1
                cat = err.category.lower()
                if "instructor" in cat or "teacher" in cat:
                    stats["missing_instructors"] += 1
                elif "room" in cat:
                    stats["missing_rooms"] += 1
                elif "capacity" in cat:
                    stats["capacity_issues"] += 1
                elif "availability" in cat:
                    stats["availability_issues"] += 1

            # Calculate health score
            score = 100.0
            has_c = len(courses) > 0
            has_g = len(groups) > 0
            has_i = len(instructors) > 0
            has_r = len(rooms) > 0
            completeness = sum([has_c, has_g, has_i, has_r])
            score -= (4 - completeness) * 10
            if len(errors) > 0:
                score -= min(40, 10 * math.log10(len(errors) + 1))
            if has_c and has_g:
                score += 10 if (has_i and has_r) else (5 if (has_i or has_r) else 0)
            score = max(0.0, min(100.0, score))

            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

            return scheduler_pb2.ValidateResponse(
                valid=score >= 70 and stats["critical_count"] == 0,
                health_score=score,
                issues=issues,
                stats=stats,
            )

        except Exception as exc:
            logger.exception("ValidateData error")
            return scheduler_pb2.ValidateResponse(
                valid=False,
                health_score=0.0,
                issues=[scheduler_pb2.ValidationIssue(
                    category="internal",
                    severity="critical",
                    message=f"Validation failed: {exc}",
                )],
            )

    # ── Ping ──────────────────────────────────────────────────

    def Ping(
        self,
        request: scheduler_pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> scheduler_pb2.PingResponse:
        return scheduler_pb2.PingResponse(status="ok", version="1.0.0")


# ── helpers ───────────────────────────────────────────────────

def _find_file(root: Path, name: str) -> Path | None:
    """Recursively find a file under root."""
    for p in root.rglob(name):
        return p
    return None
