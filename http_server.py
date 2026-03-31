"""
FastAPI HTTP server for the Scheduling Engine.

Provides REST endpoints that the EdutableStudio Next.js app can call
directly (via the AlgorithmRunner service). Runs alongside the gRPC server.

Endpoints:
  GET  /api/ping          – Health check
  POST /api/validate      – Validate scheduling data
  POST /api/schedule      – Run the scheduler (blocking, returns result)
  GET  /api/engine-status  – Engine status + version
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Schedule Engine API",
    version="2.0.0",
    description="University Timetable Scheduling Engine — HTTP interface",
)

# CORS for Next.js frontend
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Active jobs tracking ──────────────────────────────────────────────

_active_jobs: dict[str, dict] = {}


# ── Request / Response Models ─────────────────────────────────────────

class PingResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    uptime_seconds: float = 0


class ValidateRequest(BaseModel):
    courses: list[dict[str, Any]] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    instructors: list[dict[str, Any]] = Field(default_factory=list)
    rooms: list[dict[str, Any]] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    category: str
    severity: str
    message: str


class ValidateResponse(BaseModel):
    valid: bool
    health_score: float
    issues: list[ValidationIssue]
    stats: dict[str, int]


class ScheduleRequest(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    courses: list[dict[str, Any]] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    instructors: list[dict[str, Any]] = Field(default_factory=list)
    rooms: list[dict[str, Any]] = Field(default_factory=list)
    generations: int = 300
    population_size: int = 100
    seed: int = 42
    solver: str = "cpsat"  # "cpsat" (default) or "ga"


class ScheduleResponse(BaseModel):
    job_id: str
    status: str  # "completed" | "failed"
    schedule: list[dict[str, Any]] = Field(default_factory=list)
    best_hard: float = 0
    best_soft: float = 0
    elapsed_seconds: float = 0
    fitness_history: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    # CP-SAT specific fields
    violations: dict[str, int] | None = None
    rooms_assigned: int | None = None
    rooms_failed: int | None = None


class EngineStatusResponse(BaseModel):
    connected: bool = True
    host: str = ""
    version: str = "2.0.0"
    latency_ms: float | None = None
    active_runs: int = 0
    queued_runs: int = 0


# ── Startup ───────────────────────────────────────────────────────────

_start_time = time.time()


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/ping", response_model=PingResponse)
async def ping():
    return PingResponse(
        status="ok",
        version="2.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/api/engine-status", response_model=EngineStatusResponse)
async def engine_status():
    return EngineStatusResponse(
        connected=True,
        host=os.environ.get("HOSTNAME", "engine"),
        version="2.0.0",
        latency_ms=1.0,
        active_runs=len(_active_jobs),
        queued_runs=0,
    )


@app.post("/api/validate", response_model=ValidateResponse)
async def validate_data(req: ValidateRequest):
    """Validate scheduling input data without running the solver."""
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="sch_validate_"))
        (work_dir / "Course.json").write_text(json.dumps(req.courses))
        (work_dir / "Groups.json").write_text(json.dumps(req.groups))
        (work_dir / "Instructors.json").write_text(json.dumps(req.instructors))
        (work_dir / "Rooms.json").write_text(json.dumps(req.rooms))

        from src.io.time_system import QuantumTimeSystem
        from src.io.data_loader import (
            load_courses,
            load_groups,
            load_instructors,
            load_rooms,
            link_courses_and_groups,
            link_courses_and_instructors,
        )
        from src.domain.types import SchedulingContext
        from src.io.validator import InputValidator

        qts = QuantumTimeSystem()
        courses, _sk = load_courses(str(work_dir / "Course.json"))
        groups = load_groups(str(work_dir / "Groups.json"), qts)
        instructors = load_instructors(str(work_dir / "Instructors.json"), qts)
        rooms = load_rooms(str(work_dir / "Rooms.json"), qts)

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

        issues: list[ValidationIssue] = []
        stats = {
            "critical_count": 0,
            "warning_count": 0,
            "info_count": 0,
        }
        for err in errors:
            sev = (
                "critical"
                if err.severity == "ERROR"
                else ("warning" if err.severity == "WARNING" else "info")
            )
            stats[f"{sev}_count"] = stats.get(f"{sev}_count", 0) + 1
            issues.append(ValidationIssue(
                category=err.category.lower(),
                severity=sev,
                message=err.message,
            ))

        total_issues = stats["critical_count"] + stats["warning_count"]
        health = max(0, 100 - stats["critical_count"] * 20 - stats["warning_count"] * 5)

        shutil.rmtree(work_dir, ignore_errors=True)

        return ValidateResponse(
            valid=stats["critical_count"] == 0,
            health_score=health,
            issues=issues,
            stats=stats,
        )

    except Exception as exc:
        logger.exception("Validation error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/schedule", response_model=ScheduleResponse)
async def run_schedule(req: ScheduleRequest):
    """Run the scheduling algorithm. Returns when complete."""
    job_id = req.job_id
    logger.info(
        "Schedule request job_id=%s solver=%s courses=%d groups=%d instructors=%d rooms=%d",
        job_id, req.solver, len(req.courses), len(req.groups),
        len(req.instructors), len(req.rooms),
    )

    _active_jobs[job_id] = {"status": "running", "started": time.time()}

    try:
        if req.solver == "cpsat":
            return await _run_cpsat(req, job_id)
        else:
            return await _run_ga(req, job_id)
    except Exception as exc:
        logger.exception("Schedule run failed job_id=%s", job_id)
        return ScheduleResponse(
            job_id=job_id,
            status="failed",
            error=str(exc),
            elapsed_seconds=round(time.time() - _active_jobs.get(job_id, {}).get("started", time.time()), 2),
        )
    finally:
        _active_jobs.pop(job_id, None)


async def _run_cpsat(req: ScheduleRequest, job_id: str) -> ScheduleResponse:
    """Run the CP-SAT 3-phase pipeline."""
    from cpsat_solver import solve_timetable_from_json, SolveConfig

    cfg = SolveConfig(seeds=max(1, req.seed))
    result = solve_timetable_from_json(
        courses=req.courses,
        groups=req.groups,
        instructors=req.instructors,
        rooms=req.rooms,
        config=cfg,
    )

    return ScheduleResponse(
        job_id=job_id,
        status="completed" if result.success else "failed",
        schedule=result.schedule,
        best_hard=float(result.violations),
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
        violations=result.violation_details,
        rooms_assigned=result.rooms_assigned,
        rooms_failed=result.rooms_failed,
    )


async def _run_ga(req: ScheduleRequest, job_id: str) -> ScheduleResponse:
    """Run the legacy GA solver."""
    work_dir = Path(tempfile.mkdtemp(prefix="sch_run_"))
    data_dir = work_dir / "data"
    data_dir.mkdir()
    output_dir = work_dir / "output"
    output_dir.mkdir()

    try:
        (data_dir / "Course.json").write_text(json.dumps(req.courses))
        (data_dir / "Groups.json").write_text(json.dumps(req.groups))
        (data_dir / "Instructors.json").write_text(json.dumps(req.instructors))
        (data_dir / "Rooms.json").write_text(json.dumps(req.rooms))

        # Import engine (lazy for fast startup)
        from src.experiments import AdaptiveExperiment

        exp = AdaptiveExperiment(
            pop_size=req.population_size,
            ngen=req.generations,
            seed=req.seed,
            export_pdf=False,
            verbose=False,
            data_dir=str(data_dir),
            output_dir=str(output_dir),
        )

        t0 = time.time()
        result = exp.run()
        elapsed = time.time() - t0

        # Read schedule output
        schedule_data: list[dict] = []
        schedule_path = None
        for candidate in [
            output_dir / "schedule.json",
            output_dir / "ga_adaptive" / "schedule.json",
        ]:
            if candidate.exists():
                schedule_path = candidate
                break

        if schedule_path:
            raw = json.loads(schedule_path.read_text())
            if isinstance(raw, dict) and "decoded_schedule" in raw:
                schedule_data = raw["decoded_schedule"]
            elif isinstance(raw, list):
                schedule_data = raw
            else:
                schedule_data = raw.get("schedule", [])

        best_hard = float(result.get("best_hard", 0)) if isinstance(result, dict) else 0
        best_soft = float(result.get("best_soft", 0)) if isinstance(result, dict) else 0

        # Build decoded schedule with time information from quantum system
        decoded = _decode_schedule_quanta(schedule_data)

        logger.info(
            "Schedule job_id=%s completed in %.1fs hard=%d soft=%.1f entries=%d",
            job_id, elapsed, best_hard, best_soft, len(decoded),
        )

        return ScheduleResponse(
            job_id=job_id,
            status="completed",
            schedule=decoded,
            best_hard=best_hard,
            best_soft=best_soft,
            elapsed_seconds=round(elapsed, 2),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────────

def _decode_schedule_quanta(entries: list[dict]) -> list[dict]:
    """Enrich schedule entries with human-readable time info from quanta."""
    from src.io.time_system import QuantumTimeSystem

    qts = QuantumTimeSystem()
    result = []
    for entry in entries:
        enriched = dict(entry)
        quanta = entry.get("session_quanta", [])
        if quanta and not entry.get("time"):
            time_map: dict[str, list[dict[str, str]]] = {}
            for q in sorted(quanta):
                try:
                    day, t = qts.quanta_to_time(q)
                    slots = time_map.setdefault(day, [])
                    slots.append({"start": t, "end": _add_hour(t)})
                except Exception:
                    pass
            # Merge consecutive slots per day
            for day in time_map:
                time_map[day] = _merge_slots(time_map[day])
            enriched["time"] = time_map
        result.append(enriched)
    return result


def _add_hour(time_str: str) -> str:
    h, m = time_str.split(":")[:2]
    return f"{int(h) + 1:02d}:{m}"


def _merge_slots(slots: list[dict[str, str]]) -> list[dict[str, str]]:
    if not slots:
        return slots
    merged = [slots[0]]
    for s in slots[1:]:
        if merged[-1]["end"] == s["start"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    return merged


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SCH_HTTP_PORT", "8100"))
    logger.info("Starting HTTP server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
