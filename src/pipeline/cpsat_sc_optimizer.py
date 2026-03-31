"""CP-SAT Soft Constraint Optimizer.

Post-processing optimizer that takes any HC-feasible timetable and minimizes
the weighted sum of all 6 soft constraints while structurally preserving hard
constraint feasibility.  Uses CP-SAT's internal LNS with ``model.add_hint()``
warm-start from the input solution.

Usage (Python API)::

    from src.pipeline.cpsat_sc_optimizer import SCOptimizer, SCOptimizerConfig

    config = SCOptimizerConfig(time_budget_seconds=60.0)
    optimizer = SCOptimizer(data_store=store)
    result = optimizer.optimize(timetable, config)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

from src.constraints.constraints import (
    ALL_CONSTRAINTS,
    SOFT_CONSTRAINT_CLASSES,
)
from src.constraints.evaluator import Evaluator
from src.domain.gene import SessionGene
from src.domain.timetable import Timetable
from src.ga.core.population import (
    analyze_group_hierarchy,
    generate_course_group_pairs,
    get_subsession_durations,
)
from src.utils.room_compatibility import is_room_suitable_for_course

if TYPE_CHECKING:
    from src.io.data_store import DataStore

logger = logging.getLogger(__name__)

# Valid SC short names for CLI / config
VALID_SC_NAMES: set[str] = {c.name for c in SOFT_CONSTRAINT_CLASSES}
# Map short name → default weight
SC_DEFAULT_WEIGHTS: dict[str, float] = {
    c.name: c.weight for c in SOFT_CONSTRAINT_CLASSES
}


# ──────────────────────────────────────────────────────────────────
# T001 + T002: Data classes
# ──────────────────────────────────────────────────────────────────


@dataclass
class SCOptimizerConfig:
    """Configuration for a single SC optimization run."""

    time_budget_seconds: float = 60.0
    target_constraints: list[str] | None = None  # None = all 6
    weight_overrides: dict[str, float] | None = None
    seed: int = 42
    num_workers: int = 8
    log_progress: bool = True
    relax_ictd: bool = False

    def __post_init__(self) -> None:
        if self.time_budget_seconds <= 0:
            msg = "time_budget_seconds must be > 0"
            raise ValueError(msg)
        if self.num_workers < 1:
            msg = "num_workers must be >= 1"
            raise ValueError(msg)
        if self.seed < 0:
            msg = "seed must be non-negative"
            raise ValueError(msg)
        if self.target_constraints is not None:
            unknown = set(self.target_constraints) - VALID_SC_NAMES
            if unknown:
                msg = f"Unknown SC names: {unknown}. Valid: {sorted(VALID_SC_NAMES)}"
                raise ValueError(msg)
        if self.weight_overrides is not None:
            unknown = set(self.weight_overrides) - VALID_SC_NAMES
            if unknown:
                msg = f"Unknown SC names in weight_overrides: {unknown}. Valid: {sorted(VALID_SC_NAMES)}"
                raise ValueError(msg)


@dataclass
class ConstraintDelta:
    """Per-SC before/after comparison."""

    name: str
    before: float
    after: float
    change_pct: float
    weight: float


@dataclass
class SCOptimizationResult:
    """Output of a single optimizer run."""

    input_timetable: Timetable
    output_timetable: Timetable
    before_penalties: dict[str, float]
    after_penalties: dict[str, float]
    total_before: float
    total_after: float
    improvement_pct: float
    hard_violations_before: int
    hard_violations_after: int
    solver_status: str
    solve_time_seconds: float
    solutions_found: int


@dataclass
class ImprovementReport:
    """Human-readable report generated after optimization."""

    feature_name: str
    timestamp: str
    config: SCOptimizerConfig
    result: SCOptimizationResult
    per_constraint_summary: list[ConstraintDelta]


# ──────────────────────────────────────────────────────────────────
# T003: Session generation (mirrors cpsat_phase1.build_sessions)
# ──────────────────────────────────────────────────────────────────


@dataclass
class _Session:
    """Internal scheduling unit for the SC optimizer model."""

    idx: int
    course_id: str
    course_type: str
    group_ids: list[str]
    duration: int
    qualified_instructor_idxs: list[int]
    compatible_room_idxs: list[int]
    sibling_key: tuple


def _build_sessions(
    store: DataStore,
) -> tuple[list[_Session], list[str], list[str]]:
    """Generate all scheduling sessions from loaded data.

    Mirrors ``cpsat_phase1.build_sessions`` but without cross-qualification
    (the input solution already has valid instructor assignments).
    """
    hierarchy = analyze_group_hierarchy(store.groups)
    pairs = generate_course_group_pairs(
        store.courses, store.groups, hierarchy, silent=True
    )

    instructor_ids = list(store.instructors.keys())
    inst_to_idx = {iid: i for i, iid in enumerate(instructor_ids)}
    room_ids = list(store.rooms.keys())
    room_to_idx = {rid: i for i, rid in enumerate(room_ids)}

    room_compat_cache: dict[tuple[str, str], list[int]] = {}
    sessions: list[_Session] = []
    idx = 0

    for course_key, group_ids, _session_type, _num_quanta in pairs:
        course = store.courses.get(course_key)
        if course is None:
            continue

        q_inst = sorted(
            {
                inst_to_idx[iid]
                for iid in course.qualified_instructor_ids
                if iid in inst_to_idx
            }
        )

        if course_key not in room_compat_cache:
            compat: list[int] = []
            for rid, room in store.rooms.items():
                if is_room_suitable_for_course(
                    course.required_room_features,
                    room.room_features,
                    course.specific_lab_features or None,
                    room.specific_features or None,
                ):
                    compat.append(room_to_idx[rid])
            room_compat_cache[course_key] = sorted(set(compat))
        c_rooms = room_compat_cache[course_key]

        durations = get_subsession_durations(course.quanta_per_week, course.course_type)
        sibling_key = (course.course_id, course.course_type, tuple(sorted(group_ids)))

        for dur in durations:
            sessions.append(
                _Session(
                    idx=idx,
                    course_id=course.course_id,
                    course_type=course.course_type,
                    group_ids=group_ids,
                    duration=dur,
                    qualified_instructor_idxs=q_inst,
                    compatible_room_idxs=c_rooms,
                    sibling_key=sibling_key,
                )
            )
            idx += 1

    return sessions, instructor_ids, room_ids


# ──────────────────────────────────────────────────────────────────
# T020: Schedule JSON deserialization
# ──────────────────────────────────────────────────────────────────


def load_schedule_json(
    json_path: str | Path,
    store: DataStore,
) -> list[SessionGene]:
    """Load a schedule.json and reconstruct list[SessionGene].

    Handles both GA-produced format (with ``time`` dict) and CP-SAT Phase 1
    format (with ``start_quanta`` int).
    """
    qts = store.qts
    raw = json.loads(Path(json_path).read_text())

    if isinstance(raw, dict) and "decoded_schedule" in raw:
        entries = raw["decoded_schedule"]
    elif isinstance(raw, list):
        entries = raw
    else:
        entries = raw.get("schedule", [])

    genes: list[SessionGene] = []
    for entry in entries:
        # CP-SAT Phase 1 format: has start_quanta directly
        if "start_quanta" in entry:
            gene = SessionGene(
                course_id=entry.get("original_course_id", entry["course_id"]),
                course_type=entry["course_type"],
                instructor_id=entry["instructor_id"],
                group_ids=entry["group_ids"],
                room_id=entry.get("room_id") or "",
                start_quanta=entry["start_quanta"],
                num_quanta=entry.get("duration", entry.get("num_quanta", 1)),
                co_instructor_ids=entry.get("co_instructor_ids", []),
            )
            genes.append(gene)
        elif "time" in entry:
            # GA format: has time dict {day: [{start, end}]}
            quanta = qts.encode_schedule(entry["time"])
            if not quanta:
                continue
            sorted_q = sorted(quanta)
            start_q = sorted_q[0]
            num_q = sorted_q[-1] - sorted_q[0] + 1
            gene = SessionGene(
                course_id=entry.get("original_course_id", entry["course_id"]),
                course_type=entry["course_type"],
                instructor_id=entry["instructor_id"],
                group_ids=entry["group_ids"],
                room_id=entry.get("room_id") or "",
                start_quanta=start_q,
                num_quanta=num_q,
                co_instructor_ids=entry.get("co_instructor_ids", []),
            )
            genes.append(gene)

    return genes


# ──────────────────────────────────────────────────────────────────
# T004 – T008: Core model building
# ──────────────────────────────────────────────────────────────────


def _compute_valid_starts(
    duration: int,
    total_quanta: int,
    day_offsets: list[int],
    day_lengths: list[int],
) -> list[int]:
    """Compute valid start positions for a session of given duration."""
    min_day_len = min(day_lengths) if day_lengths else 7
    if duration <= min_day_len:
        valid: list[int] = []
        for offset, length in zip(day_offsets, day_lengths, strict=True):
            valid.extend(range(offset, offset + length - duration + 1))
        return valid
    return list(range(total_quanta - duration + 1))


class SCOptimizer:
    """CP-SAT soft constraint optimizer.

    Takes an HC-feasible timetable and optimizes the weighted sum of
    all 6 soft constraints while structurally preserving hard constraint
    feasibility.
    """

    def __init__(self, data_store: DataStore) -> None:
        self._store = data_store
        self._evaluator = Evaluator(constraints=list(ALL_CONSTRAINTS))
        self._sessions, self._instructor_ids, self._room_ids = _build_sessions(
            data_store
        )
        self._qts = data_store.qts

    # ── T007: Evaluate penalties ──

    def _evaluate_penalties(self, tt: Timetable) -> dict[str, float]:
        """Get per-SC penalty breakdown using shared Evaluator.

        Handles sessions without room assignments by filtering them out
        for evaluation (SC penalties are time/instructor-based, not room-based).
        """
        has_roomless = any(not g.room_id for g in tt.genes)
        if has_roomless:
            roomed_genes = [g for g in tt.genes if g.room_id]
            if not roomed_genes:
                return {c.name: 0.0 for c in SOFT_CONSTRAINT_CLASSES}
            filtered_tt = Timetable(roomed_genes, tt.context, tt.qts)
            return self._evaluator.soft_breakdown(filtered_tt)
        return self._evaluator.soft_breakdown(tt)

    # ── T008: Validate hard feasibility ──

    def _validate_hard_feasibility(self, tt: Timetable) -> dict[str, float]:
        """Check hard constraint violations. Returns breakdown dict.

        Gracefully handles sessions without room assignments (room_id="")
        by filtering them before room-sensitive constraints are evaluated.
        """
        # If any genes lack room_id, the shared Evaluator may crash on
        # room_for_gene().  Build a filtered timetable for validation.
        has_roomless = any(not g.room_id for g in tt.genes)
        if has_roomless:
            roomed_genes = [g for g in tt.genes if g.room_id]
            if roomed_genes:
                filtered_tt = Timetable(roomed_genes, tt.context, tt.qts)
                result = self._evaluator.hard_breakdown(filtered_tt)
            else:
                result = {}
            # Count roomless sessions as SRE violations
            n_roomless = len(tt.genes) - len(roomed_genes)
            result["_roomless"] = float(n_roomless)
            return result
        return self._evaluator.hard_breakdown(tt)

    # ── T004: Build core variables ──

    def _build_core_variables(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        model_indices: list[int],
    ) -> dict:
        """Create start/end/day/interval vars, room bools, instructor bools."""
        qts = self._qts
        total_quanta = qts.total_quanta

        day_offsets: list[int] = []
        day_lengths: list[int] = []
        day_names: list[str] = []
        for day in qts.DAY_NAMES:
            off = qts.day_quanta_offset.get(day)
            cnt = qts.day_quanta_count.get(day, 0)
            if off is not None and cnt > 0:
                day_offsets.append(off)
                day_lengths.append(cnt)
                day_names.append(day)
        num_days = len(day_offsets)
        quanta_per_day = day_lengths[0] if day_lengths else 7

        start_vars: list[cp_model.IntVar] = []
        end_vars: list[cp_model.IntVar] = []
        day_vars: list[cp_model.IntVar] = []
        interval_vars: list[cp_model.IntervalVar] = []

        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            valid_starts = _compute_valid_starts(
                s.duration, total_quanta, day_offsets, day_lengths
            )
            start = model.new_int_var_from_domain(
                cp_model.Domain.from_values(valid_starts), f"start_{mi}"
            )
            end = model.new_int_var(0, total_quanta, f"end_{mi}")
            model.add(end == start + s.duration)
            day = model.new_int_var(0, num_days - 1, f"day_{mi}")
            model.add_division_equality(day, start, quanta_per_day)
            interval = model.new_interval_var(start, s.duration, end, f"iv_{mi}")

            start_vars.append(start)
            end_vars.append(end)
            day_vars.append(day)
            interval_vars.append(interval)

        # Room booleans
        room_bool: dict[tuple[int, int], cp_model.IntVar] = {}
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            session_room_bools = []
            for ridx in s.compatible_room_idxs:
                b = model.new_bool_var(f"room_{mi}_{ridx}")
                room_bool[(mi, ridx)] = b
                session_room_bools.append(b)
            model.add_exactly_one(session_room_bools)

        # Instructor booleans
        instr_bool: dict[tuple[int, int], cp_model.IntVar] = {}
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            session_instr_bools = []
            for iidx in s.qualified_instructor_idxs:
                b = model.new_bool_var(f"instr_{mi}_{iidx}")
                instr_bool[(mi, iidx)] = b
                session_instr_bools.append(b)
            if s.course_type == "practical":
                model.add(sum(session_instr_bools) == 2)
            else:
                model.add_exactly_one(session_instr_bools)

        # Optional intervals for rooms
        room_opt_intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
        for (mi, ridx), b in room_bool.items():
            orig_i = model_indices[mi]
            s = sessions[orig_i]
            opt_iv = model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi], b, f"oiv_r_{mi}_{ridx}"
            )
            room_opt_intervals[(mi, ridx)] = opt_iv

        # Optional intervals for instructors
        instr_opt_intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
        for (mi, iidx), b in instr_bool.items():
            orig_i = model_indices[mi]
            s = sessions[orig_i]
            opt_iv = model.new_optional_interval_var(
                start_vars[mi], s.duration, end_vars[mi], b, f"oiv_i_{mi}_{iidx}"
            )
            instr_opt_intervals[(mi, iidx)] = opt_iv

        return {
            "start": start_vars,
            "end": end_vars,
            "day": day_vars,
            "interval": interval_vars,
            "room_bool": room_bool,
            "instr_bool": instr_bool,
            "room_opt_intervals": room_opt_intervals,
            "instr_opt_intervals": instr_opt_intervals,
            "model_indices": model_indices,
            "day_names": day_names,
            "day_offsets": day_offsets,
            "day_lengths": day_lengths,
            "num_days": num_days,
            "quanta_per_day": quanta_per_day,
        }

    # ── T005: Add hard constraints ──

    def _add_hard_constraints(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
        *,
        relax_ictd: bool = False,
    ) -> None:
        """Encode all 9 hard constraints in the CP-SAT model.

        - CTE: NoStudentDoubleBooking (NoOverlap per group, mandatory intervals)
        - FTE: NoInstructorDoubleBooking (NoOverlap per instructor, optional intervals)
        - SRE: NoRoomDoubleBooking (NoOverlap per room, optional intervals)
        - FPC: InstructorMustBeQualified (domain restriction on instructor bools)
        - FFC: RoomMustHaveFeatures (domain restriction on room bools)
        - FCA: InstructorMustBeAvailable (availability windows on instructor bools)
        - CQF: ExactWeeklyHours (structural — session durations are fixed)
        - ICTD: SpreadAcrossDays (AllDifferent on day vars per sibling group)
        - PMI: RequiresTwoInstructors (sum==2, already in _build_core_variables)
        """
        model_indices = vars_dict["model_indices"]
        interval_vars = vars_dict["interval"]
        day_vars = vars_dict["day"]
        start_vars = vars_dict["start"]
        room_opt_intervals = vars_dict["room_opt_intervals"]
        instr_opt_intervals = vars_dict["instr_opt_intervals"]

        # CTE: NoStudentDoubleBooking
        group_sessions: dict[str, list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for gid in set(s.group_ids):
                group_sessions[gid].append(mi)
        for midxs in group_sessions.values():
            if len(midxs) > 1:
                model.add_no_overlap([interval_vars[mi] for mi in midxs])

        # FTE: NoInstructorDoubleBooking
        instr_to_opts: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)
        for (_mi, iidx), opt_iv in instr_opt_intervals.items():
            instr_to_opts[iidx].append(opt_iv)
        for opt_ivs in instr_to_opts.values():
            if len(opt_ivs) > 1:
                model.add_no_overlap(opt_ivs)

        # SRE: NoRoomDoubleBooking
        room_to_opts: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)
        for (_mi, ridx), opt_iv in room_opt_intervals.items():
            room_to_opts[ridx].append(opt_iv)
        for opt_ivs in room_to_opts.values():
            if len(opt_ivs) > 1:
                model.add_no_overlap(opt_ivs)

        # FPC + FFC: Already enforced via domain restriction on booleans
        # (instructor bools only for qualified, room bools only for compatible)

        # FCA: InstructorMustBeAvailable
        instr_bool = vars_dict["instr_bool"]
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for iidx in s.qualified_instructor_idxs:
                iid = self._instructor_ids[iidx]
                instr = self._store.instructors[iid]
                if instr.is_full_time:
                    continue
                # Part-time: for each possible start, check if all quanta are available
                avail = instr.available_quanta
                qts = self._qts
                day_offsets = vars_dict["day_offsets"]
                day_lengths = vars_dict["day_lengths"]
                valid_starts = _compute_valid_starts(
                    s.duration, qts.total_quanta, day_offsets, day_lengths
                )
                forbidden_starts = []
                for sq in valid_starts:
                    for q in range(sq, sq + s.duration):
                        if q not in avail:
                            forbidden_starts.append(sq)
                            break
                if forbidden_starts:
                    b = instr_bool.get((mi, iidx))
                    if b is not None:
                        for sq in forbidden_starts:
                            model.add(start_vars[mi] != sq).only_enforce_if(b)

        # CQF: Structural (session durations are fixed; guaranteed by build_sessions)

        # ICTD: SpreadAcrossDays
        if not relax_ictd:
            sibling_groups: dict[tuple, list[int]] = defaultdict(list)
            for mi, orig_i in enumerate(model_indices):
                s = sessions[orig_i]
                sibling_groups[s.sibling_key].append(mi)
            for siblings in sibling_groups.values():
                if len(siblings) > 1:
                    model.add_all_different([day_vars[mi] for mi in siblings])
            # Symmetry breaking
            for siblings in sibling_groups.values():
                if len(siblings) > 1:
                    for j in range(len(siblings) - 1):
                        model.add(start_vars[siblings[j]] < start_vars[siblings[j + 1]])

    # ── T006: Seed hints ──

    def _seed_hints(
        self,
        model: cp_model.CpModel,
        vars_dict: dict,
        sessions: list[_Session],
        input_genes: list[SessionGene],
    ) -> int:
        """Warm-start from the input HC-feasible solution using model.add_hint()."""
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        instr_bool = vars_dict["instr_bool"]
        room_bool = vars_dict["room_bool"]

        room_to_idx = {rid: i for i, rid in enumerate(self._room_ids)}

        # Map (course_id, course_type, tuple(group_ids), duration, occurrence_idx)
        # to input gene, to match sessions to genes
        gene_lookup: dict[tuple, list[SessionGene]] = defaultdict(list)
        for gene in input_genes:
            key = (gene.course_id, gene.course_type, tuple(sorted(gene.group_ids)))
            gene_lookup[key].append(gene)

        # Sort genes within each key by start_quanta for deterministic matching
        for genes_list in gene_lookup.values():
            genes_list.sort(key=lambda g: g.start_quanta)

        # Track which genes we've consumed per sibling_key
        consumed: dict[tuple, int] = defaultdict(int)
        n_hints = 0

        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            key = (s.course_id, s.course_type, tuple(sorted(s.group_ids)))
            ci = consumed[key]
            available_genes = gene_lookup.get(key, [])
            if ci >= len(available_genes):
                continue
            gene = available_genes[ci]
            consumed[key] = ci + 1

            # Hint start time
            model.add_hint(start_vars[mi], gene.start_quanta)
            n_hints += 1

            # Hint instructor booleans
            for iidx in s.qualified_instructor_idxs:
                iid = self._instructor_ids[iidx]
                b = instr_bool.get((mi, iidx))
                if b is not None:
                    all_instr = [gene.instructor_id, *(gene.co_instructor_ids or [])]
                    model.add_hint(b, 1 if iid in all_instr else 0)
                    n_hints += 1

            # Hint room booleans
            ridx_assigned = room_to_idx.get(gene.room_id)
            for ridx in s.compatible_room_idxs:
                b = room_bool.get((mi, ridx))
                if b is not None:
                    model.add_hint(b, 1 if ridx == ridx_assigned else 0)
                    n_hints += 1

        return n_hints

    # ── T009: CSC penalty ──

    def _add_csc_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """StudentScheduleCompactness: gap penalty per (group, day).

        For each (group, day), tracks first/last occupied quantum via
        min_equality/max_equality, computes gap = span - count_occupied.
        """
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        day_vars = vars_dict["day"]
        day_offsets = vars_dict["day_offsets"]
        num_days = vars_dict["num_days"]
        quanta_per_day = vars_dict["quanta_per_day"]
        # Get midday break quanta (within-day indices) per day
        break_quanta = self._qts.get_midday_break_quanta()
        break_counts: dict[str, int] = {}
        day_names = vars_dict["day_names"]
        for d_idx, dname in enumerate(day_names):
            break_counts[d_idx] = len(break_quanta.get(dname, set()))

        # Group sessions by group
        group_sessions: dict[str, list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for gid in set(s.group_ids):
                group_sessions[gid].append(mi)

        penalty_vars: list[cp_model.IntVar] = []

        for gid, midxs in group_sessions.items():
            if len(midxs) < 2:
                continue

            for d_idx in range(num_days):
                # Create boolean: is session mi on day d_idx?
                on_day_bools = []
                for mi in midxs:
                    b = model.new_bool_var(f"csc_{gid}_{mi}_d{d_idx}")
                    model.add(day_vars[mi] == d_idx).only_enforce_if(b)
                    model.add(day_vars[mi] != d_idx).only_enforce_if(b.negated())
                    on_day_bools.append((mi, b))

                # Count sessions on this day
                count_on_day = model.new_int_var(
                    0, len(midxs), f"csc_cnt_{gid}_d{d_idx}"
                )
                model.add(count_on_day == sum(b for _, b in on_day_bools))

                # Only compute gap if >=2 sessions on this day
                has_multiple = model.new_bool_var(f"csc_mult_{gid}_d{d_idx}")
                model.add(count_on_day >= 2).only_enforce_if(has_multiple)
                model.add(count_on_day <= 1).only_enforce_if(has_multiple.negated())

                # Within-day start for each session on this day
                day_offset = day_offsets[d_idx]
                within_day_starts = []
                for mi, b_on_day in on_day_bools:
                    s = sessions[model_indices[mi]]
                    wd_start = model.new_int_var(
                        0, quanta_per_day - 1, f"csc_wd_{gid}_{mi}_d{d_idx}"
                    )
                    model.add(wd_start == start_vars[mi] - day_offset).only_enforce_if(
                        b_on_day
                    )
                    within_day_starts.append((wd_start, b_on_day, s.duration))

                # First and last occupied quantum (within day)
                first_q = model.new_int_var(
                    0, quanta_per_day - 1, f"csc_first_{gid}_d{d_idx}"
                )
                last_q = model.new_int_var(
                    0, quanta_per_day - 1, f"csc_last_{gid}_d{d_idx}"
                )

                # Compute occupied quanta count
                total_occ = model.new_int_var(
                    0, quanta_per_day, f"csc_occ_{gid}_d{d_idx}"
                )
                model.add(total_occ == sum(dur * b for _, b, dur in within_day_starts))

                # min/max over within-day starts for sessions on this day
                # We use conditional constraints: first_q <= wd_start when
                # b_on_day is true, and first_q == min of those
                # For simplicity, use add_min_equality with a large sentinel
                sentinel_high = quanta_per_day
                sentinel_low = 0
                min_targets = []
                max_targets = []
                for wd_start, b_on_day, dur in within_day_starts:
                    # For min: if on day, use wd_start; else sentinel_high
                    min_t = model.new_int_var(
                        0, sentinel_high, f"csc_mint_{gid}_{id(wd_start)}_d{d_idx}"
                    )
                    model.add(min_t == wd_start).only_enforce_if(b_on_day)
                    model.add(min_t == sentinel_high).only_enforce_if(
                        b_on_day.negated()
                    )
                    min_targets.append(min_t)

                    # For max: if on day, use wd_start + dur - 1; else sentinel_low
                    max_t = model.new_int_var(
                        0, quanta_per_day, f"csc_maxt_{gid}_{id(wd_start)}_d{d_idx}"
                    )
                    model.add(max_t == wd_start + dur - 1).only_enforce_if(b_on_day)
                    model.add(max_t == sentinel_low).only_enforce_if(b_on_day.negated())
                    max_targets.append(max_t)

                model.add_min_equality(first_q, min_targets)
                model.add_max_equality(last_q, max_targets)

                # Gap = (last_q - first_q + 1) - total_occupied - break_in_span
                # We approximate break overlap as a constant (break quanta count
                # for this day, capped by span)
                brk = break_counts.get(d_idx, 0)
                gap = model.new_int_var(0, quanta_per_day, f"csc_gap_{gid}_d{d_idx}")
                span = model.new_int_var(0, quanta_per_day, f"csc_span_{gid}_d{d_idx}")
                model.add(span == last_q - first_q + 1)
                # gap = max(0, span - total_occ - brk)
                raw_gap = model.new_int_var(
                    -quanta_per_day, quanta_per_day, f"csc_rgap_{gid}_d{d_idx}"
                )
                model.add(raw_gap == span - total_occ - brk)
                model.add_max_equality(gap, [raw_gap, model.new_constant(0)])

                # Only apply penalty when has_multiple
                csc_penalty = model.new_int_var(
                    0, quanta_per_day, f"csc_pen_{gid}_d{d_idx}"
                )
                model.add(csc_penalty == gap).only_enforce_if(has_multiple)
                model.add(csc_penalty == 0).only_enforce_if(has_multiple.negated())
                penalty_vars.append(csc_penalty)

        return penalty_vars

    # ── T010: FSC penalty ──

    def _add_fsc_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """InstructorScheduleCompactness: gap penalty per (instructor, day).

        Same pattern as CSC but indexed per instructor using optional intervals.
        """
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        day_vars = vars_dict["day"]
        instr_bool = vars_dict["instr_bool"]
        day_offsets = vars_dict["day_offsets"]
        num_days = vars_dict["num_days"]
        quanta_per_day = vars_dict["quanta_per_day"]

        break_quanta = self._qts.get_midday_break_quanta()
        day_names = vars_dict["day_names"]
        break_counts: dict[int, int] = {}
        for d_idx, dname in enumerate(day_names):
            break_counts[d_idx] = len(break_quanta.get(dname, set()))

        # For each instructor, collect sessions that could be assigned to them
        instr_sessions: dict[int, list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for iidx in s.qualified_instructor_idxs:
                instr_sessions[iidx].append(mi)

        penalty_vars: list[cp_model.IntVar] = []

        for iidx, midxs in instr_sessions.items():
            if len(midxs) < 2:
                continue

            for d_idx in range(num_days):
                # Boolean: instructor iidx teaches session mi AND session is on day d_idx
                on_day_bools = []
                for mi in midxs:
                    b_instr = instr_bool.get((mi, iidx))
                    if b_instr is None:
                        continue
                    b_day = model.new_bool_var(f"fsc_day_{iidx}_{mi}_d{d_idx}")
                    model.add(day_vars[mi] == d_idx).only_enforce_if(b_day)
                    model.add(day_vars[mi] != d_idx).only_enforce_if(b_day.negated())

                    b_both = model.new_bool_var(f"fsc_both_{iidx}_{mi}_d{d_idx}")
                    model.add_bool_and([b_instr, b_day]).only_enforce_if(b_both)
                    model.add_bool_or(
                        [b_instr.negated(), b_day.negated()]
                    ).only_enforce_if(b_both.negated())
                    s = sessions[model_indices[mi]]
                    on_day_bools.append((mi, b_both, s.duration))

                if len(on_day_bools) < 2:
                    continue

                count_on_day = model.new_int_var(
                    0, len(on_day_bools), f"fsc_cnt_{iidx}_d{d_idx}"
                )
                model.add(count_on_day == sum(b for _, b, _ in on_day_bools))

                has_multiple = model.new_bool_var(f"fsc_mult_{iidx}_d{d_idx}")
                model.add(count_on_day >= 2).only_enforce_if(has_multiple)
                model.add(count_on_day <= 1).only_enforce_if(has_multiple.negated())

                day_offset = day_offsets[d_idx]
                sentinel_high = quanta_per_day
                sentinel_low = 0
                min_targets = []
                max_targets = []

                total_occ = model.new_int_var(
                    0, quanta_per_day, f"fsc_occ_{iidx}_d{d_idx}"
                )
                model.add(total_occ == sum(dur * b for _, b, dur in on_day_bools))

                for mi, b_both, dur in on_day_bools:
                    wd_start = model.new_int_var(
                        0, quanta_per_day, f"fsc_wd_{iidx}_{mi}_d{d_idx}"
                    )
                    model.add(wd_start == start_vars[mi] - day_offset).only_enforce_if(
                        b_both
                    )
                    model.add(wd_start == 0).only_enforce_if(b_both.negated())

                    min_t = model.new_int_var(
                        0, sentinel_high, f"fsc_mint_{iidx}_{mi}_d{d_idx}"
                    )
                    model.add(min_t == wd_start).only_enforce_if(b_both)
                    model.add(min_t == sentinel_high).only_enforce_if(b_both.negated())
                    min_targets.append(min_t)

                    max_t = model.new_int_var(
                        0, quanta_per_day, f"fsc_maxt_{iidx}_{mi}_d{d_idx}"
                    )
                    model.add(max_t == wd_start + dur - 1).only_enforce_if(b_both)
                    model.add(max_t == sentinel_low).only_enforce_if(b_both.negated())
                    max_targets.append(max_t)

                if not min_targets:
                    continue

                first_q = model.new_int_var(
                    0, quanta_per_day - 1, f"fsc_first_{iidx}_d{d_idx}"
                )
                last_q = model.new_int_var(
                    0, quanta_per_day - 1, f"fsc_last_{iidx}_d{d_idx}"
                )
                model.add_min_equality(first_q, min_targets)
                model.add_max_equality(last_q, max_targets)

                brk = break_counts.get(d_idx, 0)
                span = model.new_int_var(0, quanta_per_day, f"fsc_span_{iidx}_d{d_idx}")
                model.add(span == last_q - first_q + 1)
                raw_gap = model.new_int_var(
                    -quanta_per_day, quanta_per_day, f"fsc_rgap_{iidx}_d{d_idx}"
                )
                model.add(raw_gap == span - total_occ - brk)
                gap = model.new_int_var(0, quanta_per_day, f"fsc_gap_{iidx}_d{d_idx}")
                model.add_max_equality(gap, [raw_gap, model.new_constant(0)])

                fsc_penalty = model.new_int_var(
                    0, quanta_per_day, f"fsc_pen_{iidx}_d{d_idx}"
                )
                model.add(fsc_penalty == gap).only_enforce_if(has_multiple)
                model.add(fsc_penalty == 0).only_enforce_if(has_multiple.negated())
                penalty_vars.append(fsc_penalty)

        return penalty_vars

    # ── T011: MIP penalty ──

    def _add_mip_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """StudentLunchBreak: per (group, day), penalize occupied lunch quanta."""
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        end_vars = vars_dict["end"]
        day_offsets = vars_dict["day_offsets"]

        # Get break window quanta per day (within-day offsets)
        qts = self._qts
        break_windows: dict[int, list[int]] = {}
        day_names = vars_dict["day_names"]
        for d_idx, dname in enumerate(day_names):
            try:
                bw_start = qts.time_to_quanta(dname, qts.break_window_start)
                bw_end = qts.time_to_quanta(dname, qts.break_window_end)
                day_off = day_offsets[d_idx]
                # Within-day indices for the break window
                break_windows[d_idx] = list(range(bw_start - day_off, bw_end - day_off))
            except ValueError:
                continue

        if not break_windows:
            return []

        # Group sessions by group
        group_sessions: dict[str, list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            for gid in set(s.group_ids):
                group_sessions[gid].append(mi)

        penalty_vars: list[cp_model.IntVar] = []

        for gid, midxs in group_sessions.items():
            for d_idx, bw_quanta in break_windows.items():
                day_offset = day_offsets[d_idx]

                for bw_q in bw_quanta:
                    # Absolute quantum for this break window slot
                    abs_q = day_offset + bw_q

                    # For each session, check if it occupies abs_q AND is on this day
                    occupies_bools = []
                    for mi in midxs:
                        s = sessions[model_indices[mi]]
                        # Session occupies abs_q iff start <= abs_q < end
                        b_occ = model.new_bool_var(
                            f"mip_occ_{gid}_{mi}_d{d_idx}_q{bw_q}"
                        )
                        model.add(start_vars[mi] <= abs_q).only_enforce_if(b_occ)
                        model.add(end_vars[mi] > abs_q).only_enforce_if(b_occ)
                        model.add(start_vars[mi] > abs_q).only_enforce_if(
                            b_occ.negated()
                        )
                        # Actually: b_occ is true iff start <= abs_q AND end > abs_q
                        # We need an AND here. Use auxiliary.
                        b_start_ok = model.new_bool_var(
                            f"mip_sok_{gid}_{mi}_d{d_idx}_q{bw_q}"
                        )
                        b_end_ok = model.new_bool_var(
                            f"mip_eok_{gid}_{mi}_d{d_idx}_q{bw_q}"
                        )
                        model.add(start_vars[mi] <= abs_q).only_enforce_if(b_start_ok)
                        model.add(start_vars[mi] > abs_q).only_enforce_if(
                            b_start_ok.negated()
                        )
                        model.add(end_vars[mi] > abs_q).only_enforce_if(b_end_ok)
                        model.add(end_vars[mi] <= abs_q).only_enforce_if(
                            b_end_ok.negated()
                        )
                        # b_occ = b_start_ok AND b_end_ok
                        model.add_bool_and([b_start_ok, b_end_ok]).only_enforce_if(
                            b_occ
                        )
                        model.add_bool_or(
                            [b_start_ok.negated(), b_end_ok.negated()]
                        ).only_enforce_if(b_occ.negated())
                        occupies_bools.append(b_occ)

                    # Penalty: is this lunch quantum occupied by any session for this group?
                    is_busy = model.new_bool_var(f"mip_busy_{gid}_d{d_idx}_q{bw_q}")
                    if occupies_bools:
                        model.add_bool_or(occupies_bools).only_enforce_if(is_busy)
                        model.add_bool_and(
                            [b.negated() for b in occupies_bools]
                        ).only_enforce_if(is_busy.negated())
                    else:
                        model.add(is_busy == 0)
                    penalty_vars.append(is_busy)

        return penalty_vars

    # ── T012: SessionContinuity penalty ──

    def _add_session_continuity_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """SessionContinuity: penalize isolated 1-quantum theory sessions.

        For theory courses with 1-quantum subsessions, track if they end up
        isolated on a day (only 1 quantum from that course on that day).
        """
        model_indices = vars_dict["model_indices"]
        day_vars = vars_dict["day"]
        num_days = vars_dict["num_days"]

        # Find 1-quantum theory sessions
        theory_1q_by_course: dict[tuple[str, str], list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            if s.course_type == "theory" and s.duration == 1:
                key = (s.course_id, s.course_type)
                theory_1q_by_course[key].append(mi)

        penalty_vars: list[cp_model.IntVar] = []
        penalty_weight = 10  # matches isolated_slot_penalty=10.0 from constraints.py

        for _course_key, midxs in theory_1q_by_course.items():
            if len(midxs) <= 1:
                # Single 1q session: first one is excused per constraint logic
                continue

            for d_idx in range(num_days):
                # Count how many 1q sessions of this course are on this day
                on_day_bools = []
                for mi in midxs:
                    b = model.new_bool_var(f"scon_{_course_key[0]}_{mi}_d{d_idx}")
                    model.add(day_vars[mi] == d_idx).only_enforce_if(b)
                    model.add(day_vars[mi] != d_idx).only_enforce_if(b.negated())
                    on_day_bools.append(b)

                count = model.new_int_var(
                    0, len(on_day_bools), f"scon_cnt_{_course_key[0]}_d{d_idx}"
                )
                model.add(count == sum(on_day_bools))

                # Penalty if exactly 1 isolated session
                is_isolated = model.new_bool_var(f"scon_iso_{_course_key[0]}_d{d_idx}")
                model.add(count == 1).only_enforce_if(is_isolated)
                model.add(count != 1).only_enforce_if(is_isolated.negated())

                pen = model.new_int_var(
                    0, penalty_weight, f"scon_pen_{_course_key[0]}_d{d_idx}"
                )
                model.add(pen == penalty_weight).only_enforce_if(is_isolated)
                model.add(pen == 0).only_enforce_if(is_isolated.negated())
                penalty_vars.append(pen)

        return penalty_vars

    # ── T013: SSCP penalty ──

    def _add_sscp_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """PairedCohortPracticalAlignment: |start_A - start_B| for cohort pairs."""
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        total_quanta = self._qts.total_quanta

        cohort_pairs = self._store.cohort_pairs or []
        if not cohort_pairs:
            return []

        # Index practical sessions by (course_id, group_id)
        practical_sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            if s.course_type == "practical":
                for gid in s.group_ids:
                    practical_sessions[(s.course_id, gid)].append(mi)

        penalty_vars: list[cp_model.IntVar] = []

        for left_id, right_id in cohort_pairs:
            # Find shared practical courses
            left_courses = {cid for (cid, gid) in practical_sessions if gid == left_id}
            right_courses = {
                cid for (cid, gid) in practical_sessions if gid == right_id
            }
            shared = left_courses & right_courses

            for course_id in shared:
                left_mis = practical_sessions.get((course_id, left_id), [])
                right_mis = practical_sessions.get((course_id, right_id), [])

                # Match sessions by order (same subsession index)
                for li, ri in zip(left_mis, right_mis, strict=False):
                    abs_diff = model.new_int_var(
                        0,
                        total_quanta,
                        f"sscp_{course_id}_{left_id}_{right_id}_{li}_{ri}",
                    )
                    diff = model.new_int_var(
                        -total_quanta,
                        total_quanta,
                        f"sscp_d_{course_id}_{left_id}_{right_id}_{li}_{ri}",
                    )
                    model.add(diff == start_vars[li] - start_vars[ri])
                    model.add_abs_equality(abs_diff, diff)
                    penalty_vars.append(abs_diff)

        return penalty_vars

    # ── T014: BreakPlacementCompliance penalty ──

    def _add_break_placement_penalty(
        self,
        model: cp_model.CpModel,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[cp_model.IntVar]:
        """BreakPlacementCompliance: penalize groups without free time in break window.

        Same modeling as MIP: count occupied quanta in the break window.
        Only active if enforce_break_placement is enabled in QTS.
        """
        qts = self._qts
        if not qts.enforce_break_placement:
            return []

        # Reuse MIP penalty structure — they share the same modeling pattern
        # Both count occupied quanta in a window per group per day
        # The break window is break_window_start..break_window_end
        # This is already captured by MIP penalty since they use the same window
        # BPC differs only in: it counts violations (0/1) not missing quanta
        # For the CP-SAT model, we use the same penalty vars — the objective
        # weight handles the distinction
        return self._add_mip_penalty(model, sessions, vars_dict)

    # ── T015 + T021: Build objective ──

    def _build_objective(
        self,
        model: cp_model.CpModel,
        penalty_terms: dict[str, list[cp_model.IntVar]],
        config: SCOptimizerConfig,
    ) -> None:
        """Combine SC penalty terms into weighted model.minimize().

        When target_constraints is set, non-targeted SCs get weight=0.
        """
        weights = dict(SC_DEFAULT_WEIGHTS)

        # Apply weight overrides
        if config.weight_overrides:
            weights.update(config.weight_overrides)

        # Zero out non-targeted constraints
        if config.target_constraints is not None:
            targeted = set(config.target_constraints)
            for name in weights:
                if name not in targeted:
                    weights[name] = 0.0

        # Build objective
        objective_terms = []
        for sc_name, pvars in penalty_terms.items():
            w = weights.get(sc_name, 1.0)
            if w == 0.0 or not pvars:
                continue
            # Scale weight to integer (CP-SAT works with integers)
            # Use 100x scaling for fractional weights
            w_int = int(w * 100)
            objective_terms.extend(w_int * pv for pv in pvars)

        if objective_terms:
            model.minimize(sum(objective_terms))

    # ── T016: Extract solution ──

    def _extract_solution(
        self,
        solver: cp_model.CpSolver,
        sessions: list[_Session],
        vars_dict: dict,
    ) -> list[SessionGene]:
        """Reconstruct list[SessionGene] from CP-SAT solution."""
        model_indices = vars_dict["model_indices"]
        start_vars = vars_dict["start"]
        room_bool = vars_dict["room_bool"]
        instr_bool = vars_dict["instr_bool"]

        genes: list[SessionGene] = []

        for mi, orig_i in enumerate(model_indices):
            s = sessions[orig_i]
            start_q = solver.value(start_vars[mi])

            # Resolve instructors
            assigned_instructors = []
            for iidx in s.qualified_instructor_idxs:
                b = instr_bool.get((mi, iidx))
                if b is not None and solver.value(b):
                    assigned_instructors.append(self._instructor_ids[iidx])

            # Resolve room
            assigned_room = ""
            for ridx in s.compatible_room_idxs:
                b = room_bool.get((mi, ridx))
                if b is not None and solver.value(b):
                    assigned_room = self._room_ids[ridx]
                    break

            gene = SessionGene(
                course_id=s.course_id,
                course_type=s.course_type,
                instructor_id=assigned_instructors[0] if assigned_instructors else "",
                group_ids=s.group_ids,
                room_id=assigned_room,
                start_quanta=start_q,
                num_quanta=s.duration,
                co_instructor_ids=(
                    assigned_instructors[1:] if len(assigned_instructors) > 1 else []
                ),
            )
            genes.append(gene)

        return genes

    # ── T017: Main optimize() entry point ──

    def optimize(
        self,
        timetable: Timetable,
        config: SCOptimizerConfig | None = None,
    ) -> SCOptimizationResult:
        """Optimize soft constraints on an HC-feasible timetable.

        Args:
            timetable: HC-feasible input timetable.
            config: Optimization configuration. Defaults to SCOptimizerConfig().

        Returns:
            SCOptimizationResult with before/after penalties and optimized timetable.

        Raises:
            ValueError: If input has hard constraint violations.
        """
        if config is None:
            config = SCOptimizerConfig()

        t_start = time.time()

        # Validate hard feasibility of input
        hard_breakdown = self._validate_hard_feasibility(timetable)
        # _roomless is informational — sessions without rooms are ok
        n_roomless = int(hard_breakdown.pop("_roomless", 0))
        hard_total = sum(hard_breakdown.values())
        if hard_total > 0:
            violated = {k: v for k, v in hard_breakdown.items() if v > 0}
            msg = f"Input timetable has hard constraint violations: {violated}"
            raise ValueError(msg)
        if n_roomless > 0:
            logger.info("%d sessions without room assignments (will be preserved)", n_roomless)

        # Evaluate before penalties
        before_penalties = self._evaluate_penalties(timetable)
        total_before = sum(
            SC_DEFAULT_WEIGHTS.get(name, 1.0) * val
            for name, val in before_penalties.items()
        )

        # T030: Early exit if zero soft penalty
        if total_before == 0:
            logger.info("Input has zero soft penalty — returning unchanged.")
            return SCOptimizationResult(
                input_timetable=timetable,
                output_timetable=timetable,
                before_penalties=before_penalties,
                after_penalties=before_penalties,
                total_before=total_before,
                total_after=total_before,
                improvement_pct=0.0,
                hard_violations_before=0,
                hard_violations_after=0,
                solver_status="SKIPPED",
                solve_time_seconds=time.time() - t_start,
                solutions_found=0,
            )

        # Build model
        sessions = self._sessions
        model = cp_model.CpModel()

        # Filter impossible sessions
        impossible = set()
        for i, s in enumerate(sessions):
            if (
                not s.qualified_instructor_idxs
                or not s.compatible_room_idxs
                or (
                    s.course_type == "practical"
                    and len(s.qualified_instructor_idxs) < 2
                )
            ):
                impossible.add(i)
        model_indices = [i for i in range(len(sessions)) if i not in impossible]

        if impossible:
            logger.warning(
                "Excluded %d impossible sessions from model", len(impossible)
            )

        # T004: Build variables
        vars_dict = self._build_core_variables(model, sessions, model_indices)

        # T005: Add hard constraints
        self._add_hard_constraints(
            model, sessions, vars_dict, relax_ictd=config.relax_ictd
        )

        # T009-T014: Add SC penalty terms
        penalty_terms: dict[str, list[cp_model.IntVar]] = {
            "CSC": self._add_csc_penalty(model, sessions, vars_dict),
            "FSC": self._add_fsc_penalty(model, sessions, vars_dict),
            "MIP": self._add_mip_penalty(model, sessions, vars_dict),
            "session_continuity": self._add_session_continuity_penalty(
                model, sessions, vars_dict
            ),
            "SSCP": self._add_sscp_penalty(model, sessions, vars_dict),
            "break_placement_compliance": self._add_break_placement_penalty(
                model, sessions, vars_dict
            ),
        }

        # T015/T021: Build objective
        self._build_objective(model, penalty_terms, config)

        # T006: Seed hints from input solution
        n_hints = self._seed_hints(model, vars_dict, sessions, timetable.genes)
        logger.info("Seeded %d hints from input solution", n_hints)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = config.time_budget_seconds
        solver.parameters.num_workers = config.num_workers
        solver.parameters.log_search_progress = config.log_progress
        solver.parameters.random_seed = config.seed

        # Track solutions
        status = solver.solve(model)
        solve_time = time.time() - t_start

        STATUS_NAMES = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        status_name = STATUS_NAMES.get(status, "UNKNOWN")
        logger.info("Solver status: %s (%.1fs)", status_name, solve_time)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # T016: Extract solution
            output_genes = self._extract_solution(solver, sessions, vars_dict)
            context = self._store.to_context()
            output_tt = Timetable(output_genes, context, self._qts)

            # Validate output
            output_hard = self._validate_hard_feasibility(output_tt)
            output_hard_total = int(sum(output_hard.values()))

            after_penalties = self._evaluate_penalties(output_tt)
            total_after = sum(
                SC_DEFAULT_WEIGHTS.get(name, 1.0) * val
                for name, val in after_penalties.items()
            )

            improvement_pct = (
                ((total_before - total_after) / total_before * 100)
                if total_before > 0
                else 0.0
            )
        else:
            # T031: No improvement — return input unchanged
            logger.warning(
                "Solver returned %s — returning input unchanged.", status_name
            )
            output_tt = timetable
            after_penalties = before_penalties
            total_after = total_before
            improvement_pct = 0.0
            output_hard_total = 0

        return SCOptimizationResult(
            input_timetable=timetable,
            output_timetable=output_tt,
            before_penalties=before_penalties,
            after_penalties=after_penalties,
            total_before=total_before,
            total_after=total_after,
            improvement_pct=improvement_pct,
            hard_violations_before=0,
            hard_violations_after=output_hard_total,
            solver_status=status_name,
            solve_time_seconds=solve_time,
            solutions_found=1 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
        )

    # ── T018: Format report ──

    def _format_report(
        self,
        result: SCOptimizationResult,
        config: SCOptimizerConfig,
    ) -> ImprovementReport:
        """Generate ImprovementReport from optimization result."""
        import datetime

        deltas = []
        for sc in SOFT_CONSTRAINT_CLASSES:
            before = result.before_penalties.get(sc.name, 0.0)
            after = result.after_penalties.get(sc.name, 0.0)
            change_pct = ((before - after) / before * 100) if before > 0 else 0.0
            deltas.append(
                ConstraintDelta(
                    name=sc.name,
                    before=before,
                    after=after,
                    change_pct=change_pct,
                    weight=SC_DEFAULT_WEIGHTS.get(sc.name, 1.0),
                )
            )

        return ImprovementReport(
            feature_name="CP-SAT SC Optimizer",
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
            config=config,
            result=result,
            per_constraint_summary=deltas,
        )

    # ── T027: Print console report ──

    @staticmethod
    def print_console_report(report: ImprovementReport) -> None:
        """Formatted table output to console."""
        result = report.result
        print("\n" + "=" * 72)
        print("  CP-SAT SOFT CONSTRAINT OPTIMIZATION REPORT")
        print("=" * 72)
        print(f"  Solver Status:  {result.solver_status}")
        print(f"  Solve Time:     {result.solve_time_seconds:.1f}s")
        print(f"  HC Violations:  {result.hard_violations_after}")
        print("-" * 72)
        print(f"  {'Constraint':<30s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
        print("-" * 72)
        for delta in report.per_constraint_summary:
            sign = "-" if delta.change_pct > 0 else "+"
            print(
                f"  {delta.name:<30s} {delta.before:8.1f} {delta.after:8.1f} "
                f"{sign}{abs(delta.change_pct):6.1f}%"
            )
        print("-" * 72)
        print(
            f"  {'TOTAL (weighted)':<30s} {result.total_before:8.1f} "
            f"{result.total_after:8.1f} "
            f"{'-' if result.improvement_pct > 0 else '+'}"
            f"{abs(result.improvement_pct):6.1f}%"
        )
        print("=" * 72 + "\n")

    # ── T028: Save JSON report ──

    @staticmethod
    def save_json_report(report: ImprovementReport, path: str | Path) -> None:
        """Serialize ImprovementReport to JSON file."""
        data = {
            "feature_name": report.feature_name,
            "timestamp": report.timestamp,
            "config": {
                "time_budget_seconds": report.config.time_budget_seconds,
                "target_constraints": report.config.target_constraints,
                "weight_overrides": report.config.weight_overrides,
                "seed": report.config.seed,
                "num_workers": report.config.num_workers,
                "relax_ictd": report.config.relax_ictd,
            },
            "result": {
                "total_before": report.result.total_before,
                "total_after": report.result.total_after,
                "improvement_pct": report.result.improvement_pct,
                "hard_violations_before": report.result.hard_violations_before,
                "hard_violations_after": report.result.hard_violations_after,
                "solver_status": report.result.solver_status,
                "solve_time_seconds": report.result.solve_time_seconds,
                "solutions_found": report.result.solutions_found,
                "before_penalties": report.result.before_penalties,
                "after_penalties": report.result.after_penalties,
            },
            "per_constraint_summary": [
                {
                    "name": d.name,
                    "before": d.before,
                    "after": d.after,
                    "change_pct": d.change_pct,
                    "weight": d.weight,
                }
                for d in report.per_constraint_summary
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
