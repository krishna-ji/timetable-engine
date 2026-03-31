"""Compare manual routine vs GA-produced schedule.

Maps the hand-crafted manual_routine.json onto our internal encoding,
evaluates both with the same constraint system, and reports differences.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Load our internal data ──────────────────────────────────────────
PKL = ".cache/events_with_domains.pkl"
if not Path(PKL).exists():
    print("ERROR: run solve.py once to generate", PKL)
    sys.exit(1)

with open(PKL, "rb") as f:
    pkl = pickle.load(f)

events = pkl["events"]
E = len(events)
allowed_instr = pkl["allowed_instructors"]
allowed_rooms = pkl["allowed_rooms"]
allowed_starts = pkl["allowed_starts"]
idx_to_instr = {int(k): v for k, v in pkl["idx_to_instructor"].items()}
idx_to_room = {int(k): v for k, v in pkl["idx_to_room"].items()}
instr_to_idx = {v: int(k) for k, v in pkl["idx_to_instructor"].items()}
room_to_idx = {v: int(k) for k, v in pkl["idx_to_room"].items()}

# ── Load raw data for name mapping ──────────────────────────────────
with open("data/Instructors.json") as f:
    raw_instructors = json.load(f)
with open("data/Rooms.json") as f:
    raw_rooms = json.load(f)
with open("data/Groups.json") as f:
    raw_groups = json.load(f)

instructor_name_to_id = {}
for inst in raw_instructors:
    instructor_name_to_id[inst["name"].strip().lower()] = inst["id"]

room_name_to_id = {}
for rm in raw_rooms:
    room_name_to_id[rm["name"].strip().lower()] = rm["room_id"]
    room_name_to_id[rm["room_id"].lower()] = rm["room_id"]

# ── Parse manual routine ────────────────────────────────────────────
with open("manual_routine.json") as f:
    manual = json.load(f)

def manual_class_to_group_id(ac):
    fac = ac["faculty"]
    sem_num = ac["semester"].replace("sem", "")
    sec = ac.get("section", "AB")
    return f"{fac}{sem_num}{sec}"

def manual_room_to_id(rm):
    room_name = rm["name"].strip().lower()
    if room_name in room_name_to_id:
        return room_name_to_id[room_name]
    block = rm.get("block", "")
    room_no = rm.get("room_no", "")
    if block and room_no:
        candidate = f"{block}{room_no}"
        if candidate in room_to_idx:
            return candidate
    return None

def manual_teacher_to_id(teachers):
    if not teachers:
        return None
    name = teachers[0]["name"].strip().lower()
    return instructor_name_to_id.get(name)

DAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_OFFSET = {d: i * 7 for i, d in enumerate(DAY_ORDER)}

def time_to_quanta(day, start_time_str):
    parts = start_time_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    total_minutes = (h - 10) * 60 + m
    q_within_day = total_minutes // 60
    if day not in DAY_OFFSET:
        return None
    return DAY_OFFSET[day] + q_within_day

def session_duration_hours(start_str, end_str):
    """Compute duration in whole hours from HH:MM:SS strings."""
    sh, sm = int(start_str.split(":")[0]), int(start_str.split(":")[1])
    eh, em = int(end_str.split(":")[0]), int(end_str.split(":")[1])
    return round(((eh * 60 + em) - (sh * 60 + sm)) / 60)

# ── Build event lookup by (course, type, group_key) ─────────────────
# Theory events have groups like ['BAM1A','BAM1B'], manual says 'BAM1AB'
# Need to map 'BAM1AB' → frozenset({'BAM1A','BAM1B'})
def expand_group_id(gid):
    """Expand 'BAM1AB' → {'BAM1A','BAM1B'}, 'BME1A' → {'BME1A'}."""
    if len(gid) >= 2 and gid[-2:] == "AB":
        return frozenset({gid[:-2] + "A", gid[:-2] + "B"})
    if len(gid) >= 2 and gid[-2:] == "CD":
        return frozenset({gid[:-2] + "C", gid[:-2] + "D"})
    if len(gid) >= 2 and gid[-2:] == "EF":
        return frozenset({gid[:-2] + "E", gid[:-2] + "F"})
    return frozenset({gid})

# Build: (course_id, course_type, frozenset_groups) → [event_indices]
events_by_key = defaultdict(list)
for e, ev in enumerate(events):
    key = (ev["course_id"], ev["course_type"], frozenset(ev["group_ids"]))
    events_by_key[key].append(e)

manual_entries = manual["results"]
print(f"Manual routine: {len(manual_entries)} entries")
print(f"Internal events: {E}")
print()

# ── Process manual entries ──────────────────────────────────────────
X_manual = np.full(3 * E, -1, dtype=np.int64)
matched = 0
unmatched = 0
unmatched_reasons = Counter()
matched_events = set()
# Track consumed event indices so we don't reuse them
consumed = set()

for entry in manual_entries:
    subj_code = entry["subject"]["code"]
    course_type_raw = entry["course_type"]
    course_type = "theory" if course_type_raw == "TH" else "practical"

    ac = entry["academic_class"]
    group_id = manual_class_to_group_id(ac)

    # For practicals with lab_group, map to subgroup
    lab_group = entry.get("lab_group")
    if lab_group and course_type == "practical":
        # lab_group is a string like "A" or "B"
        lg = lab_group if isinstance(lab_group, str) else str(lab_group.get("name", ""))
        lg = lg.strip().upper()
        if lg in ("A", "B", "C", "D", "E", "F"):
            # section "AB" + lab_group "A" → take first letter of section
            # BAM1AB → BAM1A, BCE1CD + "C" → BCE1C
            group_id = group_id[:-2] + lg

    day = entry["day"]
    start_time = entry["start_time"]
    end_time = entry["end_time"]
    dur_h = session_duration_hours(start_time, end_time)

    teacher_id = manual_teacher_to_id(entry.get("teacher", []))
    room_id = manual_room_to_id(entry["room"])
    q = time_to_quanta(day, start_time)

    # Find matching internal event
    expanded = expand_group_id(group_id)
    # Also try the literal group_id as a single-element set
    search_keys = [
        (subj_code, course_type, expanded),
        (subj_code, course_type, frozenset({group_id})),
    ]

    found = False
    for sk in search_keys:
        candidates = events_by_key.get(sk, [])
        for e_idx in candidates:
            if e_idx in consumed:
                continue
            ev = events[e_idx]
            # Match by duration if possible
            if ev["num_quanta"] == dur_h or not candidates:
                consumed.add(e_idx)
                matched_events.add(e_idx)
                if teacher_id and teacher_id in instr_to_idx:
                    X_manual[3 * e_idx + 0] = instr_to_idx[teacher_id]
                if room_id and room_id in room_to_idx:
                    X_manual[3 * e_idx + 1] = room_to_idx[room_id]
                if q is not None:
                    X_manual[3 * e_idx + 2] = q
                matched += 1
                found = True
                break
        if found:
            break

    if not found:
        # Try any duration match
        for sk in search_keys:
            candidates = events_by_key.get(sk, [])
            for e_idx in candidates:
                if e_idx in consumed:
                    continue
                consumed.add(e_idx)
                matched_events.add(e_idx)
                if teacher_id and teacher_id in instr_to_idx:
                    X_manual[3 * e_idx + 0] = instr_to_idx[teacher_id]
                if room_id and room_id in room_to_idx:
                    X_manual[3 * e_idx + 1] = room_to_idx[room_id]
                if q is not None:
                    X_manual[3 * e_idx + 2] = q
                matched += 1
                found = True
                break
            if found:
                break

    if not found:
        unmatched += 1
        unmatched_reasons[(subj_code, course_type, group_id)] += 1

print(f"=== MATCHING RESULTS ===")
print(f"Matched events:   {matched}/{E}")
print(f"Unmatched entries: {unmatched}/{len(manual_entries)}")
print(f"Events with assignment: {len(matched_events)}/{E}")
print()

if unmatched_reasons:
    print("Top unmatched (subject, type, group):")
    for (s, t, g), cnt in unmatched_reasons.most_common(20):
        # Check if course exists in our data
        exists = any(ev["course_id"] == s for ev in events)
        print(f"  {s} {t} {g}: {cnt}  {'(course exists)' if exists else '(MISSING course)'}")
    print()

n_no_instructor = int((X_manual[0::3] == -1).sum())
n_no_room = int((X_manual[1::3] == -1).sum())
n_no_time = int((X_manual[2::3] == -1).sum())
print(f"Events missing instructor: {n_no_instructor}/{E}")
print(f"Events missing room:       {n_no_room}/{E}")
print(f"Events missing time:       {n_no_time}/{E}")
print()

# ── Check domain compliance ─────────────────────────────────────────
print("=== DOMAIN COMPLIANCE (matched events only) ===")
oob_instr = 0
oob_room = 0
oob_room_details = []
oob_time = 0
for e in matched_events:
    i_idx = X_manual[3*e + 0]
    r_idx = X_manual[3*e + 1]
    t_idx = X_manual[3*e + 2]

    if i_idx >= 0 and int(i_idx) not in allowed_instr[e]:
        oob_instr += 1
        iid = idx_to_instr.get(int(i_idx), "?")

    if r_idx >= 0 and int(r_idx) not in allowed_rooms[e]:
        oob_room += 1
        rid = idx_to_room.get(int(r_idx), "?")
        oob_room_details.append((e, events[e]["course_id"], events[e]["course_type"],
                                 rid, [idx_to_room[x] for x in allowed_rooms[e]]))

    if t_idx >= 0 and int(t_idx) not in allowed_starts[e]:
        oob_time += 1

print(f"Out-of-domain instructor: {oob_instr}/{len(matched_events)}")
print(f"Out-of-domain room:       {oob_room}/{len(matched_events)}")
print(f"Out-of-domain time:       {oob_time}/{len(matched_events)}")
if oob_room_details:
    print("\nRoom domain violations (manual uses room GA doesn't allow):")
    for e, cid, ct, rid, allowed in oob_room_details[:15]:
        print(f"  Event {e} ({cid} {ct}): manual={rid}, allowed={allowed[:5]}...")
if oob_instr > 0:
    print("\nINSTRUCTOR DOMAIN VIOLATIONS - manual assigns teachers not in allowed list!")
print()

# ── Evaluate with constraint system ────────────────────────────────
print("=== CONSTRAINT EVALUATION ===")
# Clamp all values to valid ranges
X_eval = X_manual.copy()
max_q = max(max(s) for s in allowed_starts if s)
for e in range(E):
    if X_eval[3*e + 0] < 0:
        X_eval[3*e + 0] = allowed_instr[e][0] if allowed_instr[e] else 0
    if X_eval[3*e + 1] < 0:
        X_eval[3*e + 1] = allowed_rooms[e][0] if allowed_rooms[e] else 0
    if X_eval[3*e + 2] < 0:
        X_eval[3*e + 2] = allowed_starts[e][0] if allowed_starts[e] else 0
    # Clamp to valid range
    if X_eval[3*e + 2] > max_q:
        X_eval[3*e + 2] = max_q

from src.pipeline.scheduling_problem import prepare_vectorized_data, fast_evaluate_hard_vectorized
vec_data = prepare_vectorized_data(pkl)
# Force ALL values into valid ranges for the evaluator
for e in range(E):
    if X_eval[3*e + 0] < 0:
        X_eval[3*e + 0] = allowed_instr[e][0] if allowed_instr[e] else 0
    if X_eval[3*e + 1] < 0:
        X_eval[3*e + 1] = allowed_rooms[e][0] if allowed_rooms[e] else 0
    q = int(X_eval[3*e + 2])
    if q < 0 or q not in allowed_starts[e]:
        X_eval[3*e + 2] = allowed_starts[e][0] if allowed_starts[e] else 0
X_batch = X_eval.reshape(1, -1)
G = fast_evaluate_hard_vectorized(X_batch, vec_data)

SHORT = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD", "sib"]
print("Manual routine violations (unmatched events filled with defaults):")
for name, val in zip(SHORT, G[0]):
    print(f"  {name}: {int(val)}")
total_strict = int(G[0, :5].sum() + G[0, 6:].sum())
print(f"  TOTAL HARD (strict): {total_strict}")
print()

# ── Load GA result ──────────────────────────────────────────────────
import glob
ga_dirs = sorted(glob.glob("output/ga_adaptive/*/"))
X_ga = None
if ga_dirs:
    latest = ga_dirs[-1]
    best_x_path = Path(latest) / "best_X.npy"
    if best_x_path.exists():
        X_ga = np.load(str(best_x_path))
        G_ga = fast_evaluate_hard_vectorized(X_ga.reshape(1, -1), vec_data)
        print("GA best violations:")
        for name, val in zip(SHORT, G_ga[0]):
            print(f"  {name}: {int(val)}")
        print(f"  TOTAL HARD (strict): {int(G_ga[0, :5].sum() + G_ga[0, 6:].sum())}")
        print()

# ── FTE analysis ────────────────────────────────────────────────────
print("=== FTE (DOUBLE-BOOKING) ANALYSIS ===")
def count_fte(X):
    occ = defaultdict(set)
    for e in range(E):
        i_idx = int(X[3*e + 0])
        t_start = int(X[3*e + 2])
        nq = events[e]["num_quanta"]
        for q in range(t_start, t_start + nq):
            occ[(i_idx, q)].add(e)
    conflicts = 0
    conflict_instructors = Counter()
    for (i_idx, q), evts in occ.items():
        if len(evts) > 1:
            conflicts += len(evts) - 1
            conflict_instructors[idx_to_instr.get(i_idx, f"?{i_idx}")] += len(evts) - 1
    return conflicts, conflict_instructors

# Only count FTE for matched portion
man_fte, man_fte_instr = count_fte(X_eval)
print(f"Manual FTE (with defaults for unmatched): {man_fte}")
if man_fte_instr:
    for iid, cnt in man_fte_instr.most_common(10):
        print(f"  {iid}: {cnt}")

if X_ga is not None:
    ga_fte, ga_fte_instr = count_fte(X_ga)
    print(f"GA FTE: {ga_fte}")
    for iid, cnt in ga_fte_instr.most_common(10):
        print(f"  {iid}: {cnt}")
print()

# ── Instructor load comparison ──────────────────────────────────────
print("=== INSTRUCTOR LOAD COMPARISON ===")
def instructor_load(X):
    loads = Counter()
    for e in range(E):
        i_idx = int(X[3*e + 0])
        if i_idx >= 0:
            loads[i_idx] += events[e]["num_quanta"]
    return loads

# Only count matched events for manual
man_matched_loads = Counter()
for e in matched_events:
    i_idx = int(X_manual[3*e + 0])
    if i_idx >= 0:
        man_matched_loads[i_idx] += events[e]["num_quanta"]

print(f"Manual (matched): {len(man_matched_loads)} unique instructors, "
      f"total={sum(man_matched_loads.values())}q")
sl = sorted(man_matched_loads.values(), reverse=True)
print(f"  Max={max(sl) if sl else 0}, Mean={np.mean(sl):.1f}, Std={np.std(sl):.1f}")
print(f"  Top10: {sl[:10]}")

if X_ga is not None:
    ga_loads = instructor_load(X_ga)
    print(f"GA: {len(ga_loads)} unique instructors, "
          f"total={sum(ga_loads.values())}q")
    gl = sorted(ga_loads.values(), reverse=True)
    print(f"  Max={max(gl) if gl else 0}, Mean={np.mean(gl):.1f}, Std={np.std(gl):.1f}")
    print(f"  Top10: {gl[:10]}")
print()

# ── Teacher assignment differences ──────────────────────────────────
if X_ga is not None:
    print("=== TEACHER ASSIGNMENT DIFFERENCES ===")
    diff_teacher = 0
    same_teacher = 0
    for e in matched_events:
        man_i = int(X_manual[3*e + 0])
        ga_i = int(X_ga[3*e + 0])
        if man_i >= 0:
            if man_i == ga_i:
                same_teacher += 1
            else:
                diff_teacher += 1
                if diff_teacher <= 15:
                    ev = events[e]
                    print(f"  {ev['course_id']} {ev['course_type']} {ev['group_ids']}: "
                          f"manual={idx_to_instr.get(man_i,'?')} "
                          f"GA={idx_to_instr.get(ga_i,'?')}")
    print(f"\nSame teacher: {same_teacher}/{same_teacher+diff_teacher}")
    print(f"Different teacher: {diff_teacher}/{same_teacher+diff_teacher}")
    print()

# ── Room usage comparison ───────────────────────────────────────────
print("=== ROOM USAGE ===")
man_room_count = Counter()
for e in matched_events:
    r = int(X_manual[3*e+1])
    if r >= 0:
        man_room_count[idx_to_room.get(r, "?")] += 1

if X_ga is not None:
    ga_room_count = Counter()
    for e in range(E):
        ga_room_count[idx_to_room.get(int(X_ga[3*e+1]), "?")] += 1
    print(f"Manual rooms used: {len(man_room_count)}")
    print(f"GA rooms used: {len(ga_room_count)}")
    manual_only = set(man_room_count) - set(ga_room_count)
    ga_only = set(ga_room_count) - set(man_room_count)
    if manual_only:
        print(f"Rooms in manual but not GA: {manual_only}")
    if ga_only:
        print(f"Rooms in GA but not manual: {ga_only}")
print()

# ── Special entries ─────────────────────────────────────────────────
print("=== SPECIAL ENTRIES IN MANUAL ===")
n_elective = sum(1 for e in manual_entries if e.get("is_elective"))
n_concurrent = sum(1 for e in manual_entries if e.get("concurrent_with"))
n_no_teacher = sum(1 for e in manual_entries if not e.get("teacher"))
print(f"Elective: {n_elective}, Concurrent: {n_concurrent}, No teacher: {n_no_teacher}")

# ── Theory sharing patterns ────────────────────────────────────────
print()
print("=== THEORY SHARING PATTERNS (manual) ===")
theory_by_course = defaultdict(list)
for entry in manual_entries:
    if entry["course_type"] == "TH":
        subj = entry["subject"]["code"]
        day = entry["day"]
        time = entry["start_time"]
        group = manual_class_to_group_id(entry["academic_class"])
        teacher = entry["teacher"][0]["name"] if entry.get("teacher") else "NONE"
        theory_by_course[subj].append((day, time, group, teacher))

# How many theory courses are taught at DIFFERENT times for different groups?
shared_count = 0
separate_count = 0
for subj, sessions in theory_by_course.items():
    groups_at_times = defaultdict(set)
    for d, t, g, _ in sessions:
        groups_at_times[(d, t)].add(g)
    unique_slots = len(groups_at_times)
    unique_groups = len(set(g for _, _, g, _ in sessions))
    # If same course appears at same time for multiple groups → shared lecture
    # If different times → separate sections
    if unique_slots < unique_groups:
        shared_count += 1
    else:
        separate_count += 1

    if unique_slots > 1 and unique_groups > 1:
        pass  # Multiple timeslots expected for 5h courses

print(f"Courses with shared lectures (multi-group same slot): {shared_count}")
print(f"Courses taught separately per group: {separate_count}")

# ── KEY INSIGHT: Does manual use instructors not in our allowed list?
print()
print("=" * 60)
print("  KEY FINDINGS SUMMARY")
print("=" * 60)
print(f"1. Matched {len(matched_events)}/{E} events ({100*len(matched_events)/E:.0f}%)")
print(f"2. Room domain violations: {oob_room}/{len(matched_events)} "
      f"({100*oob_room/max(len(matched_events),1):.0f}%) — manual uses rooms our system disallows")
print(f"3. Instructor domain violations: {oob_instr}/{len(matched_events)} "
      f"({100*oob_instr/max(len(matched_events),1):.0f}%)")
print(f"4. Time domain violations: {oob_time}")
print(f"5. Constraint eval strict hard: {total_strict}")
print(f"6. Unmatched manual entries: {unmatched} (courses not in our data or matching failure)")
