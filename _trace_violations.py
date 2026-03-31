"""Trace exactly which events remain CTE-violating after repair."""
import numpy as np
import pickle
from collections import defaultdict

from src.pipeline.fast_evaluator_vectorized import (
    fast_evaluate_hard_vectorized,
    prepare_vectorized_data,
)
from src.pipeline.repair_operator_vectorized import VectorizedRepair
from src.pipeline.scheduling_problem import SchedulingProblem, _TOLERATED_HARD_COLS
from src.pipeline.pymoo_operators import RandomDomainSampling

pkl = pickle.load(open(".cache/events_with_domains.pkl", "rb"))
vr = VectorizedRepair(".cache/events_with_domains.pkl")
vd = prepare_vectorized_data(pkl)
events = pkl["events"]
ai = pkl["allowed_instructors"]
ar = pkl["allowed_rooms"]
ast_list = pkl["allowed_starts"]
E = len(events)
T = 42


class FP:
    n_var = E * 3


s = RandomDomainSampling(".cache/events_with_domains.pkl")
np.random.seed(42)
X_raw = s._do(FP(), 60)
X_rep = vr.repair_batch(X_raw, passes=4)

# Find best
p = SchedulingProblem(".cache/events_with_domains.pkl")
strict = [j for j in range(8) if j not in _TOLERATED_HARD_COLS]
best_cv, best_idx = 1e9, 0
for i in range(60):
    out = {}
    p._evaluate(X_rep[i : i + 1], out)
    G = out["G"][0]
    cv = sum(max(0, G[j]) for j in strict)
    if cv < best_cv:
        best_cv, best_idx = cv, i

print(f"Best single repaired: cv={best_cv:.0f} idx={best_idx}")
X_best = X_rep[best_idx]

# Decomposed scores
rsc, isc, gsc = vr._score_decomposed_batch(X_best.reshape(1, -1))
print(
    f"CTE_events: {(gsc[0] > 0).sum()}, "
    f"FTE_events: {(isc[0] > 0).sum()}, "
    f"SRE_events: {(rsc[0] > 0).sum()}"
)

# ── CTE violations: which events, which groups ─────────────────
print("\n=== CTE VIOLATIONS ===")
inst_n = np.clip(X_best[0::3], 0, vr.n_instructors - 1).astype(np.int64)
room_n = np.clip(X_best[1::3], 0, vr.n_rooms - 1).astype(np.int64)
time_n = X_best[2::3].astype(np.int64)

# Build group occupancy
group_to_idx = {}
for ev in events:
    for gid in ev["group_ids"]:
        if gid not in group_to_idx:
            group_to_idx[gid] = len(group_to_idx)
G = len(group_to_idx)
idx_to_group = {v: k for k, v in group_to_idx.items()}

grp_occ = np.zeros((G, T), dtype=np.int32)
for e in range(E):
    t = int(time_n[e])
    d = int(events[e]["num_quanta"])
    for gid in events[e]["group_ids"]:
        gidx = group_to_idx[gid]
        for q in range(t, min(t + d, T)):
            grp_occ[gidx, q] += 1

# Find groups with occupancy > 1
cte_groups = set()
for g in range(G):
    if (grp_occ[g] > 1).any():
        cte_groups.add(g)
        max_occ = grp_occ[g].max()
        conflict_quanta = (grp_occ[g] > 1).sum()
        gname = idx_to_group[g]
        print(f"  Group {gname}: max_occ={max_occ}, conflict_quanta={conflict_quanta}, "
              f"total_load={grp_occ[g].sum()}/{T}")

print(f"\nTotal CTE-violating groups: {len(cte_groups)}")

# For each CTE-violating event, show detailed info
print("\n=== CTE EVENTS DETAIL ===")
for e in range(E):
    if gsc[0, e] == 0:
        continue
    t = int(time_n[e])
    d = int(events[e]["num_quanta"])
    i_e = int(inst_n[e])
    r_e = int(room_n[e])
    gids = events[e]["group_ids"]
    n_starts = len(ast_list[e])
    n_inst = len(ai[e])

    # Count how many starts would be CTE-free
    cte_free = 0
    for s_cand in ast_list[e]:
        bad = False
        for gid in gids:
            gidx = group_to_idx[gid]
            for q in range(s_cand, min(s_cand + d, T)):
                # occupancy excluding self
                occ = grp_occ[gidx, q]
                if gidx in cte_groups:
                    for q2 in range(t, min(t + d, T)):
                        if q == q2:
                            occ -= 1
                if occ > 0:
                    bad = True
                    break
            if bad:
                break
        if not bad:
            cte_free += 1

    print(
        f"  e{e}: {events[e]['course_id']} {events[e]['course_type']} "
        f"dur={d} t={t} i={i_e} r={r_e} "
        f"groups={gids} starts={n_starts} inst={n_inst} "
        f"cte_free_starts={cte_free}/{n_starts} "
        f"grp_sc={gsc[0,e]}"
    )

# ── FTE violations ──────────────────────────────────────────────
print("\n=== FTE VIOLATIONS ===")
inst_occ = np.zeros((vr.n_instructors, T), dtype=np.int32)
for e in range(E):
    t = int(time_n[e])
    d = int(events[e]["num_quanta"])
    i_e = int(inst_n[e])
    for q in range(t, min(t + d, T)):
        inst_occ[i_e, q] += 1

fte_insts = set()
for i in range(vr.n_instructors):
    if (inst_occ[i] > 1).any():
        max_occ = inst_occ[i].max()
        n_conflict = (inst_occ[i] > 1).sum()
        fte_insts.add(i)
        # Find events using this instructor
        ev_list = [e for e in range(E) if int(inst_n[e]) == i]
        print(f"  Inst {i}: max_occ={max_occ}, conflict_quanta={n_conflict}, "
              f"n_events={len(ev_list)}, load={inst_occ[i].sum()}/{T}")

print(f"\nTotal FTE-violating instructors: {len(fte_insts)}")

# ── SRE violations ──────────────────────────────────────────────
print("\n=== SRE VIOLATIONS ===")
room_occ = np.zeros((vr.n_rooms, T), dtype=np.int32)
for e in range(E):
    t = int(time_n[e])
    d = int(events[e]["num_quanta"])
    r_e = int(room_n[e])
    for q in range(t, min(t + d, T)):
        room_occ[r_e, q] += 1

for r in range(vr.n_rooms):
    if (room_occ[r] > 1).any():
        max_occ = room_occ[r].max()
        n_conflict = (room_occ[r] > 1).sum()
        print(f"  Room {r}: max_occ={max_occ}, conflict_quanta={n_conflict}, "
              f"load={room_occ[r].sum()}/{T}")

# ── Pigeonhole check for BCE3 ──────────────────────────────────
print("\n=== BCE3 PIGEONHOLE DEEP ANALYSIS ===")
bce3_groups = [gid for gid in group_to_idx if gid.startswith("BCE3")]
bce3_groups.sort()
print(f"BCE3 groups: {bce3_groups}")

# For each BCE3 group, show free quanta vs events that need placement
for gid in bce3_groups:
    gidx = group_to_idx[gid]
    load = int(grp_occ[gidx].sum())
    free = int((grp_occ[gidx] == 0).sum())
    max_occ = int(grp_occ[gidx].max())
    print(f"\n  {gid}: load={load}/{T}, free_quanta={free}, max_occ={max_occ}")
    # Show events in this group
    group_events = [e for e in range(E) if gid in events[e]["group_ids"]]
    shared = [e for e in group_events if len(events[e]["group_ids"]) > 1]
    indep = [e for e in group_events if len(events[e]["group_ids"]) == 1]
    shared_q = sum(events[e]["num_quanta"] for e in shared)
    indep_q = sum(events[e]["num_quanta"] for e in indep)
    print(f"    shared: {len(shared)} events ({shared_q}q), indep: {len(indep)} events ({indep_q}q)")

    # Check inter-group instructor conflicts for independent events
    indep_instructors = defaultdict(list)
    for e in indep:
        i_e = int(inst_n[e])
        indep_instructors[i_e].append(e)
    print(f"    indep instructor counts: {dict(sorted(((i, len(es)) for i, es in indep_instructors.items()), key=lambda x: -x[1]))}")
