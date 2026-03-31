"""Analyze group conflict structure to understand CTE bottleneck."""
import pickle
import re
import numpy as np
from collections import defaultdict

with open('.cache/events_with_domains.pkl', 'rb') as f:
    data = pickle.load(f)

events = data['events']
E = len(events)
allowed_starts = data['allowed_starts']
allowed_instructors = data['allowed_instructors']

# Time domain analysis
dom_sizes = [len(s) for s in allowed_starts]
print('=== TIME DOMAIN ANALYSIS ===')
print('Time domain sizes: min=%d, max=%d, mean=%.1f' % (min(dom_sizes), max(dom_sizes), np.mean(dom_sizes)))
print('Events with tight time domains (<35 starts):', sum(1 for d in dom_sizes if d < 35))
print()

# Group analysis
group_events = defaultdict(list)
for e, ev in enumerate(events):
    for gid in ev['group_ids']:
        group_events[gid].append(e)

n_groups = len(group_events)
print('=== GROUP STRUCTURE ===')
print('Groups: %d, Events: %d, Multi-group events: %d' % (
    n_groups, E, sum(1 for ev in events if len(ev['group_ids']) > 1)
))
print()

# BCE3A deep analysis
print('=== BCE3A (tightest group: 41/42 quanta) ===')
bce3a = group_events['BCE3A']
total_q = sum(events[e]['num_quanta'] for e in bce3a)
print('Events: %d, Total quanta: %d/42, Slack: %d' % (len(bce3a), total_q, 42-total_q))
print()

# BCE3 siblings
bce3_groups = sorted([g for g in group_events if g.startswith('BCE3')])
print('BCE3 subgroups:', bce3_groups)

shared_all = set(group_events[bce3_groups[0]])
for g in bce3_groups[1:]:
    shared_all &= set(group_events[g])
shared_q_all = sum(events[e]['num_quanta'] for e in shared_all)
print('Shared across ALL %d BCE3 groups: %d events, %d quanta' % (
    len(bce3_groups), len(shared_all), shared_q_all
))
for e in sorted(shared_all):
    ev = events[e]
    n_inst = len(allowed_instructors[e])
    print('  e%d: %s %s dur=%d n_groups=%d n_inst=%d starts=%d' % (
        e, ev['course_id'], ev['course_type'], ev['num_quanta'],
        len(ev['group_ids']), n_inst, len(allowed_starts[e])
    ))
print()

# Compute per-subgroup independent events
for g in bce3_groups[:3]:
    indep = set(group_events[g]) - shared_all
    indep_q = sum(events[e]['num_quanta'] for e in indep)
    print('%s independent: %d events, %d quanta' % (g, len(indep), indep_q))
    for e in sorted(indep):
        ev = events[e]
        print('  e%d: %s %s dur=%d groups=%s inst=%d' % (
            e, ev['course_id'], ev['course_type'], ev['num_quanta'],
            ev['group_ids'], len(allowed_instructors[e])
        ))
print()

# BCE3A vs BCE3B: can independent events fit?
sa = set(group_events['BCE3A'])
sb = set(group_events['BCE3B'])
shared_ab = sa & sb
only_a = sa - sb
only_b = sb - sa
sq = sum(events[e]['num_quanta'] for e in shared_ab)
aq = sum(events[e]['num_quanta'] for e in only_a)
bq = sum(events[e]['num_quanta'] for e in only_b)
print('BCE3A vs BCE3B:')
print('  Shared: %d events (%d quanta)' % (len(shared_ab), sq))
print('  A-only: %d events (%d quanta)' % (len(only_a), aq))
print('  B-only: %d events (%d quanta)' % (len(only_b), bq))
avail = 42 - sq
print('  Available after shared: %d quanta' % avail)
print('  A-indep + B-indep = %d quanta (must fit since different subgroups)' % (aq + bq))
# The A-only and B-only events CAN overlap in time (they're different subgroups)
# but if they share an instructor, they CANNOT
# Check instructor overlap
a_insts = set()
for e in only_a:
    a_insts.update(allowed_instructors[e])
b_insts = set()
for e in only_b:
    b_insts.update(allowed_instructors[e])
print('  Instructor overlap between A-only and B-only:', len(a_insts & b_insts))
print()

# GLOBAL: Inter-group contention
print('=== INTER-GROUP CONTENTION (sibling batches) ===')
batch_map = defaultdict(list)
for gid in group_events:
    m = re.match(r'^(.*?)([A-Z])$', gid)
    if m:
        batch_map[m.group(1)].append(gid)

for batch_name in sorted(batch_map, key=lambda b: -max(
    sum(events[e]['num_quanta'] for e in group_events[g]) for g in batch_map[b]
)):
    siblings = sorted(batch_map[batch_name])
    if len(siblings) < 2:
        continue
    loads = [sum(events[e]['num_quanta'] for e in group_events[g]) for g in siblings]

    shared = set(group_events[siblings[0]])
    for g in siblings[1:]:
        shared &= set(group_events[g])
    shared_q = sum(events[e]['num_quanta'] for e in shared)
    indep_qs = []
    for g in siblings:
        indep = set(group_events[g]) - shared
        indep_qs.append(sum(events[e]['num_quanta'] for e in indep))
    avail = 42 - shared_q
    status = ''
    if max(indep_qs) > avail:
        status = ' ** INFEASIBLE: max_indep=%d > avail=%d **' % (max(indep_qs), avail)
    print('%s (%d siblings): loads=%s shared=%dq indep=%s avail=%d%s' % (
        batch_name, len(siblings), loads, shared_q, indep_qs, avail, status
    ))

print()
print('=== KEY INSIGHT: SHARED-EVENT INSTRUCTOR BOTTLENECK ===')
# The real killer: shared events (taught to ALL subgroups simultaneously)
# are locked to specific instructors. When those instructors also teach
# independent subgroup events, FTE+CTE conflicts cascade.
# Count how many shared events have single-instructor domains
for batch_name in ['BCE3', 'BCE1', 'BAR8', 'BEI6']:
    siblings = sorted(batch_map.get(batch_name, []))
    if len(siblings) < 2:
        continue
    shared = set(group_events[siblings[0]])
    for g in siblings[1:]:
        shared &= set(group_events[g])
    single_inst = 0
    for e in shared:
        if len(allowed_instructors[e]) == 1:
            single_inst += 1
    print('%s shared events: %d total, %d single-instructor (bottleneck)' % (
        batch_name, len(shared), single_inst
    ))
