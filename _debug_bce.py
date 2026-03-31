"""Debug BCE matching."""
import pickle
from collections import defaultdict

with open(".cache/events_with_domains.pkl", "rb") as f:
    pkl = pickle.load(f)

events = pkl["events"]

print("ENCE 101 theory events:")
for e, ev in enumerate(events):
    if ev["course_id"] == "ENCE 101" and ev["course_type"] == "theory":
        print(f"  Event {e}: groups={ev['group_ids']} nq={ev['num_quanta']}")

print()
print("ENSH 101 theory events:")
for e, ev in enumerate(events):
    if ev["course_id"] == "ENSH 101" and ev["course_type"] == "theory":
        print(f"  Event {e}: groups={ev['group_ids']} nq={ev['num_quanta']}")

print()
# How many events per course for BCE sections?
bce_theory = defaultdict(list)
for e, ev in enumerate(events):
    if ev["course_type"] == "theory" and any("BCE" in g for g in ev["group_ids"]):
        key = (ev["course_id"], frozenset(ev["group_ids"]))
        bce_theory[key].append((e, ev["num_quanta"]))

print("BCE theory events by (course, groups):")
for (cid, gs), evts in sorted(bce_theory.items()):
    print(f"  {cid} {sorted(gs)}: {len(evts)} events, nq={[nq for _, nq in evts]}")

# Show all rooms used by events that have OOB in manual
print()
print("Events with room violations — allowed_rooms:")
for eidx in [0, 20, 40, 66, 188, 200, 202]:
    if eidx < len(events):
        ev = events[eidx]
        ar = pkl["allowed_rooms"][eidx]
        from_idx = {int(k): v for k, v in pkl["idx_to_room"].items()}
        allowed_names = [from_idx[x] for x in ar]
        print(f"  Event {eidx} ({ev['course_id']} {ev['course_type']}): {allowed_names}")
