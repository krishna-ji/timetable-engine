"""Check room types for manual-used rooms."""
import json
from collections import Counter

with open("data/Rooms.json") as f:
    rooms = json.load(f)

target_rooms = ['E205','E101','E203','E102','E305','E304','E404',
                'F101','F102','A202','A301','A303','B304','B406','B407','B203']
print("Rooms used in manual but NOT in allowed_rooms for those events:")
for rm in rooms:
    if rm["room_id"] in target_rooms:
        print(f'  {rm["room_id"]}: type={rm["type"]}, features={rm["features"]}, '
              f'cap={rm["capacity"]}, name={rm["name"]}')

print()
print("All room types:")
types = Counter(rm["type"] for rm in rooms)
for t, c in types.most_common():
    print(f"  {t}: {c}")

print()
# Show which rooms ARE allowed for event 0 (AM651 practical) and their types
import pickle
with open(".cache/events_with_domains.pkl", "rb") as f:
    pkl = pickle.load(f)

idx_to_room = {int(k): v for k, v in pkl["idx_to_room"].items()}
ar = pkl["allowed_rooms"][0]  # Event 0
room_by_id = {rm["room_id"]: rm for rm in rooms}

print("Allowed rooms for Event 0 (AM651 practical):")
for ridx in ar:
    rid = idx_to_room[ridx]
    rm = room_by_id.get(rid, {})
    print(f'  {rid}: type={rm.get("type","?")}, features={rm.get("features",[])}')

# Check what features AM651 requires
events = pkl["events"]
ev0 = events[0]
print(f"\nEvent 0 course: {ev0['course_id']} {ev0['course_type']}")

# Check Course.json for AM651
with open("data/Course.json") as f:
    courses = json.load(f)
for c in courses:
    if c["CourseCode"] == "AM651":
        print(f"  Course: {c['CourseTitle']}")
        print(f"  PracticalRoomFeatures: {c.get('PracticalRoomFeatures','')}")
        break
