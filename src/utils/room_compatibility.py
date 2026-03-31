"""
Centralized Room Type Compatibility Logic

SINGLE SOURCE OF TRUTH for room type matching across:
- Constraints (hard.py)
- Mutation operators (mutation.py)
- Repair operators (repair.py, repair_selective.py)
- Room entity methods (room.py)

This eliminates 4+ duplicate implementations and ensures consistent behavior.
"""

from __future__ import annotations

__all__ = ["is_room_suitable_for_course", "is_room_type_compatible"]


def is_room_type_compatible(required: str, room_type: str) -> bool:
    """
    Check if room type satisfies requirement with flexible compatibility.

    Compatibility Rules:
    - Lecture courses → lecture, classroom, auditorium, seminar, tutorial
    - Practical courses → practical, lab, laboratory, computer_lab, science_lab
    - Exact matches always work

    Args:
        required: Required room type (e.g., "lecture", "practical")
            Should be lowercase and stripped, but will normalize if not.
        room_type: Actual room type (e.g., "lecture", "practical")
            Should be lowercase and stripped, but will normalize if not.

    Returns:
        True if compatible, False otherwise

    Examples:
        >>> is_room_type_compatible("lecture", "classroom")
        True
        >>> is_room_type_compatible("lecture", "auditorium")
        True
        >>> is_room_type_compatible("practical", "lab")
        True
        >>> is_room_type_compatible("practical", "lecture")
        False
        >>> is_room_type_compatible("LECTURE", "Classroom")  # Case insensitive
        True

    Note:
        This function is intentionally lenient to handle real-world scheduling
        flexibility where similar room types can often substitute for each other.
    """
    # Normalize inputs (defensive - callers should already normalize)
    req = required.lower().strip()
    room = room_type.lower().strip()

    # Exact match
    if req == room:
        return True

    # "Both" type rooms are compatible with any requirement
    if room in ["both", "multipurpose"]:
        return True

    # Lecture/theory courses: Accept lecture, classroom, auditorium, seminar, tutorial
    if req in ["lecture", "classroom", "theory"] and room in [
        "lecture",
        "classroom",
        "auditorium",
        "seminar",
        "tutorial",
    ]:
        return True

    # Practical/lab courses: Accept practical, lab variants
    return req in ["practical", "lab", "laboratory"] and room in [
        "practical",
        "lab",
        "laboratory",
        "computer_lab",
        "science_lab",
    ]


def is_room_suitable_for_course(
    required_type: str,
    room_type: str,
    course_lab_features: list[str] | None = None,
    room_specific_features: list[str] | None = None,
) -> bool:
    """Check if a room is fully suitable for a course: type AND specific features.

    For practical courses with specific lab requirements (e.g. "chemistry lab",
    "networking lab"), the room must have **at least one** of those features in
    its ``specific_features`` list.  Theory courses or practical courses
    without specific lab requirements only need the broad type match.

    Args:
        required_type: Course's ``required_room_features`` (``"lecture"`` or ``"practical"``).
        room_type: Room's ``room_features`` (``"lecture"`` or ``"practical"``).
        course_lab_features: Course's ``specific_lab_features`` (may be empty/None).
        room_specific_features: Room's ``specific_features`` (may be empty/None).

    Returns:
        ``True`` if the room satisfies both type and feature requirements.

    Examples:
        >>> is_room_suitable_for_course("practical", "practical",
        ...     ["chemistry lab"], ["chemistry lab", "physics lab"])
        True
        >>> is_room_suitable_for_course("practical", "practical",
        ...     ["chemistry lab"], ["networking lab"])
        False
        >>> is_room_suitable_for_course("practical", "practical", [], [])
        True
        >>> is_room_suitable_for_course("lecture", "lecture", None, None)
        True
    """
    # Step 1: if course has specific feature requirements, feature match
    # takes priority over the broad type check.  Example: a "practical"
    # course with PracticalRoomFeatures="Lecture Hall" should be placed in
    # any room that has the "Lecture Hall" feature, even if the room is
    # type "Lecture" rather than "Practical".
    if course_lab_features:
        if not room_specific_features:
            return False  # Course needs features but room has none
        room_feat_set = {f.lower().strip() for f in room_specific_features}
        if any(f.lower().strip() in room_feat_set for f in course_lab_features):
            return True
        # Features don't match — fall through to type check as last resort
        return False

    # Step 2: no specific feature requirements — broad type check only
    return is_room_type_compatible(required_type, room_type)
