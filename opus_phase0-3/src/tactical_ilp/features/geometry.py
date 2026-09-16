"""
Task 3.1 — Basic geometric predicates (the "neural layer" of the MVP).

Technical document §2.2 replaces a real neural encoder with a rule-based
feature extractor for the MVP.  This module is that extractor: pure, side-effect
free Python functions that turn SPADL coordinates plus a 360 freeze frame into
**discrete symbols**.  Phase 3.2/3.3 dumps their output as ground Prolog facts.

Reference frame (fixed for the whole project)
---------------------------------------------
    x in [0, 105]   0 = Barcelona's own goal line, 105 = opponent goal line
    y in [0, 68]    0 = right touchline seen from Barcelona's attacking direction

Vertical lanes (5-lane model)
    right_wing | right_half_space | central_channel | left_half_space | left_wing
     0 - 13.6      13.6 - 27.2        27.2 - 40.8       40.8 - 54.4     54.4 - 68
"""

from __future__ import annotations

import math

from tactical_ilp.config import (
    DEFENSIVE_THIRD_MAX_X,
    LANE_EDGES,
    LANE_NAMES,
    MIDDLE_THIRD_MAX_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PRESSURE_MIN_OPPONENTS,
    PRESSURE_RADIUS_M,
    TIGHT_PRESSURE_RADIUS_M,
    ZONE14_X,
    ZONE14_Y,
)

# --------------------------------------------------------------------------
# lanes / thirds / zones
# --------------------------------------------------------------------------


def lane_of(y: float) -> str:
    """Vertical lane containing ``y``."""
    y = min(max(float(y), 0.0), PITCH_WIDTH)
    for i in range(len(LANE_NAMES)):
        lo, hi = LANE_EDGES[i], LANE_EDGES[i + 1]
        if lo <= y < hi or (i == len(LANE_NAMES) - 1 and y == hi):
            return LANE_NAMES[i]
    return LANE_NAMES[-1]


def occupies_half_space(x: float, y: float) -> str:
    """
    Lane occupied by a player/ball at ``(x, y)``.

    Named after the tactical concept but returns the full lane vocabulary so a
    single predicate covers wings, half-spaces and the central channel.
    ``x`` is accepted for signature stability (lanes are x-independent) and to
    keep the Prolog-side predicate a proper 2-argument geometric function.
    """
    return lane_of(y)


def is_half_space(x: float, y: float) -> bool:
    return occupies_half_space(x, y) in ("right_half_space", "left_half_space")


def third_of(x: float) -> str:
    """Horizontal third from Barcelona's point of view."""
    x = float(x)
    if x <= DEFENSIVE_THIRD_MAX_X:
        return "defensive_third"
    if x <= MIDDLE_THIRD_MAX_X:
        return "middle_third"
    return "final_third"


def is_zone14(x: float, y: float) -> bool:
    """Central zone immediately in front of the opponent penalty area."""
    return (ZONE14_X[0] <= float(x) < ZONE14_X[1]) and (
        ZONE14_Y[0] <= float(y) < ZONE14_Y[1]
    )


def pitch_zone(x: float, y: float) -> str:
    """Compact symbol combining third and lane, e.g. 'defensive_third|left_wing'."""
    return f"{third_of(x)}|{lane_of(y)}"


# --------------------------------------------------------------------------
# pressure
# --------------------------------------------------------------------------


def under_pressure(
    opponents_within_5m: float | None = None,
    statsbomb_flag: bool = False,
    min_opponents: int = PRESSURE_MIN_OPPONENTS,
) -> bool:
    """
    Pressure model.

    Primary signal is the 360 freeze frame (how many opponents stand within
    ``PRESSURE_RADIUS_M`` of the ball).  When no freeze frame exists we fall
    back to StatsBomb's own ``under_pressure`` flag, so the predicate is total.
    """
    if opponents_within_5m is not None and not _isnan(opponents_within_5m):
        return int(opponents_within_5m) >= min_opponents
    return bool(statsbomb_flag)


def pressure_level(
    opponents_within_3m: float | None,
    opponents_within_5m: float | None,
    statsbomb_flag: bool = False,
) -> str:
    """Discrete pressure symbol: no_pressure | light_pressure | heavy_pressure."""
    if opponents_within_3m is not None and not _isnan(opponents_within_3m):
        n3 = int(opponents_within_3m)
        n5 = int(opponents_within_5m or 0)
        if n3 >= 2:
            return "heavy_pressure"
        if n3 >= 1 or n5 >= 2:
            return "light_pressure"
        return "no_pressure"
    return "light_pressure" if statsbomb_flag else "no_pressure"


# --------------------------------------------------------------------------
# passes / carries
# --------------------------------------------------------------------------


def pass_direction(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> str:
    """forward | sideways | backward, using a +/-30 degree band around the y axis."""
    dx = float(end_x) - float(start_x)
    dy = float(end_y) - float(start_y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "sideways"
    angle = math.degrees(math.atan2(dx, abs(dy) if dy else 1e-9))
    if dx > 0 and (abs(dy) <= 1e-9 or abs(dx) >= abs(dy) * 0.5774):
        return "forward"
    if dx < 0 and (abs(dy) <= 1e-9 or abs(dx) >= abs(dy) * 0.5774):
        return "backward"
    return "sideways"


def pass_length_class(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> str:
    """short (<10 m) | medium (10-25 m) | long (>25 m)."""
    d = math.hypot(float(end_x) - float(start_x), float(end_y) - float(start_y))
    if d < 10.0:
        return "short"
    if d <= 25.0:
        return "medium"
    return "long"


def is_progressive(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> bool:
    """
    Standard progressive-action rule: the action reduces the distance to the
    opponent goal by at least 25 % (own half) or moves >= 10 m forward
    (opponent half).
    """
    goal = (PITCH_LENGTH, PITCH_WIDTH / 2.0)
    d0 = math.hypot(goal[0] - float(start_x), goal[1] - float(start_y))
    d1 = math.hypot(goal[0] - float(end_x), goal[1] - float(end_y))
    if float(start_x) <= PITCH_LENGTH / 2.0:
        return d1 <= 0.75 * d0
    return (float(end_x) - float(start_x)) >= 10.0


def distance_to_own_goal(x: float, y: float) -> float:
    return math.hypot(float(x), float(y) - PITCH_WIDTH / 2.0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


__all__ = [
    "lane_of",
    "occupies_half_space",
    "is_half_space",
    "third_of",
    "is_zone14",
    "pitch_zone",
    "under_pressure",
    "pressure_level",
    "pass_direction",
    "pass_length_class",
    "is_progressive",
    "distance_to_own_goal",
    "PRESSURE_RADIUS_M",
    "TIGHT_PRESSURE_RADIUS_M",
]
