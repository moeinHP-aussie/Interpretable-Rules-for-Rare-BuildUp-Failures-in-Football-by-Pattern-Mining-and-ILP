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
from statistics import median

from tactical_ilp.config import (
    DEFENSIVE_LINE_SIZE,
    FLANK_LINE_CARRY_MIN_M,
    LONG_PASS_MIN_M,
    DIAGONAL_MAX_ANGLE_DEG,
    DIAGONAL_MIN_ANGLE_DEG,
    DIAGONAL_MIN_FORWARD_DX_M,
    DIAGONAL_MIN_LENGTH_M,
    DEFENSIVE_THIRD_MAX_X,
    LANE_EDGES,
    LANE_NAMES,
    MIDDLE_THIRD_MAX_X,
    MIDFIELD_LINE_SIZE,
    MIN_OPPONENTS_FOR_LINE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PRESSURE_MIN_OPPONENTS,
    PRESSURE_RADIUS_M,
    SWITCH_MIN_DY_M,
    TIGHT_PRESSURE_RADIUS_M,
    TRIANGLE_MAX_AREA_M2,
    TRIANGLE_MIN_ANGLE_DEG,
    TRIANGLE_MIN_AREA_M2,
    TRIANGLE_EDGE_TOLERANCE_M,
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


def third_of(x: float) -> str:
    """Horizontal third from Barcelona's point of view."""
    x = float(x)
    if x <= DEFENSIVE_THIRD_MAX_X:
        return "defensive_third"
    if x <= MIDDLE_THIRD_MAX_X:
        return "middle_third"
    return "final_third"


def pitch_zone(x: float, y: float) -> str:
    """Compact symbol combining third and lane, e.g. 'defensive_third|left_wing'."""
    return f"{third_of(x)}|{lane_of(y)}"


def flank_dribble_features(
    action_type: str,
    start_x: float | None,
    start_y: float | None,
    end_x: float | None,
    end_y: float | None,
    min_forward_m: float = FLANK_LINE_CARRY_MIN_M,
) -> frozenset[str]:
    """Return flank dribble concepts supported by one SPADL action.

    ``flank_line_carry`` requires a dribble that begins and ends in the same
    outside wing lane and advances at least ``min_forward_m`` along +x.
    ``flank_cut_inside`` begins in either wing and ends in a half-space or the
    central channel. ``flank_dribble`` marks every dribble beginning in a wing.
    Missing coordinates and non-dribble actions produce no concepts.
    """
    if action_type != "dribble" or start_y is None or pd_is_missing(start_y):
        return frozenset()
    start_lane = lane_of(start_y)
    wings = {"left_wing", "right_wing"}
    if start_lane not in wings:
        return frozenset()
    concepts = {"flank_dribble"}
    if end_y is None or pd_is_missing(end_y):
        return frozenset(concepts)
    end_lane = lane_of(end_y)
    if end_lane in {"left_half_space", "right_half_space", "central_channel"}:
        concepts.add("flank_cut_inside")
    coordinates_present = all(
        value is not None and not pd_is_missing(value)
        for value in (start_x, end_x)
    )
    if (
        coordinates_present
        and end_lane == start_lane
        and float(end_x) - float(start_x) >= min_forward_m
    ):
        concepts.add("flank_line_carry")
    return frozenset(concepts)


def action_movement_features(
    action_type: str,
    start_x: float | None,
    start_y: float | None,
    end_x: float | None,
    end_y: float | None,
) -> frozenset[str]:
    """Return the reviewed action-level carry/pass/take-on geometry tags.

    SPADL coordinates are in metres and Barcelona attacks towards +x. The
    central channel includes both half-spaces; the two outer lanes are flanks.
    """
    if any(pd_is_missing(value) for value in (start_x, start_y, end_x, end_y)):
        return frozenset()
    sx, sy, ex, ey = map(float, (start_x, start_y, end_x, end_y))
    start_lane, end_lane = lane_of(sy), lane_of(ey)
    wings = {"left_wing", "right_wing"}
    central_lanes = set(LANE_NAMES) - wings
    dx = ex - sx
    length = math.hypot(dx, ey - sy)
    concepts: set[str] = set()

    if action_type in {"dribble", "carry"}:
        if start_lane in central_lanes and length >= 3.5:
            concepts.add("central_long_carry")
        if start_lane in wings and end_lane == start_lane and dx <= -4.0:
            concepts.add("flank_backward_carry")
    elif action_type == "take_on":
        if start_lane in central_lanes:
            concepts.add("central_take_on")
    elif action_type in PASS_LIKE_GEOMETRY:
        if start_lane in central_lanes and length >= 7.0 and dx <= -5.0:
            concepts.add("central_backward_pass")
        if (
            start_lane in wings
            and end_lane in central_lanes
            and length >= 10.0
            and dx >= 5.0
        ):
            concepts.add("flank_to_center_progression")
    return frozenset(concepts)


# Pass-like SPADL action types used for spatial features; kept local to avoid
# a dependency from this geometry module on the Prolog fact writer.
PASS_LIKE_GEOMETRY = frozenset({
    "pass", "cross", "freekick_short", "freekick_crossed", "goalkick",
    "throw_in", "corner_short", "corner_crossed",
})


def pd_is_missing(value) -> bool:
    """Small scalar missing-value check without a pandas dependency."""
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return True


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
    """Bucket pressure intensity, using a stricter threshold than any-contact pressure.

    ``under_pressure`` is the binary signal (at least one opponent within 5m).
    This intensity bucket calls pressure light at >=1 opponent within 3m or
    >=2 within 5m, and heavy at >=2 within 3m. An isolated opponent in the
    outer 3-5m ring can therefore set the binary flag while remaining below
    the intensity threshold used for hesitation.
    """
    has_n3 = opponents_within_3m is not None and not _isnan(opponents_within_3m)
    has_n5 = opponents_within_5m is not None and not _isnan(opponents_within_5m)
    n3 = int(opponents_within_3m) if has_n3 else 0
    n5 = int(opponents_within_5m) if has_n5 else 0

    if n3 >= 2:
        return "heavy_pressure"
    if n3 >= 1 or n5 >= 2:
        return "light_pressure"
    if not has_n5 and statsbomb_flag:
        return "light_pressure"
    return "no_pressure"


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


def is_long_pass(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_length_m: float = LONG_PASS_MIN_M,
) -> bool:
    """Whether a pass travels at least ``min_length_m`` in Euclidean metres."""
    if any(_isnan(value) for value in (start_x, start_y, end_x, end_y)):
        return False
    return math.hypot(float(end_x) - float(start_x), float(end_y) - float(start_y)) >= min_length_m


def is_progressive(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> bool:
    """
    Zone-scaled forward-progress rule (redefined after phase-3 empirical
    review — see report.md section "بازتعریف is_progressive").

    A single threshold across the pitch can undercount the smaller advances
    typical of deep build-up. Zone-scaled thresholds keep that intent while
    measuring progress strictly along x, so lateral movement cannot qualify.

    Only forward displacement on the pitch's longitudinal x-axis counts. A
    lateral move towards the centre of the goal must not become progressive
    merely because it shortens Euclidean distance to the goal centre.

    Threshold now scales with the third the action starts in:
        defensive third (x<=35)  : forward gain >= 10% of remaining pitch length
        middle third   (35<x<=70): forward gain >= 25% of remaining pitch length
        final third    (x>70)    : forward gain >= 10 m
    """
    if _isnan(start_x) or _isnan(end_x):
        return False
    start_x, end_x = float(start_x), float(end_x)
    forward_gain = end_x - start_x
    remaining_length = PITCH_LENGTH - start_x
    if forward_gain <= 0.0 or remaining_length <= 1e-9:
        return False

    if start_x <= DEFENSIVE_THIRD_MAX_X:
        return forward_gain / remaining_length >= 0.10
    elif start_x <= MIDDLE_THIRD_MAX_X:
        return forward_gain / remaining_length >= 0.25
    else:
        return forward_gain >= 10.0


def distance_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Perpendicular distance from P to the segment AB (concept 4.1).

    Used to test whether an opponent stands "on" the passing lane between the
    ball carrier and a teammate, clamped to the segment itself so a defender
    far past either endpoint does not count as blocking.
    """
    dx, dy = float(bx) - float(ax), float(by) - float(ay)
    if dx == 0.0 and dy == 0.0:
        return math.hypot(float(px) - float(ax), float(py) - float(ay))
    t = ((float(px) - float(ax)) * dx + (float(py) - float(ay)) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = float(ax) + t * dx, float(ay) + t * dy
    return math.hypot(float(px) - cx, float(py) - cy)


# --------------------------------------------------------------------------
# Phase 4 — two-axis pass shapes (concepts 1.2 and 1.3)
# --------------------------------------------------------------------------
# pass_direction/2 (phase 3) collapses a pass into forward|sideways|backward.
# This predicate applies the project's stricter length, forward-progress, and
# angle definition directly to the raw vector; see config.py for thresholds.
def is_diagonal_pass(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_length: float = DIAGONAL_MIN_LENGTH_M,
    min_forward_dx: float = DIAGONAL_MIN_FORWARD_DX_M,
    min_angle: float = DIAGONAL_MIN_ANGLE_DEG,
    max_angle: float = DIAGONAL_MAX_ANGLE_DEG,
) -> bool:
    """Concept 1.2 — a long, forward pass at a genuinely diagonal angle."""
    if any(_isnan(v) for v in (start_x, start_y, end_x, end_y)):
        return False
    dx = float(end_x) - float(start_x)
    dy = float(end_y) - float(start_y)
    if dx < min_forward_dx:
        return False
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dy), dx))
    return length >= min_length and min_angle <= angle <= max_angle


def is_switch_of_play(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_dy: float = SWITCH_MIN_DY_M,
) -> bool:
    """Concept 1.3 — a genuine switch: lane change PLUS real lateral travel.

    A lane change alone is far too common (59% of Barcelona's passes) because
    the lanes are only 13.6 m wide.
    """
    if any(_isnan(v) for v in (start_x, start_y, end_x, end_y)):
        return False
    if abs(float(end_y) - float(start_y)) < min_dy:
        return False
    return lane_of(start_y) != lane_of(end_y)


def switch_direction(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_dy: float = SWITCH_MIN_DY_M,
) -> str | None:
    """Return the destination side for a qualifying directional switch.

    A switch requires at least ``min_dy`` lateral metres and a lane change.
    The endpoint must be in one of the two half-spaces or wings; a switch
    ending in the central channel is not assigned a side.
    """
    if not is_switch_of_play(start_x, start_y, end_x, end_y, min_dy):
        return None
    destination_lane = lane_of(end_y)
    if destination_lane in {"right_half_space", "right_wing"}:
        return "right"
    if destination_lane in {"left_half_space", "left_wing"}:
        return "left"
    return None


# --------------------------------------------------------------------------
# Phase 4 — triangle geometry (concept 3.6)
# --------------------------------------------------------------------------
def triangle_area(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """Area of the triangle ABC in square metres (shoelace formula)."""
    return abs(
        (float(bx) - float(ax)) * (float(cy) - float(ay))
        - (float(cx) - float(ax)) * (float(by) - float(ay))
    ) / 2.0


def is_meaningful_triangle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    min_area: float = TRIANGLE_MIN_AREA_M2,
    max_area: float = TRIANGLE_MAX_AREA_M2,
    min_angle: float = TRIANGLE_MIN_ANGLE_DEG,
) -> bool:
    """Does ABC enclose enough space to be a tactical shape rather than a line?

    Concept 3.6a. The area bounds reject tiny/oversized shapes, while a
    minimum-angle bound independently rejects needle-like triangles.
    """
    a = (float(ax), float(ay))
    b = (float(bx), float(by))
    c = (float(cx), float(cy))
    area = triangle_area(*a, *b, *c)
    if not min_area <= area <= max_area:
        return False

    # Reject needle-shaped triangles even when a small off-axis displacement
    # happens to push their area above the minimum.
    def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    sides = (distance(b, c), distance(a, c), distance(a, b))
    if min(sides) <= 1e-9:
        return False
    angles = []
    for opposite, side1, side2 in (
        (sides[0], sides[1], sides[2]),
        (sides[1], sides[0], sides[2]),
        (sides[2], sides[0], sides[1]),
    ):
        cosine = (side1 * side1 + side2 * side2 - opposite * opposite) / (2 * side1 * side2)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return min(angles) >= min_angle


def point_in_triangle(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    edge_tolerance: float = TRIANGLE_EDGE_TOLERANCE_M,
) -> bool:
    """Is P meaningfully inside triangle ABC?  (concept 3.6b)

    Uses the sign-of-cross-product test. Points on or within ``edge_tolerance``
    metres of an edge count as outside: a defender near a passing lane is
    intercepting it, not trapped.
    """

    def cross(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    d1 = cross(ax, ay, bx, by, px, py)
    d2 = cross(bx, by, cx, cy, px, py)
    d3 = cross(cx, cy, ax, ay, px, py)
    if min(
        distance_point_to_segment(px, py, ax, ay, bx, by),
        distance_point_to_segment(px, py, bx, by, cx, cy),
        distance_point_to_segment(px, py, cx, cy, ax, ay),
    ) <= edge_tolerance:
        return False
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# --------------------------------------------------------------------------
# Phase 4 — defensive / midfield line clustering (concept 0.1)
# --------------------------------------------------------------------------
# ARCHITECTURAL NOTE (glossary v5, explicit decision):
# This is an *internal helper of the fact-extraction stage*, never a predicate
# in bias.pl.  Dependent concepts (1.1, 1.5, 7.5, 7.6b) call it while facts are
# being written and dump only their own result.  Popper must never learn rules
# over the raw x-rank approximation itself.
def defensive_line_x(
    opponent_xs: list[float],
    line_size: int = DEFENSIVE_LINE_SIZE,
    min_opponents: int = MIN_OPPONENTS_FOR_LINE,
) -> float | None:
    """Median x of the ``line_size`` deepest opponents, if plausible.

    Barcelona always attacks toward +x in the project reference frame, so an
    opponent defending Barcelona's attack faces toward decreasing x and keeps
    its deepest line closest to its own goal at the greatest visible x values.
    This is a positional proxy: 360 frames carry no player_id for non-actors,
    so we cannot verify that these really are the centre-backs.
    """
    xs = sorted(float(x) for x in opponent_xs if not _isnan(x))
    if len(xs) < min_opponents:
        return None
    take = xs[-line_size:]
    line_x = float(median(take))
    # A 360 freeze frame around a deep build-up often shows only the opponent
    # press, not the distant back four. Do not relabel those midfielders as a
    # defensive line: require the estimate to reach the opponent half.
    if line_x < PITCH_LENGTH / 2.0:
        return None
    return line_x


def midfield_line_x(
    opponent_xs: list[float],
    defensive_size: int = DEFENSIVE_LINE_SIZE,
    midfield_size: int = MIDFIELD_LINE_SIZE,
    min_opponents: int = MIN_OPPONENTS_FOR_LINE,
) -> float | None:
    """Median x of the midfield band immediately in front of the back line.

    In Barcelona's +x frame, the opponent midfield line is at a lower x than
    its defensive line.
    """
    xs = sorted(float(x) for x in opponent_xs if not _isnan(x))
    if len(xs) < min_opponents or defensive_line_x(xs, defensive_size, min_opponents) is None:
        return None
    ordered = list(reversed(xs))
    band = ordered[defensive_size : defensive_size + midfield_size]
    # The midfield line is defined from the full configured band. A partial
    # band (e.g. only two visible opponents for a 3-player line) is too
    # unstable to use as a geometric boundary.
    if len(band) < midfield_size:
        return None
    return float(median(band))


def crosses_line(start_x: float, end_x: float, line_x: float | None) -> bool:
    """Did the ball travel from behind a line to beyond it?  (concept 1.1)"""
    if line_x is None or _isnan(start_x) or _isnan(end_x):
        return False
    return float(start_x) < float(line_x) < float(end_x)


def beyond_line(end_x: float, line_x: float | None) -> bool:
    """Did the ball end up past the whole defensive line?  (concept 1.5)"""
    if line_x is None or _isnan(end_x):
        return False
    return float(end_x) > float(line_x)


def opponent_line_height(line_x: float | None) -> str | None:
    """Bucket the visible opponent back line by distance from its own goal.

    In Barcelona's reference frame the opponent goal is at ``PITCH_LENGTH``.
    A line at least one third of the pitch away from that goal is high; a line
    closer to goal is low.  This gives the glossary's high/low distinction an
    explicit, reproducible boundary.
    """
    if line_x is None or _isnan(line_x):
        return None
    boundary = PITCH_LENGTH * (2.0 / 3.0)
    return "high" if float(line_x) <= boundary else "low"


def half_of(x: float, is_barca: bool = True) -> str:
    """Return own_half/opponent_half for the acting team at an action.

    Event coordinates are expressed in the selected team's fixed reference
    frame. ``is_barca`` is a historical argument name and means selected-team
    action; opponent actions use the opposite half of that same x-axis.
    """
    x = float(x)
    midpoint = PITCH_LENGTH / 2.0
    own_half = x <= midpoint if is_barca else x >= midpoint
    return "own_half" if own_half else "opponent_half"


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
    "third_of",
    "pitch_zone",
    "under_pressure",
    "pressure_level",
    "pass_direction",
    "pass_length_class",
    "is_long_pass",
    "is_progressive",
    "is_diagonal_pass",
    "is_switch_of_play",
    "switch_direction",
    "triangle_area",
    "is_meaningful_triangle",
    "point_in_triangle",
    "defensive_line_x",
    "midfield_line_x",
    "crosses_line",
    "beyond_line",
    "opponent_line_height",
    "half_of",
    "distance_point_to_segment",
    "PRESSURE_RADIUS_M",
    "TIGHT_PRESSURE_RADIUS_M",
]
