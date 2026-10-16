"""
Phase 4 — tactical concept extraction.

Turns the concept glossary (``docs/tactical_concepts_glossary.md``) into
computable functions.  Everything here produces **ground facts**; the genuine
Prolog *rules* live in ``bk.pl`` and are written by hand.

What is implemented, and what deliberately is not, is documented in
``docs/phase4_concept_disposition.md``.  The short version: a concept is
implemented when the 360 frame plus SPADL can answer it without inventing
data.  The hard limit throughout is that **non-actor players in a 360 frame
have no player_id** -- only anonymous ``(x, y, teammate, keeper)`` points --
so every concept about "which opponent" is positional, never identity-based.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pandas as pd

from tactical_ilp.config import (
    FREQUENT_INVOLVEMENT_MIN,
    HESITATION_THRESHOLD_S,
    ONE_TOUCH_MAX_SECONDS,
    SUPPORT_CORRIDOR_M,
    TEMPO_SLOW_MAX,
    TRIANGLE_RADIUS_M,
)
from tactical_ilp.features.frames import (
    Point,
    actor_point,
    frame_points,
    near,
    opponents,
    teammates,
)
from tactical_ilp.features.geometry import (
    beyond_line,
    crosses_line,
    defensive_line_x,
    distance_point_to_segment,
    is_meaningful_triangle,
    lane_of,
    opponent_line_height,
    point_in_triangle,
    pressure_level,
    third_of,
    triangle_area,
)


# ==========================================================================
# concept 4.1 / 4.5 — support (open passing lane), not distance
# ==========================================================================
def open_supporters(
    points: list[Point], corridor: float = SUPPORT_CORRIDOR_M
) -> int:
    """Concept 4.1/4.5 — how many teammates have an unobstructed passing lane.

    User's corrected definition: a supporter is defined by an OPEN passing
    lane to the ball carrier, not by distance.  A teammate only counts if no
    opponent stands within ``corridor`` metres of the straight line between
    them and the carrier.  This replaces the earlier (wrong) definition that
    derived "support" from passes_to/3, which only reflected who happened to
    receive the ball, not real geometric support.
    """
    carrier = actor_point(points)
    if carrier is None:
        return 0
    mates = teammates(points)
    opps = opponents(points)
    count = 0
    for m in mates:
        blocked = any(
            distance_point_to_segment(o.x, o.y, carrier.x, carrier.y, m.x, m.y)
            < corridor
            for o in opps
        )
        if not blocked:
            count += 1
    return count


# ==========================================================================
# concept 3.6 — triangulation around the ball carrier
# ==========================================================================
@dataclass(frozen=True)
class TriangleObservation:
    """One triangle found around a ball carrier at one action."""

    area: float
    traps_opponent: bool
    n_trapped: int


def triangles_at_action(
    points: list[Point], radius: float = TRIANGLE_RADIUS_M
) -> list[TriangleObservation]:
    """Concept 3.6a/3.6b — triangles the carrier forms with nearby teammates.

    The carrier is always one vertex: the glossary asks specifically for the
    small shapes *around the ball*, not for formation-wide geometry.  The
    radius restriction is what keeps this affordable (see config).

    3.6a (the event) and 3.6b (the consequence) stay separable: every triangle
    is reported with its ``traps_opponent`` flag rather than only the trapping
    ones, so Popper can learn the relation between them instead of having it
    baked into the definition.
    """
    carrier = actor_point(points)
    if carrier is None:
        return []
    mates = near(teammates(points), carrier, radius)
    if len(mates) < 2:
        return []

    opps = opponents(points)
    out: list[TriangleObservation] = []
    for p, q in itertools.combinations(mates, 2):
        if not is_meaningful_triangle(carrier.x, carrier.y, p.x, p.y, q.x, q.y):
            continue
        trapped = sum(
            1
            for o in opps
            if point_in_triangle(o.x, o.y, carrier.x, carrier.y, p.x, p.y, q.x, q.y)
        )
        out.append(
            TriangleObservation(
                area=triangle_area(carrier.x, carrier.y, p.x, p.y, q.x, q.y),
                traps_opponent=trapped > 0,
                n_trapped=trapped,
            )
        )
    return out


def triangulation_summary(points: list[Point]) -> dict:
    """Per-action summary of concept 3.6, ready to be dumped as facts."""
    tris = triangles_at_action(points)
    trapping = [t for t in tris if t.traps_opponent]
    return {
        "n_triangles": len(tris),
        "n_trapping": len(trapping),
        "max_trapped": max((t.n_trapped for t in tris), default=0),
        # "several overlapping triangles" from the glossary: a single triangle
        # is often accidental, two or more trapping ones are a real cage.
        "multi_trap": len(trapping) >= 2,
    }


# ==========================================================================
# concept 0.1 / 1.1 / 1.5 — opponent lines (positional proxy)
# ==========================================================================
def line_summary(points: list[Point], start_x: float, end_x: float) -> dict:
    """Concepts 1.1 and 1.5, both built on the 0.1 clustering helper.

    0.1 itself is *not* exported as a fact -- see the architectural note in
    geometry.py.  Only the tactical conclusions are.
    """
    opps = opponents(points, include_keeper=False)
    xs = [o.x for o in opps]
    dline = defensive_line_x(xs)
    return {
        "has_line": dline is not None,
        "breaks_defensive_line": crosses_line(start_x, end_x, dline),
        "beyond_defensive_line": beyond_line(end_x, dline),
        "defensive_line_x": dline,
        "line_height": opponent_line_height(dline),
    }


def rest_defense(points: list[Point]) -> str | None:
    """Count outfield teammates at least 5 m behind the ball carrier."""
    carrier = actor_point(points)
    if carrier is None:
        return None
    behind = sum(
        1 for mate in teammates(points)
        if not mate.keeper and carrier.x - mate.x >= 5.0
    )
    return "protected" if behind >= 2 else "exposed"


def overload_zones(points: list[Point], advantage: int = 1) -> list[str]:
    """Zones where visible teammates outnumber visible opponents by `advantage`."""
    counts: dict[tuple[str, str], list[int]] = {}
    for p in points:
        key = (third_of(p.x), lane_of(p.y))
        sides = counts.setdefault(key, [0, 0])
        sides[0 if p.teammate else 1] += 1
    return [
        f"{third}_{lane}"
        for (third, lane), (mates, opps) in counts.items()
        if mates - opps >= advantage and mates >= 2
    ]


def visible_team_shape(points: list[Point]) -> tuple[str, str] | None:
    """Bucket visible teammate width and depth; not a full-team formation."""
    own = [p for p in points if p.teammate]
    if len(own) < 4:
        return None
    width = max(p.y for p in own) - min(p.y for p in own)
    depth = max(p.x for p in own) - min(p.x for p in own)
    width_bucket = "narrow" if width < 25.0 else "balanced" if width < 45.0 else "wide"
    depth_bucket = "compact" if depth < 20.0 else "balanced" if depth < 35.0 else "stretched"
    return width_bucket, depth_bucket


def second_defender_cover(points: list[Point], radius: float = 15.0) -> bool:
    """Whether a second visible opponent covers goal-side of the nearest one."""
    carrier = actor_point(points)
    if carrier is None:
        return False
    ordered = sorted(
        opponents(points, include_keeper=False),
        key=lambda p: math.hypot(p.x - carrier.x, p.y - carrier.y),
    )
    if len(ordered) < 2:
        return False
    first, second = ordered[:2]
    second_dist = math.hypot(second.x - carrier.x, second.y - carrier.y)
    return second_dist <= radius and second.x >= first.x


def defensive_line_alignment(points: list[Point], max_spread: float = 4.0) -> str | None:
    """Alignment of the four deepest visible opponents, in metres of x-spread."""
    xs = sorted((p.x for p in opponents(points, include_keeper=False)), reverse=True)
    if len(xs) < 4:
        return None
    spread = xs[0] - xs[3]
    return "aligned" if spread <= max_spread else "stretched"


def structural_spacing(points: list[Point]) -> list[tuple[str, str, str]]:
    """Bucket local visible-player footprint by side and pitch dimension.

    This is an independent positional proxy for structural spacing, not a
    role-to-role distance: freeze frames do not identify non-actor players.
    Each side needs at least three visible non-keeper points.
    """
    result: list[tuple[str, str, str]] = []
    for side, players in (
        ("barca", [p for p in points if p.teammate and not p.keeper]),
        ("opponent", [p for p in points if not p.teammate and not p.keeper]),
    ):
        if len(players) < 3:
            continue
        length_span = max(p.x for p in players) - min(p.x for p in players)
        width_span = max(p.y for p in players) - min(p.y for p in players)
        length_bucket = "compact" if length_span < 12.0 else "balanced" if length_span < 24.0 else "stretched"
        width_bucket = "narrow" if width_span < 15.0 else "balanced" if width_span < 30.0 else "wide"
        result.extend(
            ((side, "longitudinal", length_bucket), (side, "lateral", width_bucket))
        )
    return result


# ==========================================================================
# concept 6 — rhythm, timing, decision making
# ==========================================================================
def tempo_bucket(
    n_actions: int, duration_s: float, fast_min: float | None = 1.07
) -> str | None:
    """Concept 6.2 — sequence tempo, discretised.

    The production extractor passes its data-calibrated P70 threshold.
    ``1.07`` remains the direct-call fallback for utility compatibility.
    Kept as a bucket rather than a raw float so Popper gets a finite vocabulary.
    """
    if duration_s is None or duration_s <= 0 or n_actions is None:
        return None
    rate = float(n_actions) / float(duration_s)
    if rate <= TEMPO_SLOW_MAX:
        return "slow"
    if fast_min is not None and rate >= fast_min:
        return "fast"
    return "medium"


def is_hesitation(dwell_s: float | None, pressure: str) -> bool:
    """Concept 6.1 — did the player dwell unusually long on the ball?

    Compared against the mean+1sd of *its own pressure level*, not a global
    mean.  Measured on the real data the levels differ a lot (1.05 s under
    heavy pressure vs 1.70 s unpressured), so a global threshold would make
    hesitation-under-pressure almost undetectable.

    Note this is the *event*, deliberately separate from its consequence, so
    that Popper can discover "hesitation leads to X" rather than being told.
    """
    if dwell_s is None:
        return False
    try:
        d = float(dwell_s)
    except (TypeError, ValueError):
        return False
    if d != d or d < 0:  # NaN guard
        return False
    threshold = HESITATION_THRESHOLD_S.get(pressure, HESITATION_THRESHOLD_S["unknown"])
    return d > threshold


def is_one_touch(dwell_s: float | None) -> bool:
    """Approximate first-time play using the configured event-time threshold."""
    if dwell_s is None:
        return False
    try:
        d = float(dwell_s)
    except (TypeError, ValueError):
        return False
    return 0 <= d <= ONE_TOUCH_MAX_SECONDS


def frequently_involved(touch_count: int) -> bool:
    """Concept 3.4 — repeated involvement, as a boolean rather than a count."""
    return int(touch_count) >= FREQUENT_INVOLVEMENT_MIN


__all__ = [
    "TriangleObservation",
    "triangles_at_action",
    "triangulation_summary",
    "open_supporters",
    "line_summary",
    "rest_defense",
    "overload_zones",
    "visible_team_shape",
    "second_defender_cover",
    "defensive_line_alignment",
    "structural_spacing",
    "tempo_bucket",
    "is_hesitation",
    "is_one_touch",
    "frequently_involved",
]
