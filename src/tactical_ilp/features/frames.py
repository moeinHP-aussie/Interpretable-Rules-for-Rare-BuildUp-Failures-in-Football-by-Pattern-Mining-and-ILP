"""
Task 4.1 — Per-player freeze-frame geometry.

Why this module exists
----------------------
Phase 1 (:mod:`tactical_ilp.data.freeze_frames`) reduces every 360 freeze frame
to pressure counts (``ff_opponents_5m`` and ``ff_opponents_3m``) and the number
of visible players. It does not retain individual player positions in that
table. Those aggregates are enough for pressure predicates but not for the
phase-4 concepts, which are about the *shape* a few players form:

* concept 0.1  — which opponents form the defensive / midfield line
* concept 3.6a — do three teammates near the ball form a triangle
* concept 3.6b — is an opponent trapped inside that triangle

So this module re-reads the raw ``frames_<match>.parquet`` files and exposes the
individual points again, in the SPADL reference frame (the selected team attacks +x).

The anonymity constraint (documented in the concept glossary, §2.3پ)
--------------------------------------------------------------------
StatsBomb 360 frames contain **no ``player_id`` for non-actor players**.  The
raw columns are exactly ``event_id, teammate, actor, keeper, x, y``.  Of the 22
players in a frame, only the actor and the keepers are identifiable; everyone
else is an anonymous point.

Every function here therefore works on *anonymous points*.  We never claim to
know that a given point is "the left centre-back" — only positional facts such
as "one of the two deepest opponents".  This is a hard data limit, not a
shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from tactical_ilp.config import PITCH_LENGTH, PITCH_WIDTH, RAW_DIR, cache_path


@dataclass(frozen=True)
class Point:
    """One player in a freeze frame — deliberately anonymous (see module doc)."""

    x: float
    y: float
    teammate: bool
    actor: bool
    keeper: bool


# --------------------------------------------------------------------------
# reference-frame alignment
# --------------------------------------------------------------------------
# The transform below was NOT guessed -- it was measured.  For every match we
# compared the actor's frame position against the same action's SPADL
# start_x/start_y and tested three candidate transforms.  Result (8 matches,
# ~17k actions, identical in every one):
#
#     selected-team actions : x unchanged, y mirrored -> median error 0.061 m
#     opponent actions  : x mirrored,  y unchanged  -> median error 0.061 m
#
# The residual 0.061 m is pure 120x80 -> 105x68 rescaling rounding, not a
# geometric mismatch (the p95 error equals the median).
#
# The intuitive guess -- "mirror the opponent frame through 180 degrees" --
# is WRONG and produces ~30 m errors.  socceraction's SPADL converter mirrors
# only ONE axis per team, so a naive 180-degree rotation double-flips y for
# opponents.  ``tests/test_phase4_frames.py`` pins this down so the mistake
# cannot silently come back.
#
# The measurement above only covered home fixtures. An earlier revision added a
# further rotation for away fixtures; it was wrong (see data/spadl_converter.py)
# and has been removed: the same transform now applies to every match.
def _to_spadl_frame(
    x: float, y: float, is_barca: bool, selected_team_home: bool | None = None
) -> tuple[float, float]:
    """Convert StatsBomb 360 pitch coordinates to the project reference frame.

    StatsBomb 360 uses a 120x80 pitch and is normalised per ACTING team.
    The project uses 105x68 with the selected team always attacking +x, in
    every match (home or away -- socceraction's converter already guarantees
    this for the SPADL actions, see ``data/spadl_converter.py``). Hence the
    transform depends only on whether the acting team is the selected one;
    ``is_barca`` is the historical column name for that flag.

    ``selected_team_home`` is accepted for backward compatibility only and is
    deliberately ignored: home/away must not change the geometry.
    """
    sx = float(x) / 120.0 * PITCH_LENGTH
    sy = float(y) / 80.0 * PITCH_WIDTH
    if is_barca:
        sy = PITCH_WIDTH - sy
    else:
        sx = PITCH_LENGTH - sx
    return sx, sy


@lru_cache(maxsize=64)
def _frames_for_match(match_id: int) -> pd.DataFrame:
    """Raw 360 rows of one match, cached (one row per visible player)."""
    path = cache_path("frames", match_id)
    if not path.exists():
        path = RAW_DIR / f"frames_{int(match_id)}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "teammate", "actor", "keeper", "x", "y"])
    return pd.read_parquet(path)


def frame_points(match_id: int, event_id: str, is_selected_team: bool) -> list[Point]:
    """All visible players of one event, in the project reference frame.

    ``is_selected_team`` is the acting team's match-specific membership flag
    from the merged SPADL table and selects the mirroring, see
    :func:`_to_spadl_frame`.

    Note that ``teammate`` is relative to the *acting* team, so for an opponent
    action ``teammate=True`` means an opponent of the acting team.
    """
    frames = _frames_for_match(match_id)
    if frames.empty or not event_id:
        return []
    rows = frames[frames.event_id == event_id]
    out: list[Point] = []
    for r in rows.itertuples():
        sx, sy = _to_spadl_frame(r.x, r.y, is_selected_team)
        out.append(
            Point(
                x=sx,
                y=sy,
                teammate=bool(r.teammate),
                actor=bool(r.actor),
                keeper=bool(r.keeper),
            )
        )
    return out


# --------------------------------------------------------------------------
# helpers used by the tactical concepts
# --------------------------------------------------------------------------
def teammates(points: list[Point], include_actor: bool = False) -> list[Point]:
    """Points belonging to the acting team."""
    return [p for p in points if p.teammate and (include_actor or not p.actor)]


def opponents(points: list[Point], include_keeper: bool = True) -> list[Point]:
    """Points belonging to the defending team."""
    return [p for p in points if not p.teammate and (include_keeper or not p.keeper)]


def actor_point(points: list[Point]) -> Point | None:
    """The ball carrier, the only outfield player we can actually name."""
    for p in points:
        if p.actor:
            return p
    return None


def near(points: list[Point], origin: Point, radius: float) -> list[Point]:
    """Points within ``radius`` metres of ``origin``."""
    r2 = radius * radius
    return [
        p
        for p in points
        if (p.x - origin.x) ** 2 + (p.y - origin.y) ** 2 <= r2 and p is not origin
    ]
