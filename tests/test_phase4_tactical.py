"""
Phase 4 tests — tactical concepts, frame geometry and the Prolog rules.

Several of these are *regression tests for real bugs found while building
phase 4*.  They are marked with a BUG comment so the mistake cannot silently
come back.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from tactical_ilp.config import (
    DIAGONAL_MIN_FORWARD_DX_M,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PROLOG_DIR,
    SWITCH_MIN_DY_M,
    TRIANGLE_MAX_AREA_M2,
    TRIANGLE_MIN_AREA_M2,
    TRIANGLE_RADIUS_M,
)
from tactical_ilp.features.frames import Point, _to_spadl_frame, near, opponents, teammates
from tactical_ilp.features.geometry import (
    beyond_line,
    crosses_line,
    defensive_line_x,
    distance_point_to_segment,
    is_diagonal_pass,
    is_meaningful_triangle,
    is_switch_of_play,
    switch_direction,
    is_long_pass,
    midfield_line_x,
    half_of,
    opponent_line_height,
    point_in_triangle,
    triangle_area,
    flank_dribble_features,
)
from tactical_ilp.features.tactical import (
    defensive_line_alignment,
    frequently_involved,
    is_hesitation,
    is_one_touch,
    line_summary,
    overload_zones,
    open_supporters,
    rest_defense,
    second_defender_cover,
    tempo_bucket,
    triangles_at_action,
    triangulation_summary,
    visible_team_shape,
)
from tactical_ilp.prolog.tactical_facts import _dwell_bucket

FACTS_ALL = PROLOG_DIR / "facts_all.pl"
FACTS_TACTICAL = PROLOG_DIR / "facts_tactical.pl"
BK = PROLOG_DIR / "bk.pl"


def test_flank_dribble_lane_transition_features():
    assert flank_dribble_features("dribble", 20, 4, 26, 5) == {
        "flank_dribble", "flank_line_carry"
    }
    assert flank_dribble_features("dribble", 20, 4, 24, 20) == {
        "flank_dribble", "flank_cut_inside"
    }
    assert flank_dribble_features("dribble", 20, 4, 22, 6) == {"flank_dribble"}
    assert flank_dribble_features("pass", 20, 4, 30, 5) == set()
    assert flank_dribble_features("dribble", 20, 30, 26, 35) == set()
    assert flank_dribble_features("dribble", None, 4, None, None) == {
        "flank_dribble"
    }


def test_long_pass_uses_euclidean_threshold_of_thirty_metres():
    assert is_long_pass(10, 10, 40, 10)
    assert is_long_pass(10, 10, 10, 40)
    assert not is_long_pass(10, 10, 39.999, 10)
    assert not is_long_pass(10, 10, float("nan"), 10)


# ==========================================================================
# 4.1 — frame reference alignment
# ==========================================================================
def test_barca_frames_mirror_only_y():
    """BUG REGRESSION: the intuitive '180 degree rotation for the opponent'
    assumption is wrong and produced ~30 m position errors.  Measured on 8
    matches: Barcelona frames mirror y only, opponent frames mirror x only."""
    # a point at the top-left of the 120x80 frame
    x, y = 0.0, 0.0
    bx, by = _to_spadl_frame(x, y, is_barca=True)
    assert bx == pytest.approx(0.0)
    assert by == pytest.approx(PITCH_WIDTH)


def test_opponent_frames_mirror_only_x():
    x, y = 0.0, 0.0
    ox, oy = _to_spadl_frame(x, y, is_barca=False)
    assert ox == pytest.approx(PITCH_LENGTH)
    assert oy == pytest.approx(0.0)


def test_frame_transform_is_not_a_180_rotation():
    """The wrong transform would flip BOTH axes for one of the teams."""
    bx, by = _to_spadl_frame(30.0, 20.0, is_barca=True)
    ox, oy = _to_spadl_frame(30.0, 20.0, is_barca=False)
    # if either were a full 180 rotation, both coordinates would be mirrored
    assert not (bx != pytest.approx(30.0 / 120 * PITCH_LENGTH) and by != pytest.approx(20.0 / 80 * PITCH_WIDTH) is False)
    assert bx == pytest.approx(30.0 / 120 * PITCH_LENGTH)     # x untouched
    assert oy == pytest.approx(20.0 / 80 * PITCH_WIDTH)       # y untouched


def test_transform_stays_on_the_pitch():
    for x in (0.0, 60.0, 120.0):
        for y in (0.0, 40.0, 80.0):
            for barca in (True, False):
                sx, sy = _to_spadl_frame(x, y, barca)
                assert -0.01 <= sx <= PITCH_LENGTH + 0.01
                assert -0.01 <= sy <= PITCH_WIDTH + 0.01


# ==========================================================================
# triangle geometry (concept 3.6)
# ==========================================================================
def test_triangle_area_right_triangle():
    assert triangle_area(0, 0, 3, 0, 0, 4) == pytest.approx(6.0)


def test_collinear_points_have_zero_area():
    assert triangle_area(0, 0, 5, 5, 10, 10) == pytest.approx(0.0)


def test_degenerate_triangle_is_rejected():
    """Three players almost in a line form no passing geometry."""
    assert not is_meaningful_triangle(0, 0, 5, 0.1, 10, 0.2)


def test_reasonable_triangle_is_accepted():
    area = triangle_area(0, 0, 8, 0, 4, 6)
    assert TRIANGLE_MIN_AREA_M2 <= area <= TRIANGLE_MAX_AREA_M2
    assert is_meaningful_triangle(0, 0, 8, 0, 4, 6)


def test_point_inside_triangle_is_detected():
    assert point_in_triangle(3, 2, 0, 0, 10, 0, 5, 8)


def test_point_outside_triangle_is_rejected():
    assert not point_in_triangle(20, 20, 0, 0, 10, 0, 5, 8)


def test_point_on_edge_counts_as_outside():
    """A defender standing on the passing line is intercepting, not trapped."""
    assert not point_in_triangle(5, 0, 0, 0, 10, 0, 5, 8)


# ==========================================================================
# line clustering (concept 0.1) and its consumers (1.1, 1.5)
# ==========================================================================
def test_defensive_line_is_the_deepest_band():
    # Barcelona attacks toward +x; the opponent's deepest/back line is
    # therefore the band nearest its goal at the greatest visible x values.
    xs = [50, 51, 52, 65, 66, 67, 80, 81, 82, 83]
    assert defensive_line_x(xs) == pytest.approx(81.5)


def test_midfield_line_sits_ahead_of_the_defensive_line():
    xs = [50, 51, 52, 65, 66, 67, 80, 81, 82, 83]
    d = defensive_line_x(xs)
    m = midfield_line_x(xs)
    assert m < d


def test_opponent_line_height_is_relative_to_its_own_goal():
    assert opponent_line_height(70.0) == "high"
    assert opponent_line_height(80.0) == "low"
    assert opponent_line_height(None) is None


def test_structural_frame_proxies_use_visible_points():
    points = [
        Point(20, 30, True, True, False),
        Point(10, 28, True, False, False),
        Point(5, 32, True, False, False),
        Point(24, 29, True, False, False),
        Point(21, 31, False, False, False),
        Point(55, 30, False, False, False),
    ]
    assert rest_defense(points) == "protected"
    assert "defensive_third_central_channel" in overload_zones(points)
    assert visible_team_shape(points) is not None


def test_second_defender_and_defensive_line_alignment():
    points = [
        Point(50, 30, True, True, False),
        Point(49, 30, False, False, False),
        Point(55, 31, False, False, False),
        Point(70, 20, False, False, False),
        Point(71, 40, False, False, False),
        Point(72, 25, False, False, False),
        Point(73, 35, False, False, False),
    ]
    assert second_defender_cover(points)
    assert defensive_line_alignment(points) == "aligned"


def test_line_needs_enough_visible_opponents():
    assert defensive_line_x([10, 11, 12]) is None


def test_crosses_line_requires_going_from_behind_to_beyond():
    assert crosses_line(20.0, 40.0, 30.0)
    assert not crosses_line(40.0, 50.0, 30.0)     # started already beyond
    assert not crosses_line(10.0, 20.0, 30.0)     # never reached it


def test_beyond_line_is_about_the_end_point_only():
    assert beyond_line(40.0, 30.0)
    assert not beyond_line(20.0, 30.0)


def test_none_line_never_matches():
    assert not crosses_line(20.0, 40.0, None)
    assert not beyond_line(40.0, None)


# ==========================================================================
# two-axis pass shapes (concepts 1.2, 1.3)
# ==========================================================================
def test_diagonal_needs_both_axes():
    assert is_diagonal_pass(0, 0, 15, 15)
    assert not is_diagonal_pass(0, 0, 15, 1)      # forward only
    assert not is_diagonal_pass(0, 0, 1, 15)      # sideways only


def test_diagonal_threshold_is_the_configured_one():
    d = DIAGONAL_MIN_FORWARD_DX_M
    # Keep total length above 12 m so this isolates the forward-dx boundary.
    assert is_diagonal_pass(0, 0, d, 11.0)
    assert not is_diagonal_pass(0, 0, d - 0.1, 11.0)


def test_switch_needs_lane_change_and_distance():
    """BUG REGRESSION: a lane change alone matches 59% of passes, which is not
    a switch of play."""
    assert is_switch_of_play(50, 5, 55, 5 + SWITCH_MIN_DY_M + 1)
    assert not is_switch_of_play(50, 5, 55, 12)   # lane change but too short
    assert switch_direction(50, 5, 55, 50) == "left"
    assert switch_direction(50, 60, 55, 25) == "right"
    assert switch_direction(50, 5, 55, 20) is None  # no qualifying switch
    assert switch_direction(50, 5, 55, 34) is None  # ends centrally


# ==========================================================================
# rhythm / timing (concept 6)
# ==========================================================================
def test_tempo_buckets_are_ordered():
    assert tempo_bucket(10, 20.0) == "slow"       # 0.5 /s
    assert tempo_bucket(10, 10.0) == "medium"     # 1.0 /s
    assert tempo_bucket(10, 5.0) == "fast"        # 2.0 /s


def test_tempo_rejects_degenerate_durations():
    assert tempo_bucket(10, 0.0) is None
    assert tempo_bucket(10, None) is None


def test_hesitation_threshold_depends_on_pressure():
    """The whole point of concept 6.1: 2.5 s is hesitation under heavy
    pressure but normal when unpressured."""
    assert is_hesitation(2.5, "heavy_pressure")
    assert not is_hesitation(2.5, "no_pressure")


def test_hesitation_handles_missing_values():
    assert not is_hesitation(None, "no_pressure")
    assert not is_hesitation(float("nan"), "no_pressure")
    assert not is_hesitation(-1.0, "no_pressure")


def test_one_touch_is_a_short_dwell():
    assert is_one_touch(0.4)
    assert not is_one_touch(3.0)


def test_instant_dwell_bucket_is_broader_than_one_touch():
    assert _dwell_bucket(0.75) == "instant"
    assert not is_one_touch(0.75)


def test_frequent_involvement_is_boolean():
    assert frequently_involved(3)
    assert not frequently_involved(2)


# ==========================================================================
# triangulation on synthetic frames
# ==========================================================================
def _frame(carrier, mates, opps):
    pts = [Point(carrier[0], carrier[1], True, True, False)]
    pts += [Point(x, y, True, False, False) for x, y in mates]
    pts += [Point(x, y, False, False, False) for x, y in opps]
    return pts


def test_triangulation_finds_a_triangle():
    pts = _frame((50, 30), [(56, 32), (53, 37)], [])
    tris = triangles_at_action(pts)
    assert len(tris) == 1
    assert not tris[0].traps_opponent


def test_triangulation_detects_a_trapped_opponent():
    # all teammates kept inside TRIANGLE_RADIUS_M of the carrier at (50, 30)
    pts = _frame((50, 30), [(59, 30), (52, 38)], [(54, 32)])
    tris = triangles_at_action(pts)
    assert len(tris) == 1
    assert tris[0].traps_opponent


def test_triangulation_needs_two_teammates():
    pts = _frame((50, 30), [(56, 32)], [])
    assert triangles_at_action(pts) == []


def test_radius_restriction_actually_restricts():
    """The radius is what keeps concept 3.6 affordable -- teammates far away
    must not create triangles."""
    far = TRIANGLE_RADIUS_M * 3
    pts = _frame((50, 30), [(50 + far, 30), (50, 30 + far)], [])
    assert triangles_at_action(pts) == []


def test_summary_reports_multi_trap():
    # three teammates all within the radius -> C(3,2) = 3 candidate triangles
    pts = _frame((50, 30), [(58, 30), (50, 38), (57, 36)], [(52, 34)])
    summary = triangulation_summary(pts)
    assert summary["n_triangles"] >= 2
    assert summary["multi_trap"] is True


def test_line_summary_uses_opponents_only():
    pts = _frame(
        (72, 30),
        [(25, 30), (30, 35)],
        [(50, 20), (51, 30), (52, 40), (65, 20), (66, 30), (67, 40), (80, 20), (81, 30), (82, 40), (83, 50)],
    )
    out = line_summary(pts, start_x=70.0, end_x=85.0)
    assert out["has_line"]
    assert out["breaks_defensive_line"]
    assert out["defensive_line_x"] == pytest.approx(81.5)


def test_half_labels_are_relative_to_the_acting_team():
    assert half_of(40, is_barca=True) == "own_half"
    assert half_of(60, is_barca=True) == "opponent_half"
    assert half_of(60, is_barca=False) == "own_half"
    assert half_of(40, is_barca=False) == "opponent_half"


# ==========================================================================
# concept 4.1 / 4.5 — support via open passing lane
# ==========================================================================
def test_distance_point_to_segment_on_the_line():
    assert distance_point_to_segment(5, 0, 0, 0, 10, 0) == pytest.approx(0.0)


def test_distance_point_to_segment_clamped_to_endpoints():
    """A defender far past either endpoint should not count as blocking."""
    d = distance_point_to_segment(20, 0, 0, 0, 10, 0)
    assert d == pytest.approx(10.0)


def test_open_supporter_counted_when_no_opponent_blocks():
    pts = _frame((50, 30), [(60, 30)], [])
    assert open_supporters(pts) == 1


def test_supporter_not_counted_when_opponent_blocks_the_lane():
    """BUG REGRESSION target: support must come from an open lane, not from
    passes_to/3 or raw distance -- a nearby teammate whose lane is blocked is
    not a supporter."""
    pts = _frame((50, 30), [(60, 30)], [(55, 30)])
    assert open_supporters(pts) == 0


def test_supporter_far_from_the_lane_still_counts():
    pts = _frame((50, 30), [(60, 30)], [(55, 45)])
    assert open_supporters(pts) == 1


def test_open_supporters_needs_a_carrier():
    pts = [Point(60, 30, True, False, False)]
    assert open_supporters(pts) == 0


# ==========================================================================
# generated artefacts and the Prolog rules
# ==========================================================================
def _swipl_available() -> bool:
    try:
        return subprocess.run(["swipl", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _count(goal: str) -> int:
    script = (
        f"consult('{FACTS_ALL.as_posix()}'), consult('{FACTS_TACTICAL.as_posix()}'), "
        f"consult('{BK.as_posix()}'), "
        f"aggregate_all(count, {goal}, N), write(N), nl, halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return int(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not FACTS_TACTICAL.exists(), reason="phase 4 facts not generated")
def test_tactical_facts_file_exists_and_is_nonempty():
    text = FACTS_TACTICAL.read_text(encoding="utf-8")
    assert "hesitation(" in text
    assert "forms_triangle(" in text
    assert len(text) > 10_000


@pytest.mark.skipif(not FACTS_TACTICAL.exists(), reason="phase 4 facts not generated")
def test_tactical_facts_carry_no_raw_floats():
    """Popper cannot generalise over thousands of distinct float constants, so
    every numeric concept must arrive discretised."""
    for line in FACTS_TACTICAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        assert "." not in line[:-1].split("(", 1)[-1] or not any(
            ch.isdigit() for ch in line.split("(", 1)[-1].split(",")[-1].rstrip(").")
        ) or line.split("(", 1)[-1].rstrip(").").split(",")[-1].strip().isdigit()


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog files not generated")
def test_bk_loads_without_errors():
    script = (
        f"consult('{FACTS_ALL.as_posix()}'), consult('{FACTS_TACTICAL.as_posix()}'), "
        f"consult('{BK.as_posix()}'), "
        "write(ok), nl, halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog files not generated")
@pytest.mark.parametrize(
    "goal",
    [
        "line_breaking_pass(_)",
        "diagonal_pass(_)",
        "line_pass(_)",
        "switch_to_right(_)",
        "switch_to_left(_)",
        "aerial_pass(_)",
        "long_pass(_)",
        "pass_in_behind(_)",
        "connecting_pass(_)",
        "dropped_between_lines(_)",
        "give_and_continue(_,_) ",
        "layoff(_)",
        "triangulation(_)",
        "triangle_trap(_)",
        "rondo_cage(_)",
        "pressured_cage(_)",
        "supported_action(_)",
        "dead_end_support(_)",
        "opponent_visible_majority(_)",
        "persistent_half_space_occupier(_,_)",
        "fast_sequence(_)",
        "slow_sequence(_)",
        "first_time_play(_)",
        "quick_line_break(_)",
        "dwelt_long(_)",
        "counter_pressed(_)",
        "press_resistance(_)",
        "keeper_build_up_start(_)",
        "keeper_played_open_play(_)",
    ],
)
def test_every_rule_has_solutions(goal):
    """Every rule in the active phase-5 vocabulary must have solutions."""
    assert _count(goal) > 0


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog facts not generated")
def test_offside_event_is_only_an_observed_result():
    assert _count("offside_event(_)") == _count("action_result(_, offside)")


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog facts not generated")
def test_keeper_build_up_start_excludes_claims_and_long_goal_kicks():
    assert _count("(keeper_build_up_start(A), action_type(A, keeper_claim))") == 0
    assert _count("(keeper_build_up_start(A), pass_length(A, long))") == 0


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog facts not generated")
def test_corner_is_known_to_be_outside_this_build_up_sample():
    assert _count("corner_action(_)") == 0


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog facts not generated")
def test_trapping_is_a_subset_of_triangulation():
    """3.6b cannot happen without 3.6a -- a sanity check on the separation."""
    assert _count("triangle_trap(_)") <= _count("triangulation(_)")


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not on PATH")
@pytest.mark.skipif(not (FACTS_ALL.exists() and FACTS_TACTICAL.exists() and BK.exists()), reason="phase 3/4 Prolog facts not generated")
def test_cage_is_a_subset_of_trapping():
    assert _count("rondo_cage(_)") <= _count("triangle_trap(_)")
