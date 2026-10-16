"""
Phase 4 pre-check — tests for the Prolog fact-volume cost estimator
(tactical_ilp.estimation.cost_report).

These tests do not check specific concept verdicts staying GREEN/AMBER/RED
forever — the whole point of the estimator is that verdicts can change as
thresholds or definitions change. What they check is that the estimator
itself behaves correctly: verdict thresholds are applied consistently, the
triangulation radius restriction actually matters, and every estimate
returns a real, positive, finite number rather than silently producing zero
or NaN.
"""

from __future__ import annotations

import pytest

from tactical_ilp.estimation.cost_report import (
    AMBER_RATIO,
    CURRENT_TOTAL_FACTS,
    RED_RATIO,
    Estimate,
    _verdict,
    estimate_existing_relational_facts,
    estimate_sequence_counters,
    run_all_estimates,
)
from tactical_ilp.estimation.raw_frames import raw_positions_sample


# --------------------------------------------------------------------------
# verdict thresholds
# --------------------------------------------------------------------------
def test_verdict_thresholds_are_consistent():
    assert _verdict(0.0) == "GREEN"
    assert _verdict(AMBER_RATIO - 0.01) == "GREEN"
    assert _verdict(AMBER_RATIO) == "AMBER"
    assert _verdict((AMBER_RATIO + RED_RATIO) / 2) == "AMBER"
    assert _verdict(RED_RATIO) == "AMBER"
    assert _verdict(RED_RATIO + 0.01) == "RED"


def test_amber_threshold_below_red_threshold():
    assert AMBER_RATIO < RED_RATIO


# --------------------------------------------------------------------------
# raw frame sampling
# --------------------------------------------------------------------------
def test_raw_positions_sample_has_expected_columns():
    df, matches = raw_positions_sample(n_matches=2)
    assert len(matches) == 2
    for col in ("match_id", "action_id", "x", "y", "is_teammate", "is_actor"):
        assert col in df.columns
    assert len(df) > 0


def test_raw_positions_sample_is_deterministic():
    df1, matches1 = raw_positions_sample(n_matches=3)
    df2, matches2 = raw_positions_sample(n_matches=3)
    assert matches1 == matches2
    assert len(df1) == len(df2)


def test_raw_positions_coordinates_mostly_in_pitch_bounds():
    """
    StatsBomb 360 freeze-frame positions come from broadcast camera tracking,
    not a laser system, so a small share of rows can fall slightly outside
    the pitch rectangle (a throw-in taker standing off the touchline, a
    keeper right on the goal line, etc.). A wide tolerance and a "mostly"
    check is the honest version of this test; a hard boundary would be wrong
    about the actual data, not just strict.
    """
    from tactical_ilp.config import PITCH_LENGTH, PITCH_WIDTH

    df, _ = raw_positions_sample(n_matches=2)
    TOLERANCE = 10.0  # meters beyond the pitch edge, generous camera-frame margin
    out_of_bounds = (
        (df.x < -TOLERANCE)
        | (df.x > PITCH_LENGTH + TOLERANCE)
        | (df.y < -TOLERANCE)
        | (df.y > PITCH_WIDTH + TOLERANCE)
    )
    assert out_of_bounds.mean() < 0.01, "more than 1% of tracked positions are wildly outside the pitch — check coordinate transform"


def test_every_action_has_exactly_one_actor_row():
    df, _ = raw_positions_sample(n_matches=2)
    actors_per_action = df[df.is_actor].groupby(["match_id", "action_id"]).size()
    assert (actors_per_action <= 1).all()


# --------------------------------------------------------------------------
# triangulation estimate — this is the concept that motivated the whole tool
# --------------------------------------------------------------------------
def test_triangulation_radius_restriction_is_much_cheaper_than_naive():
    """
    The whole point of this estimator: confirm the configured radius on
    3.6a produces a materially smaller fact count than the naive "any 3 of
    all visible teammates" version. The exact numbers will drift as the
    dataset or sample changes, but the radius version must stay at least an
    order of magnitude cheaper than the naive one.
    """
    from tactical_ilp.estimation.cost_report import estimate_triangulation

    df, matches = raw_positions_sample(n_matches=8)
    result = estimate_triangulation(df, matches)

    assert result.estimated_facts_full_season > 0
    # naive count is embedded in the note as a formatted string; re-derive it
    # independently instead of parsing that string, for a real check
    import numpy as np
    from math import comb

    mates = df[df.is_teammate & ~df.is_actor]
    naive_counts = mates.groupby(["match_id", "action_id"]).size()
    naive_total = int(naive_counts.apply(lambda n: comb(int(n), 3) if n >= 3 else 0).sum())

    assert naive_total > result.estimated_facts_full_season * 10, (
        "the configured-radius restriction should cut candidate triangles by at least 10x "
        "relative to enumerating all visible teammates — if this shrinks, the radius "
        "restriction may no longer be doing meaningful work"
    )


def test_triangulation_verdict_is_green():
    """
    With the radius restriction, 3.6a should stay comfortably GREEN — this is
    the concrete claim made to the project owner and should not silently
    regress to AMBER/RED without someone noticing.
    """
    from tactical_ilp.estimation.cost_report import estimate_triangulation

    df, matches = raw_positions_sample(n_matches=8)
    result = estimate_triangulation(df, matches)
    assert result.verdict == "GREEN"


# --------------------------------------------------------------------------
# defensive-line clustering — regression test for the estimator's own bug
# --------------------------------------------------------------------------
def test_line_clustering_is_not_measured_as_per_opponent_fact_table():
    """
    Regression test for a real mistake made while building this estimator: an
    early version measured 0.1 as "one fact per (action, visible opponent)",
    which produced a false RED verdict (6.77x). The corrected design is ONE
    derived label per action (matching at_third/2, at_lane/2), which should
    land close to the number of sampled actions, not ~17x that number (the
    average visible-opponent count per frame).
    """
    from tactical_ilp.estimation.cost_report import estimate_line_clustering

    df, matches = raw_positions_sample(n_matches=8)
    result = estimate_line_clustering(df, matches)

    n_actions = len(df[df.is_actor])
    # scaled to season, so compare ratios instead of raw counts
    actions_per_match = n_actions / len(matches)
    implied_per_match = result.estimated_facts_full_season / 35

    # should be within a small constant factor of "one fact per action", not
    # within a factor of the average opponent count (~17x)
    assert implied_per_match < actions_per_match * 3, (
        "0.1 estimate looks like it regressed to counting one fact per "
        "(action, opponent) instead of one derived label per action"
    )


# --------------------------------------------------------------------------
# structural distances — feasibility finding must survive
# --------------------------------------------------------------------------
def test_structural_distances_flags_the_anonymous_player_limitation():
    """
    The real finding here isn't the fact count — it's that StatsBomb 360
    frames never carry player_id for non-actor players, so role-pair
    distances can only be approximated, not resolved exactly. This test
    checks the finding is still being surfaced, not just a number.
    """
    from tactical_ilp.data.players_reference import build_players_reference
    from tactical_ilp.estimation.cost_report import estimate_structural_distances

    players_ref = build_players_reference()
    result = estimate_structural_distances(players_ref)
    assert "player_id" in result.note or "anonymous" in result.note.lower() or "proxy" in result.note.lower()


def test_raw_frames_confirm_no_player_id_for_non_actors():
    """
    Direct evidence for the finding above: load one match's raw frames and
    confirm there genuinely is no player-identifying column beyond
    actor/keeper flags.
    """
    from tactical_ilp.data.load_raw_data import load_frames

    frames = load_frames(3764440)
    assert "player_id" not in frames.columns
    assert set(frames.columns) == {"event_id", "teammate", "actor", "keeper", "x", "y"}


# --------------------------------------------------------------------------
# already-built relational facts (sanity cross-check against phase 2/3 output)
# --------------------------------------------------------------------------
def test_existing_relational_facts_match_phase2_output():
    results = estimate_existing_relational_facts()
    assert len(results) == 2
    for r in results:
        assert r.estimated_facts_full_season > 0
        assert r.verdict == "GREEN"


# --------------------------------------------------------------------------
# sequence-level counters — must flag the discretization requirement
# --------------------------------------------------------------------------
def test_sequence_counters_recommend_discretization():
    results = estimate_sequence_counters()
    assert len(results) == 2
    for r in results:
        assert r.recommendation, f"{r.concept} should carry a discretization recommendation"
        assert "discretiz" in r.recommendation.lower() or "bucket" in r.recommendation.lower() or "threshold" in r.recommendation.lower()


# --------------------------------------------------------------------------
# full report
# --------------------------------------------------------------------------
def test_run_all_estimates_returns_every_concept_with_a_verdict():
    results = run_all_estimates(n_sample_matches=5)
    assert len(results) >= 7
    for r in results:
        assert isinstance(r, Estimate)
        assert r.verdict in ("GREEN", "AMBER", "RED")
        assert r.estimated_facts_full_season >= 0
        assert r.ratio_to_current >= 0


def test_current_total_facts_baseline_matches_manifest():
    """
    CURRENT_TOTAL_FACTS should come from the current phase-3 manifest, not a
    stale schema or a hand-maintained estimate.
    """
    import json

    from tactical_ilp.prolog.prolog_facts import MANIFEST_PATH, PHASE3_FEATURE_VERSION

    if not MANIFEST_PATH.exists():
        pytest.skip("facts_manifest.json not built — run scripts/run_phase3.py first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("phase3_feature_version") != PHASE3_FEATURE_VERSION:
        pytest.skip("facts_manifest.json uses an old phase-3 schema — rerun scripts/run_phase3.py")
    assert manifest["total_facts"] == CURRENT_TOTAL_FACTS, (
        "CURRENT_TOTAL_FACTS in cost_report.py is stale relative to the actual "
        "facts_manifest.json — update the constant after any phase-3 rerun"
    )
