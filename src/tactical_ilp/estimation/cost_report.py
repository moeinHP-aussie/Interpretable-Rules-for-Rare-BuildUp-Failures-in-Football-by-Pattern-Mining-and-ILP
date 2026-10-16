"""
Phase 4 pre-check -- Prolog fact-volume / search-space cost estimator.

Why this exists
----------------
The tactical concepts glossary (docs/tactical_concepts_glossary.md) lists
candidate concepts. Some of them, if translated into Prolog naively, would
generate a huge number of ground facts or high-arity predicates -- which
directly blows up Popper's hypothesis search space. The triangulation concept
(3.6) is the concrete example that triggered this module: a naive "every
combination of nearby teammates" implementation would have produced ~750k
facts (~8x the project's entire current fact base of 96,609), while a
the calibrated teammate radius from ``config.TRIANGLE_RADIUS_M`` keeps the
candidate count manageable while including realistic short-passing distances.

This module runs that same kind of check for every concept that plausibly
risks a large fact count, using real project data (not guesses), and prints a
report with a traffic-light verdict per concept:

    GREEN  (< 0.5x current total facts)   -- safe as a flat fact table
    AMBER  (0.5x - 3x)                    -- fine, but should be discretized
                                              or scoped down before use
    RED    (> 3x)                          -- must not be dumped as raw
                                              per-combination facts; needs a
                                              threshold, a discrete predicate,
                                              or a narrower scope

Concepts that are pure single-action lookups (progressive, pressure_level,
pass_direction, ...) are not measured here -- their cost is already known
(one fact per action, same order of magnitude as facts already in
facts_all.pl) and is not a search-space risk.

Usage
-----
    uv run python -m tactical_ilp.estimation.cost_report

or from tests / other scripts:

    from tactical_ilp.estimation.cost_report import run_all_estimates
    results = run_all_estimates()
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from math import comb

import numpy as np
import pandas as pd

from tactical_ilp.config import PITCH_LENGTH, PROCESSED_DIR, TRIANGLE_RADIUS_M
from tactical_ilp.data.players_reference import build_players_reference
from tactical_ilp.estimation.raw_frames import raw_positions_sample
from tactical_ilp.prolog.prolog_facts import (
    FACTS_ALL_PATH,
    MANIFEST_PATH,
    PHASE3_FEATURE_VERSION,
)

_FALLBACK_TOTAL_FACTS = 96_609  # last packaged manifest; used before phase 3 runs


def _current_total_facts() -> int:
    """Use the current phase-3 manifest so schema changes refresh the baseline."""
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            total = int(manifest["total_facts"])
            if (
                total > 0
                and manifest.get("phase3_feature_version") == PHASE3_FEATURE_VERSION
            ):
                return total
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return _FALLBACK_TOTAL_FACTS


CURRENT_TOTAL_FACTS = _current_total_facts()
AMBER_RATIO = 0.5
RED_RATIO = 3.0


@dataclass
class Estimate:
    concept: str
    label: str
    method: str
    estimated_facts_full_season: int
    ratio_to_current: float
    verdict: str
    note: str = ""
    recommendation: str = ""


def _verdict(ratio: float) -> str:
    if ratio < AMBER_RATIO:
        return "GREEN"
    if ratio <= RED_RATIO:
        return "AMBER"
    return "RED"


def _scale_to_season(sample_count: int, sample_matches: list[int], total_matches: int = 35) -> int:
    """Linear scale-up from a sample of matches to the full 35-match season."""
    if not sample_matches:
        return 0
    return round(sample_count * (total_matches / len(sample_matches)))


# ---------------------------------------------------------------------------
# 3.6 -- Ball-near triangulation (the concept that triggered this whole check)
# ---------------------------------------------------------------------------
def estimate_triangulation(sample_df: pd.DataFrame, sample_matches: list[int]) -> Estimate:
    """
    3.6a (event only): count candidate triangles = ball carrier + 2 teammates
    within TEAMMATE_RADIUS_M of the ball, per action. Naive worst case (any 3
    of all visible teammates) is reported in the note for comparison.
    """
    RADIUS = TRIANGLE_RADIUS_M
    mates = sample_df[sample_df.is_teammate & ~sample_df.is_actor]
    actors = sample_df[sample_df.is_actor][["match_id", "action_id", "x", "y"]].rename(
        columns={"x": "ax", "y": "ay"}
    )
    joined = mates.merge(actors, on=["match_id", "action_id"], how="inner")
    joined["dist"] = np.hypot(joined.x - joined.ax, joined.y - joined.ay)
    nearby_counts = (
        joined[joined.dist <= RADIUS].groupby(["match_id", "action_id"]).size()
    )
    triangles = nearby_counts.apply(lambda n: comb(int(n), 2) if n >= 2 else 0)
    sample_total = int(triangles.sum())
    season_total = _scale_to_season(sample_total, sample_matches)

    # naive worst case for comparison, using ALL visible teammates (not just nearby)
    all_mate_counts = mates.groupby(["match_id", "action_id"]).size()
    naive = all_mate_counts.apply(lambda n: comb(int(n), 3) if n >= 3 else 0)
    naive_season = _scale_to_season(int(naive.sum()), sample_matches)

    ratio = season_total / CURRENT_TOTAL_FACTS
    return Estimate(
        concept="3.6a",
        label="مثلث‌سازی اطراف توپ‌دار (رخداد)",
        method=f"C(n,2) triangles per action, n = teammates within {RADIUS}m of ball carrier",
        estimated_facts_full_season=season_total,
        ratio_to_current=round(ratio, 4),
        verdict=_verdict(ratio),
        note=(
            f"naive (any 3 of ALL visible teammates, no radius limit) would be "
            f"~{naive_season:,} facts (~{naive_season / CURRENT_TOTAL_FACTS:.1f}x current total) "
            f"-- confirms the radius restriction is what keeps this concept cheap"
        ),
        recommendation=(
            f"use the calibrated {RADIUS:g}m-radius restriction; do not enumerate "
            "all visible teammates"
        ),
    )


# ---------------------------------------------------------------------------
# 0.1 -- Defensive-line clustering (light solution: x-coordinate ranking)
# ---------------------------------------------------------------------------
def estimate_line_clustering(sample_df: pd.DataFrame, sample_matches: list[int]) -> Estimate:
    """
    IMPORTANT DESIGN CORRECTION (found while building this estimator): the
    glossary's own text calls 0.1 "a Python ranking function" -- exactly the
    same kind of helper as third_of()/lane_of() in tactical_ilp.features.geometry,
    which are NOT dumped as one fact per (action, other player). They produce
    ONE derived label per action (at_third/2, at_lane/2: one fact per action).

    A first, wrong version of this estimator measured "one fact per (action,
    visible opponent)" -- i.e. treating the ranking as if every opponent's
    line membership were separately asserted. That is not what the glossary
    proposes and it is not how any comparable predicate in facts_all.pl is
    built; it produced a false RED verdict (6.77x). The correct measurement
    is: one derived fact per action (e.g. "ball played from in front of /
    behind / between the opponent's two lines"), matching every other
    per-action geometric predicate already in the fact base.
    """
    n_actions_with_ff = len(sample_df[sample_df.is_actor])
    season_total = _scale_to_season(n_actions_with_ff, sample_matches)
    ratio = season_total / CURRENT_TOTAL_FACTS
    return Estimate(
        concept="0.1",
        label="خوشه‌بندی خط دفاعی/هافبک (راه‌حل سبک)",
        method="ONE derived label per action (like at_third/2), not one fact per (action, opponent)",
        estimated_facts_full_season=season_total,
        ratio_to_current=round(ratio, 4),
        verdict=_verdict(ratio),
        note=(
            "CORRECTED after an initial wrong measurement (see docstring): the x-rank "
            "clustering is a Python helper function, consumed by dependent concepts "
            "(1.1, 2.3, 7.5, 7.6ب) at query time -- it should not itself become a large "
            "per-opponent fact table. Cost is now measured as one label per action, same "
            "order of magnitude as at_third/at_lane."
        ),
        recommendation="implement as a Python function (geometry.py), not a dumped Prolog fact table; safe as-is",
    )


# ---------------------------------------------------------------------------
# 2.3پ -- Inter-player structural distances
# ---------------------------------------------------------------------------
def estimate_structural_distances(players_ref: pd.DataFrame) -> Estimate:
    """
    IMPORTANT FEASIBILITY FINDING (found while building this estimator, not
    previously documented in the glossary): StatsBomb 360 freeze-frame rows
    for non-actor players carry ONLY (x, y, teammate, keeper) -- there is no
    player_id for anyone except the ball-carrier (actor=True) and the
    goalkeeper (keeper=True). The other ~9-10 players per frame are anonymous
    dots. This means distance_between(central_defender, central_defender, ...)
    as literally proposed CANNOT be built by looking up two named roles and
    measuring their distance -- there is no way to know which anonymous dot
    is "the left center-back" versus "the right full-back" from the frame
    data alone.

    What IS buildable: distances between POSITION-PATTERN proxies (e.g. "the
    two deepest opponents" as a proxy for center-backs, matching concept
    0.1's own light-solution approach), not distances keyed on verified
    roles. For the ACTOR specifically we do have a verified role (from
    has_role via player_id), so actor-to-nearest-opponent-line distances are
    fully buildable; teammate-pair-to-teammate-pair distances for the 10
    non-actor players are only approximate.

    Given this, the fact-count ceiling below is informative but secondary to
    the feasibility finding: even at a favorable count, only a proxy version
    of most of the 6 named distances in the glossary is achievable with this
    dataset, not the exact role-pair version as written.
    """
    ROLE_PAIRS = [
        ("central_defender", "central_defender"),
        ("central_midfielder", "central_midfielder"),
        ("full_back", "full_back"),
        ("full_back", "winger"),
        ("central_defender", "defensive_midfielder"),
        ("defensive_midfielder", "forward"),
    ]
    merged = pd.read_parquet(PROCESSED_DIR / "spadl_with_360_all_matches.parquet")
    n_actions_with_ff = int(merged.has_freeze_frame.sum())
    season_total = n_actions_with_ff * len(ROLE_PAIRS)
    ratio = season_total / CURRENT_TOTAL_FACTS
    return Estimate(
        concept="2.3پ",
        label="فاصله‌های ساختاری بین بازیکنان",
        method=f"ceiling: {len(ROLE_PAIRS)} role-pairs x actions-with-360 ({n_actions_with_ff:,}) -- see feasibility note, this ceiling is secondary",
        estimated_facts_full_season=season_total,
        ratio_to_current=round(ratio, 4),
        verdict=_verdict(ratio),
        note=(
            "FEASIBILITY ISSUE, not just a volume issue: raw 360 frames give NO "
            "player_id for the ~10 non-actor players per frame (only x, y, "
            "teammate/keeper flags) -- so true role-pair distances (e.g. "
            "'the two central defenders') cannot be resolved by role lookup. Only "
            "position-pattern PROXIES (e.g. 'two deepest opponents') are buildable, "
            "which is the same approximation concept 0.1 already accepts for lines."
        ),
        recommendation=(
            "scope down to position-pattern proxy pairs only (reuse 0.1's ranking "
            "logic per pair), and drop or de-prioritize named-role pairs that need "
            "player identity we don't have; re-estimate the proxy version's count "
            "once its exact definition is fixed, since it will differ from this ceiling"
        ),
    )


# ---------------------------------------------------------------------------
# 3.1/3.2 -- one-two and third-man (already-built relational facts, sanity check)
# ---------------------------------------------------------------------------
def estimate_existing_relational_facts() -> list[Estimate]:
    """
    These already exist as parquet outputs from phase 2 -- reported here for
    a single consistent before/after comparison table, not re-measured.
    """
    pairs = pd.read_parquet(PROCESSED_DIR / "sequence_pairs.parquet")
    triples = pd.read_parquet(PROCESSED_DIR / "sequence_triples.parquet")
    out = []
    for name, label, df, arity_note in [
        ("3.1 (passes_to)", "یک‌ودو / لینک پاس", pairs, "retired from active phase-3 Prolog export; pair parquet remains available"),
        ("3.2 (chain)", "سه‌نفره / مرد سوم", triples, "retired from active phase-3 Prolog export; triple parquet remains available"),
    ]:
        ratio = len(df) / CURRENT_TOTAL_FACTS
        out.append(
            Estimate(
                concept=name,
                label=label,
                method="already built in phase 2/3 -- exact count from parquet",
                estimated_facts_full_season=len(df),
                ratio_to_current=round(ratio, 4),
                verdict=_verdict(ratio),
                note=arity_note,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Sequence-level counters that the glossary flags as needing discretization
# ---------------------------------------------------------------------------
def estimate_sequence_counters() -> list[Estimate]:
    """
    3.4 (give-and-continue involvement), 5.1 (half-space occupation streak),
    6.2 (tempo) -- these are one raw number per (player, sequence) or per
    sequence. Cheap in COUNT, but the glossary's v3 concern is about VALUE
    CARDINALITY (a raw integer fact has as many distinct values as there are
    integers in the data, which blows up Popper's argument-matching space
    even with few facts). We report count here and flag the value-diversity
    problem separately since count alone understates the real risk.
    """
    sequences = pd.read_parquet(PROCESSED_DIR / "sequences.parquet")
    involved = pd.read_parquet(PROCESSED_DIR / "sequences_with_subsets.parquet")
    n_seq = len(sequences)

    out = []
    # 6.2 tempo: one fact per sequence (mean inter-action gap)
    ratio = n_seq / CURRENT_TOTAL_FACTS
    out.append(
        Estimate(
            concept="6.2",
            label="سرعت انتقال توپ (تمپو)",
            method="one raw numeric fact per sequence",
            estimated_facts_full_season=n_seq,
            ratio_to_current=round(ratio, 4),
            verdict=_verdict(ratio),
            note=(
                f"fact COUNT is tiny ({n_seq}) and GREEN by volume, but as a raw "
                f"float this predicate has up to {n_seq} distinct values -- Popper "
                f"cannot generalize over it as-is"
            ),
            recommendation="discretize into tempo_bucket(Seq, slow|medium|fast) like pressure_level, not a raw float fact",
        )
    )
    # 3.4 give-and-continue involvement count: one fact per (player, sequence) pair, same shape as involved_in
    n_pairs = len(involved) if "player_pairs" not in involved.columns else int(
        involved["n_pairs"].sum()
    )
    ratio2 = n_pairs / CURRENT_TOTAL_FACTS
    out.append(
        Estimate(
            concept="3.4",
            label="Give-and-continue (شمارش دخالت مکرر)",
            method="one raw integer fact per (player, sequence), same order as involved_in/2 (2,915 facts)",
            estimated_facts_full_season=n_pairs,
            ratio_to_current=round(ratio2, 4),
            verdict=_verdict(ratio2),
            note="count is GREEN, but same raw-integer problem as 6.2 applies",
            recommendation="discretize to frequently_involved(Player, Seq) boolean with an empirically chosen threshold, per v3 decision",
        )
    )
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all_estimates(n_sample_matches: int = 8) -> list[Estimate]:
    sample_df, sample_matches = raw_positions_sample(n_matches=n_sample_matches)
    players_ref = build_players_reference()

    results: list[Estimate] = []
    results.append(estimate_triangulation(sample_df, sample_matches))
    results.append(estimate_line_clustering(sample_df, sample_matches))
    results.append(estimate_structural_distances(players_ref))
    results.extend(estimate_existing_relational_facts())
    results.extend(estimate_sequence_counters())
    return results


def print_report(results: list[Estimate]) -> None:
    print("=" * 100)
    print(
        f"PHASE 4 PRE-CHECK -- Prolog fact-volume estimate "
        f"(baseline: {CURRENT_TOTAL_FACTS:,} facts currently in facts_all.pl)"
    )
    print("=" * 100)
    print()
    header = f"{'concept':<10} {'label':<38} {'est. facts':>12} {'ratio':>8}  verdict"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.concept:<10} {r.label:<38} {r.estimated_facts_full_season:>12,} "
            f"{r.ratio_to_current:>7.2f}x  {r.verdict}"
        )
    print()
    for r in results:
        if r.note or r.recommendation:
            print(f"[{r.concept}] {r.label}")
            if r.method:
                print(f"    method : {r.method}")
            if r.note:
                print(f"    note   : {r.note}")
            if r.recommendation:
                print(f"    action : {r.recommendation}")
            print()

    reds = [r for r in results if r.verdict == "RED"]
    ambers = [r for r in results if r.verdict == "AMBER"]
    print("-" * 100)
    print(f"summary: {len(results)} concepts checked, {len(reds)} RED, {len(ambers)} AMBER, "
          f"{len(results) - len(reds) - len(ambers)} GREEN")
    if reds:
        print("RED concepts must not be added as flat fact tables without narrowing scope first:")
        for r in reds:
            print(f"  - {r.concept} {r.label}")


if __name__ == "__main__":
    results = run_all_estimates()
    print_report(results)
