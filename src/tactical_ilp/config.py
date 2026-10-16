"""
Central configuration for the symbolic tactical analysis project.

All paths are resolved relative to the project root so the code behaves
identically on Windows, Linux and macOS.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("TACTICAL_ILP_DATA", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROLOG_DIR = DATA_DIR / "prolog"
REPORTS_DIR = DATA_DIR / "reports"
DOCS_DIR = PROJECT_ROOT / "history_docs"

for _d in (RAW_DIR, PROCESSED_DIR, PROLOG_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset identity (technical document, decision #4)
# --------------------------------------------------------------------------
COMPETITION_ID = int(os.environ.get("TACTICAL_ILP_COMPETITION_ID", "11"))
SEASON_ID = int(os.environ.get("TACTICAL_ILP_SEASON_ID", "90"))
TEAM_NAME = os.environ.get("TACTICAL_ILP_TEAM_NAME", "Barcelona")
TEAM_ID = int(os.environ.get("TACTICAL_ILP_TEAM_ID", "217"))

# --------------------------------------------------------------------------
# Pitch geometry (SPADL standard: 105 x 68, attacking direction = +x)
# --------------------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32

# 5 vertical lanes. y = 0 is the RIGHT touchline from the attacking team's
# point of view (because the team attacks towards +x).
LANE_EDGES = (0.0, 13.6, 27.2, 40.8, 54.4, 68.0)
LANE_NAMES = (
    "right_wing",
    "right_half_space",
    "central_channel",
    "left_half_space",
    "left_wing",
)

# Horizontal thirds
DEFENSIVE_THIRD_MAX_X = PITCH_LENGTH / 3.0          # 35.0
MIDDLE_THIRD_MAX_X = 2.0 * PITCH_LENGTH / 3.0       # 70.0

# --------------------------------------------------------------------------
# Phase 2 — build-up sequence definition
# --------------------------------------------------------------------------
# Start from anywhere in the selected team's own half. Successful exit requires
# controlled entry into one of the approved attacking zones or a shot.
OWN_HALF_MAX_X = PITCH_LENGTH / 2.0             # 52.5 m
OPPONENT_THIRD_START_X = 2.0 * PITCH_LENGTH / 3.0  # 70.0 m (reference line)
BUILD_UP_START_MAX_X = OWN_HALF_MAX_X
BUILD_UP_DEEP_ENTRY_X = 78.0
BUILD_UP_ZONE_ENTRY_X = 70.0
BUILD_UP_ZONE14_MAX_X = 87.5
BUILD_UP_ZONE14_Y_MIN = 27.2
BUILD_UP_ZONE14_Y_MAX = 40.8
BUILD_UP_LEFT_HALF_SPACE_MIN_Y = 40.8
BUILD_UP_LEFT_HALF_SPACE_MAX_Y = 54.4
BUILD_UP_RIGHT_HALF_SPACE_MIN_Y = 13.6
BUILD_UP_RIGHT_HALF_SPACE_MAX_Y = 27.2
MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 25

# --------------------------------------------------------------------------
# Phase 2 — operational definition of a NEGATIVE OUTCOME
# (see docs/negative_outcome_definition.md — these constants ARE the definition)
# --------------------------------------------------------------------------
MIDFIELD_REGAIN_WINDOW_S = 5.0  # own-half midfield loss: time allowed to regain

# --------------------------------------------------------------------------
# Phase 3 — pressure model
# --------------------------------------------------------------------------
PRESSURE_RADIUS_M = 5.0          # opponents inside this radius count as pressure
TIGHT_PRESSURE_RADIUS_M = 3.0
PRESSURE_MIN_OPPONENTS = 1

# --------------------------------------------------------------------------
# Phase 1 — players reference
# --------------------------------------------------------------------------
MIN_APPEARANCES_FOR_REFERENCE = 5   # a player must appear in >= N matches

# --------------------------------------------------------------------------
# Phase 2.4 — negative sampling
# --------------------------------------------------------------------------
NEG_SAMPLES_PER_POSITIVE = 2
NEG_SAMPLING_RANDOM_SEED = 20_2021
MAX_GLOBAL_MATCH_SHARE = 0.05

# --------------------------------------------------------------------------
# Phase 4 — tactical concepts (see docs/tactical_concepts_glossary.md)
# --------------------------------------------------------------------------
# Concept 3.6a: only triangles formed by teammates near the ball count. The
# radius bounds pair enumeration to local support shapes rather than the whole
# visible formation.
#
# The earlier 10 m radius made a 150 m² area ceiling unreachable (the maximum
# area with two 10 m sides is 50 m²).  An 18 m local radius lets the area cap
# distinguish larger support shapes while avoiding pitch-wide combinations.
TRIANGLE_RADIUS_M = 18.0

# A triangle must enclose enough space, avoid needle-like angles, and remain
# local to the carrier. Area and minimum-angle checks work together: area alone
# does not reject every nearly-collinear shape.
TRIANGLE_MIN_AREA_M2 = 8.0
TRIANGLE_MAX_AREA_M2 = 150.0
TRIANGLE_MIN_ANGLE_DEG = 20.0
TRIANGLE_EDGE_TOLERANCE_M = 1.0

# Concept 0.1: how many of the deepest opponents are treated as "the defensive
# line" and the next band as "the midfield line".  Positional approximation
# only -- 360 frames carry no player_id for non-actors, so a true role lookup
# is impossible (see the glossary, §2.3پ).
DEFENSIVE_LINE_SIZE = 4
MIDFIELD_LINE_SIZE = 3

# Minimum number of opponents that must be visible in a frame before the line
# clustering is trusted at all.
MIN_OPPONENTS_FOR_LINE = 6

# Version stamp for phase-1 derived data. Bump this when conversion or
# freeze-frame linkage changes; raw event/lineup/frame caches remain reusable.
PHASE1_VERSION = "phase1-v7-orientation-fix-ms-time-360-linkage"

# Matches whose 360 file does not link to the event stream (share of frame event
# ids found among the events below this value) give no pressure features. They
# are excluded from the merged table by default; set to 0 to keep them (their
# 360 features are then NaN).
EXCLUDE_UNLINKED_360_MATCHES = os.environ.get("TACTICAL_ILP_EXCLUDE_UNLINKED_360", "1") != "0"
MIN_FRAME_LINKAGE = 0.50


def cache_path(kind: str, match_id: int | str, suffix: str = "parquet") -> Path:
    """Canonical cache file path, e.g. data/raw/events_3773386.parquet"""
    return RAW_DIR / f"{kind}_{match_id}.{suffix}"

# Concept 6.2 (tempo): slow is a fixed low-tempo threshold; fast starts at
# the 70th percentile, covering the upper three deciles (deciles 8–10).
TEMPO_SLOW_MAX = 0.50
TEMPO_FAST_PERCENTILE = 70

# Concept 6.3: event-time gap proxy for immediate/first-time play. StatsBomb
# does not record literal touch counts, so this is an operational threshold.
ONE_TOUCH_MAX_SECONDS = 0.50

# A 360 snapshot is used for a visible-player count comparison only when it
# contains at least this many players across both teams.
MIN_VISIBLE_PLAYERS_FOR_MAJORITY = 7
# A sequence-level visible-opponent-majority signal requires this many valid
# snapshots before its per-sequence majority ratio is evaluated.
MIN_VALID_FRAMES_FOR_SEQUENCE_MAJORITY = 2

# Concept 6.1 (hesitation): a dwell time is "long" when it exceeds
# mean + 1 sd *of its own pressure level*.  Measured on the real data -- the
# levels differ a lot (heavy 1.05 s vs none 1.70 s on average), so a single
# global threshold would hide hesitation under pressure entirely.
HESITATION_THRESHOLD_S = {
    "heavy_pressure": 2.14,
    "light_pressure": 2.65,
    "no_pressure": 2.87,
    "unknown": 3.70,
}

# Concept 3.4: a player counts as "frequently involved" in a sequence when they
# touch the ball at least this many times.
FREQUENT_INVOLVEMENT_MIN = 3

# Concept 1.2 (diagonal pass): sufficient total distance, forward progress,
# and an angle that is neither nearly vertical nor nearly horizontal.
DIAGONAL_MIN_LENGTH_M = 12.0
DIAGONAL_MIN_FORWARD_DX_M = 5.0
DIAGONAL_MIN_ANGLE_DEG = 20.0
DIAGONAL_MAX_ANGLE_DEG = 70.0

# Concept 1.3 (switch of play): a lane change is not enough on its own (59% of
# passes change lane).  Requiring 25 m of lateral travel as well isolates the
# real switches: 5.6% of passes.
SWITCH_MIN_DY_M = 25.0
FLANK_LINE_CARRY_MIN_M = 5.0
LONG_PASS_MIN_M = 30.0

# Concept 4.1 (support): a teammate counts as supporting when no opponent
# stands within this many metres of the straight line to the ball carrier.
# 2 m approximates a defender's effective interception reach.
SUPPORT_CORRIDOR_M = 2.0

# Concept 4.5 (passing options): calibrated bucket edges for "few / some /
# many" open supporters, from the real distribution (median 3, p75 5, p90 6).
PASSING_OPTIONS_FEW_MAX = 1
PASSING_OPTIONS_MANY_MIN = 5

# Concept 5.1 (half-space occupation): a player counts as persistently
# occupying a half-space when they touch the ball from a half-space lane at
# least this many times inside one sequence.  Calibrated on the real
# distribution: median is 0, p75 is 1, so >=2 already picks out a minority
# with a genuinely repeated pattern rather than a single incidental touch.
HALF_SPACE_MIN_TOUCHES = 2
