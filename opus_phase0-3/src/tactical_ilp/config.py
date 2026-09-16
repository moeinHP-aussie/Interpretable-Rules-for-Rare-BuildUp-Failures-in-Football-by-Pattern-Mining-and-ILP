"""
Central configuration for the neuro-symbolic tactical analysis project.

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
DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, PROLOG_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset identity (technical document, decision #4)
# --------------------------------------------------------------------------
COMPETITION_ID = 11          # La Liga
SEASON_ID = 90               # 2020/2021
TEAM_NAME = "Barcelona"
TEAM_ID = 217                # StatsBomb team_id for FC Barcelona

# --------------------------------------------------------------------------
# Pitch geometry (SPADL standard: 105 x 68, attacking direction = +x)
# --------------------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

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

# Zone 14: the central zone immediately in front of the penalty area.
# Pitch split into 6 x 3 = 18 zones; zone 14 is x in [70, 87.5), y in [22.67, 45.33)
ZONE14_X = (70.0, 87.5)
ZONE14_Y = (PITCH_WIDTH / 3.0, 2.0 * PITCH_WIDTH / 3.0)

# --------------------------------------------------------------------------
# Phase 2 — build-up sequence definition
# --------------------------------------------------------------------------
# A build-up sequence starts when Barcelona takes controlled possession in its
# own defensive third and ends when the ball leaves the defensive third,
# possession is lost, or play stops.
BUILD_UP_START_MAX_X = DEFENSIVE_THIRD_MAX_X   # 35.0 m
BUILD_UP_EXIT_X = DEFENSIVE_THIRD_MAX_X        # crossing this line = build-up completed
MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 25

# --------------------------------------------------------------------------
# Phase 2 — operational definition of a NEGATIVE OUTCOME
# (see docs/negative_outcome_definition.md — these constants ARE the definition)
# --------------------------------------------------------------------------
NEG_TURNOVER_WINDOW_S = 8.0     # ball lost within N seconds of the sequence end
NEG_DANGER_ZONE_MAX_X = 52.5    # turnover inside our own half
NEG_OPP_SHOT_WINDOW_S = 12.0    # opponent shot within N seconds of the turnover
NEG_OPP_ENTRY_WINDOW_S = 12.0   # opponent reaches our final third within N seconds

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


def cache_path(kind: str, match_id: int | str, suffix: str = "parquet") -> Path:
    """Canonical cache file path, e.g. data/raw/events_3773386.parquet"""
    return RAW_DIR / f"{kind}_{match_id}.{suffix}"
