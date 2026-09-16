"""
Phase 0 — environment and infrastructure.

Tasks 0.1 .. 0.4 of the execution plan.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


# --------------------------------------------------------------------------
# Task 0.1 — python project and packages
# --------------------------------------------------------------------------
def test_core_imports():
    import mplsoccer  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import socceraction  # noqa: F401
    import statsbombpy  # noqa: F401

    assert True


def test_project_package_importable():
    import tactical_ilp
    from tactical_ilp import config

    assert config.PROJECT_ROOT.exists()
    assert config.COMPETITION_ID == 11
    assert config.SEASON_ID == 90


def test_socceraction_pandera_stack_is_healthy():
    """
    Regression guard: socceraction 1.5 pulls pandera 0.17, which crashes with
    multimethod >= 2.0 (``cannot import name 'overload'``).  pyproject pins
    ``multimethod<2.0``; this test makes the breakage visible immediately.
    """
    from socceraction.data.statsbomb import StatsBombLoader  # noqa: F401
    import socceraction.spadl as spadl

    assert hasattr(spadl.statsbomb, "convert_to_actions")


# --------------------------------------------------------------------------
# Task 0.2 — Prolog / ILP toolchain
# --------------------------------------------------------------------------
def test_swipl_installed_and_recent_enough(swipl_available):
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    out = subprocess.run(["swipl", "--version"], capture_output=True, text=True)
    assert out.returncode == 0
    # janus_swi (Popper's Prolog bridge) needs >= 9.1.12
    version = out.stdout.split("version")[1].split()[0]
    major, minor, patch = (int(p) for p in version.split(".")[:3])
    assert (major, minor, patch) >= (9, 1, 12), f"SWI-Prolog {version} is too old for Popper"


def test_swipl_can_run_a_goal(swipl_available):
    if not swipl_available:
        pytest.skip("SWI-Prolog not on PATH")
    out = subprocess.run(
        ["swipl", "-g", "X is 2+3, format('~w~n',[X]), halt."],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "5" in out.stdout


@pytest.mark.slow
def test_popper_installed(popper_dir):
    """
    Runs Popper's own tutorial example.  Skipped unless a Popper checkout is
    reachable (env var ``POPPER_DIR`` or ``<project>/external/Popper``).
    """
    if popper_dir is None:
        pytest.skip("Popper checkout not found — set POPPER_DIR")
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    result = subprocess.run(
        ["uv", "run", "popper.py", "examples/iggp-rps-next-score"],
        cwd=popper_dir,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0
    assert "SOLUTION" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# Task 0.3 / 0.4 — dataset reachability
# --------------------------------------------------------------------------
def test_barca_matches_available(barca_match_ids):
    assert len(barca_match_ids) >= 20


def test_360_data_available_for_at_least_one_match(barca_match_ids):
    from tactical_ilp.data.load_raw_data import load_frames

    found = False
    for match_id in barca_match_ids:
        frames = load_frames(match_id)
        if len(frames) > 0:
            found = True
            break
    assert found


@pytest.mark.network
def test_statsbomb_open_data_reachable():
    """Only meaningful on a fresh machine before the cache is built."""
    from statsbombpy import sb

    from tactical_ilp.config import COMPETITION_ID, SEASON_ID, TEAM_NAME

    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    barca = matches[
        (matches["home_team"] == TEAM_NAME) | (matches["away_team"] == TEAM_NAME)
    ]
    assert len(barca) >= 20
