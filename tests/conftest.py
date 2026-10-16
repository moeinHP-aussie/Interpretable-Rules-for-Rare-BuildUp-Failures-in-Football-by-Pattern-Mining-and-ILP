"""
Shared pytest fixtures.

Design rule: the tests must be runnable **offline** once the cache exists.
Anything that would touch the network is marked ``@pytest.mark.network`` so it
can be skipped with ``-m "not network"``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tactical_ilp import config


# --------------------------------------------------------------------------
# environment helpers
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    return config.PROJECT_ROOT


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return config.RAW_DIR


@pytest.fixture(scope="session")
def swipl_available() -> bool:
    return shutil.which("swipl") is not None


@pytest.fixture(scope="session")
def popper_dir() -> Path | None:
    """Location of a local Popper checkout, if the user provided one."""
    import os

    env = os.environ.get("POPPER_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        config.PROJECT_ROOT / "external" / "Popper",
        config.PROJECT_ROOT.parent / "Popper",
    ]
    for c in candidates:
        if c.exists() and (c / "popper.py").exists():
            return c
    return None


# --------------------------------------------------------------------------
# phase 1 data
# --------------------------------------------------------------------------
def _require(path: Path, hint: str):
    if not path.exists():
        pytest.skip(f"missing artefact {path.name} — run `{hint}` first")


@pytest.fixture(scope="session")
def barca_match_ids() -> list[int]:
    from tactical_ilp.data.load_raw_data import MATCHES_CACHE, barca_match_ids

    _require(MATCHES_CACHE, "python scripts/run_phase1.py")
    return barca_match_ids()


@pytest.fixture(scope="session")
def spadl_all_matches() -> pd.DataFrame:
    from tactical_ilp.data.spadl_converter import SPADL_ALL_PATH

    _require(SPADL_ALL_PATH, "python scripts/run_phase1.py")
    return pd.read_parquet(SPADL_ALL_PATH)


@pytest.fixture(scope="session")
def sample_spadl_df(spadl_all_matches) -> pd.DataFrame:
    first = spadl_all_matches.match_id.iloc[0]
    return spadl_all_matches[spadl_all_matches.match_id == first]


@pytest.fixture(scope="session")
def merged_df() -> pd.DataFrame:
    from tactical_ilp.data.freeze_frames import MERGED_ALL_PATH

    _require(MERGED_ALL_PATH, "python scripts/run_phase1.py")
    return pd.read_parquet(MERGED_ALL_PATH)


@pytest.fixture(scope="session")
def players_reference_df() -> pd.DataFrame:
    from tactical_ilp.data.players_reference import PLAYERS_REFERENCE_PATH

    _require(PLAYERS_REFERENCE_PATH, "python scripts/run_phase1.py")
    return pd.read_parquet(PLAYERS_REFERENCE_PATH)


# --------------------------------------------------------------------------
# phase 2 sequences
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sequences_df() -> pd.DataFrame:
    from tactical_ilp.sequences.sequence_builder import SEQUENCES_PATH

    _require(SEQUENCES_PATH, "python scripts/run_phase2.py")
    return pd.read_parquet(SEQUENCES_PATH)


@pytest.fixture(scope="session")
def sequence_actions_df() -> pd.DataFrame:
    from tactical_ilp.sequences.sequence_builder import SEQUENCE_ACTIONS_PATH

    _require(SEQUENCE_ACTIONS_PATH, "python scripts/run_phase2.py")
    return pd.read_parquet(SEQUENCE_ACTIONS_PATH)


@pytest.fixture(scope="session")
def sample_sequences(sequences_df, sequence_actions_df) -> list[list[int]]:
    """Sequences of the first three matches, as lists of action ids."""
    matches = sorted(sequences_df.match_id.unique())[:3]
    sids = set(sequences_df[sequences_df.match_id.isin(matches)].sequence_id)
    sub = sequence_actions_df[sequence_actions_df.sequence_id.isin(sids)]
    return [list(g.action_id) for _, g in sub.groupby("sequence_id")]


@pytest.fixture(scope="session")
def labeled_sequences_df() -> pd.DataFrame:
    from tactical_ilp.sequences.outcome_labeler import LABELED_PATH

    _require(LABELED_PATH, "python scripts/run_phase2.py")
    return pd.read_parquet(LABELED_PATH)


@pytest.fixture(scope="session")
def outcome_definition_doc() -> dict:
    from tactical_ilp.docs_sync import load_outcome_definition_doc

    return load_outcome_definition_doc()


@pytest.fixture(scope="session")
def sampling_strategy_doc() -> dict:
    from tactical_ilp.docs_sync import load_sampling_strategy_doc

    return load_sampling_strategy_doc()


@pytest.fixture(scope="session")
def sequences_with_subsets() -> pd.DataFrame:
    from tactical_ilp.sequences.player_subsets import SEQ_WITH_SUBSETS_PATH

    _require(SEQ_WITH_SUBSETS_PATH, "python scripts/run_phase2.py")
    return pd.read_parquet(SEQ_WITH_SUBSETS_PATH)


@pytest.fixture(scope="session")
def ilp_examples_df() -> pd.DataFrame:
    from tactical_ilp.sequences.negative_sampling import EXAMPLES_PATH

    _require(EXAMPLES_PATH, "python scripts/run_phase2.py")
    return pd.read_parquet(EXAMPLES_PATH)


@pytest.fixture(scope="session")
def pos_examples_df(ilp_examples_df) -> pd.DataFrame:
    return ilp_examples_df[ilp_examples_df.example_class == "pos"]


@pytest.fixture(scope="session")
def neg_examples_df(ilp_examples_df) -> pd.DataFrame:
    return ilp_examples_df[ilp_examples_df.example_class == "neg"]


# --------------------------------------------------------------------------
# phase 3 prolog facts
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def facts_file_path() -> Path:
    """Fact file of one sample match (task 3.2)."""
    files = sorted(config.PROLOG_DIR.glob("facts_match_*.pl"))
    if not files:
        pytest.skip("no sample fact file — run `python scripts/run_phase3.py` first")
    return files[0]


@pytest.fixture(scope="session")
def final_facts_file() -> Path:
    from tactical_ilp.prolog.prolog_facts import FACTS_ALL_PATH

    _require(FACTS_ALL_PATH, "python scripts/run_phase3.py")
    return FACTS_ALL_PATH


def consult_prolog(path: Path) -> subprocess.CompletedProcess:
    """Load a .pl file in SWI-Prolog and return the finished process."""
    return subprocess.run(
        ["swipl", "-g", f"consult('{path.as_posix()}'), halt.", "-t", "halt(1)"],
        capture_output=True,
        text=True,
        timeout=600,
    )
