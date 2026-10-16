"""
Keeps the human-readable definition documents and the machine-readable
constants in :mod:`tactical_ilp.config` from drifting apart.

The execution plan asks (task 2.2) that the "negative outcome" definition be
written down with *explicit numeric / spatial criteria*, not a vague qualitative
description.  A markdown file alone cannot guarantee that, so the test suite
parses the document and compares every parameter it mentions with the value the
pipeline actually used.
"""

from __future__ import annotations

import re
from pathlib import Path

from tactical_ilp import config

DOCS_DIR = Path(config.DOCS_DIR)

OUTCOME_DOC = DOCS_DIR / "negative_outcome_definition.md"
SAMPLING_DOC = DOCS_DIR / "negative_sampling_strategy.md"

OUTCOME_PARAMETERS = {
    "BUILD_UP_START_MAX_X": config.BUILD_UP_START_MAX_X,
    "BUILD_UP_DEEP_ENTRY_X": config.BUILD_UP_DEEP_ENTRY_X,
    "BUILD_UP_ZONE_ENTRY_X": config.BUILD_UP_ZONE_ENTRY_X,
    "BUILD_UP_ZONE14_MAX_X": config.BUILD_UP_ZONE14_MAX_X,
    "BUILD_UP_ZONE14_Y_MIN": config.BUILD_UP_ZONE14_Y_MIN,
    "BUILD_UP_ZONE14_Y_MAX": config.BUILD_UP_ZONE14_Y_MAX,
    "BUILD_UP_LEFT_HALF_SPACE_MIN_Y": config.BUILD_UP_LEFT_HALF_SPACE_MIN_Y,
    "BUILD_UP_LEFT_HALF_SPACE_MAX_Y": config.BUILD_UP_LEFT_HALF_SPACE_MAX_Y,
    "BUILD_UP_RIGHT_HALF_SPACE_MIN_Y": config.BUILD_UP_RIGHT_HALF_SPACE_MIN_Y,
    "BUILD_UP_RIGHT_HALF_SPACE_MAX_Y": config.BUILD_UP_RIGHT_HALF_SPACE_MAX_Y,
    "DEFENSIVE_THIRD_MAX_X": config.DEFENSIVE_THIRD_MAX_X,
    "OWN_HALF_MAX_X": config.OWN_HALF_MAX_X,
    "MIDFIELD_REGAIN_WINDOW_S": config.MIDFIELD_REGAIN_WINDOW_S,
}

SAMPLING_PARAMETERS = {
    "NEG_SAMPLES_PER_POSITIVE": config.NEG_SAMPLES_PER_POSITIVE,
    "NEG_SAMPLING_RANDOM_SEED": config.NEG_SAMPLING_RANDOM_SEED,
    "MAX_GLOBAL_MATCH_SHARE": config.MAX_GLOBAL_MATCH_SHARE,
}


def _parse(path: Path, expected: dict[str, float]) -> dict:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "has_explicit_criteria": False,
            "parameters": {},
            "mismatches": list(expected),
        }

    text = path.read_text(encoding="utf-8")
    found: dict[str, float] = {}
    mismatches: list[str] = []
    for name, value in expected.items():
        m = re.search(rf"{name}\s*=\s*(-?\d+(?:\.\d+)?)", text)
        if m is None:
            mismatches.append(name)
            continue
        found[name] = float(m.group(1))
        if abs(found[name] - float(value)) > 1e-9:
            mismatches.append(name)

    return {
        "path": str(path),
        "exists": True,
        "parameters": found,
        "mismatches": mismatches,
        "has_explicit_criteria": len(found) == len(expected) and not mismatches,
    }


def load_outcome_definition_doc() -> dict:
    return _parse(OUTCOME_DOC, OUTCOME_PARAMETERS)


def load_sampling_strategy_doc() -> dict:
    return _parse(SAMPLING_DOC, SAMPLING_PARAMETERS)


if __name__ == "__main__":
    for d in (load_outcome_definition_doc(), load_sampling_strategy_doc()):
        print(d["path"])
        print("  explicit criteria:", d["has_explicit_criteria"])
        print("  parameters       :", d["parameters"])
        print("  mismatches       :", d["mismatches"])
