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
    "NEG_TURNOVER_WINDOW_S": config.NEG_TURNOVER_WINDOW_S,
    "NEG_DANGER_ZONE_MAX_X": config.NEG_DANGER_ZONE_MAX_X,
    "NEG_OPP_SHOT_WINDOW_S": config.NEG_OPP_SHOT_WINDOW_S,
    "NEG_OPP_ENTRY_WINDOW_S": config.NEG_OPP_ENTRY_WINDOW_S,
    "BUILD_UP_EXIT_X": config.BUILD_UP_EXIT_X,
}

SAMPLING_PARAMETERS = {
    "NEG_SAMPLES_PER_POSITIVE": config.NEG_SAMPLES_PER_POSITIVE,
    "NEG_SAMPLING_RANDOM_SEED": config.NEG_SAMPLING_RANDOM_SEED,
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
