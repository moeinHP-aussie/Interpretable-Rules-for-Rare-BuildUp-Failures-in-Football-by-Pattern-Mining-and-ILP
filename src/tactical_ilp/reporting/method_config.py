"""Select the Phase-5 model used to build the coach-facing report.

The hybrid sequential arm is the default. Set ``TACTICAL_ILP_COACH_METHOD`` to
``full_tactical`` only when intentionally reviewing the comparison baseline.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path

from tactical_ilp.config import PROLOG_DIR, REPORTS_DIR


@dataclass(frozen=True)
class CoachMethod:
    key: str
    title: str
    task_dir: Path
    training_log: Path
    training_report: Path | None


METHODS = {
    "full_tactical": CoachMethod(
        key="full_tactical",
        title="واژگان تاکتیکی کامل",
        task_dir=PROLOG_DIR / "popper",
        training_log=REPORTS_DIR / "phase5_popper_output.txt",
        training_report=None,
    ),
    "hybrid_sequential": CoachMethod(
        key="hybrid_sequential",
        title="الگوهای ترتیبی ترکیبی",
        task_dir=PROLOG_DIR / "popper_sequential",
        training_log=REPORTS_DIR / "phase5_sequential_training_popper_output.txt",
        training_report=REPORTS_DIR / "phase5_sequential_training.json",
    ),
}

DEFAULT_COACH_METHOD = "hybrid_sequential"


def get_coach_method() -> CoachMethod:
    """Return the configured arm, defaulting to the hybrid sequential arm."""
    key = os.environ.get("TACTICAL_ILP_COACH_METHOD", DEFAULT_COACH_METHOD).strip()
    try:
        return METHODS[key]
    except KeyError:
        choices = ", ".join(sorted(METHODS))
        raise ValueError(
            f"TACTICAL_ILP_COACH_METHOD must be one of: {choices}; got {key!r}"
        ) from None


def coach_source_fingerprint(method: CoachMethod | None = None) -> str:
    """Fingerprint the exact model artifacts a coach is asked to review."""
    method = method or get_coach_method()
    paths = [
        method.task_dir / "bk.pl",
        method.task_dir / "bias.pl",
        method.task_dir / "exs.pl",
        method.task_dir / "exs_holdout.pl",
        method.training_log,
    ]
    if method.training_report is not None:
        paths.append(method.training_report)
    cv_report = REPORTS_DIR / (
        "phase5_sequential_cv.json" if method.key == "hybrid_sequential"
        else "phase5_cross_validation.json"
    )
    paths.append(cv_report)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()[:20]
