"""
Phase 6, task 6.3 -- Coach Validation.

The technical document is explicit that this step is "غیرقابل‌حذف" (the one
metric that cannot be removed): even a rule with excellent Support,
Confidence and Lift must be read and approved by a human who understands
football before it reaches a report. Statistics can confirm a pattern is
common and correlated with bad outcomes; they cannot confirm it is a
*real, actionable, football* reason, as opposed to a coincidence of how the
2020/21 season happened to go, or an artefact of one of the many proxy
definitions documented in docs/tactical_concepts_glossary.md and
docs/phase4_concept_disposition.md (e.g. every "which opponent" predicate is
positional, not identity-based, because 360 frames carry no player_id for
non-actor players).

This module does not decide anything. It:
  1. writes a YAML file, one entry per rule, with every computed metric next
     to empty approved/rejected/note fields for the coach to fill in
  2. reads that file back and validates it (task 6.3's acceptance test)

The file is meant to be opened, read, and edited by hand -- not filled in
programmatically. A rule with validation_status still "pending" after this
step has NOT been coach-validated and must not appear in the final report
(phase 7).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tactical_ilp.config import REPORTS_DIR
from tactical_ilp.evaluation.rule_metrics import Rule
from tactical_ilp.reporting.rule_translator import parse_body

COACH_VALIDATION_PATH = REPORTS_DIR / "phase6_coach_validation.yaml"

VALID_STATUSES = {"approved", "rejected", "pending"}


def _rule_to_dict(rule: Rule) -> dict:
    phrases = [literal.phrase for literal in parse_body(rule.as_prolog_body())]
    human_readable = "در این توالی: " + "؛ سپس ".join(phrases) + "."
    return {
        "name": rule.name,
        "source": rule.source,
        "category": rule.category,
        "prolog_body": rule.as_prolog_body(),
        "candidate_metrics": rule.candidate_metrics,
        "pattern_metadata": rule.pattern_metadata,
        "discovery_fold": rule.discovery_fold,
        "human_readable": human_readable,
        "metrics": {
            "train": {
                "support": rule.train.support if rule.train else None,
                "confidence": rule.train.confidence if rule.train else None,
                "lift": rule.train.lift if rule.train else None,
            },
            "holdout": {
                "support": rule.holdout.support if rule.holdout else None,
                "confidence": rule.holdout.confidence if rule.holdout else None,
                "lift": rule.holdout.lift if rule.holdout else None,
            },
            "holdout_drop_ratio": rule.holdout_drop_ratio,
        },
        "flags": {
            "low_support": rule.flagged_low_support,
            "holdout_confidence_collapsed": rule.flagged_holdout_drop,
        },
        # the only fields a human should actually edit:
        "validation_status": "pending",
        "validation_note": "",
    }


def write_coach_validation_file(
    rules: list[Rule],
    out_path: Path = COACH_VALIDATION_PATH,
    source_method: str = "hybrid_sequential",
    source_fingerprint: str | None = None,
) -> Path:
    """Task 6.3: produce the file a coach reads and edits by hand."""
    doc = {
        "source_method": source_method,
        "source_fingerprint": source_fingerprint,
        "instructions": (
            "برای هر قانون: validation_status را به approved یا rejected تغییر دهید. "
            "نوشتن validation_note اختیاری است. "
            "قانون‌هایی که flags.holdout_confidence_collapsed=true دارند احتمالاً باید "
            "rejected شوند مگر دلیل قانع‌کننده‌ای برای نگه‌داشتنشان داشته باشید."
        ),
        "candidate_selection_warning": (
            "قواعد با category=intermediate_candidate از نامزدهای میان‌مرحله‌ای CV انتخاب شده‌اند. "
            "کاندیداهای الگوی فولد نیز فقط در صورت وجود همتای معنایی دقیق در واژگان مدل نهایی وارد می‌شوند. "
            "امتیازهای holdout/test گزارش CV برای رتبه‌بندی استفاده شده‌اند و selection-biased هستند، نه برآورد مستقل؛ "
            "امتیاز Phase 6 نیز به‌تنهایی جای ارزیابی روی مسابقات تازه را نمی‌گیرد."
            if any(r.category == "intermediate_candidate" for r in rules) else None
        ),
        "candidate_selection_warning_en": (
            "Intermediate candidates are ranked using reported CV holdout/test evidence; this selection is biased and is not an independent estimate. Fold-local patterns are included only when their ordered signature matches the final vocabulary. Phase 6 scores are for review and do not replace evaluation on new matches."
            if any(r.category == "intermediate_candidate" for r in rules) else None
        ),
        "rules": [_rule_to_dict(r) for r in rules],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False, width=100)
    return out_path


def read_coach_validation_file(path: Path = COACH_VALIDATION_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run scripts/run_phase6.py first, then edit it by hand"
        )
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc.get("rules", [])


def validate_coach_file(
    path: Path = COACH_VALIDATION_PATH,
    expected_method: str | None = None,
    expected_fingerprint: str | None = None,
) -> tuple[bool, list[str]]:
    """Task 6.3 acceptance test, as a reusable function (also used directly
    by tests/test_phase6_evaluation.py)."""
    problems: list[str] = []
    try:
        rules = read_coach_validation_file(path)
    except FileNotFoundError as exc:
        return False, [str(exc)]

    if not rules:
        return False, ["no rules found in the coach validation file"]

    if expected_method is not None:
        with path.open(encoding="utf-8") as fh:
            source_doc = yaml.safe_load(fh) or {}
        source_method = source_doc.get("source_method")
        if source_method != expected_method:
            problems.append(
                f"coach file source_method is {source_method!r}; expected {expected_method!r}"
            )
        source_fingerprint = source_doc.get("source_fingerprint")
        if expected_fingerprint is not None and source_fingerprint != expected_fingerprint:
            problems.append(
                "coach file source_fingerprint does not match the current model artifacts"
            )

    for r in rules:
        status = r.get("validation_status")
        if status not in VALID_STATUSES:
            problems.append(f"{r.get('name')}: invalid validation_status {status!r}")
        if status == "pending":
            problems.append(f"{r.get('name')}: still pending -- not yet reviewed by a coach")
    return (len(problems) == 0), problems


def approved_rules(path: Path = COACH_VALIDATION_PATH) -> list[dict]:
    """Only the rules a human has actually signed off on -- this is the input
    phase 7 (final report) is allowed to read from."""
    return [r for r in read_coach_validation_file(path) if r.get("validation_status") == "approved"]
