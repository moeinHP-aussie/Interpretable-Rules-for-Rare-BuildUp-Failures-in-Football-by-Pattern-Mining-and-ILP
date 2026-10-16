"""
Phase 6 -- filter, evaluate, and interpret the rules Popper discovered in
phase 5.

This module implements exactly the three metrics the technical document
(section 6) requires before a rule can reach a coach:

    Support     raw count of sequences the rule's premises match
                (not a ratio -- a rule that fires 2 times is not trustworthy
                no matter how confident it looks)
    Confidence  P(negative outcome | rule's premises hold)
                = TP / (TP + FP)  on the sequences the rule matched
    Lift        Confidence / team base rate
                > 1 means the pattern is riskier than average;
                ~= 1 means the pattern is common AND healthy (the exact
                failure mode section 2.7 of the technical document warns
                about -- a frequent, well-executed pattern like a deep
                midfielder supporting a pressed defender must NOT be
                reported as dangerous just because it is frequent)

Every rule is scored twice on the current one-split diagnostic: once on the
training matches and once on matches excluded from that particular Popper fit.
Because this split has been inspected across multiple runs, it is not an
independent final estimate. Use grouped match-level cross-validation in
scripts/run_phase5_cv.py as the primary internal estimate. A rule whose
confidence collapses on the diagnostic split is flagged, not silently dropped.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tactical_ilp.config import PROLOG_DIR

POPPER_DIR = PROLOG_DIR / "popper"
BK_FILE = POPPER_DIR / "bk.pl"
TRAIN_FILE = POPPER_DIR / "exs.pl"
HOLDOUT_FILE = POPPER_DIR / "exs_holdout.pl"

MIN_SUPPORT = 3  # technical document section 6: "3 to 5 occurrences minimum"
HOLDOUT_DROP_FLAG_RATIO = 0.5  # execution plan task 6.0: flag drops > 50%


@dataclass
class RuleMetrics:
    """One split's (train or holdout) Support/Confidence/Lift for one rule."""

    split: str
    support: int          # TP + FP: total sequences the rule matched
    true_positives: int
    false_positives: int
    total_positives: int  # all negative-outcome sequences in this split
    total_negatives: int
    confidence: float
    base_rate: float
    lift: float


@dataclass
class Rule:
    """A learned (or candidate) hypothesis, plus its metrics on both splits."""

    name: str
    body: list[str]              # e.g. ["seq_dropped_between_lines", "not_seq_first_time_play", ...]
    source: str                  # "popper" | "manual_candidate" | "synthetic_test"
    train: RuleMetrics | None = None
    holdout: RuleMetrics | None = None
    holdout_drop_ratio: float | None = None
    flagged_low_support: bool = False
    flagged_holdout_drop: bool = False
    validation_status: str = "pending"     # "approved" | "rejected" | "pending"
    validation_note: str = ""
    category: str = "primary"
    candidate_metrics: dict | None = None
    pattern_metadata: dict = field(default_factory=dict)
    discovery_fold: int | str | None = None

    def as_prolog_body(self) -> str:
        return ", ".join(f"{lit}(S)" for lit in self.body)


def _run_swipl_goal(goal: str, timeout: int = 300) -> str:
    result = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-g", goal],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"swipl failed:\n{result.stderr}")
    return result.stdout


def _score_on_split(body: list[str], exs_file: Path, bk_file: Path = BK_FILE) -> RuleMetrics:
    """Support / Confidence / Lift of one rule body on one examples file."""
    if not exs_file.exists():
        raise FileNotFoundError(exs_file)

    lits = ", ".join(f"{p}(S)" for p in body)
    goal = f"""
consult('{bk_file.as_posix()}'),
consult('{exs_file.as_posix()}'),
findall(P, pos(negative_outcome(P)), Pos), length(Pos, NP),
findall(N, neg(negative_outcome(N)), Neg), length(Neg, NN),
include([S]>>({lits}), Pos, TPs), length(TPs, TP),
include([S]>>({lits}), Neg, FPs), length(FPs, FP),
format("~w ~w ~w ~w~n", [NP, NN, TP, FP]),
halt.
"""
    out = _run_swipl_goal(goal)
    line = [ln for ln in out.strip().splitlines() if ln.strip()][-1]
    n_pos, n_neg, tp, fp = (int(x) for x in line.split())

    support = tp + fp
    confidence = tp / support if support else 0.0
    base_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0
    lift = confidence / base_rate if base_rate else 0.0

    return RuleMetrics(
        split=exs_file.stem,
        support=support,
        true_positives=tp,
        false_positives=fp,
        total_positives=n_pos,
        total_negatives=n_neg,
        confidence=round(confidence, 4),
        base_rate=round(base_rate, 4),
        lift=round(lift, 4),
    )


def score_rule(
    name: str,
    body: list[str],
    source: str = "popper",
    train_file: Path = TRAIN_FILE,
    holdout_file: Path = HOLDOUT_FILE,
    bk_file: Path = BK_FILE,
    min_support: int = MIN_SUPPORT,
) -> Rule:
    """Task 6.1 + 6.0 + 6.2 combined: score a rule on both splits and flag it."""
    rule = Rule(name=name, body=body, source=source)
    rule.train = _score_on_split(body, train_file, bk_file)
    rule.holdout = _score_on_split(body, holdout_file, bk_file)

    rule.flagged_low_support = rule.train.support < min_support

    if rule.train.confidence > 0:
        drop = (rule.train.confidence - rule.holdout.confidence) / rule.train.confidence
        rule.holdout_drop_ratio = round(drop, 4)
        rule.flagged_holdout_drop = drop > HOLDOUT_DROP_FLAG_RATIO
    else:
        rule.holdout_drop_ratio = None
        rule.flagged_holdout_drop = False

    return rule


# ---------------------------------------------------------------------------
# task 6.2 -- lift correctness checks (synthetic + a known real healthy pattern)
# ---------------------------------------------------------------------------
def compute_lift(confidence: float, base_rate: float) -> float:
    """Pure function version, for the synthetic unit test (task 6.2 acceptance
    criterion a) -- does not touch Prolog at all."""
    if base_rate == 0:
        return 0.0
    return confidence / base_rate


KNOWN_HEALTHY_PATTERN = "seq_supported_action"
# Why this one: technical document section 2.7 gives "a defensive midfielder
# helping a pressed teammate" as its worked example of a frequent, healthy
# pattern that must not be flagged as dangerous. seq_supported_action (bk.pl:
# "the ball carrier had an open passing lane to a nearby teammate") is the
# closest predicate we have to that example, and it is frequent by
# previous dataset construction. Its count and lift are historical; see
# docs/phase6_evaluation_report.md for the run that produced this number.
KNOWN_HEALTHY_PATTERN_MAX_LIFT = 1.3  # execution plan task 6.2 acceptance criterion


def known_healthy_pattern_metrics(
    bk_file: Path = BK_FILE, train_file: Path = TRAIN_FILE
) -> RuleMetrics | None:
    """Task 6.2 acceptance criterion (b): a known-healthy pattern should have
    lift close to 1. Returns None if bk.pl / exs.pl are not built yet, so
    tests can skip cleanly instead of failing on a missing fixture."""
    if not bk_file.exists() or not train_file.exists():
        return None
    return _score_on_split([KNOWN_HEALTHY_PATTERN], train_file, bk_file)
