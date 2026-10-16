"""
Phase 7 -- every number the report prints, computed here from the data.

The report template never contains a literal number.  Anything it states about
the dataset (how many sequences, what share are negative, which rules were
rejected and why) comes out of :func:`collect` and nothing else.  That is what
prevents the failure this project already had once: a hand-typed label-count
table drifted from the data after the outcome threshold changed. The current
definition document therefore describes the rule, while this module computes
all current dataset counts directly from the fresh run.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tactical_ilp import config
from tactical_ilp.evaluation.coach_validation import (
    COACH_VALIDATION_PATH,
    read_coach_validation_file,
    validate_coach_file,
)
from tactical_ilp.reporting.rule_translator import RuleSentence, translate
from tactical_ilp.reporting.method_config import coach_source_fingerprint, get_coach_method

PROCESSED = config.PROCESSED_DIR
ACTIVE_METHOD = get_coach_method()
POPPER_DIR = Path(ACTIVE_METHOD.task_dir)


@dataclass
class RuleMatch:
    """Which sequences one rule fires on -- the raw material for the pitch plots."""
    name: str
    matched: list[str]                 # sequence ids, whole dataset
    matched_negative: list[str]
    n_total: int
    n_negative_total: int

    @property
    def confidence(self) -> float:
        return len(self.matched_negative) / len(self.matched) if self.matched else 0.0

    @property
    def base_rate(self) -> float:
        return self.n_negative_total / self.n_total

    @property
    def lift(self) -> float:
        return self.confidence / self.base_rate if self.base_rate else 0.0


@dataclass
class RejectedRule:
    name: str
    prolog_body: str
    note: str
    train_conf: float
    holdout_conf: float
    train_support: int
    holdout_support: int


@dataclass
class ReportFacts:
    # dataset
    n_sequences: int
    n_matches: int
    n_negative: int
    n_non_negative: int
    negative_rate: float
    ilp_pos: int
    ilp_neg: int
    holdout_pos: int
    holdout_neg: int
    # rules
    approved: list[dict]
    rejected: list[RejectedRule]
    pending: list[str]
    sentences: dict[str, RuleSentence]
    training_match_ids: set[int] = field(default_factory=set)
    holdout_match_ids: set[int] = field(default_factory=set)
    negative_reason_counts: dict[str, int] = field(default_factory=dict)
    matches: dict[str, RuleMatch] = field(default_factory=dict)
    # provenance
    fingerprint: str = ""
    validation_ok: bool = True
    validation_problems: list[str] = field(default_factory=list)
    config_params: dict[str, float] = field(default_factory=dict)
    method_key: str = "hybrid_sequential"
    method_title: str = "الگوهای ترتیبی ترکیبی"
    model_fingerprint: str = ""


def _count_examples(path: Path, kind: str) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith(kind + "("))


def _example_match_ids(path: Path, labeled: pd.DataFrame) -> set[int]:
    """Recover match-grouped train/holdout membership from the generated exs."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    sequence_ids = set(re.findall(r"(?:pos|neg)\(negative_outcome\(([^)]+)\)\)\.", text))
    sequence_matches = labeled.assign(sequence_id=labeled.sequence_id.astype(str))
    selected = sequence_matches[sequence_matches.sequence_id.isin(sequence_ids)]
    return set(selected.match_id.astype(int))


def fingerprint() -> str:
    """Hash of everything that legitimately changes the report's content.

    The generator stores this next to the report; if it is unchanged there is
    nothing new to say, and ``--check`` can tell CI the report is up to date.
    """
    h = hashlib.sha256()
    inputs = [
        COACH_VALIDATION_PATH, PROCESSED / "labeled_sequences.parquet",
        POPPER_DIR / "bk.pl", POPPER_DIR / "exs.pl", POPPER_DIR / "exs_holdout.pl",
    ]
    if ACTIVE_METHOD.training_report is not None:
        inputs.append(ACTIVE_METHOD.training_report)
    for p in inputs:
        h.update(str(p).encode("utf-8"))
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def collect() -> ReportFacts:
    lab = pd.read_parquet(PROCESSED / "labeled_sequences.parquet")
    training_match_ids = _example_match_ids(POPPER_DIR / "exs.pl", lab)
    holdout_match_ids = _example_match_ids(POPPER_DIR / "exs_holdout.pl", lab)
    if training_match_ids & holdout_match_ids:
        raise ValueError("Phase-5 example files contain matches in both train and holdout splits")
    n_neg = int((lab["label"] == "negative").sum())
    reason_counts = {
        str(reason): int(count)
        for reason, count in lab.loc[lab.label == "negative", "negative_reason"]
        .value_counts(dropna=False)
        .items()
    }
    ok, problems = validate_coach_file(
        expected_method=ACTIVE_METHOD.key,
        expected_fingerprint=coach_source_fingerprint(ACTIVE_METHOD),
    )
    stale_source = [
        p for p in problems
        if p.startswith("coach file source_method is")
        or p.startswith("coach file source_fingerprint")
    ]
    if stale_source:
        raise ValueError(
            "coach validation file does not match the active Phase-5 model artifacts; "
            "run scripts/run_phase6.py for the active method before building the report: "
            + "; ".join(stale_source)
        )

    approved, rejected, pending = [], [], []
    for r in read_coach_validation_file():
        status = r.get("validation_status")
        if status == "approved":
            approved.append(r)
        elif status == "rejected":
            m = r["metrics"]
            rejected.append(RejectedRule(
                r["name"], r["prolog_body"], (r.get("validation_note") or "").strip(),
                m["train"]["confidence"], m["holdout"]["confidence"],
                m["train"]["support"], m["holdout"]["support"]))
        else:
            pending.append(r["name"])

    return ReportFacts(
        n_sequences=len(lab),
        n_matches=int(lab["match_id"].nunique()),
        n_negative=n_neg,
        n_non_negative=len(lab) - n_neg,
        negative_rate=n_neg / len(lab),
        negative_reason_counts=reason_counts,
        ilp_pos=_count_examples(POPPER_DIR / "exs.pl", "pos"),
        ilp_neg=_count_examples(POPPER_DIR / "exs.pl", "neg"),
        holdout_pos=_count_examples(POPPER_DIR / "exs_holdout.pl", "pos"),
        holdout_neg=_count_examples(POPPER_DIR / "exs_holdout.pl", "neg"),
        approved=approved,
        rejected=rejected,
        pending=pending,
        sentences={r["name"]: translate(r) for r in approved},
        training_match_ids=training_match_ids,
        holdout_match_ids=holdout_match_ids,
        fingerprint=fingerprint(),
        validation_ok=ok,
        validation_problems=problems,
        config_params={
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
        },
        method_key=ACTIVE_METHOD.key,
        method_title=ACTIVE_METHOD.title,
        model_fingerprint=coach_source_fingerprint(ACTIVE_METHOD),
    )


# ---------------------------------------------------------------------------
# which sequences does a rule fire on?  (SWI-Prolog, same BK the rules came from)
# ---------------------------------------------------------------------------
_PROLOG_TMPL = r"""
:- consult('{bk}').
:- consult('{facts_all}').
rule(S) :- {body}.
:- forall(( sequence(S), rule(S) ), format("~w~n", [S])).
:- halt.
"""


@lru_cache(maxsize=1)
def _sequence_labels() -> dict[str, str]:
    labeled = pd.read_parquet(PROCESSED / "labeled_sequences.parquet")
    return dict(zip(labeled.sequence_id.astype(str), labeled.label.astype(str)))


class PrologEvaluationError(RuntimeError):
    """SWI-Prolog failed or printed nothing usable.  Never treated as '0 matches'."""


def rule_matches(rule: dict, n_total: int, n_neg_total: int) -> RuleMatch:
    """Evaluate a rule over ALL sequences (train + holdout) with SWI-Prolog.

    Deduplicated in Python: the `seq_*` wrappers can succeed several times per
    sequence (one per satisfying action), which inflated an earlier count to 1101
    'matches' out of 1066 sequences.

    A failed Prolog run raises instead of returning an empty match list: a rule
    silently reported as matching 0 sequences would look like a real result.
    """
    import tempfile

    prog = _PROLOG_TMPL.format(
        bk=(POPPER_DIR / "bk.pl").as_posix(),
        facts_all=(config.PROLOG_DIR / "facts_all.pl").as_posix(),
        body=rule["prolog_body"],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as fh:
        fh.write(prog)
        tmp = fh.name
    try:
        out = subprocess.run(["swipl", "-q", tmp], capture_output=True, text=True, timeout=300)
    finally:
        Path(tmp).unlink(missing_ok=True)

    real_errors = [l for l in out.stderr.splitlines() if l.startswith("ERROR")]
    if out.returncode != 0 or real_errors:
        raise PrologEvaluationError(
            f"swipl failed for rule {rule['name']!r} (rc={out.returncode}): "
            + " | ".join(real_errors[:3] or [out.stderr[:200]])
        )

    labels: dict[str, str] = {}
    for line in out.stdout.splitlines():
        sequence_id = line.strip()
        if sequence_id.startswith("s_"):
            labels[sequence_id] = ""
    if not labels:
        raise PrologEvaluationError(
            f"rule {rule['name']!r} matched no sequence at all -- expected for a broken "
            f"evaluation, not for an approved rule (it had support in train)"
        )
    label_by_id = _sequence_labels()
    # The BK can contain sequences without outcome labels. Report metrics only
    # over the project's labelled population, matching the original intent of
    # the old sequence_label/2 join.
    matched = sorted(sid for sid in labels if sid in label_by_id)
    if not matched:
        raise PrologEvaluationError(
            f"rule {rule['name']!r} matched no labelled sequence"
        )
    neg = [sid for sid in matched if label_by_id[sid] == "negative"]
    return RuleMatch(rule["name"], matched, neg, n_total, n_neg_total)


# ---------------------------------------------------------------------------
# witnesses: WHICH action satisfied each literal (drives the pitch plots and the
# "where on the pitch does this happen" statistics).
#
# Only literals whose bk.pl definition is action-level have a witness; the
# sequence-level ones (fast_sequence, tempo) do not, and are drawn as a label.
# Negated literals have no witness by construction (they assert absence).
# ---------------------------------------------------------------------------
ACTION_LEVEL = {
    "seq_dropped_between_lines": "dropped_between_lines",
    "seq_dead_end_support": "dead_end_support",
    "seq_dwelt_long": "dwelt_long",
    "seq_diagonal_pass": "diagonal_pass",
    "seq_counter_attack_potential": "counter_attack_potential",
    "seq_first_time_play": "first_time_play",
    "seq_line_pass": "line_pass",
}


def literal_witnesses(base_predicate: str, sequence_ids: list[str]) -> pd.DataFrame:
    """Actions (sequence_id, match_id, action_id, order) that satisfy one literal.

    Returns an empty frame for sequence-level predicates.  Raises if Prolog
    fails -- same rule as :func:`rule_matches`.
    """
    import tempfile

    action_pred = ACTION_LEVEL.get(base_predicate)
    cols = ["sequence_id", "match_id", "action_id", "order"]
    if action_pred is None or not sequence_ids:
        return pd.DataFrame(columns=cols)

    ids = ",".join(f"'{s}'" for s in sequence_ids)
    prog = (
        f":- consult('{(POPPER_DIR / 'bk.pl').as_posix()}').\n"
        f"want(S) :- member(S, [{ids}]).\n"
        f":- forall(( want(S), in_sequence(A,S), action_order(A,N), {action_pred}(A) ),\n"
        f"          format(\"~w ~w ~w~n\", [S,A,N])).\n"
        f":- halt.\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as fh:
        fh.write(prog)
        tmp = fh.name
    try:
        out = subprocess.run(["swipl", "-q", tmp], capture_output=True, text=True, timeout=300)
    finally:
        Path(tmp).unlink(missing_ok=True)
    errs = [l for l in out.stderr.splitlines() if l.startswith("ERROR")]
    if out.returncode != 0 or errs:
        raise PrologEvaluationError(f"witness query for {base_predicate} failed: {errs[:3]}")

    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith("s_"):
            _, mid, aid = parts[1].split("_")
            rows.append((parts[0], int(mid), int(aid), int(parts[2])))
    return pd.DataFrame(rows, columns=cols).drop_duplicates().reset_index(drop=True)


def attach_positions(w: pd.DataFrame) -> pd.DataFrame:
    """Add start/end coordinates (SPADL metres, Barcelona attacks +x)."""
    if w.empty:
        return w.assign(start_x=[], start_y=[], end_x=[], end_y=[])
    sp = pd.read_parquet(
        PROCESSED / "spadl_with_360_all_matches.parquet",
        columns=["match_id", "action_id", "start_x", "start_y", "end_x", "end_y",
                 "type_name", "result_name", "player_id"],
    )
    return w.merge(sp, on=["match_id", "action_id"], how="left")


def baseline_start_x() -> pd.Series:
    """start_x of EVERY action in EVERY build-up sequence (the comparison population).

    A location statistic about one rule's actions is only meaningful against
    this.  Measured on the real data, 99.9 % of all sequence actions start in our
    defensive third, so "100 % of this rule's actions are in the defensive third"
    carries no information -- an earlier draft of this report stated exactly that.
    """
    sp = pd.read_parquet(PROCESSED / "spadl_with_360_all_matches.parquet",
                         columns=["match_id", "action_id", "start_x"])
    sa = pd.read_parquet(PROCESSED / "sequence_actions.parquet")
    return sa.merge(sp, on=["match_id", "action_id"])["start_x"]


def location_vs_baseline(w: pd.DataFrame, base: pd.Series) -> dict | None:
    """Median start_x of a rule's witnessing actions vs. the median of all actions.

    Returns None when there are no actions.  ``delta`` > 0 means the rule's
    actions are further up the pitch than a typical build-up action.
    """
    if w.empty:
        return None
    med, bmed = float(w["start_x"].median()), float(base.median())
    return {"n": len(w), "median": med, "baseline_median": bmed, "delta": med - bmed}


def sequence_actions_with_positions(sequence_ids: list[str]) -> pd.DataFrame:
    """Actions with pitch coordinates, event/result labels, and actor identity."""
    sa = pd.read_parquet(PROCESSED / "sequence_actions.parquet")
    sa = sa[sa["sequence_id"].isin(sequence_ids)].rename(columns={"order_in_sequence": "order"})
    actions = attach_positions(sa[["sequence_id", "match_id", "action_id", "order"]])
    players_path = PROCESSED / "players_reference.parquet"
    if players_path.exists():
        players = pd.read_parquet(players_path, columns=["player_id", "display_name"])
        players = players.drop_duplicates("player_id")
        actions = actions.merge(players, on="player_id", how="left")
    else:
        actions["display_name"] = None
    return actions


def pick_examples(
    actions: pd.DataFrame,
    labels: dict[str, str],
    witness_orders_by_seq: dict[str, dict[str, set[int]]],
    want_negative: bool,
    k: int = 1,
    *,
    expected_witness_literals: int = 1,
    exclude_sequence_ids: set[str] | None = None,
) -> list[str]:
    """Choose high-evidence, readable examples while preferring unused sequences.

    Candidates first rank by the fraction and number of rule literals with an
    action witness, then by cross-literal overlap and spatial spread per action.
    Excluded sequences are used only as a fallback when there are not enough
    unused candidates. Ties break on sequence id for deterministic reports.
    """
    rows = []
    excluded = exclude_sequence_ids or set()
    for sid, grp in actions.groupby("sequence_id"):
        if (labels.get(sid) == "negative") != want_negative:
            continue
        spread = float(np.hypot(grp.start_x.max() - grp.start_x.min(),
                                grp.start_y.max() - grp.start_y.min()))
        witnesses = [v for v in witness_orders_by_seq.get(sid, {}).values() if v]
        n_lits = len(witnesses)
        coverage = n_lits / max(expected_witness_literals, 1)
        shared_orders = len(set.intersection(*witnesses)) if len(witnesses) > 1 else 0
        action_count = max(len(grp), 1)
        readability = spread / np.sqrt(action_count)
        match_id = grp["match_id"].iloc[0] if "match_id" in grp else None
        score = (coverage, n_lits, shared_orders, readability)
        rows.append((score, sid, match_id, sid in excluded))
    rows.sort(key=lambda row: (row[3], tuple(-part for part in row[0]), row[1]))
    picked = []
    used_matches = set()
    for _, sid, match_id, _ in rows:
        if match_id not in used_matches:
            picked.append(sid)
            used_matches.add(match_id)
            if len(picked) >= k:
                return picked
    for _, sid, _, _ in rows:
        if sid not in picked:
            picked.append(sid)
            if len(picked) >= k:
                break
    return picked
