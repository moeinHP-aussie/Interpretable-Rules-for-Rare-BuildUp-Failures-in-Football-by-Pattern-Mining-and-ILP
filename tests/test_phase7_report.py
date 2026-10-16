"""Phase 7 -- the report generator's guarantees.

These tests exist because the report is meant to regenerate itself with nobody
editing prose.  Each test pins one way that promise could silently break.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tactical_ilp.evaluation.coach_validation import approved_rules
from tactical_ilp.reporting import build_report, facts as F
from tactical_ilp.reporting.concept_dictionary import (
    CONCEPTS, SHORT_LABELS, UnknownPredicateError, lookup, split_negation,
)
from tactical_ilp.reporting.rule_translator import parse_body, translate

needs_swipl = pytest.mark.skipif(shutil.which("swipl") is None, reason="swipl not installed")


# ---------------------------------------------------------------- dictionary
def test_every_concept_has_a_latin_figure_label():
    assert set(CONCEPTS) <= set(SHORT_LABELS)


def test_no_persian_text_reaches_a_figure_label():
    for lab in SHORT_LABELS.values():
        assert re.fullmatch(r"[A-Za-z0-9 \-]+", lab), lab   # matplotlib cannot shape Persian


def test_unknown_predicate_fails_loudly_never_guesses():
    with pytest.raises(UnknownPredicateError):
        lookup("seq_totally_new_predicate")


def test_every_approved_rule_predicate_is_in_the_dictionary():
    """A newly approved Popper rule with an unreviewed predicate must break THIS test."""
    for r in approved_rules():
        for lit in parse_body(r["prolog_body"]):
            assert lit.base in CONCEPTS


def test_no_phrase_claims_same_player_or_same_moment():
    """seq_* is an existential lift over the sequence; a phrase must not overclaim."""
    banned = ["همان بازیکن", "همان لحظه", "هم‌زمان"]
    for c in CONCEPTS.values():
        for text in (c.fa_present, c.fa_absent or ""):
            assert not any(b in text for b in banned), (c.predicate, text)


def test_negation_split():
    assert split_negation("not_seq_dwelt_long") == ("seq_dwelt_long", True)
    assert split_negation("fast_sequence") == ("fast_sequence", False)


# ---------------------------------------------------------------- translator
def test_translation_is_deterministic():
    r = approved_rules()[0]
    assert translate(r).headline == translate(r).headline


def test_translator_uses_the_negated_phrase_for_not_literals():
    r = next(x for x in approved_rules() if "not_seq_dwelt_long" in x["prolog_body"])
    lit = next(l for l in parse_body(r["prolog_body"]) if l.negated)
    assert lit.phrase == lookup("seq_dwelt_long").fa_absent


def test_headline_does_not_invent_a_definition_of_negative_outcome():
    """An earlier draft said 'گل‌خوردن‌نزدیک' (near-conceded goal), which the project never defined."""
    for r in approved_rules():
        assert "گل" not in translate(r).headline


# ------------------------------------------------------------------ template
def test_template_source_contains_no_hardcoded_dataset_numbers():
    """The three numbers that drifted in the old docs must never be literals here."""
    src = Path(build_report.__file__).read_text(encoding="utf-8")
    for bad in ("1066", "1,066", "1101", "143", "191", "17.92", "13.41", "۱۰۶۶", "۱۴۳", "۱۹۱"):
        assert bad not in src, f"literal {bad!r} in build_report.py"


# ----------------------------------------------------------------- generation
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if shutil.which("swipl") is None:
        pytest.skip("swipl not installed")
    facts = F.collect()
    doc, state = build_report.render(facts)
    return facts, doc, state


@needs_swipl
def test_report_is_one_self_contained_file(built):
    _, doc, _ = built
    assert "src=\"http" not in doc and "href=\"http" not in doc      # no external resource
    assert doc.count("data:image/png;base64,") >= 4


@needs_swipl
def test_report_numbers_match_the_data(built):
    facts, doc, _ = built
    fa = build_report.fa
    assert fa(facts.n_sequences) in doc
    assert fa(facts.n_negative) in doc
    assert build_report.pct(facts.negative_rate) in doc


@needs_swipl
def test_only_approved_rules_appear_as_findings(built):
    facts, doc, _ = built
    findings = doc.split("الگوهای تأییدشده")[1].split("پایداری روی داده")[0]
    for rr in facts.rejected:
        assert rr.prolog_body not in findings


@needs_swipl
def test_rejected_rule_is_listed_with_the_coach_reason(built):
    facts, doc, _ = built
    for rr in facts.rejected:
        assert rr.name in doc and rr.note[:20] in doc


@needs_swipl
def test_regeneration_is_idempotent(built):
    facts, doc, _ = built
    doc2, _ = build_report.render(F.collect())
    assert doc == doc2       # no timestamps, no shuffled examples


@needs_swipl
def test_prolog_failure_is_an_error_not_zero_matches():
    bad = {"name": "broken", "prolog_body": "no_such_predicate_xyz(S)"}
    with pytest.raises(F.PrologEvaluationError):
        F.rule_matches(bad, 1066, 143)


@needs_swipl
def test_rule_match_counts_are_deduplicated(built):
    """seq_* wrappers succeed once per satisfying action; counts must be per SEQUENCE."""
    facts, _, _ = built
    for r in facts.approved:
        m = F.rule_matches(r, facts.n_sequences, facts.n_negative)
        assert len(m.matched) == len(set(m.matched)) <= facts.n_sequences
        assert set(m.matched_negative) <= set(m.matched)


@needs_swipl
def test_location_statistic_is_relative_to_baseline(built):
    """'100 % in the defensive third' was uninformative (baseline is 99.9 %)."""
    facts, _, _ = built
    base = F.baseline_start_x()
    assert (base <= 35).mean() > 0.99          # documents WHY absolute zone shares are not reported


def test_check_flag_detects_a_stale_report(tmp_path, monkeypatch):
    monkeypatch.setattr(build_report, "REPORT_PATH", tmp_path / "r.html")
    (tmp_path / "r.html").write_text("<html>old</html>", encoding="utf-8")
    assert build_report.main(["--check"]) == 1


# ------------------------------------------------- approved-but-risky rules
def _risky_rule():
    return {"name": "x", "prolog_body": "", "flags": {"holdout_confidence_collapsed": True, "low_support": False}}


def test_collapsed_rule_gets_a_visible_caution_even_if_coach_approved():
    """Found by experiment: approving the overfit rule 2a produced a green 'approved'
    card saying 'much higher than normal (lift 3.09)' with no warning at all."""
    notes = build_report.risk_notes(_risky_rule())
    assert notes and "فروپاشید" in notes[0]


def test_healthy_rule_has_no_caution():
    ok = {"flags": {"holdout_confidence_collapsed": False, "low_support": False}}
    assert build_report.risk_notes(ok) == []


def test_low_support_alone_also_triggers_a_caution():
    r = {"flags": {"holdout_confidence_collapsed": False, "low_support": True}}
    assert len(build_report.risk_notes(r)) == 1


@needs_swipl
def test_risk_banner_appears_in_html_when_a_collapsed_rule_is_approved(monkeypatch):
    """End to end: flip the real rejected rule to approved in memory and render."""
    real = F.collect()
    import yaml
    from tactical_ilp.evaluation import coach_validation as cv

    rules = cv.read_coach_validation_file()
    for r in rules:
        if r["name"].startswith("candidate_2a"):
            r["validation_status"], r["validation_note"] = "approved", "test"
    monkeypatch.setattr(cv, "read_coach_validation_file", lambda path=cv.COACH_VALIDATION_PATH: rules)
    monkeypatch.setattr(F, "read_coach_validation_file", lambda path=cv.COACH_VALIDATION_PATH: rules)
    doc, _ = build_report.render(F.collect())
    assert 'class="risk"' in doc and "finding risky" in doc
    assert len(real.approved) == 2          # the real file was not touched
