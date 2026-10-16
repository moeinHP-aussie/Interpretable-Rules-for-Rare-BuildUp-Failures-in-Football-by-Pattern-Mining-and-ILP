"""
Phase 7 -- turn an approved Prolog rule into Persian sentences, mechanically.

Nothing in here is football prose.  Football wording lives in
:mod:`concept_dictionary`; this module only (a) parses the rule body,
(b) looks each literal up, (c) joins the phrases, and (d) chooses a
strength/stability wording from the numbers with fixed, documented thresholds.

Consequence: change a rule or add one, re-run, and the sentence changes with
it.  No report text is ever edited by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tactical_ilp.reporting.concept_dictionary import Concept, lookup, split_negation

# ---------------------------------------------------------------------------
# thresholds for the wording that depends on data.  They are constants (not
# magic numbers inline) so the report can print them in its methodology box.
# ---------------------------------------------------------------------------
LIFT_STRONG = 2.0          # lift >= this on train  -> "بسیار بیشتر از حالت عادی"
LIFT_MODERATE = 1.3        # lift >= this           -> "بیشتر از حالت عادی"
STABLE_DROP_MAX = 0.25     # holdout_drop_ratio <= this -> "پایدار"
WEAKENED_DROP_MAX = 0.50   # <= this -> "ضعیف‌تر شده"; above -> "فروپاشیده"

_LITERAL = re.compile(r"\s*([a-z_0-9]+)\(S\)\s*")


@dataclass(frozen=True)
class Literal:
    name: str            # as written in the rule, e.g. "not_seq_first_time_play"
    base: str            # predicate without the not_ prefix
    negated: bool
    concept: Concept

    @property
    def phrase(self) -> str:
        if self.negated:
            if self.concept.fa_absent is None:
                # never invent a negative phrase the dictionary did not review
                return f"«{self.concept.fa_present}» برقرار نبوده است"
            return self.concept.fa_absent
        return self.concept.fa_present


def parse_body(prolog_body: str) -> list[Literal]:
    """``a(S), not_b(S)`` -> [Literal, Literal].  Fails on anything unexpected."""
    parts = [p for p in re.split(r"\)\s*,", prolog_body) if p.strip()]
    out: list[Literal] = []
    for raw in parts:
        text = raw if raw.rstrip().endswith(")") else raw + ")"
        m = _LITERAL.fullmatch(text)
        if not m:
            raise ValueError(f"cannot parse literal {raw!r} in body {prolog_body!r}")
        name = m.group(1)
        base, negated = split_negation(name)
        # 'not_' is only a negation marker when the remainder is a known
        # predicate; otherwise the name itself is the predicate.
        try:
            concept = lookup(base)
        except KeyError:
            base, negated = name, False
            concept = lookup(base)
        out.append(Literal(name, base, negated, concept))
    return out


def strength_word(lift: float) -> str:
    if lift >= LIFT_STRONG:
        return "بسیار بیشتر از حالت عادی"
    if lift >= LIFT_MODERATE:
        return "بیشتر از حالت عادی"
    if lift > 1.0:
        return "کمی بیشتر از حالت عادی"
    return "بیشتر از حالت عادی نیست"


def stability_word(drop_ratio: float | None, holdout_conf: float | None) -> str:
    if drop_ratio is None or holdout_conf is None:
        return "روی داده‌ٔ کنترل سنجیده نشده"
    if holdout_conf == 0.0:
        return "روی داده‌ٔ کنترل کاملاً فروپاشید"
    if drop_ratio <= STABLE_DROP_MAX:
        return "روی داده‌ٔ کنترل هم پایدار ماند"
    if drop_ratio <= WEAKENED_DROP_MAX:
        return "روی داده‌ٔ کنترل ضعیف‌تر شد"
    return "روی داده‌ٔ کنترل به‌شدت افت کرد"


@dataclass
class RuleSentence:
    name: str
    literals: list[Literal]
    headline: str                 # one sentence: the pattern
    evidence: str                 # one sentence: numbers in words
    caveats: list[str] = field(default_factory=list)   # proxy notes, deduplicated
    glossary_refs: list[str] = field(default_factory=list)


def _fa_num(x: float, nd: int = 0) -> str:
    s = f"{x:.{nd}f}"
    return s.translate(str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫"))


def translate(rule: dict) -> RuleSentence:
    """``rule`` is one entry of ``approved_rules()``."""
    lits = parse_body(rule["prolog_body"])
    clauses = [l.phrase for l in lits]
    headline = "الگوی مشاهده‌شده: " + "؛ سپس ".join(clauses) + "."

    tr = rule["metrics"]["train"]
    ho = rule["metrics"]["holdout"]
    drop = rule["metrics"].get("holdout_drop_ratio")
    evidence = (
        f"شواهد آماری نشان می‌دهد در دادهٔ آموزش، از {_fa_num(tr['support'])} سکانسِ دارای این الگو، "
        f"{_fa_num(tr['confidence'] * 100)}٪ به پیامد منفی رسیده‌اند؛ نرخ مشاهده‌شده "
        f"{_fa_num(tr['lift'], 2)} برابر نرخ پایه بوده است. در دادهٔ نگه‌داشته‌شده، "
        f"{stability_word(drop, ho['confidence'])} "
        f"({_fa_num(ho['support'])} سکانس، اطمینان {_fa_num(ho['confidence'] * 100)}٪)."
    )

    caveats: list[str] = []
    for l in lits:
        if l.concept.proxy_note and l.concept.proxy_note not in caveats:
            caveats.append(l.concept.proxy_note)
    refs = [l.concept.glossary_ref for l in lits]
    return RuleSentence(rule["name"], lits, headline, evidence, caveats, refs)
