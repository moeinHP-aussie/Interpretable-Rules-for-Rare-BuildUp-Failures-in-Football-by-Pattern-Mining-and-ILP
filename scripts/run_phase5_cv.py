"""Grouped match-level cross-validation for Popper rule discovery.

Each fold trains Popper only on the context-matched examples from its training
matches, then evaluates the whole learned program on *all labelled sequences*
from unseen matches. This keeps the natural outcome prevalence in evaluation
and measures the discovery procedure rather than rescoring a hand-picked rule.

Run after phases 1-5 have produced processed parquet files, phase-4 Prolog facts,
and a Popper checkout:

    POPPER_DIR=/content/Popper uv run python scripts/run_phase5_cv.py --folds 3 --timeout 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import PROLOG_DIR, REPORTS_DIR  # noqa: E402
from tactical_ilp.prolog.legacy_43_inputs import build_legacy_43_bias  # noqa: E402
from tactical_ilp.prolog.phase5_vocabulary import POPPER_MAX_CLAUSES, build_bk_wrapper  # noqa: E402
from tactical_ilp.reporting.method_config import METHODS, coach_source_fingerprint  # noqa: E402
from phase5_cv_common import (  # noqa: E402
    COMPARISON_PROTOCOL_REVISION,
    EVALUABLE_LABELS,
    _build_intermediate_candidate_report,
    _evaluate_hypothesis_snapshots,
    _write_examples,
    holdout_population,
)

PROCESSED_DIR = PROLOG_DIR.parent / "processed"
MATCHED_PATH = PROCESSED_DIR / "ilp_examples.parquet"
LABELLED_PATH = PROCESSED_DIR / "labeled_sequences.parquet"
POPPER_DIR = PROLOG_DIR / "popper"
# Parent for the fold task directories created below. Fold tasks stay beside
# the existing Prolog data so the CV run can reuse the cached facts.
CV_DIR = PROLOG_DIR
REPORT_PATH = REPORTS_DIR / "phase5_cross_validation.json"
LOG_DIR = REPORTS_DIR / "phase5_cross_validation"
COACH_VALIDATION_PATH = REPORTS_DIR / "phase6_coach_validation.yaml"
MIN_CV_FOLDS_FOR_REPORT = 2
MIN_POSITIVE_LIFT_FOLDS_FOR_REPORT = 2
MIN_CV_LIFT_FOR_REPORT = 1.0
MIN_AUTO_APPROVE_LIFT = 1.25
MIN_AUTO_APPROVE_OOF_SUPPORT = 20
MIN_PENDING_OOF_SUPPORT = 8
MIN_MATCHES_FOR_EVIDENCE = 2
DEFAULT_CV_TIMEOUT_SECONDS = 600


def _balanced_group_folds(
    sampled: pd.DataFrame, all_matches: list[str], n_folds: int, seed: int
) -> list[set[str]]:
    """Greedily balance positive/negative counts while keeping matches intact."""
    counts: dict[str, tuple[int, int]] = {}
    for match_id, group in sampled.groupby("match_id"):
        counts[str(match_id)] = (
            int((group.label == "negative").sum()),
            int((group.label == "non_negative").sum()),
        )
    # A match can have no sampled negative examples and still belongs in the
    # out-of-fold test population. Keep every labelled match in the partition.
    for match_id in all_matches:
        counts.setdefault(str(match_id), (0, 0))
    if len(counts) < n_folds:
        raise ValueError(f"need at least {n_folds} distinct matches; found {len(counts)}")

    rng = random.Random(seed)
    tie_order = list(counts)
    rng.shuffle(tie_order)
    rank = {match_id: i for i, match_id in enumerate(tie_order)}
    ordered = sorted(
        counts,
        key=lambda m: (-(counts[m][0] + counts[m][1]), rank[m]),
    )

    folds: list[set[str]] = [set() for _ in range(n_folds)]
    totals = [[0, 0, 0] for _ in range(n_folds)]  # positives, negatives, matches
    target_pos = sum(p for p, _ in counts.values()) / n_folds
    target_neg = sum(n for _, n in counts.values()) / n_folds
    target_matches = len(counts) / n_folds
    match_limit = int(target_matches + 0.999999)

    for match_id in ordered:
        pos, neg = counts[match_id]
        candidates = range(n_folds)
        # Put at least one group in each fold before optimizing the balance.
        empty = [i for i in candidates if not folds[i]]
        if empty:
            chosen = empty[0]
        else:
            under_limit = [i for i in candidates if totals[i][2] < match_limit]
            if under_limit:
                candidates = under_limit
            def score(i: int) -> tuple[float, int]:
                p, n, m = totals[i]
                balance = ((p + pos - target_pos) / max(1.0, target_pos)) ** 2
                balance += ((n + neg - target_neg) / max(1.0, target_neg)) ** 2
                balance += 2.0 * ((m + 1 - target_matches) / max(1.0, target_matches)) ** 2
                return balance, i

            chosen = min(candidates, key=score)
        folds[chosen].add(match_id)
        totals[chosen][0] += pos
        totals[chosen][1] += neg
        totals[chosen][2] += 1

    if any(not fold for fold in folds):
        raise RuntimeError("fold builder produced an empty fold")
    return folds


def _parse_solution(output: str) -> tuple[list[list[str]], bool]:
    if re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?NO SOLUTION\s*$", output):
        return [], False
    marker = output.rfind("SOLUTION")
    provisional = marker < 0
    if provisional:
        # Under the hard per-fold budget, keep the latest complete incumbent
        # instead of discarding an otherwise usable fold.
        marker = output.rfind("New best hypothesis:")
        if marker < 0:
            raise RuntimeError("Popper printed neither SOLUTION nor a provisional hypothesis")
    final = output[marker:]
    clauses: list[list[str]] = []
    pattern = re.compile(r"negative_outcome\(\w+\)\s*:-\s*([^\n.]+)\.")
    for match in pattern.finditer(final):
        body = [part.strip().split("(", 1)[0] for part in match.group(1).split(",")]
        if body and all(re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in body):
            clauses.append(body)
    if not clauses:
        raise RuntimeError("Popper printed SOLUTION but no parseable negative_outcome clause")
    return clauses, provisional


def _run_popper(task_dir: Path, popper_home: Path, timeout: int) -> tuple[str, bool]:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is not on PATH")
    worker = ROOT / "scripts" / "run_phase5.py"
    if not worker.exists():
        raise FileNotFoundError(f"bounded Popper worker missing: {worker}")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["POPPER_DIR"] = str(popper_home.resolve())
    command = [
        uv, "run", "--project", str(popper_home), "python", "-u", str(worker),
        "--_popper-worker", str(task_dir), "--noisy", "--timeout", str(timeout),
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def forward_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=forward_output, name="popper-output-reader", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout + 30
    captured: list[str] = []
    hard_timed_out = False
    stream_closed = False

    while not stream_closed:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            hard_timed_out = True
            break
        try:
            line = lines.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if line is None:
            stream_closed = True
            break
        captured.append(line)
        print(line, end="", flush=True)

    if hard_timed_out:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()
        reader.join(timeout=5)
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                captured.append(line)
        note = f"CV_FOLD_HARD_TIMEOUT after {timeout + 30}s; using latest incumbent if available.\n"
        captured.append(note)
        print(note, end="", flush=True)
    else:
        proc.wait()

    output = "".join(captured)
    expected_no_solution = bool(re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?NO SOLUTION\s*$", output))
    complete_solution = bool(
        re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?\*+\s*SOLUTION\s*\*+\s*$", output)
        and re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?\*{10,}\s*$", output)
    )
    if proc.returncode != 0 and not hard_timed_out and not expected_no_solution and not complete_solution:
        raise RuntimeError(f"Popper failed in {task_dir.name} (rc={proc.returncode}):\n{output[-8000:]}")
    if proc.returncode != 0 and complete_solution:
        print(f"Popper worker exited with status {proc.returncode} after emitting a complete SOLUTION; keeping that result.", flush=True)
    return output, hard_timed_out


def _filter_coach_candidates(report: dict) -> dict:
    """Separate replicated rules from rare candidates needing more exposure.

    Rules supported by at least two completed folds and sufficiently strong
    out-of-fold metrics are pre-approved. Candidates with too few or no
    out-of-fold firings remain pending; zero firings means unobserved, not
    failed. Only rules with enough out-of-fold exposure and poor lift are
    classified as contradicted.
    """
    if not COACH_VALIDATION_PATH.exists():
        report["coach_rule_screen"] = {
            "status": "no_phase6_coach_file",
            "message": "Run phase 6 first; no coach candidates were available to screen.",
            "excluded_rules": [],
        }
        return report

    import yaml
    from tactical_ilp.reporting.concept_dictionary import lookup, split_negation

    with COACH_VALIDATION_PATH.open(encoding="utf-8") as fh:
        coach_doc = yaml.safe_load(fh) or {}
    baseline = METHODS["full_tactical"]
    expected_fingerprint = coach_source_fingerprint(baseline)
    if (
        coach_doc.get("source_method") != baseline.key
        or coach_doc.get("source_fingerprint") != expected_fingerprint
    ):
        report["coach_rule_screen"] = {
            "status": "not_applied_method_or_source_mismatch",
            "expected_method": baseline.key,
            "coach_source_method": coach_doc.get("source_method"),
            "message": (
                "This CV runner screens only a current full_tactical coach form. "
                "The coach YAML was left unchanged because it belongs to another "
                "method or model artifact set."
            ),
            "excluded_rules": [],
        }
        return report
    rules = coach_doc.get("rules", [])
    stability = {item["canonical_body"]: item for item in report["rule_stability"]}
    known_bodies = {
        ",".join(sorted(
            part.strip().split("(", 1)[0]
            for part in rule.get("prolog_body", "").split(",")
            if part.strip()
        ))
        for rule in rules
    }
    train_examples = POPPER_DIR / "exs.pl"
    holdout_examples = POPPER_DIR / "exs_holdout.pl"
    if not train_examples.exists() or not holdout_examples.exists():
        raise FileNotFoundError("phase-5 exs.pl and exs_holdout.pl are required to build coach metrics")

    # Repeated grouped-CV clauses must be eligible for the coach form even if
    # the full-data Popper run did not select them as its final hypothesis.
    # Existing phase-6 entries are reused; new CV clauses receive the metric
    # schema consumed by phase 7 plus their grouped-CV evidence.
    cv_new_index = 1
    for cv_rule in report["rule_stability"]:
        canonical = cv_rule["canonical_body"]
        if canonical in known_bodies:
            continue
        body = [part.strip() for part in cv_rule["display_body"].split(",") if part.strip()]
        train_score = _score_program([body], POPPER_DIR / "bk.pl", train_examples)
        holdout_score = _score_program([body], POPPER_DIR / "bk.pl", holdout_examples)
        train_conf = train_score["precision"]
        holdout_conf = holdout_score["precision"]
        drop = ((train_conf - holdout_conf) / train_conf) if train_conf else None
        rules.append({
            "name": f"cv_discovered_{cv_new_index}",
            "source": "popper_match_grouped_cv",
            "prolog_body": ", ".join(f"{predicate}(S)" for predicate in body),
            "human_readable": None,
            "metrics": {
                "train": {
                    "support": train_score["support"],
                    "confidence": train_conf,
                    "lift": train_score["lift"],
                },
                "holdout": {
                    "support": holdout_score["support"],
                    "confidence": holdout_conf,
                    "lift": holdout_score["lift"],
                },
                "holdout_drop_ratio": round(drop, 4) if drop is not None else None,
            },
            "flags": {
                "low_support": train_score["support"] < 3,
                "holdout_confidence_collapsed": drop is not None and drop > 0.5,
            },
            "validation_status": "pending",
            "validation_note": "",
        })
        known_bodies.add(canonical)
        cv_new_index += 1
    positive_lift_folds: dict[str, set[int]] = defaultdict(set)
    complete_folds: dict[str, set[int]] = defaultdict(set)
    complete_clause_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {key: 0 for key in ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")}
    )
    complete_clause_test_matches: dict[str, set[str]] = defaultdict(set)
    complete_clause_firing_match_counts: dict[str, int] = defaultdict(int)
    for fold_result in report["fold_results"]:
        fold_id = int(fold_result["fold"])
        complete = not fold_result["hard_timed_out"] and not fold_result["used_provisional_hypothesis"]
        for clause in fold_result["clauses"]:
            canonical = clause["canonical"]
            if complete:
                complete_folds[canonical].add(fold_id)
                for key in complete_clause_counts[canonical]:
                    complete_clause_counts[canonical][key] += int(clause["test_metrics"][key])
                complete_clause_test_matches[canonical].update(map(str, fold_result["test_matches"]))
                complete_clause_firing_match_counts[canonical] += int(
                    clause["test_metrics"].get("triggered_match_count", 0)
                )
            if complete and clause["test_metrics"]["lift"] > MIN_CV_LIFT_FOR_REPORT:
                positive_lift_folds[canonical].add(fold_id)
    coach_rules = []
    excluded = []
    auto_approved = 0
    pending_candidates = 0
    manual_decisions = 0

    for rule in rules:
        body = [part.strip() for part in rule.get("prolog_body", "").split(",") if part.strip()]
        canonical = ",".join(sorted(body))
        reasons = []

        # The coach form cannot safely pre-approve a predicate without a
        # reviewed football description; phase 7 must not invent wording.
        unreviewed = []
        for literal in body:
            predicate = literal.split("(", 1)[0]
            base, _ = split_negation(predicate)
            try:
                lookup(base)
            except KeyError:
                unreviewed.append(base)
        if unreviewed:
            reasons.extend(f"no reviewed football wording for {name}" for name in sorted(set(unreviewed)))
            excluded.append({"rule": rule, "classification": "insufficient_evidence", "reasons": reasons})
            continue

        auto_note_prefix = "تأیید پیش‌فرض خودکار:"
        still_automatic = (
            rule.get("validation_origin") == "automated_match_cv"
            and rule.get("validation_status") == "approved"
            and str(rule.get("validation_note", "")).startswith(auto_note_prefix)
        )
        manually_decided = rule.get("validation_status") in {"approved", "rejected"} and not still_automatic
        if manually_decided:
            # A coach's explicit prior decision remains authoritative.
            if rule.get("validation_status") == "approved":
                coach_rules.append(rule)
                manual_decisions += 1
            else:
                excluded.append({
                    "rule": rule,
                    "classification": "coach_rejected",
                    "reasons": ["rule was explicitly rejected by the coach"],
                })
            continue

        full_found = len(complete_folds.get(canonical, set()))
        good_folds = len(positive_lift_folds.get(canonical, set()))
        counts = complete_clause_counts[canonical]
        support = counts["support"]
        total = counts["n_positive"] + counts["n_negative"]
        base_rate = counts["n_positive"] / total if total else 0.0
        precision = counts["tp"] / support if support else None
        cv_metrics = {
            **counts,
            "base_rate": base_rate,
            "precision": precision,
            "recall": counts["tp"] / counts["n_positive"] if counts["n_positive"] else 0.0,
            "lift": precision / base_rate if precision is not None and base_rate else None,
            "evidence_status": "observed" if support else "unobserved_no_out_of_fold_firings",
            "complete_folds_found": full_found,
            "evaluated_match_count": len(complete_clause_test_matches[canonical]),
            "triggered_match_count": complete_clause_firing_match_counts[canonical],
        }
        flags = rule.get("flags", {})
        # A rule is contradicted only after it fired enough times to make a
        # low lift informative. Sparse evidence (including zero firings) is
        # retained as a pending discovery candidate.
        sufficiently_observed = (
            support >= MIN_PENDING_OOF_SUPPORT
            and cv_metrics["triggered_match_count"] >= MIN_MATCHES_FOR_EVIDENCE
        )
        if sufficiently_observed and cv_metrics["lift"] is not None and cv_metrics["lift"] <= MIN_CV_LIFT_FOR_REPORT:
            reasons.append(
                f"out-of-fold lift {cv_metrics['lift']:.3f} is at or below baseline after "
                f"{support} firings"
            )
            excluded.append({"rule": rule, "classification": "contradicted", "reasons": reasons})
            continue

        stable = (
            full_found >= MIN_CV_FOLDS_FOR_REPORT
            and good_folds >= MIN_POSITIVE_LIFT_FOLDS_FOR_REPORT
            and cv_metrics["support"] >= MIN_AUTO_APPROVE_OOF_SUPPORT
            and cv_metrics["triggered_match_count"] >= MIN_MATCHES_FOR_EVIDENCE
            and cv_metrics["lift"] is not None
            and cv_metrics["lift"] >= MIN_AUTO_APPROVE_LIFT
        )
        if stable:
            approved_rule = dict(rule)
            approved_rule["validation_status"] = "approved"
            approved_rule["validation_origin"] = "automated_match_cv"
            approved_rule["validation_note"] = (
                "تأیید پیش‌فرض خودکار: قانون در دست‌کم دو fold کامل کشف شده، "
                f"پشتیبانی خارج از آموزش {cv_metrics['support']} و lift برابر "
                f"{cv_metrics['lift']:.3f} داشته است. مربی می‌تواند این تصمیم را تغییر دهد."
            )
            approved_rule["cross_validation"] = dict(cv_metrics)
            approved_rule["cross_validation"]["complete_folds_found"] = full_found
            coach_rules.append(approved_rule)
            auto_approved += 1
            continue

        promising = (
            full_found >= 1
            and cv_metrics["support"] >= MIN_PENDING_OOF_SUPPORT
            and cv_metrics["triggered_match_count"] >= MIN_MATCHES_FOR_EVIDENCE
            and cv_metrics["lift"] is not None
            and cv_metrics["lift"] > MIN_CV_LIFT_FOR_REPORT
        )
        pending_rule = dict(rule)
        # Revoke only a previous automated approval when the current CV no
        # longer meets its gate; preserve any coach-authored decision above.
        if pending_rule.get("validation_origin") == "automated_match_cv":
            pending_rule["validation_status"] = "pending"
            pending_rule["validation_note"] = ""
            pending_rule.pop("validation_origin", None)
        if not full_found:
            pending_rule["cv_candidate_status"] = "not_rediscovered_in_cv"
            pending_rule["cv_evidence_note"] = (
                "The clause was not selected by any complete CV fold. This is not a failed test; "
                "the coach-form train/holdout scores are diagnostic and the clause needs fresh matches."
            )
        elif support == 0:
            pending_rule["cv_candidate_status"] = "rare_unobserved"
            pending_rule["cv_evidence_note"] = (
                f"The clause had zero out-of-fold firings across {cv_metrics['evaluated_match_count']} "
                "held-out matches, so CV precision and lift are undefined. "
                "It is retained as a rare discovery candidate, not validated."
            )
        elif (
            support < MIN_PENDING_OOF_SUPPORT
            or cv_metrics["triggered_match_count"] < MIN_MATCHES_FOR_EVIDENCE
        ):
            pending_rule["cv_candidate_status"] = "rare_low_support"
            pending_rule["cv_evidence_note"] = (
                f"There were {support} out-of-fold firings across "
                f"{cv_metrics['triggered_match_count']} distinct matches; evidence is too sparse "
                "for approval or rejection. Review and seek validation on new matches."
            )
        else:
            pending_rule["cv_candidate_status"] = "promising_single_fold" if promising else "review_candidate"
        pending_rule["cross_validation"] = dict(cv_metrics)
        coach_rules.append(pending_rule)
        pending_candidates += 1

    coach_doc["rules"] = coach_rules
    # Keep the coach form short. Full exclusions and reasons live in the JSON
    # CV audit report below, not in a section the coach has to review.
    coach_doc.pop("excluded_rules", None)
    coach_doc["automated_screen"] = {
        "method": "holdout_flag_and_match_grouped_cv",
        "auto_approval": {
            "minimum_complete_folds_found": MIN_CV_FOLDS_FOR_REPORT,
            "minimum_complete_folds_with_lift_above_baseline": MIN_POSITIVE_LIFT_FOLDS_FOR_REPORT,
            "minimum_oof_support": MIN_AUTO_APPROVE_OOF_SUPPORT,
            "minimum_distinct_oof_matches": MIN_MATCHES_FOR_EVIDENCE,
            "minimum_oof_lift": MIN_AUTO_APPROVE_LIFT,
            "grouped_cv_overrides_diagnostic_holdout": True,
        },
        "pending_candidate": {
            "minimum_complete_folds_found": 0,
            "minimum_oof_support": 0,
            "note": "Sparse and unobserved rules are retained as pending; zero support makes precision and lift undefined.",
        },
        "note": "Rules meeting the stable CV gate are pre-approved by default. Sparse or zero-fire rules remain pending: zero firings means unobserved and precision/lift are undefined, not that the rule failed. Rules with sufficient firings and lift at or below baseline are classified as contradicted. Pending candidates require review and fresh-match validation.",
    }
    with COACH_VALIDATION_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(coach_doc, fh, allow_unicode=True, sort_keys=False, width=110)

    report["coach_rule_screen"] = {
        "status": "complete",
        "criteria": {
            "auto_approval_minimum_complete_folds_found": MIN_CV_FOLDS_FOR_REPORT,
            "auto_approval_minimum_complete_folds_with_lift_above_baseline": MIN_POSITIVE_LIFT_FOLDS_FOR_REPORT,
            "auto_approval_minimum_oof_support": MIN_AUTO_APPROVE_OOF_SUPPORT,
            "auto_approval_minimum_oof_lift": MIN_AUTO_APPROVE_LIFT,
            "pending_minimum_complete_folds_found": 0,
            "pending_minimum_oof_support": 0,
            "minimum_oof_support_for_contradiction": MIN_PENDING_OOF_SUPPORT,
            "minimum_distinct_oof_matches_for_evidence": MIN_MATCHES_FOR_EVIDENCE,
        },
        "auto_approved_rule_count": auto_approved,
        "pending_candidate_count": pending_candidates,
        "rare_unobserved_or_low_support_candidates": sum(
            rule.get("cv_candidate_status") in {"rare_unobserved", "rare_low_support", "not_rediscovered_in_cv"}
            for rule in coach_rules
        ),
        "preserved_manual_decision_count": manual_decisions,
        "coach_form_rule_count": len(coach_rules),
        "excluded_rules": [
            {
                "name": item["rule"].get("name"),
                "prolog_body": item["rule"].get("prolog_body"),
                "classification": item["classification"],
                "reasons": item["reasons"],
            }
            for item in excluded
        ],
    }
    return report


def _score_program(
    clauses: list[list[str]],
    bk: Path,
    examples: Path,
    sequence_to_match: dict[str, str] | None = None,
) -> dict[str, int | float]:
    """Score a disjunctive Popper program against one complete examples file."""
    esc = lambda p: p.as_posix().replace("'", "\\'")
    goals = [
        f"consult('{esc(bk)}')",
        f"consult('{esc(examples)}')",
    ]
    for i, body in enumerate(clauses):
        conjunction = ", ".join(f"{literal}(S)" for literal in body)
        goals.append(f"assertz((cv_clause_{i}(S) :- {conjunction}))")
    if clauses:
        disjunction = "; ".join(f"cv_clause_{i}(S)" for i in range(len(clauses)))
        goals.extend([
            f"assertz((cv_predict(S) :- ({disjunction})))",
            "findall(S, pos(negative_outcome(S)), Pos), length(Pos, NP)",
            "findall(S, neg(negative_outcome(S)), Neg), length(Neg, NN)",
            "findall(S, (pos(negative_outcome(S)), once(cv_predict(S))), TPs), length(TPs, TP)",
            "findall(S, (neg(negative_outcome(S)), once(cv_predict(S))), FPs), length(FPs, FP)",
        ])
    else:
        goals.extend([
            "findall(S, pos(negative_outcome(S)), Pos), length(Pos, NP)",
            "findall(S, neg(negative_outcome(S)), Neg), length(Neg, NN)",
            "TP = 0", "FP = 0",
        ])
    goals.extend(["format('~w ~w ~w ~w', [NP, NN, TP, FP])", "halt"])
    if sequence_to_match is not None:
        goals[-2] = "format('~w ~w ~w ~w~n', [NP, NN, TP, FP])"
        goals[-1:-1] = [
            "findall(S, (pos(negative_outcome(S)), once(cv_predict(S))), TPs)",
            "findall(S, (neg(negative_outcome(S)), once(cv_predict(S))), FPs)",
            "append(TPs, FPs, Predicted), sort(Predicted, UniquePredicted)",
            "forall(member(S, UniquePredicted), format('PRED ~w~n', [S]))",
        ]
    proc = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-q", "-g", ", ".join(goals)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode or "ERROR:" in proc.stderr:
        raise RuntimeError(f"SWI-Prolog scoring failed:\n{proc.stderr}\n{proc.stdout}")
    output_lines = proc.stdout.strip().splitlines()
    tokens = output_lines[0].split() if output_lines else []
    if len(tokens) != 4:
        raise RuntimeError(f"unexpected Prolog scoring output: {proc.stdout!r}")
    n_pos, n_neg, tp, fp = map(int, tokens)
    support = tp + fp
    base_rate = n_pos / (n_pos + n_neg) if n_pos + n_neg else 0.0
    precision = tp / support if support else 0.0
    recall = tp / n_pos if n_pos else 0.0
    result = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": n_pos - tp,
        "tn": n_neg - fp,
        "base_rate": round(base_rate, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "lift": round(precision / base_rate, 6) if base_rate else 0.0,
    }
    if sequence_to_match is not None:
        predicted = {
            line.split(maxsplit=1)[1]
            for line in output_lines[1:]
            if line.startswith("PRED ") and len(line.split(maxsplit=1)) == 2
        }
        result["triggered_match_count"] = len({
            str(sequence_to_match[sequence_id])
            for sequence_id in predicted
            if sequence_id in sequence_to_match
        })
    return result


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def run(folds_count: int, seed: int, popper_home: Path, timeout: int) -> dict:
    required = [
        MATCHED_PATH, LABELLED_PATH, PROLOG_DIR / "facts_all.pl",
        PROLOG_DIR / "facts_tactical.pl", PROLOG_DIR / "bk.pl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing cached phase outputs:\n  " + "\n  ".join(missing))

    # Rebuild the shared inputs from the current source. Do not let a bias.pl
    # left by an earlier Colab upload silently reintroduce a screened feature.
    build_bk_wrapper(POPPER_DIR / "bk.pl")
    build_legacy_43_bias(POPPER_DIR / "bias.pl")

    labelled = pd.read_parquet(LABELLED_PATH)
    matched = pd.read_parquet(MATCHED_PATH)
    if not {"sequence_id", "match_id", "label"}.issubset(labelled.columns):
        raise ValueError("labeled_sequences.parquet needs sequence_id, match_id, label columns")
    sampled_ids = set(matched.sequence_id)
    train_pool_all = labelled[labelled.sequence_id.isin(sampled_ids)].copy()
    train_pool = train_pool_all[train_pool_all.label.isin(EVALUABLE_LABELS)].copy()
    all_matches = sorted(labelled.match_id.astype(str).unique())
    match_folds = _balanced_group_folds(train_pool, all_matches, folds_count, seed)

    CV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict] = []
    out_of_fold = {key: 0 for key in ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")}
    clause_folds: dict[str, set[int]] = defaultdict(set)
    clause_spellings: dict[str, list[str]] = defaultdict(list)
    clause_oof: dict[str, dict[str, int]] = defaultdict(
        lambda: {key: 0 for key in ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")}
    )
    intermediate_snapshots: list[dict] = []
    intermediate_rule_snapshots: list[dict] = []

    for fold_index, test_matches in enumerate(match_folds, start=1):
        fold_name = f"fold_{fold_index:02d}"
        task_dir = PROLOG_DIR / f"popper_cv_{fold_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        raw_train_rows = train_pool_all[~train_pool_all.match_id.astype(str).isin(test_matches)]
        n_train_inconclusive = int((~raw_train_rows.label.isin(EVALUABLE_LABELS)).sum())
        train_rows = raw_train_rows[raw_train_rows.label.isin(EVALUABLE_LABELS)]
        test_rows, n_test_inconclusive = holdout_population(labelled, test_matches)
        _write_examples(train_rows, task_dir / "exs.pl")
        _write_examples(test_rows, task_dir / "exs_holdout.pl")
        shutil.copy2(POPPER_DIR / "bk.pl", task_dir / "bk.pl")
        shutil.copy2(POPPER_DIR / "bias.pl", task_dir / "bias.pl")

        print(
            f"Starting {fold_name}/{folds_count}: Popper search budget {timeout}s "
            f"(+ {30}s hard-stop margin).",
            flush=True,
        )
        popper_started = time.monotonic()
        output, hard_timed_out = _run_popper(task_dir, popper_home, timeout)
        popper_seconds = time.monotonic() - popper_started
        (LOG_DIR / f"{fold_name}_popper_output.txt").write_text(output, encoding="utf-8")
        fold_candidates, fold_rule_candidates = _evaluate_hypothesis_snapshots(
            output, fold_index, task_dir / "bk.pl", task_dir / "exs.pl", task_dir / "exs_holdout.pl"
        )
        intermediate_snapshots.extend(fold_candidates)
        intermediate_rule_snapshots.extend(fold_rule_candidates)
        clauses, provisional = _parse_solution(output)
        metrics = _score_program(clauses, task_dir / "bk.pl", task_dir / "exs_holdout.pl")
        for key in out_of_fold:
            out_of_fold[key] += int(metrics[key])

        canonical_rules = []
        sequence_to_match = dict(zip(
            test_rows.sequence_id.astype(str), test_rows.match_id.astype(str)
        ))
        for body in clauses:
            canonical = ",".join(sorted(body))
            clause_folds[canonical].add(fold_index)
            clause_spellings[canonical].append(", ".join(body))
            single_metrics = _score_program(
                [body], task_dir / "bk.pl", task_dir / "exs_holdout.pl", sequence_to_match
            )
            for key in clause_oof[canonical]:
                clause_oof[canonical][key] += int(single_metrics[key])
            canonical_rules.append({"body": body, "canonical": canonical, "test_metrics": single_metrics})

        fold_results.append({
            "fold": fold_index,
            "train_matches": sorted(set(train_pool.match_id.astype(str)) - test_matches),
            "test_matches": sorted(test_matches),
            "n_train_matched_examples": int(len(train_rows)),
            "n_train_inconclusive_excluded": n_train_inconclusive,
            "n_test_all_labelled_sequences": int(len(test_rows)),
            "n_test_inconclusive_excluded": n_test_inconclusive,
            "clauses": canonical_rules,
            "test_metrics": metrics,
            "popper_seconds": popper_seconds,
            "hard_timed_out": hard_timed_out,
            "used_provisional_hypothesis": provisional,
            "no_solution": not clauses,
        })
        print(
            f"{fold_name}: train={len(train_rows)} matched examples, "
            f"test={len(test_rows)} all labelled sequences, "
            f"rules={len(clauses)}, support={metrics['support']}, "
            f"precision={metrics['precision']:.3f}, lift={metrics['lift']:.3f}, "
            f"popper_seconds={popper_seconds:.1f}, provisional={provisional}, hard_timeout={hard_timed_out}",
            flush=True,
        )

    total = out_of_fold
    support = total["support"]
    n_all = total["n_positive"] + total["n_negative"]
    base_rate = total["n_positive"] / n_all if n_all else 0.0
    precision = total["tp"] / support if support else 0.0
    recall = total["tp"] / total["n_positive"] if total["n_positive"] else 0.0
    aggregate = {
        **total,
        "base_rate": round(base_rate, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "lift": round(precision / base_rate, 6) if base_rate else 0.0,
    }
    stability = []
    for body, found_in in sorted(clause_folds.items(), key=lambda item: (-len(item[1]), item[0])):
        counts = clause_oof[body]
        clause_support = counts["support"]
        clause_total = counts["n_positive"] + counts["n_negative"]
        clause_base = counts["n_positive"] / clause_total if clause_total else 0.0
        clause_precision = counts["tp"] / clause_support if clause_support else 0.0
        clause_recall = counts["tp"] / counts["n_positive"] if counts["n_positive"] else 0.0
        stability.append({
            "canonical_body": body,
            "folds_found": len(found_in),
            "fold_ids": sorted(found_in),
            "display_body": max(set(clause_spellings[body]), key=clause_spellings[body].count),
            "out_of_fold_metrics_on_folds_where_discovered": {
                **counts,
                "base_rate": round(clause_base, 6),
                "precision": round(clause_precision, 6),
                "recall": round(clause_recall, 6),
                "lift": round(clause_precision / clause_base, 6) if clause_base else 0.0,
            },
        })
    report = {
        "method": "full_tactical_grouped_match_k_fold_cv",
        "comparison_protocol_revision": COMPARISON_PROTOCOL_REVISION,
        "popper_max_clauses": POPPER_MAX_CLAUSES,
        "body_predicate_count": sum(
            line.strip().startswith("body_pred(")
            for line in (POPPER_DIR / "bias.pl").read_text(encoding="utf-8").splitlines()
        ),
        "folds": folds_count,
        "seed": seed,
        "training_sampling": "phase2 context-matched ilp_examples.parquet, training matches only",
        "evaluation_population": "all labeled sequences in held-out matches (natural prevalence)",
        "data_fingerprint": _fingerprint([
            LABELLED_PATH, MATCHED_PATH, PROLOG_DIR / "facts_all.pl",
            PROLOG_DIR / "facts_tactical.pl", PROLOG_DIR / "bk.pl",
            POPPER_DIR / "bias.pl", POPPER_DIR / "bk.pl",
        ]),
        "n_matches": int(labelled.match_id.nunique()),
        "n_labelled_sequences": int(len(labelled)),
        "search_budget_seconds_per_fold": timeout,
        "hard_stop_seconds_per_fold": timeout + 30,
        "out_of_fold_metrics": aggregate,
        "rule_stability": stability,
        "intermediate_candidates": {
            **_build_intermediate_candidate_report(intermediate_snapshots, folds_count),
            "individual_rules": _build_intermediate_candidate_report(intermediate_rule_snapshots, folds_count)["candidates"],
        },
        "fold_results": fold_results,
        "interpretation": (
            "The out-of-fold score evaluates the rule-discovery procedure. "
            "Rules may differ by fold; inspect rule_stability and per-fold metrics. "
            "This remains internal validation, not an independent new-season test."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _filter_coach_candidates(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3, help="number of match-grouped folds (default: 3)")
    parser.add_argument("--seed", type=int, default=202021, help="deterministic fold-assignment seed")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_CV_TIMEOUT_SECONDS,
        help=f"Popper search budget per fold in seconds; hard stop adds 30s (default: {DEFAULT_CV_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()
    if args.folds < 3:
        parser.error("--folds must be at least 3")
    popper_dir = os.environ.get("POPPER_DIR")
    if not popper_dir:
        parser.error("POPPER_DIR must point to the Popper checkout, e.g. /content/Popper")
    try:
        report = run(args.folds, args.seed, Path(popper_dir), args.timeout)
    except Exception as exc:
        print(f"CROSS-VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    m = report["out_of_fold_metrics"]
    print("\n" + "=" * 78)
    print("MATCH-GROUPED CROSS-VALIDATION — OUT-OF-FOLD RESULT")
    print("=" * 78)
    print(f"held-out sequences : {m['n_positive'] + m['n_negative']} (all labelled sequences)")
    print(f"base rate          : {m['base_rate']:.3f}")
    print(f"support            : {m['support']} (TP={m['tp']}, FP={m['fp']})")
    print(f"precision / recall : {m['precision']:.3f} / {m['recall']:.3f}")
    print(f"lift               : {m['lift']:.3f}")
    print("\nRule recurrence across folds:")
    for rule in report["rule_stability"]:
        print(f"  {rule['folds_found']}/{report['folds']} folds: {rule['display_body']}")
    candidates = report["intermediate_candidates"]["candidates"]
    print(f"\nExploratory intermediate hypotheses: {len(candidates)} distinct programs")
    for candidate in candidates[:10]:
        m_candidate = candidate["holdout_metrics_on_discovery_folds"]
        print(
            f"  {candidate['folds_found']}/{report['folds']} folds; "
            f"support={m_candidate['support']}, precision={m_candidate['precision']:.3f}, "
            f"lift={m_candidate['lift']:.3f}: {candidate['display_program'].replace(chr(10), ' | ')}"
        )
    print("Individual clauses from those intermediate programs:")
    for candidate in report["intermediate_candidates"]["individual_rules"][:10]:
        m_candidate = candidate["holdout_metrics_on_discovery_folds"]
        print(
            f"  {candidate['folds_found']}/{report['folds']} folds; "
            f"support={m_candidate['support']}, precision={m_candidate['precision']:.3f}, "
            f"lift={m_candidate['lift']:.3f}: {candidate['display_program']}"
        )
    print("  Exploratory ranking only; candidate scores are selection-biased. See the JSON report for per-fold details.")
    print(f"\nDetailed report -> {REPORT_PATH}")
    print(f"Per-fold Popper logs -> {LOG_DIR}")
    print("Use out-of-fold metrics as the main internal estimate; do not select rules by one old holdout score.")
    screen = report["coach_rule_screen"]
    if screen["status"] == "complete":
        print(
            f"Coach form: auto-approved={screen['auto_approved_rule_count']}, "
            f"pending candidates={screen['pending_candidate_count']}, "
            f"omitted={len(screen['excluded_rules'])}. Omission reasons are in the JSON report."
        )
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged("phase5_cross_validation", main))
