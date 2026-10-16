"""
Phase 6 — filter, evaluate, and interpret the rules Popper found in phase 5.

    uv run python scripts/run_phase6.py

Implements the four tasks in the execution plan's phase 6 section:

  6.0  one-split diagnostic re-validation
       (the split itself was built in phase 5; repeated inspection means it
       is not the final independent estimate -- use run_phase5_cv.py)
  6.1  Support + Confidence, with the 3-occurrence minimum filter
  6.2  Lift vs the team baseline, with the synthetic + known-healthy-pattern
       sanity checks
  6.3  writes data/reports/phase6_coach_validation.yaml for you to fill in
       by hand -- this script does NOT decide approve/reject for you

Run this after the selected Phase-5 arm has completed successfully. It does
not call Popper again; it reads the final SOLUTION block from that arm's log,
then scores rules against the matching task directory and train/holdout split.
"""

from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import REPORTS_DIR                                # noqa: E402
from tactical_ilp.evaluation.coach_validation import (                     # noqa: E402
    COACH_VALIDATION_PATH,
    write_coach_validation_file,
)
from tactical_ilp.evaluation.rule_metrics import (                         # noqa: E402
    MIN_SUPPORT,
    compute_lift,
    known_healthy_pattern_metrics,
    score_rule,
)
from tactical_ilp.evaluation.rule_metrics import Rule                       # noqa: E402
from tactical_ilp.evaluation.rule_registry import ALL_RULES                 # noqa: E402
from tactical_ilp.reporting.method_config import (                           # noqa: E402
    coach_source_fingerprint,
    get_coach_method,
)


def current_popper_rules(log_path: Path) -> tuple[list[Rule], list[str]]:
    """Read only the clauses in the final SOLUTION block of this Popper run.

    Popper prints many interim ``New best hypothesis`` programs while it
    searches. They are progress diagnostics, not separately discovered rules
    and must not be copied into the coach-review file. The final SOLUTION
    program is the one bounded by the max_clauses setting in bias.pl.
    """
    from tactical_ilp.config import REPORTS_DIR
    from tactical_ilp.reporting.concept_dictionary import lookup, split_negation

    path = log_path
    if not path.exists():
        return [], [f"missing Popper output log: {path}"]
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    rules: list[Rule] = []
    unreviewed: list[str] = []
    malformed: list[str] = []
    lines = text.splitlines()
    solution_header = re.compile(
        r"^\s*(?:\d+(?:\.\d+)?s\s+)?\*+\s*SOLUTION\s*\*+\s*$",
        re.IGNORECASE,
    )
    solution_start = next(
        (index for index in range(len(lines) - 1, -1, -1) if solution_header.match(lines[index])),
        None,
    )
    if solution_start is None:
        return [], [f"no final SOLUTION block found in {path.name}"]

    final_lines = []
    for line in lines[solution_start + 1 :]:
        # Popper closes the final solution with a banner made only of a time
        # prefix and asterisks. Stop there so later log text cannot be parsed.
        if re.match(r"^\s*(?:\d+(?:\.\d+)?s\s+)?\*{10,}\s*$", line):
            break
        final_lines.append(line)
    final_solution = "\n".join(final_lines)
    pattern = re.compile(r"negative_outcome\(\w+\)\s*:-\s*([^\n]*?)(?:\s*\.|$)")
    for match in pattern.finditer(final_solution):
        body_text = match.group(1).strip()
        literals = [part.strip() for part in body_text.split(",") if part.strip()]
        body: list[str] = []
        valid = bool(literals)
        for literal in literals:
            atom = re.fullmatch(r"([a-z][a-z0-9_]*)\(\w+\)", literal)
            if not atom:
                valid = False
                break
            name = atom.group(1)
            base, _ = split_negation(name)
            try:
                lookup(base)
            except KeyError:
                unreviewed.append(base)
            body.append(name)
        if not valid:
            malformed.append(body_text)
            continue
        if any(existing.body == body for existing in ALL_RULES):
            continue
        if any(existing.body == body for existing in rules):
            continue
        rules.append(Rule(name=f"current_popper_{len(rules) + 1}", body=body, source="popper_current_run"))
    diagnostics = [
        *(f"no reviewed wording: {name}" for name in sorted(set(unreviewed))),
        *(f"could not parse clause: {body}" for body in malformed),
    ]
    return rules, diagnostics


def _candidate_rows(value, inherited_fold=None) -> list[dict]:
    """Flatten candidate containers while retaining mapping keys as names."""
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_candidate_rows(item, inherited_fold))
        return rows
    if not isinstance(value, dict):
        return []
    candidate_keys = {
        "body", "clauses", "display_program", "canonical_program", "program",
        "candidate_program", "predicate", "pattern_predicate", "holdout_metrics",
        "holdout_metrics_on_discovery_folds", "test_metrics", "test_support",
    }
    if candidate_keys.intersection(value):
        row = dict(value)
        if inherited_fold is not None:
            row.setdefault("fold", inherited_fold)
        return [row]
    rows = []
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            for row in _candidate_rows(item, inherited_fold):
                row.setdefault("candidate_key", str(key))
                rows.append(row)
    return rows


def _candidate_body(candidate: dict) -> list[str] | None:
    """Parse one candidate clause from the known CV schema variants."""
    value = candidate.get("body")
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return [x.strip().removesuffix("(S)") for x in value if x.strip()]
    clauses = candidate.get("clauses")
    if isinstance(clauses, list):
        if len(clauses) != 1:
            return None  # the coach form represents one rule, not a disjunction
        clause = clauses[0]
        if isinstance(clause, list) and clause:
            return [str(x).strip().removesuffix("(S)") for x in clause]
        if isinstance(clause, dict):
            return _candidate_body(clause)

    for key in ("display_program", "candidate_program", "program", "rule", "canonical_program"):
        text = candidate.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if ";" in text:
            return None
        if ":-" in text:
            text = text.split(":-", 1)[1]
        text = text.strip().rstrip(".")
        literals = [part.strip() for part in text.split(",") if part.strip()]
        body = []
        for literal in literals:
            atom = re.fullmatch(r"([a-z][a-z0-9_]*)(?:\(\w+\))?", literal)
            if not atom:
                body = []
                break
            body.append(atom.group(1))
        if body:
            return body

    predicate = candidate.get("pattern_predicate", candidate.get("predicate"))
    if isinstance(predicate, str) and re.fullmatch(r"(?:seq_)?[a-z][a-z0-9_]*", predicate):
        return [predicate if predicate.startswith("seq_") else f"seq_{predicate}"]
    return None


def _candidate_test_metrics(candidate: dict) -> dict | None:
    """Return support/lift only from holdout or test evidence."""
    sources = []
    for key in ("holdout_metrics_on_discovery_folds", "holdout_metrics", "test_metrics", "test", "holdout"):
        value = candidate.get(key)
        if isinstance(value, dict):
            sources.append(value)
    for key in ("metrics", "evaluation_metrics"):
        value = candidate.get(key)
        if isinstance(value, dict):
            for nested_key in ("holdout", "test", "holdout_metrics"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    sources.append(nested)
            if any(k in value for k in ("holdout_lift", "test_lift")):
                sources.append(value)

    direct = {}
    for source_key, metric_key in (("holdout_support", "support"), ("test_support", "support"),
                                   ("holdout_lift", "lift"), ("test_lift", "lift")):
        if source_key in candidate:
            direct[metric_key] = candidate[source_key]
    if direct:
        sources.insert(0, direct)
    for metrics in sources:
        support = metrics.get("support", metrics.get("holdout_support", metrics.get("test_support")))
        lift = metrics.get("lift", metrics.get("holdout_lift", metrics.get("test_lift")))
        try:
            support_value, lift_value = int(support), float(lift)
        except (TypeError, ValueError):
            continue
        if support_value > 0 and math.isfinite(lift_value) and lift_value > 0:
            return {"support": support_value, "lift": lift_value,
                    "precision": metrics.get("precision"), "folds_found": candidate.get("folds_found")}

    # Some fold-local reports keep only one test metric record per discovery fold.
    fold_metrics = []
    for fold in candidate.get("folds", []) if isinstance(candidate.get("folds"), list) else []:
        metrics = fold.get("holdout_metrics", fold.get("test_metrics", {})) if isinstance(fold, dict) else {}
        if not isinstance(metrics, dict):
            continue
        try:
            support, lift = int(metrics.get("support", 0)), float(metrics.get("lift", 0))
        except (TypeError, ValueError):
            continue
        if support > 0 and math.isfinite(lift) and lift > 0:
            fold_metrics.append({"support": support, "lift": lift, "fold": fold.get("fold")})
    if fold_metrics:
        # Retain the strongest valid discovery-fold test result as the candidate's
        # ranking evidence; its selection bias is made explicit in the form.
        return max(fold_metrics, key=lambda item: (item["lift"], item["support"]))
    return None


def _candidate_discovery_fold(candidate: dict) -> str | None:
    """Choose the fold carrying the strongest positive test evidence."""
    rows = []
    folds = candidate.get("folds", [])
    for row in folds if isinstance(folds, list) else []:
        if not isinstance(row, dict):
            continue
        metrics = row.get("holdout_metrics", row.get("test_metrics", {}))
        if not isinstance(metrics, dict):
            continue
        try:
            support, lift = int(metrics.get("support", 0)), float(metrics.get("lift", 0))
        except (TypeError, ValueError):
            continue
        if support > 0 and math.isfinite(lift) and lift > 0:
            rows.append((lift, support, str(row.get("fold", ""))))
    best = max(rows, default=None)
    return best[2] if best and best[2] else None


def _pattern_features_by_fold(report: dict) -> dict[tuple[str, str], dict]:
    result = {}
    for fold_row in report.get("fold_results", []) if isinstance(report.get("fold_results"), list) else []:
        if not isinstance(fold_row, dict):
            continue
        fold_id = str(fold_row.get("fold", ""))
        for key in ("selected_patterns", "selected_fold_local_patterns", "pattern_features"):
            features = fold_row.get(key, [])
            for feature in _candidate_rows(features):
                predicate = feature.get("predicate", feature.get("pattern_predicate"))
                if isinstance(predicate, str):
                    result[(fold_id, predicate)] = feature
    return result


def _pattern_signature(feature: dict | None) -> tuple | None:
    if not isinstance(feature, dict):
        return None
    pattern = feature.get("pattern")
    if isinstance(pattern, list):
        return tuple(tuple(str(item) for item in event) for event in pattern if isinstance(event, list))
    chain = feature.get("pattern_chain")
    if isinstance(chain, list):
        signature = []
        for event in chain:
            items = event.get("items", []) if isinstance(event, dict) else []
            signature.append(tuple(
                str(item.get("symbol") or "@".join(str(item.get(key) or "") for key in ("predicate", "action_type", "lane", "third")))
                for item in items if isinstance(item, dict)
            ))
        return tuple(signature) if signature else None
    return None


def _candidate_sources(report: dict) -> list[dict]:
    rows = []
    for key in ("exploratory_intermediate_hypotheses", "candidate_programs", "selected_fold_local_patterns"):
        rows.extend(_candidate_rows(report.get(key)))
    ledger = report.get("intermediate_candidates", {})
    if isinstance(ledger, dict):
        for key in ("candidates", "individual_rules", "exploratory_intermediate_hypotheses", "candidate_programs"):
            rows.extend(_candidate_rows(ledger.get(key)))
    for fold in report.get("fold_results", []) if isinstance(report.get("fold_results"), list) else []:
        if isinstance(fold, dict):
            rows.extend(_candidate_rows(fold.get("selected_fold_local_patterns"), fold.get("fold")))
    return rows


def cv_intermediate_rules(report_path: Path, limit: int | None = None) -> list[Rule]:
    """Return all technically mappable intermediate CV candidates for review.

    Missing/weak CV evidence is not a hard gate for noisy, rare football events;
    Phase 6 rescoring and coach-facing evidence labels expose those limitations.
    """
    from tactical_ilp.reporting.concept_dictionary import lookup, split_negation

    paths = [report_path]
    fallback = REPORTS_DIR / ("phase5_cross_validation.json" if report_path.name == "phase5_sequential_cv.json" else "phase5_sequential_cv.json")
    if fallback != report_path:
        paths.append(fallback)

    pooled: dict[tuple[str, ...], tuple[Rule, dict]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fold_features = _pattern_features_by_fold(report)
        global_features = {}
        training_path = REPORTS_DIR / "phase5_sequential_training.json"
        if training_path.exists():
            try:
                training = json.loads(training_path.read_text(encoding="utf-8"))
                global_features = {
                    item.get("predicate"): item
                    for item in training.get("pattern_facts", {}).get("features", [])
                    if isinstance(item, dict) and item.get("predicate")
                }
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        for candidate in _candidate_sources(report):
            metrics = _candidate_test_metrics(candidate)
            if metrics is None:
                metrics = {"support": None, "lift": None, "precision": None,
                           "evidence_status": "no_usable_cv_holdout_metrics"}
            body = _candidate_body(candidate)
            if not body:
                continue
            fold = metrics.get("fold", candidate.get("fold")) or _candidate_discovery_fold(candidate)
            supplied = candidate.get("pattern_metadata", {})
            supplied = supplied if isinstance(supplied, dict) else {}
            # Fold-local pattern IDs are ranks, not stable names. Translate a
            # candidate to the final training predicate only when its full
            # ordered signature matches exactly.
            for position, predicate in enumerate(list(body)):
                base, negated = split_negation(predicate)
                if not re.fullmatch(r"seq_pattern_\d+", base):
                    continue
                local_feature = supplied.get(base)
                if local_feature is None and fold is not None:
                    local_feature = fold_features.get((str(fold), base))
                if local_feature is None and candidate.get("predicate") == base:
                    local_feature = candidate
                if local_feature is None:
                    # A fold-local rank has meaning only with its fold's
                    # ordered pattern definition; never equate it by number.
                    if fold is not None:
                        body = []
                        break
                    continue
                signature = _pattern_signature(local_feature)
                matches = [name for name, feature in global_features.items()
                           if signature is not None and _pattern_signature(feature) == signature]
                if not matches:
                    body = []
                    break
                mapped = sorted(matches)[0]
                body[position] = f"not_{mapped}" if negated else mapped
            if not body:
                continue
            valid = True
            for predicate in body:
                base, _ = split_negation(predicate)
                try:
                    lookup(base)
                except KeyError:
                    valid = False
                    break
                if re.fullmatch(r"seq_pattern_\d+", base) and base not in global_features:
                    # A fold-local pattern absent from the final global task
                    # cannot safely be approved for Phase 7.
                    valid = False
                    break
            if not valid:
                continue

            candidate = dict(candidate)
            pattern_metadata = {}
            if isinstance(supplied, dict):
                pattern_metadata.update(supplied)
            for predicate in body:
                base, _ = split_negation(predicate)
                if not re.fullmatch(r"seq_pattern_\d+", base):
                    continue
                feature = (fold_features.get((str(fold), base)) if fold is not None else None) or global_features.get(base)
                if isinstance(feature, dict):
                    pattern_metadata[base] = feature

            source = "fold_local_pattern_candidate" if any(re.fullmatch(r"seq_pattern_\d+", p.removeprefix("not_")) for p in body) else "cv_intermediate_candidate"
            rule = Rule(
                name="intermediate_candidate",
                body=body,
                source=source,
                category="intermediate_candidate",
                candidate_metrics=metrics,
                pattern_metadata=pattern_metadata,
                discovery_fold=fold,
            )
            key = tuple(sorted(body))
            previous = pooled.get(key)
            rank = (metrics.get("lift") or float("-inf"), metrics.get("support") or 0)
            old_rank = ((previous[1].get("lift") or float("-inf"), previous[1].get("support") or 0)
                        if previous else None)
            if previous is None or rank > old_rank:
                pooled[key] = (rule, metrics)
        # Prefer the requested sequential CV report. The standard CV report is
        # a fallback only when the preferred source yielded no usable candidates.
        if pooled:
            break

    def candidate_rank(item: tuple[Rule, dict]) -> tuple[float, int, tuple[str, ...]]:
        metrics = item[1]
        lift = metrics.get("lift")
        support = metrics.get("support")
        # Missing CV evidence is allowed through for Phase-6 rescoring, but it
        # must sort after candidates with usable evidence rather than crash.
        lift_rank = float(lift) if isinstance(lift, (int, float)) and math.isfinite(lift) else float("-inf")
        support_rank = int(support) if isinstance(support, (int, float)) and math.isfinite(support) else -1
        return (-lift_rank, -support_rank, tuple(item[0].body))

    ranked = sorted(pooled.values(), key=candidate_rank)
    selected = []
    selected_rows = ranked if limit is None else ranked[:limit]
    for index, (rule, _metrics) in enumerate(selected_rows, start=1):
        rule.name = f"intermediate_candidate_{index}"
        selected.append(rule)
    return selected


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    method = get_coach_method()
    bk_file = Path(method.task_dir) / "bk.pl"
    train_file = Path(method.task_dir) / "exs.pl"
    holdout_file = Path(method.task_dir) / "exs_holdout.pl"
    required = [bk_file, train_file, holdout_file, Path(method.training_log)]
    if method.training_report is not None:
        required.append(method.training_report)
    missing = [path for path in required if not path.exists()]
    if missing:
        print(
            f"missing inputs for coach method {method.key!r}: "
            + ", ".join(str(path) for path in missing)
            + "; run the matching Phase-5 training script first"
        )
        return 1

    banner(f"COACH REPORT SOURCE — {method.key} ({method.title})")

    banner("TASK 6.2a — synthetic lift sanity check (no Prolog, pure arithmetic)")
    synthetic_lift = compute_lift(confidence=0.1341, base_rate=0.1341)
    print(f"confidence == base_rate  ->  lift = {synthetic_lift:.4f}  (must be ~1.0)")
    if abs(synthetic_lift - 1.0) > 0.1:
        print("FAIL: synthetic lift check did not return ~1.0 -- compute_lift() is broken")
        return 1
    print("PASS")

    banner("TASK 6.2b — known-healthy-pattern lift check (real data)")
    healthy = known_healthy_pattern_metrics(
        bk_file=bk_file, train_file=train_file
    )
    if healthy is None:
        print("could not score the known-healthy pattern (missing files)")
    else:
        print(f"seq_supported_action: support={healthy.support} confidence={healthy.confidence} lift={healthy.lift}")
        print("(technical document section 2.7's worked example: a frequent, healthy pattern")
        print(" -- e.g. a teammate offering an open passing lane -- should NOT read as dangerous)")
        if healthy.lift < 1.3:
            print(f"PASS -- lift {healthy.lift} < 1.3, this frequent pattern is correctly read as safe")
        else:
            print(f"WARNING -- lift {healthy.lift} >= 1.3, investigate before trusting other lift numbers")

    banner("TASKS 6.0 + 6.1 — Support / Confidence / Lift for every registered rule, both splits")
    current_rules, parser_diagnostics = current_popper_rules(Path(method.training_log))
    registered_bodies = {tuple(rule.body) for rule in ALL_RULES}
    discovered_rules = [r for r in current_rules if tuple(r.body) not in registered_bodies]
    cv_report = REPORTS_DIR / (
        "phase5_sequential_cv.json" if method.key == "hybrid_sequential"
        else "phase5_cross_validation.json"
    )
    intermediate_rules = cv_intermediate_rules(cv_report)
    known_bodies = registered_bodies | {tuple(rule.body) for rule in discovered_rules}
    intermediate_rules = [r for r in intermediate_rules if tuple(r.body) not in known_bodies]
    rules_to_score = [*ALL_RULES, *discovered_rules, *intermediate_rules]
    if not rules_to_score:
        print("No final SOLUTION rules were found for the selected method; coach form was not overwritten.")
        for diagnostic in parser_diagnostics:
            print(f"  {diagnostic}")
        return 1
    if discovered_rules:
        print("Additional distinct clause(s) from the final Popper SOLUTION block:")
        for rule in discovered_rules:
            print(f"  {rule.name}: negative_outcome(S) :- {rule.as_prolog_body()}.")
    if intermediate_rules:
        print(f"Added {len(intermediate_rules)} technically mappable intermediate CV clauses from {cv_report.name}.")
        print("Weak or missing CV support/lift is not a hard exclusion; Phase 6 train/holdout metrics and limitations accompany each candidate.")
    if parser_diagnostics:
        print("Parser diagnostics (only final-SOLUTION clauses are considered):")
        for diagnostic in parser_diagnostics:
            print(f"  {diagnostic}")

    scored = []
    for rule in rules_to_score:
        r = score_rule(
            rule.name,
            rule.body,
            source=(rule.source if rule.category == "intermediate_candidate" else method.key),
            train_file=train_file,
            holdout_file=holdout_file,
            bk_file=bk_file,
        )
        r.category = rule.category
        r.candidate_metrics = rule.candidate_metrics
        r.pattern_metadata = rule.pattern_metadata
        r.discovery_fold = rule.discovery_fold
        scored.append(r)
        print(f"\n{r.name}  ({r.source})")
        print(f"  body: negative_outcome(S) :- {r.as_prolog_body()}.")
        print(f"  TRAIN   support={r.train.support:>4}  confidence={r.train.confidence:.4f}  lift={r.train.lift:.4f}")
        print(f"  HOLDOUT support={r.holdout.support:>4}  confidence={r.holdout.confidence:.4f}  lift={r.holdout.lift:.4f}")
        if r.flagged_low_support:
            print(f"  [FLAG] train support {r.train.support} < minimum {MIN_SUPPORT}")
        if r.flagged_holdout_drop:
            print(f"  [FLAG] confidence dropped {r.holdout_drop_ratio:.0%} from train to holdout -- likely overfit")

    banner("TASK 6.3 — writing the coach validation file")
    model_fingerprint = coach_source_fingerprint(method)
    out = write_coach_validation_file(
        scored,
        source_method=method.key,
        source_fingerprint=model_fingerprint,
    )
    print(f"wrote -> {out}")
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_coach_review_ui import write_ui

    ui_path = write_ui(out, Path(__file__).resolve().parents[1] / "coach_review.html")
    print(f"wrote coach review UI -> {ui_path}")
    print()
    print("این فایل فرم اولیه است؛ گزارش CV بازوی انتخاب‌شده را برای بررسی پایداری بخوانید.")
    print("CV status مربی را خودکار تعیین نمی‌کند؛ تأیید یا رد باید با دلیل فوتبالی ثبت شود.")
    print("قواعد pending را بررسی کنید و برای هر تصمیم دستی یک یادداشت فوتبالی بنویسید.")
    if any(item.startswith("no reviewed wording:") for item in parser_diagnostics):
        print("قواعد دارای predicate بدون شرح بازبینی‌شده را تا افزودن شرح در concept_dictionary.py تأیید نکنید؛ فاز ۷ متن آن‌ها را نمی‌سازد.")
    print("قانون‌هایی که هنوز pending هستند وارد گزارش نهایی فاز ۷ نمی‌شوند.")

    summary = {
        "n_rules_scored": len(scored),
        "n_current_run_rules_added": len(discovered_rules),
        "n_flagged_low_support": sum(r.flagged_low_support for r in scored),
        "n_flagged_holdout_drop": sum(r.flagged_holdout_drop for r in scored),
        "synthetic_lift_check": "pass" if abs(synthetic_lift - 1.0) <= 0.1 else "fail",
        "known_healthy_pattern_lift": healthy.lift if healthy else None,
        "coach_validation_file": str(out.relative_to(Path(__file__).resolve().parents[1])),
        "source_method": method.key,
        "source_fingerprint": model_fingerprint,
    }
    (REPORTS_DIR / "phase6_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsummary -> {REPORTS_DIR / 'phase6_summary.json'}")

    banner("PHASE 6 DONE (initial coach form written)")
    cv_script = "run_phase5_sequential_cv.py" if method.key == "hybrid_sequential" else "run_phase5_cv.py"
    print(f"next: review the {method.key} coach form (after {cv_script}), then run phase 7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
