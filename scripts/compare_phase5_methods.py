"""Compare miner-only, CART, and hybrid miner+Popper OOF results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import REPORTS_DIR  # noqa: E402
from phase5_cv_common import COMPARISON_PROTOCOL_REVISION  # noqa: E402

BASELINE_PATH = REPORTS_DIR / "phase5_cross_validation.json"
SEQUENTIAL_CV_PATH = REPORTS_DIR / "phase5_sequential_cv.json"
SEQUENTIAL_TRAINING_PATH = REPORTS_DIR / "phase5_sequential_training.json"
DECISION_TREE_PATH = REPORTS_DIR / "phase5_decision_tree_cv.json"
OUTPUT_JSON = REPORTS_DIR / "phase5_method_comparison.json"
OUTPUT_MD = REPORTS_DIR / "phase5_method_comparison.md"


def _clause_body(clause: object) -> list[str]:
    """Return body predicates from either CV list clauses or report mappings."""
    if isinstance(clause, dict):
        body = clause.get("body", [])
    elif isinstance(clause, (list, tuple)):
        body = clause
    else:
        return []
    if not isinstance(body, (list, tuple)):
        return []
    return [str(predicate) for predicate in body]


def _add_common_metrics(result: dict) -> dict:
    """Add common classifier scores when the report includes a confusion matrix."""
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not all(key in metrics for key in ("tp", "fp", "fn", "tn")):
        return result
    tp, fp, fn, tn = (int(metrics[key]) for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    common = {
        "support": tp + fp,
        "specificity": specificity,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }
    positive = int(metrics.get("n_positive", tp + fn))
    negative = int(metrics.get("n_negative", tn + fp))
    base_rate = positive / (positive + negative) if positive + negative else 0.0
    common["base_rate"] = base_rate
    common["lift"] = precision / base_rate if base_rate else 0.0
    result["metrics"] = {**metrics, **common}
    return result

def _read_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}
    return value if isinstance(value, dict) else {"_read_error": "report root is not an object"}


def _fold_signature(folds: list[dict]) -> tuple[tuple[str, ...], ...]:
    signature = tuple(
        tuple(sorted(map(str, fold.get("test_matches", []))))
        for fold in folds
    )
    return signature if signature and all(signature) else ()


def _test_population_signature(folds: list[dict]) -> tuple[str, ...]:
    hashes = tuple(str(fold.get("test_population_fingerprint", "")) for fold in folds)
    return hashes if hashes and all(len(value) == 64 for value in hashes) else ()


def _baseline_result(report: dict | None) -> dict:
    if report is None:
        return {"status": "not_run", "display": "Report not available", "source": "report_missing", "report": str(BASELINE_PATH), "metrics": None}
    if report.get("_read_error"):
        return {"status": "invalid_report", "display": "Report could not be read", "source": "invalid_report", "report": str(BASELINE_PATH), "metrics": None, "report_error": report["_read_error"]}

    folds = report.get("fold_results", [])
    rule_folds: dict[str, int] = {}
    folds = report.get("fold_results", [])
    for fold in folds:
        complete = not (
            fold.get("no_solution")
            or fold.get("used_provisional_hypothesis")
            or fold.get("hard_timed_out")
        )
        if not complete:
            continue
        for clause in fold.get("clauses", []):
            metrics = clause.get("test_metrics", {})
            canonical = clause.get("canonical") or ",".join(sorted(clause.get("body", [])))
            if canonical and float(metrics.get("lift", 0)) > 1.0 and int(metrics.get("support", 0)) > 0:
                rule_folds[canonical] = rule_folds.get(canonical, 0) + 1
    stable_rules = [body for body, count in rule_folds.items() if count >= 2]
    if not folds:
        for rule in report.get("rule_stability", []):
            metrics = rule.get("out_of_fold_metrics_on_folds_where_discovered", {})
            if int(rule.get("folds_found", 0)) >= 2 and float(metrics.get("lift", 0)) > 1.0 and int(metrics.get("support", 0)) > 0:
                stable_rules.append(rule)
    no_solution = bool(folds) and all(
        fold.get("no_solution") or fold.get("used_provisional_hypothesis") or fold.get("hard_timed_out")
        for fold in folds
    )
    if not stable_rules or no_solution:
        return {
            "status": "no_stable_program",
            "display": "No stable tactical program found (MDL prune / NO SOLUTION)",
            "source": "phase5_cross_validation.json",
            "report": str(BASELINE_PATH),
            "metrics": report.get("out_of_fold_metrics"),
            "stable_rule_count": len(stable_rules),
            "no_solution_or_provisional_folds": sum(
                bool(fold.get("no_solution") or fold.get("used_provisional_hypothesis") or fold.get("hard_timed_out"))
                for fold in folds
            ),
        }
    metrics = report.get("out_of_fold_metrics", {})
    return {
        "status": "stable_rules_found",
        "display": f"Stable tactical rules found (Precision: {float(metrics.get('precision', 0)):.0%}, Lift: {float(metrics.get('lift', 0)):.2f}x)",
        "source": "phase5_cross_validation.json",
        "report": str(BASELINE_PATH),
        "metrics": metrics,
        "stable_rule_count": len(stable_rules),
    }


def _sequential_result(cv_report: dict | None, training_report: dict | None) -> dict:
    if cv_report is None:
        return {"status": "not_run", "display": "Report not available", "source": "report_missing", "cv_report": str(SEQUENTIAL_CV_PATH), "metrics": None}
    if cv_report.get("_read_error"):
        return {"status": "invalid_report", "display": "Report could not be read", "source": "invalid_report", "cv_report": str(SEQUENTIAL_CV_PATH), "metrics": None, "report_error": cv_report["_read_error"]}

    metrics = cv_report.get("out_of_fold_metrics", {})
    fold_results = cv_report.get("fold_results", [])
    max_clauses = max((len(fold.get("clauses", [])) for fold in fold_results), default=0)
    clauses = (training_report or {}).get("learned_clauses", [])
    clause_count = len(clauses) or max_clauses
    precision = float(metrics.get("precision", 0.0))
    lift = float(metrics.get("lift", 0.0))
    return {
        "status": "successful_sequential_program" if clause_count else "no_stable_program",
        "display": (
            f"{clause_count}-clause sequential program (OOF Precision: {precision:.0%}, OOF Lift: {lift:.2f}x)"
            if clause_count else "No sequential program found in the available CV folds"
        ),
        "source": "phase5_sequential_cv.json",
        "cv_report": str(SEQUENTIAL_CV_PATH),
        "training_report": str(SEQUENTIAL_TRAINING_PATH) if training_report else None,
        "metrics": metrics,
        "clauses": clause_count,
        "patterns": sorted({
            name for fold in fold_results for clause in fold.get("clauses", [])
            for name in _clause_body(clause) if name.startswith("seq_pattern_")
        }),
        "held_out_sequences": cv_report.get("n_labelled_sequences"),
    }


def _miner_only_result(cv_report: dict | None) -> dict:
    """Read the no-Popper pattern-union baseline scored inside hybrid CV."""
    if cv_report is None or cv_report.get("_read_error"):
        return {"status": "not_available", "display": "Miner-only fold scores missing", "metrics": None}
    metrics = cv_report.get("miner_only_out_of_fold_metrics")
    folds = cv_report.get("fold_results", [])
    if not metrics or not folds or any("miner_only_test_metrics" not in fold for fold in folds):
        return {
            "status": "not_run",
            "display": "Miner-only scores absent; rerun sequential CV with the updated runner",
            "source": "phase5_sequential_cv.json",
            "metrics": None,
        }
    return {
        "status": "evaluated",
        "display": "OR of fold-local mined patterns; no Popper",
        "source": "phase5_sequential_cv.json",
        "metrics": metrics,
        "folds": len(folds),
        "seed": cv_report.get("seed"),
    }


def _decision_tree_result(report: dict | None) -> dict:
    if report is None:
        return {"status": "not_run", "display": "Report not available", "source": "report_missing", "report": str(DECISION_TREE_PATH), "metrics": None}
    if report.get("_read_error"):
        return {"status": "invalid_report", "display": "Report could not be read", "source": "invalid_report", "report": str(DECISION_TREE_PATH), "metrics": None, "report_error": report["_read_error"]}
    metrics = report.get("out_of_fold_metrics", {})
    return {
        "status": "evaluated",
        "display": f"Depth-{report.get('max_depth', '?')} CART (OOF balanced accuracy: {float(metrics.get('balanced_accuracy', 0)):.3f})",
        "source": "phase5_decision_tree_cv.json",
        "report": str(DECISION_TREE_PATH),
        "metrics": metrics,
        "folds": report.get("folds"),
        "seed": report.get("seed"),
    }


def run() -> dict:
    # Never let a failed comparison leave a stale report that looks current.
    OUTPUT_JSON.unlink(missing_ok=True)
    OUTPUT_MD.unlink(missing_ok=True)
    baseline_report = _read_report(BASELINE_PATH)
    sequential_report = _read_report(SEQUENTIAL_CV_PATH)
    tree_report = _read_report(DECISION_TREE_PATH)
    baseline = _baseline_result(baseline_report)
    sequential = _sequential_result(sequential_report, _read_report(SEQUENTIAL_TRAINING_PATH))
    miner_only = _miner_only_result(sequential_report)
    decision_tree = _decision_tree_result(tree_report)
    baseline = _add_common_metrics(baseline)
    sequential = _add_common_metrics(sequential)
    miner_only = _add_common_metrics(miner_only)
    decision_tree = _add_common_metrics(decision_tree)
    hybrid_folds = (sequential_report or {}).get("fold_results", [])
    tree_folds = (tree_report or {}).get("fold_results", [])
    split_signatures = {
        "miner_only": _fold_signature(hybrid_folds)
        if hybrid_folds and all("miner_only_test_metrics" in fold for fold in hybrid_folds) else (),
        "hybrid_sequential": _fold_signature(hybrid_folds),
        "decision_tree": _fold_signature(tree_folds),
    }
    complete_signatures = {key: value for key, value in split_signatures.items() if value}
    protocol_checks = {
        "hybrid_sequential": (sequential_report or {}).get("comparison_protocol_revision") == COMPARISON_PROTOCOL_REVISION,
        "decision_tree": (tree_report or {}).get("comparison_protocol_revision") == COMPARISON_PROTOCOL_REVISION,
    }
    split_alignment = {
        "status": "matched" if len(complete_signatures) == 3 and len(set(complete_signatures.values())) == 1 else "mismatch_or_incomplete",
        "reports_with_fold_splits": sorted(complete_signatures),
        "protocol_revision": COMPARISON_PROTOCOL_REVISION,
        "protocol_checks": protocol_checks,
    }
    test_population_signatures = {
        "miner_only": _test_population_signature(hybrid_folds)
        if hybrid_folds and all("miner_only_test_metrics" in fold for fold in hybrid_folds) else (),
        "hybrid_sequential": _test_population_signature(hybrid_folds),
        "decision_tree": _test_population_signature(tree_folds),
    }
    complete_test_populations = {key: value for key, value in test_population_signatures.items() if value}
    test_population_alignment = {
        "status": "matched" if len(complete_test_populations) == 3 and len(set(complete_test_populations.values())) == 1 else "mismatch_or_incomplete",
        "reports_with_fingerprints": sorted(complete_test_populations),
    }
    comparison_ready = (
        split_alignment["status"] == "matched"
        and test_population_alignment["status"] == "matched"
        and all(protocol_checks.values())
        and miner_only.get("status") == "evaluated"
    )
    result = {
        "methods": {
            "miner_only": miner_only,
            "decision_tree": decision_tree,
            "hybrid_sequential": sequential,
        },
        "secondary_reference_arm0_full_tactical": baseline,
        "comparison_basis": "The three primary methods use the same held-out match folds and the same match/sequence/label test population, verified by a per-fold SHA-256 fingerprint, at natural prevalence. Miner-only applies an OR rule to exactly the compressed fold-local pattern set offered to hybrid Popper; this isolates the value of Popper's rule search and base-vocabulary combinations. CART uses the 34 Boolean sequence features. Mining uses all labelled training sequences, while CART and Popper use the context-matched training examples. Arm 0 is retained as a secondary reference, not part of the primary comparison.",
        "comparison_ready": comparison_ready,
        "validation_scope": "Descriptive pooled out-of-fold performance from one fixed three-fold match-grouped split; no significance test or independent-season generalization claim is made.",
        "fold_split_alignment": split_alignment,
        "test_population_alignment": test_population_alignment,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = [
        "# مقایسهٔ روش‌های فاز ۵",
        "",
        f"وضعیت مقایسهٔ سه بازوی اصلی: **{'آماده' if comparison_ready else 'نیازمند اجرای هم‌پروتکل'}** (پروتکل `{COMPARISON_PROTOCOL_REVISION}`).",
        "",
        "معیارها از پیش‌بینی‌های out-of-fold تجمیع‌شده روی مسابقه‌های آزمون و با شیوع طبیعی پیامد محاسبه می‌شوند. این گزارش توصیفی است؛ با یک اجرای سه‌fold به‌تنهایی ادعای معناداری آماری یا تعمیم به فصل دیگر مطرح نمی‌شود.",
        "",
        "| روش اصلی | OOF N | TP / FP / FN / TN | Support | Precision | Recall | Specificity | F1 | Lift | Balanced accuracy | MCC | وضعیت |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        "",
    ]
    for label, method in (("ماینر تنها (OR الگوهای fold-local)", miner_only), ("درخت تصمیم CART", decision_tree), ("هیبرید ماینر + Popper", sequential)):
        metrics = method.get("metrics") or {}
        oof_n = int(metrics.get("n_positive", 0)) + int(metrics.get("n_negative", 0))
        confusion = " / ".join(str(metrics.get(key, "—")) for key in ("tp", "fp", "fn", "tn"))
        values = [str(oof_n) if oof_n else "—", confusion, str(metrics.get("support", "—"))]
        values += [f"{float(metrics[key]):.3f}" if key in metrics else "—" for key in ("precision", "recall", "specificity", "f1", "lift", "balanced_accuracy", "mcc")]
        rows.append(f"| {label} | " + " | ".join(values) + f" | {method['display']} |")
    rows.append("")
    rows.append(f"هم‌ترازی foldهای سه روش اصلی: **{split_alignment['status']}**.")
    rows.append(f"همسانی دقیق مسابقه‌ها، توالی‌ها و برچسب‌های آزمون (SHA-256 هر fold): **{test_population_alignment['status']}**.")
    if not comparison_ready:
        missing = [name for name, ok in protocol_checks.items() if not ok]
        if split_alignment["status"] != "matched":
            missing.append("match-fold alignment")
        if test_population_alignment["status"] != "matched":
            missing.append("held-out sequence/label fingerprint alignment; rerun both current CV cells")
        if miner_only.get("status") != "evaluated":
            missing.append("fold-local miner-only OOF scores; rerun the updated sequential CV cell")
        rows.append("آمادگی مقایسه تأیید نشد: " + "; ".join(missing) + ".")
    rows.append(f"بازوی صفرم (`full_tactical` Popper، خارج از مقایسهٔ اصلی): {baseline['display']}.")
    if sequential.get("metrics", {}).get("patterns"):
        rows += ["", "الگوهای گزارش‌شده: " + ", ".join(f"`{name}`" for name in sequential["metrics"]["patterns"]) + "."]
    OUTPUT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Comparison report -> {OUTPUT_MD}")
    print(f"Machine-readable comparison -> {OUTPUT_JSON}")
    print("\n".join(rows[4:6]))
    return result


if __name__ == "__main__":
    run()
