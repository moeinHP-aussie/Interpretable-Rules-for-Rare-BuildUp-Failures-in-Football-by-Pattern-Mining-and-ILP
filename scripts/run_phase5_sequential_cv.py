"""Match-grouped CV for the hybrid ordered-pattern tactical method.

Every fold mines patterns on training matches, offers mined patterns plus the
complete revised base vocabulary to Popper, and evaluates on unseen matches.
Pattern components remain available as standalone predicates; only redundant
same-clause pattern/component combinations are forbidden by the bias.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import PROCESSED_DIR, PROLOG_DIR, REPORTS_DIR  # noqa: E402
from tactical_ilp.prolog.phase5_vocabulary import POPPER_MAX_BODY, POPPER_MAX_CLAUSES, POPPER_MAX_VARS, POPPER_WRAPPED_ACTION_PREDS, build_bk_wrapper  # noqa: E402
from tactical_ilp.prolog.patterns_only_inputs import build_hybrid_bias, select_compressing_patterns  # noqa: E402
from tactical_ilp.sequences.sequential_patterns import (  # noqa: E402
    MiningConfig,
    build_pattern_sequences,
    decode_itemsets,
    fit_patterns,
    pattern_matches,
    search_space_proxy,
    write_pattern_facts,
)
from phase5_cv_common import (  # noqa: E402
    COMPARISON_PROTOCOL_REVISION,
    EVALUABLE_LABELS,
    LABELLED_PATH,
    MATCHED_PATH,
    _balanced_group_folds,
    _build_intermediate_candidate_report,
    _evaluate_hypothesis_snapshots,
    _intermediate_candidate_markdown,
    _parse_solution,
    _run_popper,
    _score_program,
    _write_examples,
    holdout_population,
    test_population_fingerprint,
)

INPUT_CACHE = PROCESSED_DIR / "sequential_method" / "pattern_sequences.parquet"
LOG_DIR = REPORTS_DIR / "phase5_sequential_cv"
REPORT_PATH = REPORTS_DIR / "phase5_sequential_cv.json"
REPORT_MD = REPORTS_DIR / "phase5_sequential_cv.md"


def _score_no_rules(examples: pd.DataFrame) -> dict[str, int | float]:
    n_pos = int((examples.label == "negative").sum())
    n_neg = int((examples.label == "non_negative").sum())
    n = n_pos + n_neg
    return {
        "n_positive": n_pos, "n_negative": n_neg, "support": 0,
        "tp": 0, "fp": 0, "fn": n_pos, "tn": n_neg,
        "base_rate": round(n_pos / n, 6) if n else 0.0,
        "precision": 0.0, "recall": 0.0, "lift": 0.0,
    }


def _metrics_sum(rows: list[dict], metric_key: str = "test_metrics") -> dict:
    keys = ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")
    total = {key: sum(int(row[metric_key][key]) for row in rows) for key in keys}
    n = total["n_positive"] + total["n_negative"]
    precision = total["tp"] / total["support"] if total["support"] else 0.0
    recall = total["tp"] / total["n_positive"] if total["n_positive"] else 0.0
    base = total["n_positive"] / n if n else 0.0
    return {
        **total,
        "base_rate": round(base, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "lift": round(precision / base, 6) if base else 0.0,
    }


def _score_miner_only(test_rows: pd.DataFrame, pattern_events: dict[str, tuple], selected: list, max_gap: int) -> dict[str, int | float]:
    """Evaluate the fold-local pattern union with no Popper or base predicates."""
    actual = [int(row.label == "negative") for row in test_rows.itertuples(index=False)]
    predicted = [
        int(any(
            pattern_matches(pattern_events[str(row.sequence_id)], candidate.pattern, max_gap)
            for candidate in selected
        ))
        for row in test_rows.itertuples(index=False)
    ]
    tp = sum(y == 1 and p == 1 for y, p in zip(actual, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(actual, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(actual, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(actual, predicted))
    positives, negatives = tp + fn, tn + fp
    support = tp + fp
    base_rate = positives / len(actual) if actual else 0.0
    precision = tp / support if support else 0.0
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    return {
        "n_positive": positives,
        "n_negative": negatives,
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "base_rate": base_rate,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": (tp + tn) / len(actual) if actual else 0.0,
        "mcc": ((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5))
        if (tp + fp) and (tp + fn) and (tn + fp) and (tn + fn) else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
    }


def _write_markdown(report: dict) -> None:
    metrics = report["out_of_fold_metrics"]
    miner = report["miner_only_out_of_fold_metrics"]
    rows = [
        "# Hybrid ordered-pattern match-grouped CV",
        "",
        "Fold-local ordered combinations of action-level tactical predicates were mined from training matches. Each complete pattern is one atomic Popper feature. The complete revised base vocabulary remains available alongside patterns; same-clause pattern/component combinations are forbidden.",
        "",
        f"- Folds: {report['folds']}; seed: `{report['seed']}`; Popper timeout/fold: {report['timeout_seconds']} s.",
        f"- Held-out sequences: {metrics['n_positive'] + metrics['n_negative']}; base rate: {metrics['base_rate']:.3f}.",
        f"- Predicted support: {metrics['support']} (TP={metrics['tp']}, FP={metrics['fp']}); precision: {metrics['precision']:.3f}; recall: {metrics['recall']:.3f}; lift: {metrics['lift']:.3f}.",
        f"- Miner-only (OR of the same fold-local selected patterns; no Popper): support {miner['support']} (TP={miner['tp']}, FP={miner['fp']}); precision {miner['precision']:.3f}; recall {miner['recall']:.3f}; lift {miner['lift']:.3f}.",
        "- Metrics pool all held-out matches across folds. Pattern mining and selection use training matches only; Popper training uses the project's context-matched sample.",
        f"- Pattern events come from the {len(POPPER_WRAPPED_ACTION_PREDS)} currently active Popper action-level concepts; sequence summaries, player summaries, and absence complements are not treated as ordered events.",
        "- This estimates internal match-grouped performance, not cross-season generalization.",
        "",
        "## Fold details",
        "",
        "| Fold | Patterns selected | Predicates offered | Popper seconds | Provisional | Hard timeout | Support | Precision | Recall | Lift |",
        "|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for fold in report["fold_results"]:
        m = fold["test_metrics"]
        rows.append(
            f"| {fold['fold']} | {len(fold['selected_patterns'])} | {fold['body_predicate_count']} | "
            f"{fold['popper_seconds']:.1f} | {fold['provisional']} | {fold['hard_timeout']} | "
            f"{m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['lift']:.3f} |"
        )
    rows += ["", "## Selected fold-local patterns", ""]
    for fold in report["fold_results"]:
        rows.extend([f"### Fold {fold['fold']}", ""])
        if not fold["selected_patterns"]:
            rows.extend(["No pattern passed the configured filters.", ""])
            continue
        for item in fold["selected_patterns"]:
            rows.append(
                f"- `{item['predicate']}`: `{item['pattern_text']}`; target support "
                f"{item['training_target_support']}, control support {item['training_control_support']}, "
                f"smoothed growth {item['growth_rate']:.2f}x (log2={item['log2_growth']:.2f}); "
                f"target/control rates {item['target_rate']:.1%}/{item['control_rate']:.1%}."
            )
        rows.append("")
    rows.extend(_intermediate_candidate_markdown(report))
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(folds_count: int, seed: int, popper_home: Path, timeout: int, config: MiningConfig, force_inputs: bool) -> dict:
    # Keep the notebook from rendering a previous run if this run fails early.
    REPORT_PATH.unlink(missing_ok=True)
    REPORT_MD.unlink(missing_ok=True)
    missing = [str(path) for path in (LABELLED_PATH, MATCHED_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing phases 1--5 outputs:\n  " + "\n  ".join(missing))
    pattern_frame = build_pattern_sequences(
        force=force_inputs,
        excluded_suffix_actions=config.excluded_suffix_actions,
        vocabulary="tactical",
        cache_path=INPUT_CACHE,
    )
    pattern_events = {
        str(row.sequence_id): decode_itemsets(row.itemsets_json)
        for row in pattern_frame.itertuples(index=False)
    }
    labelled = pd.read_parquet(LABELLED_PATH)
    matched = pd.read_parquet(MATCHED_PATH)
    sampled_ids = set(matched.sequence_id.astype(str))
    train_pool_all = labelled[labelled.sequence_id.astype(str).isin(sampled_ids)].copy()
    train_pool = train_pool_all[train_pool_all.label.isin(EVALUABLE_LABELS)].copy()
    all_matches = sorted(labelled.match_id.astype(str).unique())
    folds = _balanced_group_folds(train_pool, all_matches, folds_count, seed)
    PROLOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    clause_folds: dict[tuple[int | None, str], set[int]] = defaultdict(set)
    intermediate_snapshots: list[dict] = []
    intermediate_rule_snapshots: list[dict] = []
    for index, test_matches in enumerate(folds, 1):
        fold_name = f"fold_{index:02d}"
        # Keep each task beside data/prolog/facts_all.pl because the shared
        # generated BK wrapper resolves its ../facts_*.pl consult paths there.
        task_dir = PROLOG_DIR / f"popper_sequential_cv_{fold_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        raw_train_rows = train_pool_all[~train_pool_all.match_id.astype(str).isin(test_matches)]
        n_train_inconclusive = int((~raw_train_rows.label.isin(EVALUABLE_LABELS)).sum())
        train_rows = raw_train_rows[raw_train_rows.label.isin(EVALUABLE_LABELS)]
        test_rows, n_test_inconclusive = holdout_population(labelled, test_matches)
        train_matches = sorted(set(train_pool.match_id.astype(str)) - test_matches)
        _write_examples(train_rows, task_dir / "exs.pl")
        _write_examples(test_rows, task_dir / "exs_holdout.pl")

        diagnostics: dict[str, int | float] = {}
        started = time.monotonic()
        selected = fit_patterns(pattern_frame, train_matches, config, diagnostics)
        selected = select_compressing_patterns(selected, config.max_selected_patterns)
        miner_only_metrics = _score_miner_only(test_rows, pattern_events, selected, config.max_gap)
        pattern_fallback = not selected
        diagnostics["no_pattern_fallback"] = pattern_fallback
        diagnostics["patterns_after_vocabulary_compression"] = len(selected)
        diagnostics["mining_seconds"] = time.monotonic() - started
        fact_info = write_pattern_facts(pattern_frame, selected, task_dir / "facts_patterns.pl", config)
        bk_path = build_bk_wrapper(task_dir / "bk.pl")
        with bk_path.open("a", encoding="utf-8") as stream:
            stream.write("\n:- consult('facts_patterns.pl').\n")
        predicates = tuple(item["predicate"] for item in fact_info["features"])

        bias_info = build_hybrid_bias(task_dir / "bias.pl", fact_info["features"])
        if pattern_fallback:
            print(f"{fold_name}: no pattern met mining thresholds; using full tactical vocabulary for this fold.", flush=True)
        print(f"Starting hybrid-pattern {fold_name}/{folds_count}: {len(predicates)} mined patterns + {bias_info['retained_base_predicates']} retained base predicates, total={bias_info['body_predicates']}, {timeout}s Popper budget.", flush=True)
        started = time.monotonic()
        output, hard_timeout = _run_popper(task_dir, popper_home, timeout)
        popper_seconds = time.monotonic() - started
        (LOG_DIR / f"{fold_name}_popper_output.txt").write_text(output, encoding="utf-8")
        fold_candidates, fold_rule_candidates = _evaluate_hypothesis_snapshots(
            output, index, bk_path, task_dir / "exs.pl", task_dir / "exs_holdout.pl"
        )
        intermediate_snapshots.extend(fold_candidates)
        intermediate_rule_snapshots.extend(fold_rule_candidates)
        clauses, provisional = _parse_solution(output)
        metrics = _score_program(clauses, bk_path, task_dir / "exs_holdout.pl")

        for body in clauses:
            canonical_body = ",".join(sorted(body))
            fold_local = any(predicate.startswith("seq_pattern_") for predicate in body)
            # Pattern IDs are fold-local names, not shared semantic IDs.
            # Keep pattern-containing rules scoped to their discovery fold.
            clause_folds[(index if fold_local else None, canonical_body)].add(index)
        fold_result = {
            "fold": index,
            "train_matches": train_matches,
            "test_matches": sorted(test_matches),
            "n_train_matched_examples": int(len(train_rows)),
            "n_train_inconclusive_excluded": n_train_inconclusive,
            "n_test_all_labelled_sequences": int(len(test_rows)),
            "n_test_inconclusive_excluded": n_test_inconclusive,
            "test_population_fingerprint": test_population_fingerprint(test_rows),
            "selected_patterns": fact_info["features"],
            "pattern_mining": diagnostics,
            "pattern_predicate_count": len(predicates),
            "miner_only_test_metrics": miner_only_metrics,
            "pattern_fallback": pattern_fallback,
            "effective_method": "full_tactical_fallback_no_patterns" if pattern_fallback else "hybrid_sequential",
            "body_predicate_count": int(bias_info["body_predicates"]),
            "hybrid_vocabulary": {key: value for key, value in bias_info.items() if key != "path"},
            "search_space_proxy": search_space_proxy(int(bias_info["body_predicates"]), POPPER_MAX_BODY, POPPER_MAX_CLAUSES),
            "clauses": clauses,
            "test_metrics": metrics,
            "popper_seconds": popper_seconds,
            "hard_timeout": hard_timeout,
            "provisional": provisional,
            "no_solution": not clauses,
        }
        results.append(fold_result)
        print(
            f"{fold_name}: patterns={len(predicates)}, support={metrics['support']}, "
            f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
            f"lift={metrics['lift']:.3f}, provisional={provisional}, hard_timeout={hard_timeout}",
            flush=True,
        )

    report = {
        "method": "hybrid_sequential_full_tactical_grouped_match_k_fold_cv",
        "comparison_protocol_revision": COMPARISON_PROTOCOL_REVISION,
        "folds": folds_count,
        "seed": seed,
        "timeout_seconds": timeout,
        "n_matches": int(labelled.match_id.nunique()),
        "n_labelled_sequences": int(len(labelled)),
        "training_sampling": "phase-2 context-matched examples from training matches",
        "evaluation_population": "all labelled sequences in held-out matches at natural prevalence",
        "mining_population": "all labelled sequences from training matches only, refit per fold",
        "label_target": "negative outcome; non_negative is the control class",
        "pattern_input": str(INPUT_CACHE),
        "mining_config": config.__dict__,
        "body_predicates_offered": "fold-local selected seq_pattern_* atoms plus the complete 34-predicate revised base vocabulary; aerial_pass and set-piece action events are excluded, while pattern components remain available as standalone predicates",
        "ordered_event_vocabulary": "selected 2-to-3-event patterns mined from the active action-level tactical predicates",
        "popper_max_body": POPPER_MAX_BODY,
        "popper_max_vars": POPPER_MAX_VARS,
        "popper_max_clauses": POPPER_MAX_CLAUSES,
        "out_of_fold_metrics": _metrics_sum(results),
        "miner_only_out_of_fold_metrics": _metrics_sum(results, "miner_only_test_metrics"),
        "rule_stability": [
            {
                "canonical_body": body,
                "fold_local_pattern_scope": fold_scope,
                "folds_found": len(found),
                "fold_ids": sorted(found),
            }
            for (fold_scope, body), found in sorted(
                clause_folds.items(),
                key=lambda item: (-len(item[1]), item[0][0] or 0, item[0][1]),
            )
        ],
        "intermediate_candidates": {
            **_build_intermediate_candidate_report(intermediate_snapshots, folds_count),
            "individual_rules": _build_intermediate_candidate_report(intermediate_rule_snapshots, folds_count)["candidates"],
        },
        "fold_results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report)
    print(f"\nDetailed quality report -> {REPORT_PATH}\nReadable report -> {REPORT_MD}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202021)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-target-support", type=int, default=5)
    parser.add_argument("--min-growth-rate", type=float, default=1.5)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--max-patterns", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=50_000)
    parser.add_argument("--exclude-suffix-actions", type=int, default=1)
    parser.add_argument("--force-pattern-inputs", action="store_true")
    args = parser.parse_args()
    if args.folds < 3:
        parser.error("--folds must be at least 3")
    popper_home = os.environ.get("POPPER_DIR")
    if not popper_home:
        parser.error("POPPER_DIR must point to the Popper checkout")
    config = MiningConfig(
        max_gap=args.max_gap,
        min_support=args.min_support,
        min_target_support=args.min_target_support,
        min_growth_rate=args.min_growth_rate,
        min_pattern_events=args.min_events,
        max_pattern_events=args.max_events,
        max_selected_patterns=args.max_patterns,
        max_candidates=args.max_candidates,
        excluded_suffix_actions=args.exclude_suffix_actions,
    )
    try:
        run(args.folds, args.seed, Path(popper_home), args.timeout, config, args.force_pattern_inputs)
    except Exception as exc:
        print(f"SEQUENTIAL CV FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged("phase5_sequential_cv", main))
