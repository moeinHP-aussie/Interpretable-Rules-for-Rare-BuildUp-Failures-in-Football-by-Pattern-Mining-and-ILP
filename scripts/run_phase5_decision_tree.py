"""Primary Arm 2: shallow CART on phase-5 grouped match folds.

The tree uses the same sequence-level Boolean tactical vocabulary and the
same context-matched training pool / match-grouped folds as full_tactical
Popper. It is intentionally capped at depth 3 and needs no extra dependency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tactical_ilp.config import PROLOG_DIR, REPORTS_DIR  # noqa: E402
from tactical_ilp.prolog.phase5_vocabulary import (  # noqa: E402
    POPPER_WRAPPED_ACTION_PREDS,
    SEQUENCE_PREDS_1,
    WRAPPED_PLAYER_SEQ_PREDS,
)
from phase5_cv_common import (  # noqa: E402
    COMPARISON_PROTOCOL_REVISION,
    EVALUABLE_LABELS,
    _balanced_group_folds,
    holdout_population,
    test_population_fingerprint,
)

PROCESSED_DIR = PROLOG_DIR.parent / "processed"
MATCHED_PATH = PROCESSED_DIR / "ilp_examples.parquet"
LABELLED_PATH = PROCESSED_DIR / "labeled_sequences.parquet"
POPPER_BK = PROLOG_DIR / "popper" / "bk.pl"
REPORT_PATH = REPORTS_DIR / "phase5_decision_tree_cv.json"
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10


def _feature_names() -> list[str]:
    return (
        [f"seq_{name}" for name in POPPER_WRAPPED_ACTION_PREDS]
        + [f"seq_{name}" for name in WRAPPED_PLAYER_SEQ_PREDS]
        + list(SEQUENCE_PREDS_1)
    )


def _prolog_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "\\'")


def _load_feature_matrix(features: list[str]) -> dict[str, tuple[int, ...]]:
    """Evaluate the active Popper sequence predicates once for all sequences."""
    if not POPPER_BK.exists():
        raise FileNotFoundError(f"missing phase-5 background knowledge: {POPPER_BK}")
    feature_terms = ",".join(features)
    goal = (
        f"consult('{_prolog_quote(POPPER_BK)}'),"
        f"findall(S,sequence(S),Raw),sort(Raw,Sequences),"
        f"forall(member(S,Sequences),(write(S),write('|'),"
        f"forall(member(P,[{feature_terms}]),"
        f"( (call(P,S)->write(1);write(0)) )),nl)),halt"
    )
    proc = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-q", "-g", goal],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    if proc.returncode or "ERROR:" in proc.stderr:
        raise RuntimeError(f"SWI-Prolog feature extraction failed:\n{proc.stderr}\n{proc.stdout}")
    matrix: dict[str, tuple[int, ...]] = {}
    for line in proc.stdout.splitlines():
        sequence_id, separator, values = line.partition("|")
        if not separator or len(values) != len(features) or set(values) - {"0", "1"}:
            raise RuntimeError(f"unexpected Prolog feature row: {line!r}")
        matrix[sequence_id] = tuple(int(value) for value in values)
    if not matrix:
        raise RuntimeError("Prolog returned no sequences for decision-tree features")
    return matrix


def _gini(labels: list[int]) -> float:
    if not labels:
        return 0.0
    positive = sum(labels) / len(labels)
    return 2.0 * positive * (1.0 - positive)


def _fit_tree(
    rows: list[tuple[tuple[int, ...], int]],
    feature_names: list[str],
    depth: int = 0,
) -> dict:
    labels = [label for _, label in rows]
    positives = sum(labels)
    node = {"n": len(rows), "negative_outcome": positives, "non_negative": len(rows) - positives}
    if (
        depth >= MAX_DEPTH
        or positives == 0
        or positives == len(rows)
        or len(rows) < 2 * MIN_SAMPLES_LEAF
    ):
        node["leaf"] = True
        return node

    parent_impurity = _gini(labels)
    best: tuple[float, int, list, list] | None = None
    for feature_index, feature_name in enumerate(feature_names):
        left = [row for row in rows if row[0][feature_index] == 0]
        right = [row for row in rows if row[0][feature_index] == 1]
        if min(len(left), len(right)) < MIN_SAMPLES_LEAF:
            continue
        weighted = (len(left) * _gini([label for _, label in left])
                    + len(right) * _gini([label for _, label in right])) / len(rows)
        gain = parent_impurity - weighted
        candidate = (gain, feature_index, left, right)
        if best is None or gain > best[0] + 1e-12:
            best = candidate

    if best is None or best[0] <= 1e-12:
        node["leaf"] = True
        return node
    _, feature_index, left_rows, right_rows = best
    node.update({
        "leaf": False,
        "feature": feature_names[feature_index],
        "when_absent": _fit_tree(left_rows, feature_names, depth + 1),
        "when_present": _fit_tree(right_rows, feature_names, depth + 1),
    })
    return node


def _predict(tree: dict, values: tuple[int, ...], feature_index: dict[str, int]) -> int:
    node = tree
    while not node["leaf"]:
        child = "when_present" if values[feature_index[node["feature"]]] else "when_absent"
        node = node[child]
    # A tie predicts the negative-outcome class, fixed before evaluation.
    return int(node["negative_outcome"] >= node["non_negative"])


def _metrics(actual: list[int], predicted: list[int]) -> dict[str, int | float]:
    tp = sum(y == 1 and p == 1 for y, p in zip(actual, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(actual, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(actual, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(actual, predicted))
    n = len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
    }


def _tree_text(node: dict, indent: str = "") -> list[str]:
    if node["leaf"]:
        predicted = "negative outcome" if node["negative_outcome"] >= node["non_negative"] else "non-negative"
        return [f"{indent}leaf: {predicted} (n={node['n']}, negative={node['negative_outcome']}, non_negative={node['non_negative']})"]
    rows = [f"{indent}if {node['feature']} is present:"]
    rows.extend(_tree_text(node["when_present"], indent + "  "))
    rows.append(f"{indent}else:")
    rows.extend(_tree_text(node["when_absent"], indent + "  "))
    return rows


def run(folds_count: int, seed: int) -> dict:
    REPORT_PATH.unlink(missing_ok=True)
    missing = [str(path) for path in (MATCHED_PATH, LABELLED_PATH, POPPER_BK) if not path.exists()]
    if missing:
        raise FileNotFoundError("run phases 1-5 first; missing:\n  " + "\n  ".join(missing))
    labelled = pd.read_parquet(LABELLED_PATH)
    matched = pd.read_parquet(MATCHED_PATH)
    required = {"sequence_id", "match_id", "label"}
    if not required.issubset(labelled.columns) or not required.issubset(matched.columns):
        raise ValueError("labelled and matched parquet files need sequence_id, match_id, label")
    sampled_ids = set(matched.sequence_id)
    train_pool_all = labelled[labelled.sequence_id.isin(sampled_ids)].copy()
    train_pool = train_pool_all[train_pool_all.label.isin(EVALUABLE_LABELS)].copy()
    all_matches = sorted(labelled.match_id.astype(str).unique())
    folds = _balanced_group_folds(train_pool, all_matches, folds_count, seed)
    features = _feature_names()
    matrix = _load_feature_matrix(features)
    missing_sequences = sorted(set(labelled.sequence_id.astype(str)) - set(matrix))
    if missing_sequences:
        raise ValueError(f"Prolog feature matrix misses {len(missing_sequences)} labelled sequences")

    index = {name: i for i, name in enumerate(features)}
    results = []
    totals = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for fold_number, test_matches in enumerate(folds, start=1):
        raw_train = train_pool_all[~train_pool_all.match_id.astype(str).isin(test_matches)]
        n_train_inconclusive = int((~raw_train.label.isin(EVALUABLE_LABELS)).sum())
        train = raw_train[raw_train.label.isin(EVALUABLE_LABELS)]
        test, n_test_inconclusive = holdout_population(labelled, test_matches)
        train_rows = [
            (matrix[str(row.sequence_id)], int(row.label == "negative"))
            for row in train.itertuples(index=False)
        ]
        test_rows = [
            (matrix[str(row.sequence_id)], int(row.label == "negative"))
            for row in test.itertuples(index=False)
        ]
        if not train_rows or not test_rows:
            raise ValueError(f"fold {fold_number} has no training or test sequences")
        tree = _fit_tree(train_rows, features)
        actual = [label for _, label in test_rows]
        predicted = [_predict(tree, values, index) for values, _ in test_rows]
        scores = _metrics(actual, predicted)
        for key in totals:
            totals[key] += int(scores[key])
        results.append({
            "fold": fold_number,
            "train_matches": sorted(set(train.match_id.astype(str))),
            "test_matches": sorted(test_matches),
            "n_train_context_matched_examples": len(train),
            "n_train_inconclusive_excluded": n_train_inconclusive,
            "n_test_all_labelled_sequences": len(test),
            "n_test_inconclusive_excluded": n_test_inconclusive,
            "test_population_fingerprint": test_population_fingerprint(test),
            "metrics": scores,
            "tree": tree,
            "readable_tree": "\n".join(_tree_text(tree)),
        })

    aggregate = _metrics_from_counts(totals)
    report = {
        "method": "depth_3_cart_full_tactical_grouped_match_k_fold_cv",
        "comparison_protocol_revision": COMPARISON_PROTOCOL_REVISION,
        "model": "binary CART decision tree; maximum depth 3; minimum leaf size 10",
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "feature_count": len(features),
        "features": features,
        "feature_source": "same sequence-level Boolean predicates offered to full_tactical Popper",
        "training_sampling": "phase-2 context-matched examples, restricted to training matches",
        "evaluation_population": "all labelled sequences in held-out matches at natural prevalence",
        "folds": folds_count,
        "seed": seed,
        "n_matches": int(labelled.match_id.nunique()),
        "n_labelled_sequences": int(len(labelled)),
        "out_of_fold_metrics": aggregate,
        "fold_results": results,
        "interpretation": "Classical, interpretable baseline. Compare on the same grouped folds and input features; this internal validation is not an independent new-season test.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, int | float]:
    return _metrics(
        [1] * counts["tp"] + [1] * counts["fn"] + [0] * counts["fp"] + [0] * counts["tn"],
        [1] * counts["tp"] + [0] * counts["fn"] + [1] * counts["fp"] + [0] * counts["tn"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202021)
    args = parser.parse_args()
    if args.folds < 3:
        parser.error("--folds must be at least 3")
    try:
        report = run(args.folds, args.seed)
    except Exception as exc:
        print(f"DECISION-TREE BASELINE FAILED: {exc}", file=sys.stderr)
        return 1
    metrics = report["out_of_fold_metrics"]
    print("\nMATCH-GROUPED CROSS-VALIDATION — DECISION-TREE BASELINE")
    print(f"out-of-fold sequences: {metrics['n_positive'] + metrics['n_negative']}")
    print(f"accuracy / balanced accuracy: {metrics['accuracy']:.3f} / {metrics['balanced_accuracy']:.3f}")
    print(f"precision / recall / F1: {metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}")
    print(f"MCC: {metrics['mcc']:.3f}")
    print(f"Detailed report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
