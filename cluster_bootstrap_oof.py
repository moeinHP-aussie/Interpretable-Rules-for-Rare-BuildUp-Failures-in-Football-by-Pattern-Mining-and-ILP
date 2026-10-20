"""Reconstruct archived OOF predictions and estimate match-clustered uncertainty.

This is post-processing only: it does not refit the miner, Popper, or CART.
It applies the fold-specific saved programs/trees to the saved fold test data,
checks their confusion counts against the archived CV reports, and bootstraps
matches within their original test fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np


TEAM_LABELS = {
    "barcelona_2020_21": "Barcelona 2020/21",
    "leverkusen_2023_24": "Bayer Leverkusen 2023/24",
}
ARM_ORDER = ("hybrid", "miner_only", "popper_no_patterns", "cart")
CONFUSION_KEYS = ("tp", "fp", "fn", "tn")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _examples(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    pattern = re.compile(r"(pos|neg)\(negative_outcome\(([^)]+)\)\)\.")
    for line in path.read_text(encoding="utf-8").splitlines():
        found = pattern.fullmatch(line.strip())
        if found:
            rows[found.group(2)] = int(found.group(1) == "pos")
    return rows


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "\\'")


def _normalize_clauses(clauses: list) -> list[list[str]]:
    normalized = []
    for clause in clauses:
        body = clause.get("body", []) if isinstance(clause, dict) else clause
        if isinstance(body, (list, tuple)) and body:
            normalized.append([str(predicate) for predicate in body])
    return normalized


def _prolog_predictions(bk: Path, examples: Path, clauses: list[list[str]]) -> list[tuple[str, int, int]]:
    """Return (sequence_id, actual_label, prediction) from saved Prolog facts."""
    goals = [f"consult('{_quote(bk)}')", f"consult('{_quote(examples)}')"]
    for index, body in enumerate(clauses):
        goals.append(
            f"assertz((archived_clause_{index}(S) :- "
            + ", ".join(f"{predicate}(S)" for predicate in body)
            + "))"
        )
    if clauses:
        disjunction = ";".join(f"archived_clause_{index}(S)" for index in range(len(clauses)))
        goals.append(f"assertz((archived_predict(S) :- ({disjunction})))")
    else:
        goals.append("assertz((archived_predict(_) :- fail))")
    goals.append(
        "forall((pos(negative_outcome(S));neg(negative_outcome(S))),"
        "((pos(negative_outcome(S))->Y=1;Y=0),"
        "(archived_predict(S)->P=1;P=0),"
        "format('~w|~w|~w~n',[S,Y,P])))"
    )
    goals.append("halt")
    process = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-q", "-g", ",".join(goals)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    if process.returncode or "ERROR:" in process.stderr:
        raise RuntimeError(f"SWI-Prolog prediction reconstruction failed:\n{process.stderr}\n{process.stdout}")
    rows = []
    for line in process.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 3 and fields[1] in ("0", "1") and fields[2] in ("0", "1"):
            rows.append((fields[0], int(fields[1]), int(fields[2])))
    if not rows:
        raise RuntimeError(f"No predictions reconstructed from {examples}")
    return rows


def _feature_matrix(bk: Path, features: list[str]) -> dict[str, tuple[int, ...]]:
    feature_terms = ",".join(features)
    goal = (
        f"consult('{_quote(bk)}'),findall(S,sequence(S),Raw),sort(Raw,Sequences),"
        f"forall(member(S,Sequences),(write(S),write('|'),"
        f"forall(member(P,[{feature_terms}]),((call(P,S)->write(1);write(0)))),nl)),halt"
    )
    process = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-q", "-g", goal],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    if process.returncode or "ERROR:" in process.stderr:
        raise RuntimeError(f"CART feature reconstruction failed:\n{process.stderr}\n{process.stdout}")
    matrix = {}
    for line in process.stdout.splitlines():
        sequence_id, separator, bits = line.partition("|")
        if separator and len(bits) == len(features) and not (set(bits) - {"0", "1"}):
            matrix[sequence_id] = tuple(int(bit) for bit in bits)
    if not matrix:
        raise RuntimeError(f"No CART feature rows reconstructed from {bk}")
    return matrix


def _predict_tree(node: dict, values: tuple[int, ...], feature_index: dict[str, int]) -> int:
    while not node["leaf"]:
        present = values[feature_index[node["feature"]]]
        node = node["when_present"] if present else node["when_absent"]
    return int(node["negative_outcome"] >= node["non_negative"])


def _counts_by_match(rows: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    """Each vector is TP, FP, FN, TN."""
    counts: dict[str, list[int]] = {}
    for sequence_id, actual, predicted in rows:
        match_id = sequence_id.split("_")[1]
        cell = counts.setdefault(match_id, [0, 0, 0, 0])
        if actual and predicted:
            cell[0] += 1
        elif not actual and predicted:
            cell[1] += 1
        elif actual:
            cell[2] += 1
        else:
            cell[3] += 1
    return counts


def _sum_counts(counts: dict[str, list[int]]) -> list[int]:
    return [sum(row[index] for row in counts.values()) for index in range(4)]


def _lift(counts: list[int] | np.ndarray) -> float:
    tp, fp, fn, tn = (float(value) for value in counts)
    total = tp + fp + fn + tn
    predicted_positive = tp + fp
    actual_positive = tp + fn
    if total == 0 or predicted_positive == 0 or actual_positive == 0:
        return 0.0
    return (tp / predicted_positive) / (actual_positive / total)


def _draw_matches(fold_matches: dict[int, list[str]], rng: random.Random) -> list[str]:
    draw: list[str] = []
    for matches in fold_matches.values():
        draw.extend(rng.choices(matches, k=len(matches)))
    return draw


def _totals_on_draw(match_counts: dict[str, list[int]], draw: list[str]) -> list[int]:
    total = [0, 0, 0, 0]
    for match_id in draw:
        total = [left + right for left, right in zip(total, match_counts[match_id])]
    return total


def _bootstrap(
    arm_counts: dict[str, dict[str, list[int]]],
    fold_matches: dict[int, list[str]],
    reps: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = random.Random(seed)
    lifts = {arm: [] for arm in ARM_ORDER}
    paired_deltas = {arm: [] for arm in ARM_ORDER if arm != "hybrid"}
    for _ in range(reps):
        # Use one common match resample for all arms so lift differences are paired.
        draw = _draw_matches(fold_matches, rng)
        sampled = {arm: _totals_on_draw(arm_counts[arm], draw) for arm in ARM_ORDER}
        for arm in ARM_ORDER:
            lifts[arm].append(_lift(sampled[arm]))
        for arm in paired_deltas:
            paired_deltas[arm].append(_lift(sampled["hybrid"]) - _lift(sampled[arm]))
    intervals = {
        arm: {
            "ci95_percentile": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "median": float(np.median(values)),
        }
        for arm, values in lifts.items()
    }
    differences = {
        arm: {
            "hybrid_minus_comparator_lift_ci95_percentile": [
                float(x) for x in np.quantile(values, [0.025, 0.975])
            ],
            "median": float(np.median(values)),
        }
        for arm, values in paired_deltas.items()
    }
    return intervals, differences


def _match_summary(counts: dict[str, list[int]]) -> dict:
    positive_by_match = {match: row[0] + row[1] for match, row in counts.items()}
    flagged = [value for value in positive_by_match.values() if value > 0]
    predicted_total = sum(flagged)
    return {
        "test_matches": len(counts),
        "matches_with_at_least_one_positive_prediction": len(flagged),
        "positive_predictions": predicted_total,
        "max_positive_predictions_in_one_match": max(flagged, default=0),
        "share_of_positive_predictions_from_most_flagged_match": (
            max(flagged) / predicted_total if predicted_total else None
        ),
        "median_positive_predictions_among_flagged_matches": (
            float(np.median(flagged)) if flagged else 0.0
        ),
    }


def _validate_report(name: str, counts: list[int], report: dict) -> bool:
    metrics = report["miner_only_out_of_fold_metrics"] if name == "miner_only" else report["out_of_fold_metrics"]
    return all(int(metrics[key]) == int(value) for key, value in zip(CONFUSION_KEYS, counts))


def _matches_metrics(counts: list[int], metrics: dict) -> bool:
    return all(int(metrics[key]) == int(value) for key, value in zip(CONFUSION_KEYS, counts))


def analyze_team(results_root: Path, team: str, reps: int, seed: int) -> tuple[dict, list[dict]]:
    team_root = results_root / "data" / "team_runs" / team
    reports = team_root / "reports"
    prolog = team_root / "prolog"
    hybrid_report = _read_json(reports / "phase5_sequential_cv.json")
    miner_report = hybrid_report
    no_pattern_report = _read_json(reports / "phase5_cross_validation.json")
    cart_report = _read_json(reports / "phase5_decision_tree_cv.json")
    fold_sets: dict[int, list[str]] = {}
    arm_counts: dict[str, dict[str, list[int]]] = {arm: {} for arm in ARM_ORDER}
    csv_rows: list[dict] = []
    fold_validation: dict[str, dict[str, bool]] = {}

    # Reconstruct the same 34 Boolean inputs used by the archived CART folds.
    cart_features = cart_report["features"]
    cart_index = {name: index for index, name in enumerate(cart_features)}
    first_cart_bk = prolog / "popper_cv_fold_01" / "bk.pl"
    matrix = _feature_matrix(first_cart_bk, cart_features)

    for hybrid_fold in hybrid_report["fold_results"]:
        fold = int(hybrid_fold["fold"])
        fold_sets[fold] = sorted(map(str, hybrid_fold["test_matches"]))
        hybrid_dir = prolog / f"popper_sequential_cv_fold_{fold:02d}"
        base_dir = prolog / f"popper_cv_fold_{fold:02d}"

        model_rows = {
            "hybrid": _prolog_predictions(hybrid_dir / "bk.pl", hybrid_dir / "exs_holdout.pl", _normalize_clauses(hybrid_fold["clauses"])),
            "miner_only": _prolog_predictions(
                hybrid_dir / "bk.pl",
                hybrid_dir / "exs_holdout.pl",
                [[pattern["predicate"]] for pattern in hybrid_fold["selected_patterns"]],
            ),
            "popper_no_patterns": _prolog_predictions(
                base_dir / "bk.pl",
                base_dir / "exs_holdout.pl",
                _normalize_clauses(no_pattern_report["fold_results"][fold - 1].get("clauses", [])),
            ),
        }

        labels = _examples(base_dir / "exs_holdout.pl")
        hybrid_labels = _examples(hybrid_dir / "exs_holdout.pl")
        if labels != hybrid_labels:
            raise RuntimeError(f"OOF sequence IDs or labels differ in {team}, fold {fold}")
        cart_fold = cart_report["fold_results"][fold - 1]
        cart_rows = []
        for sequence_id, actual in labels.items():
            if sequence_id not in matrix:
                raise RuntimeError(f"CART features missing for {sequence_id}")
            predicted = _predict_tree(cart_fold["tree"], matrix[sequence_id], cart_index)
            cart_rows.append((sequence_id, actual, predicted))
        model_rows["cart"] = cart_rows

        expected_ids = set(labels)
        fold_validation[str(fold)] = {}
        for method, rows in model_rows.items():
            if {row[0] for row in rows} != expected_ids:
                raise RuntimeError(f"OOF sequence population mismatch in {team}, fold {fold}, {method}")
            if {sequence_id: actual for sequence_id, actual, _ in rows} != labels:
                raise RuntimeError(f"OOF labels differ in {team}, fold {fold}, {method}")
            fold_counts = _sum_counts(_counts_by_match(rows))
            if method == "hybrid":
                expected_fold_metrics = hybrid_fold["test_metrics"]
            elif method == "miner_only":
                expected_fold_metrics = hybrid_fold["miner_only_test_metrics"]
            elif method == "popper_no_patterns":
                expected_fold_metrics = no_pattern_report["fold_results"][fold - 1]["test_metrics"]
            else:
                expected_fold_metrics = cart_fold["metrics"]
            fold_validation[str(fold)][method] = _matches_metrics(fold_counts, expected_fold_metrics)
            if not fold_validation[str(fold)][method]:
                raise RuntimeError(f"Fold confusion counts differ in {team}, fold {fold}, {method}")
            per_match = _counts_by_match(rows)
            for match_id, count in per_match.items():
                if match_id in arm_counts[method]:
                    raise RuntimeError(f"Match {match_id} occurs in multiple OOF folds")
                arm_counts[method][match_id] = count
                tp, fp, fn, tn = count
                csv_rows.append({
                    "team": TEAM_LABELS[team], "method": method, "fold": fold,
                    "match_id": match_id, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "positive_predictions": tp + fp,
                })

    reports_by_arm = {
        "hybrid": hybrid_report,
        "miner_only": miner_report,
        "popper_no_patterns": no_pattern_report,
        "cart": cart_report,
    }
    checks = {}
    for arm, counts in arm_counts.items():
        pooled = _sum_counts(counts)
        expected_metrics = (
            reports_by_arm[arm]["miner_only_out_of_fold_metrics"]
            if arm == "miner_only" else reports_by_arm[arm]["out_of_fold_metrics"]
        )
        checks[arm] = {
            "reproduces_archived_confusion_counts": _validate_report(arm, pooled, reports_by_arm[arm]),
            "reconstructed_tp_fp_fn_tn": pooled,
            "archived_tp_fp_fn_tn": [int(expected_metrics[key]) for key in CONFUSION_KEYS],
        }
        if not checks[arm]["reproduces_archived_confusion_counts"]:
            raise RuntimeError(f"Reconstructed confusion counts do not match archived report for {team}/{arm}")

    bootstrap_ci, paired_differences = _bootstrap(arm_counts, fold_sets, reps, seed)
    summary = {
        "team": TEAM_LABELS[team],
        "matches": len(fold_sets and set().union(*(set(x) for x in fold_sets.values()))),
        "sequences": int(hybrid_report["n_labelled_sequences"]),
        "folds": int(hybrid_report["folds"]),
        "bootstrap_seed": seed,
        "fold_stratified_resampling": {str(fold): len(matches) for fold, matches in fold_sets.items()},
        "fold_reconstruction_checks": fold_validation,
        "reconstruction_checks": checks,
        "methods": {},
        "paired_lift_differences": paired_differences,
    }
    for arm in ARM_ORDER:
        pooled = _sum_counts(arm_counts[arm])
        summary["methods"][arm] = {
            "tp_fp_fn_tn": pooled,
            "precision": pooled[0] / (pooled[0] + pooled[1]) if pooled[0] + pooled[1] else None,
            "recall": pooled[0] / (pooled[0] + pooled[2]) if pooled[0] + pooled[2] else None,
            "lift": _lift(pooled),
            "match_spread": _match_summary(arm_counts[arm]),
            "match_cluster_bootstrap_lift": bootstrap_ci[arm],
        }
    return summary, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Results & Reports folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "Fold-stratified percentile cluster bootstrap of archived OOF predictions",
        "resampling_unit": "match; all sequences from each sampled match are retained together",
        "replicates": args.replicates,
        "seed": args.seed,
        "interval": "two-sided 95% percentile interval",
        "scope": "conditional on the saved fold-specific fitted models and the archived fold assignment; no model refitting",
        "teams": {},
    }
    all_csv = []
    for index, team in enumerate(TEAM_LABELS):
        summary, csv_rows = analyze_team(args.results_root, team, args.replicates, args.seed + index)
        output["teams"][team] = summary
        all_csv.extend(csv_rows)
    (args.output_dir / "match_cluster_bootstrap_oof.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "oof_match_confusion_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("team", "method", "fold", "match_id", "tp", "fp", "fn", "tn", "positive_predictions"))
        writer.writeheader()
        writer.writerows(all_csv)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
