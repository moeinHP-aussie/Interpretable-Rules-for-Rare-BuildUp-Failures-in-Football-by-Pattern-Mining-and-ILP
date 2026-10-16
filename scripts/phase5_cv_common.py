"""Method-neutral helpers shared by the isolated Phase 5 experiments."""

from __future__ import annotations

import hashlib
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tactical_ilp.config import PROCESSED_DIR

ROOT = Path(__file__).resolve().parents[1]
MATCHED_PATH = PROCESSED_DIR / "ilp_examples.parquet"
LABELLED_PATH = PROCESSED_DIR / "labeled_sequences.parquet"
COMPARISON_PROTOCOL_REVISION = "phase5-three-primary-arms-v16-shared-test-population-hash"


EVALUABLE_LABELS = ("negative", "non_negative")


def holdout_population(labelled: pd.DataFrame, test_matches: set[str]) -> tuple[pd.DataFrame, int]:
    """Return held-out sequences with a supported binary label.

    The current labeler emits only ``negative`` and ``non_negative``. The
    explicit filter also protects CV from stale or externally supplied rows.
    Returns (rows, number of other-label rows dropped).
    """
    rows = labelled[labelled.match_id.astype(str).isin(test_matches)]
    keep = rows.label.isin(EVALUABLE_LABELS)
    return rows[keep], int((~keep).sum())


def test_population_fingerprint(rows: pd.DataFrame) -> str:
    """Stable digest of held-out match IDs, sequence IDs, and labels."""
    pairs = sorted(
        (str(row.match_id), str(row.sequence_id), str(row.label))
        for row in rows[["match_id", "sequence_id", "label"]].itertuples(index=False)
    )
    if not pairs:
        return ""
    payload = "\n".join(f"{match_id}\t{sequence_id}\t{label}" for match_id, sequence_id, label in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_examples(frame: pd.DataFrame, path: Path) -> None:
    unexpected = sorted(set(frame.label.astype(str)) - set(EVALUABLE_LABELS))
    if unexpected:
        raise ValueError(
            f"refusing to write examples with labels {unexpected}: only "
            f"{EVALUABLE_LABELS} are valid pos/neg examples; remove other labels first"
        )
    ordered = frame.sort_values("sequence_id")
    positive_ids = ordered.loc[ordered.label == "negative", "sequence_id"]
    negative_ids = ordered.loc[ordered.label == "non_negative", "sequence_id"]
    lines = [f"pos(negative_outcome({sid}))." for sid in positive_ids]
    lines.extend(f"neg(negative_outcome({sid}))." for sid in negative_ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _balanced_group_folds(sampled: pd.DataFrame, all_matches: list[str], n_folds: int, seed: int) -> list[set[str]]:
    counts: dict[str, tuple[int, int]] = {}
    for match_id, group in sampled.groupby("match_id"):
        counts[str(match_id)] = (int((group.label == "negative").sum()), int((group.label == "non_negative").sum()))
    for match_id in all_matches:
        counts.setdefault(str(match_id), (0, 0))
    if len(counts) < n_folds:
        raise ValueError(f"need at least {n_folds} distinct matches; found {len(counts)}")
    rng = random.Random(seed)
    tie_order = list(counts)
    rng.shuffle(tie_order)
    rank = {match_id: i for i, match_id in enumerate(tie_order)}
    ordered = sorted(counts, key=lambda m: (-(counts[m][0] + counts[m][1]), rank[m]))
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    totals = [[0, 0, 0] for _ in range(n_folds)]
    target_pos = sum(p for p, _ in counts.values()) / n_folds
    target_neg = sum(n for _, n in counts.values()) / n_folds
    target_matches = len(counts) / n_folds
    match_limit = int(target_matches + 0.999999)
    for match_id in ordered:
        pos, neg = counts[match_id]
        candidates = range(n_folds)
        empty = [i for i in candidates if not folds[i]]
        if empty:
            chosen = empty[0]
        else:
            under_limit = [i for i in candidates if totals[i][2] < match_limit]
            if under_limit:
                candidates = under_limit
            def score(i: int) -> tuple[float, int]:
                p, n, m = totals[i]
                value = ((p + pos - target_pos) / max(1.0, target_pos)) ** 2
                value += ((n + neg - target_neg) / max(1.0, target_neg)) ** 2
                value += 2.0 * ((m + 1 - target_matches) / max(1.0, target_matches)) ** 2
                return value, i
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
        marker = output.rfind("New best hypothesis:")
        if marker < 0:
            raise RuntimeError("Popper printed neither SOLUTION nor a provisional hypothesis")
    clauses = []
    pattern = re.compile(r"negative_outcome\(\w+\)\s*:-\s*([^\n.]+)\.")
    for match in pattern.finditer(output[marker:]):
        body = [part.strip().split("(", 1)[0] for part in match.group(1).split(",")]
        if body and all(re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in body):
            clauses.append(body)
    if not clauses:
        raise RuntimeError("Popper printed SOLUTION but no parseable negative_outcome clause")
    return clauses, provisional


def _parse_hypothesis_snapshots(output: str) -> list[dict]:
    """Extract complete Popper incumbents, including non-final snapshots.

    Popper's printed tp/fn/tn/fp are discovery-training diagnostics. They are
    kept as such; callers must score each parsed program on the fold holdout.
    """
    marker = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?New best hypothesis:\s*$")
    starts = list(marker.finditer(output))
    if not starts:
        return []
    boundary = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?\*{5,}.*$|^\s*\*{5,}.*$")
    clause_pattern = re.compile(r"negative_outcome\(\w+\)\s*:-\s*([^\n.]+)\.")
    stats_pattern = re.compile(r"\btp\s*:\s*(\d+)\s+fn\s*:\s*(\d+)\s+tn\s*:\s*(\d+)\s+fp\s*:\s*(\d+)\s+size\s*:\s*(\d+)\s+mdl\s*:\s*(\d+)")
    snapshots: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for index, start in enumerate(starts):
        end_limit = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        segment = output[start.end():end_limit]
        boundary_match = boundary.search(segment)
        if boundary_match:
            segment = segment[:boundary_match.start()]
        stats_match = stats_pattern.search(segment)
        clauses: list[list[str]] = []
        for match in clause_pattern.finditer(segment):
            body = [part.strip().split("(", 1)[0] for part in match.group(1).split(",")]
            if body and all(re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in body):
                clauses.append(body)
        if not clauses:
            continue
        canonical = tuple(sorted(",".join(sorted(body)) for body in clauses))
        if canonical in seen:
            continue
        seen.add(canonical)
        train = None
        if stats_match:
            tp, fn, tn, fp, size, mdl = map(int, stats_match.groups())
            total_pos, total_neg = tp + fn, tn + fp
            support = tp + fp
            precision = tp / support if support else 0.0
            base = total_pos / (total_pos + total_neg) if total_pos + total_neg else 0.0
            train = {
                "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                "size": size, "mdl": mdl,
                "n_positive": total_pos, "n_negative": total_neg,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(tp / total_pos, 6) if total_pos else 0.0,
                "lift": round(precision / base, 6) if base else 0.0,
            }
        snapshots.append({"clauses": clauses, "train_metrics": train})
    return snapshots


def _evaluate_hypothesis_snapshots(
    output: str, fold: int, bk: Path, train_examples: Path, holdout_examples: Path
) -> tuple[list[dict], list[dict]]:
    """Score incumbent programs and their constituent single-clause rules."""
    hypotheses = _parse_hypothesis_snapshots(output)
    program_scores = _score_programs(
        [item["clauses"] for item in hypotheses], bk, holdout_examples
    )
    programs = []
    rule_occurrences: dict[tuple[str, ...], int] = defaultdict(int)
    rule_bodies: dict[tuple[str, ...], list[str]] = {}
    for hypothesis, score in zip(hypotheses, program_scores):
        programs.append({
            **hypothesis, "fold": fold, "holdout_metrics": score,
        })
        for body in hypothesis["clauses"]:
            key = tuple(sorted(body))
            rule_occurrences[key] += 1
            rule_bodies.setdefault(key, body)
    unique_rules = list(rule_bodies.values())
    train_scores = _score_programs([[body] for body in unique_rules], bk, train_examples)
    holdout_scores = _score_programs([[body] for body in unique_rules], bk, holdout_examples)
    rules = [
        {
            "clauses": [body],
            "fold": fold,
            "train_metrics": train_score,
            "holdout_metrics": holdout_score,
            "distinct_incumbent_programs_in_fold": rule_occurrences[tuple(sorted(body))],
        }
        for body, train_score, holdout_score in zip(unique_rules, train_scores, holdout_scores)
    ]
    return programs, rules


def _build_intermediate_candidate_report(fold_snapshots: list[dict], fold_count: int) -> dict:
    """Pool holdout scores for every distinct incumbent program.

    Since candidates are selected by looking at many Popper snapshots, these
    metrics are exploratory and selection-biased, unlike the primary OOF score.
    """
    by_program: dict[tuple[str, ...], dict] = {}
    for snapshot in fold_snapshots:
        clauses = snapshot["clauses"]
        program_key = tuple(sorted(",".join(sorted(body)) for body in clauses))
        has_fold_local_pattern = any(
            predicate.startswith("seq_pattern_")
            for body in clauses
            for predicate in body
        )
        # seq_pattern_NN is assigned independently inside each fold. The same
        # number can denote different event chains, so never pool candidates
        # across folds by that display name alone.
        fold_scope = f"fold_{int(snapshot['fold']):02d}" if has_fold_local_pattern else None
        key = ((fold_scope,) + program_key) if fold_scope else program_key
        item = by_program.setdefault(key, {
            "canonical_program": " ; ".join(program_key),
            "display_program": "\n".join(", ".join(body) for body in clauses),
            "fold_local_pattern_scope": fold_scope,
            "folds": [],
            "pooled_holdout_counts": {name: 0 for name in ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")},
        })
        item["folds"].append({
            "fold": snapshot["fold"],
            "train_metrics": snapshot.get("train_metrics"),
            "holdout_metrics": snapshot["holdout_metrics"],
        })
        for name in item["pooled_holdout_counts"]:
            item["pooled_holdout_counts"][name] += int(snapshot["holdout_metrics"][name])

    candidates = []
    for item in by_program.values():
        counts = item["pooled_holdout_counts"]
        total = counts["n_positive"] + counts["n_negative"]
        support = counts["support"]
        precision = counts["tp"] / support if support else None
        base = counts["n_positive"] / total if total else 0.0
        item["holdout_metrics_on_discovery_folds"] = {
            **counts,
            "base_rate": round(base, 6),
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(counts["tp"] / counts["n_positive"], 6) if counts["n_positive"] else 0.0,
            "lift": round(precision / base, 6) if precision is not None and base else None,
            "evidence_status": "observed" if support else "unobserved_no_holdout_firings",
        }
        item["folds_found"] = len(item["folds"])
        candidates.append(item)
    candidates.sort(key=lambda item: (
        -item["folds_found"],
        -(item["holdout_metrics_on_discovery_folds"]["lift"]
          if item["holdout_metrics_on_discovery_folds"]["lift"] is not None else -1.0),
        -item["holdout_metrics_on_discovery_folds"]["support"],
        item["canonical_program"],
    ))
    return {
        "configured_fold_count": fold_count,
        "training_metrics_population": "Popper sampled context-matched training examples; individual-clause train metrics are rescored on that same file",
        "holdout_metrics_population": "all labelled sequences from the held-out matches at natural prevalence",
        "selection_warning": (
            "Exploratory only: many incumbent hypotheses were inspected and these same fold holdouts were used to rank them. "
            "Metrics are pooled only over folds where that candidate was discovered; do not treat them as an unbiased estimate. "
            "Confirm shortlisted candidates on new matches or a nested/fresh evaluation."
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _intermediate_candidate_markdown(report: dict, limit: int = 20) -> list[str]:
    """Create a compact, clearly caveated Markdown view of incumbent candidates."""
    ledger = report.get("intermediate_candidates", {})
    rows = ["", "## Exploratory intermediate Popper candidates", "", ledger.get("selection_warning", "")]
    candidates = ledger.get("candidates", [])
    if not candidates:
        rows.extend(["", "No parseable intermediate hypotheses were emitted."])
        return rows
    rows.extend(["", "| Folds found | Holdout support* | Precision* | Recall* | Lift* | Program |", "|---:|---:|---:|---:|---:|---|"])
    for item in candidates[:limit]:
        metrics = item["holdout_metrics_on_discovery_folds"]
        program = "<br>".join(f"`{clause}`" for clause in item["display_program"].splitlines())
        precision = f"{metrics['precision']:.3f}" if metrics["precision"] is not None else "—"
        lift = f"{metrics['lift']:.3f}" if metrics["lift"] is not None else "—"
        rows.append(
            f"| {item['folds_found']}/{ledger.get('configured_fold_count', '?')} | {metrics['support']} | "
            f"{precision} | {metrics['recall']:.3f} | {lift} | {program} |"
        )
    rows.extend(["", "*Candidate scores pool only folds where the candidate was discovered; use as a discovery shortlist, not a performance estimate."])
    rules = ledger.get("individual_rules", [])
    if rules:
        rows.extend([
            "", "### Individual clauses found inside intermediate hypotheses", "",
            "| Folds found | Holdout support* | Precision* | Recall* | Lift* | Clause |",
            "|---:|---:|---:|---:|---:|---|",
        ])
        for item in rules[:limit]:
            metrics = item["holdout_metrics_on_discovery_folds"]
            clause = item["display_program"].replace("`", "")
            precision = f"{metrics['precision']:.3f}" if metrics["precision"] is not None else "—"
            lift = f"{metrics['lift']:.3f}" if metrics["lift"] is not None else "—"
            rows.append(
                f"| {item['folds_found']}/{ledger.get('configured_fold_count', '?')} | {metrics['support']} | "
                f"{precision} | {metrics['recall']:.3f} | {lift} | `{clause}` |"
            )
    return rows


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
    proc = subprocess.Popen([uv, "run", "--project", str(popper_home), "python", "-u", str(worker), "--_popper-worker", str(task_dir), "--noisy", "--timeout", str(timeout)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, start_new_session=(os.name == "posix"))
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
        else:
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


def _score_program(clauses: list[list[str]], bk: Path, examples: Path) -> dict[str, int | float]:
    esc = lambda p: p.as_posix().replace("'", "\\'")
    goals = [f"consult('{esc(bk)}')", f"consult('{esc(examples)}')"]
    for i, body in enumerate(clauses):
        goals.append(f"assertz((cv_clause_{i}(S) :- {', '.join(f'{literal}(S)' for literal in body)}))")
    if clauses:
        disjunction = "; ".join(f"cv_clause_{i}(S)" for i in range(len(clauses)))
        goals.append(f"assertz((cv_predict(S) :- ({disjunction})))")
        goals.extend(["findall(S, (pos(negative_outcome(S)), once(cv_predict(S))), TPs), length(TPs, TP)", "findall(S, (neg(negative_outcome(S)), once(cv_predict(S))), FPs), length(FPs, FP)"])
    else:
        goals.extend(["TP = 0", "FP = 0"])
    goals.extend(["findall(S, pos(negative_outcome(S)), Pos), length(Pos, NP)", "findall(S, neg(negative_outcome(S)), Neg), length(Neg, NN)", "format('~w ~w ~w ~w', [NP, NN, TP, FP])", "halt"])
    proc = subprocess.run(["swipl", "-D", "encoding=utf8", "-q", "-g", ", ".join(goals)], capture_output=True, text=True, timeout=900)
    if proc.returncode or "ERROR:" in proc.stderr:
        raise RuntimeError(f"SWI-Prolog scoring failed:\n{proc.stderr}\n{proc.stdout}")
    tokens = proc.stdout.strip().split()
    if len(tokens) != 4:
        raise RuntimeError(f"unexpected Prolog scoring output: {proc.stdout!r}")
    n_pos, n_neg, tp, fp = map(int, tokens)
    support = tp + fp
    base_rate = n_pos / (n_pos + n_neg) if n_pos + n_neg else 0.0
    precision = tp / support if support else 0.0
    recall = tp / n_pos if n_pos else 0.0
    return {"n_positive": n_pos, "n_negative": n_neg, "support": support, "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": n_neg - fp, "base_rate": round(base_rate, 6), "precision": round(precision, 6), "recall": round(recall, 6), "lift": round(precision / base_rate, 6) if base_rate else 0.0}


def _score_programs(programs: list[list[list[str]]], bk: Path, examples: Path) -> list[dict[str, int | float]]:
    """Score many distinct hypotheses in one SWI-Prolog process."""
    if not programs:
        return []
    esc = lambda p: p.as_posix().replace("'", "\\'")
    goals = [f"consult('{esc(bk)}')", f"consult('{esc(examples)}')"]
    for i, clauses in enumerate(programs):
        for j, body in enumerate(clauses):
            goals.append(
                f"assertz((cv_candidate_{i}_clause_{j}(S) :- "
                f"{', '.join(f'{literal}(S)' for literal in body)}))"
            )
        disjunction = "; ".join(f"cv_candidate_{i}_clause_{j}(S)" for j in range(len(clauses)))
        goals.append(f"assertz((cv_candidate_{i}(S) :- ({disjunction})))")
    goals.extend((
        "findall(S, pos(negative_outcome(S)), Pos), length(Pos, NP)",
        "findall(S, neg(negative_outcome(S)), Neg), length(Neg, NN)",
    ))
    for i in range(len(programs)):
        goals.extend((
            f"findall(S, (pos(negative_outcome(S)), once(cv_candidate_{i}(S))), TPs_{i}), length(TPs_{i}, TP_{i})",
            f"findall(S, (neg(negative_outcome(S)), once(cv_candidate_{i}(S))), FPs_{i}), length(FPs_{i}, FP_{i})",
            f"format('~w ~w ~w ~w~n', [NP, NN, TP_{i}, FP_{i}])",
        ))
    goals.append("halt")
    proc = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-q", "-g", ", ".join(goals)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode or "ERROR:" in proc.stderr:
        raise RuntimeError(f"SWI-Prolog batch scoring failed:\n{proc.stderr}\n{proc.stdout}")
    lines = [line.split() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != len(programs) or any(len(line) != 4 for line in lines):
        raise RuntimeError(f"unexpected batch Prolog scoring output: {proc.stdout!r}")
    results = []
    for tokens in lines:
        n_pos, n_neg, tp, fp = map(int, tokens)
        support = tp + fp
        base = n_pos / (n_pos + n_neg) if n_pos + n_neg else 0.0
        precision = tp / support if support else 0.0
        results.append({
            "n_positive": n_pos, "n_negative": n_neg, "support": support,
            "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": n_neg - fp,
            "base_rate": round(base, 6), "precision": round(precision, 6),
            "recall": round(tp / n_pos, 6) if n_pos else 0.0,
            "lift": round(precision / base, 6) if base else 0.0,
        })
    return results
