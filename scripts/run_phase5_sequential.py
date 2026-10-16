"""Train the hybrid ordered-pattern tactical Popper model.

Run after phases 1--5 have generated the processed sequences, Prolog facts,
and context-matched examples:

    POPPER_DIR=/content/Popper uv run python scripts/run_phase5_sequential.py

Each selected 2-to-3-event pattern is one atomic Popper feature. The complete
base tactical, sequence, and player vocabulary remains available alongside
patterns; the bias prevents a pattern from being conjoined with its own
tactical components in one clause.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import PROCESSED_DIR, PROLOG_DIR, REPORTS_DIR  # noqa: E402
from tactical_ilp.prolog.phase5_vocabulary import (  # noqa: E402
    POPPER_MAX_BODY,
    POPPER_MAX_CLAUSES,
    POPPER_MAX_VARS,
    POPPER_EXCLUDED_ACTION_PREDS,
    build_bk_wrapper,
)
from tactical_ilp.prolog.patterns_only_inputs import (  # noqa: E402
    build_hybrid_bias,
    select_compressing_patterns,
)
from tactical_ilp.sequences.sequential_patterns import (  # noqa: E402
    MiningConfig,
    build_pattern_sequences,
    fit_patterns,
    search_space_proxy,
    write_pattern_facts,
)
from phase5_cv_common import LABELLED_PATH, MATCHED_PATH, _parse_solution, _run_popper, _write_examples  # noqa: E402

TASK_DIR = PROLOG_DIR / "popper_sequential"
INPUT_CACHE = PROCESSED_DIR / "sequential_method" / "pattern_sequences.parquet"
REPORT_PATH = REPORTS_DIR / "phase5_sequential_training.json"
LOG_PATH = REPORTS_DIR / "phase5_sequential_training_popper_output.txt"
PREP_PATH = REPORTS_DIR / "phase5_sequential_prepared.json"


def prepare_inputs(config: MiningConfig, force_inputs: bool) -> dict:
    """Mine fold-local patterns and write the sequential Popper task only."""
    PREP_PATH.unlink(missing_ok=True)
    REPORT_PATH.unlink(missing_ok=True)
    LOG_PATH.unlink(missing_ok=True)
    required = [LABELLED_PATH, MATCHED_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing phases 1--5 outputs:\n  " + "\n  ".join(missing))

    pattern_frame = build_pattern_sequences(
        force=force_inputs,
        excluded_suffix_actions=config.excluded_suffix_actions,
        vocabulary="tactical",
        cache_path=INPUT_CACHE,
    )
    labelled = pd.read_parquet(LABELLED_PATH)
    matched = pd.read_parquet(MATCHED_PATH)
    matched_ids = set(matched.sequence_id.astype(str))
    examples = labelled[labelled.sequence_id.astype(str).isin(matched_ids)].copy()
    matches = sorted(examples.match_id.astype(str).unique())
    split_rng = random.Random(202021)
    shuffled_matches = list(matches)
    split_rng.shuffle(shuffled_matches)
    n_holdout = max(1, round(len(shuffled_matches) * 0.25))
    holdout_matches = set(shuffled_matches[:n_holdout])
    train_examples = examples[~examples.match_id.astype(str).isin(holdout_matches)].copy()
    holdout_examples = examples[examples.match_id.astype(str).isin(holdout_matches)].copy()
    train_matches = sorted(train_examples.match_id.astype(str).unique())

    diagnostics: dict[str, int | float] = {}
    selected = fit_patterns(pattern_frame, train_matches, config, diagnostics)
    selected = select_compressing_patterns(selected, config.max_selected_patterns)
    pattern_fallback = not selected
    diagnostics["no_pattern_fallback"] = pattern_fallback

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    _write_examples(train_examples, TASK_DIR / "exs.pl")
    _write_examples(holdout_examples, TASK_DIR / "exs_holdout.pl")
    pattern_info = write_pattern_facts(pattern_frame, selected, TASK_DIR / "facts_patterns.pl", config)
    bk_path = build_bk_wrapper(TASK_DIR / "bk.pl")
    with bk_path.open("a", encoding="utf-8") as stream:
        stream.write("\n:- consult('facts_patterns.pl').\n")
    pattern_predicates = tuple(item["predicate"] for item in pattern_info["features"])
    bias_info = build_hybrid_bias(TASK_DIR / "bias.pl", pattern_info["features"])

    print("SEQUENTIAL METHOD — ordered patterns plus the complete tactical vocabulary", flush=True)
    if pattern_fallback:
        print("No ordered pattern met the configured thresholds; this run will use the full tactical vocabulary.", flush=True)
    print(f"training examples: {len(train_examples)} ({int((train_examples.label == 'negative').sum())} negative outcomes)", flush=True)
    print(f"mined 2-to-3-event features: {len(pattern_predicates)}; component predicates kept independently active: {len(bias_info['pattern_component_predicates'])}", flush=True)
    print(f"hybrid Popper vocabulary: {bias_info['body_predicates']} body predicates; max body: {POPPER_MAX_BODY}; max clauses: {POPPER_MAX_CLAUSES}", flush=True)
    print(f"search-space proxy: {search_space_proxy(int(bias_info['body_predicates']), POPPER_MAX_BODY, POPPER_MAX_CLAUSES)}", flush=True)
    prepared = {
        "method": "hybrid_sequential",
        "effective_method": "full_tactical_fallback_no_patterns" if pattern_fallback else "hybrid_sequential",
        "pattern_fallback": pattern_fallback,
        "evaluation_split": {
            "seed": 202021,
            "holdout_fraction": 0.25,
            "train_matches": train_matches,
            "holdout_matches": sorted(holdout_matches),
            "training_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
        },
        "mining_population": "labelled sequences in training matches only",
        "pattern_input": str(INPUT_CACHE),
        "pattern_facts": pattern_info,
        "mining_config": config.__dict__,
        "mining_diagnostics": diagnostics,
        "body_predicate_count": bias_info["body_predicates"],
        "hybrid_vocabulary": {key: value for key, value in bias_info.items() if key != "path"},
        "popper_max_body": POPPER_MAX_BODY,
        "popper_max_vars": POPPER_MAX_VARS,
        "popper_max_clauses": POPPER_MAX_CLAUSES,
        "search_space_proxy": search_space_proxy(int(bias_info["body_predicates"]), POPPER_MAX_BODY, POPPER_MAX_CLAUSES),
        "task_dir": str(TASK_DIR),
    }
    PREP_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREP_PATH.write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("exs.pl", "exs_holdout.pl", "bias.pl", "bk.pl", "facts_patterns.pl"):
        if not (TASK_DIR / name).is_file():
            raise FileNotFoundError(f"pattern preparation did not create {TASK_DIR / name}")
    print(f"Pattern preparation complete -> {PREP_PATH}", flush=True)
    return prepared


def run_popper(popper_home: Path, timeout: int, prepared: dict) -> dict:
    """Run Popper against an already prepared, method-specific task directory."""
    REPORT_PATH.unlink(missing_ok=True)
    LOG_PATH.unlink(missing_ok=True)
    for name in ("exs.pl", "exs_holdout.pl", "bias.pl", "bk.pl", "facts_patterns.pl"):
        if not (TASK_DIR / name).is_file():
            raise FileNotFoundError(f"missing prepared hybrid input {TASK_DIR / name}; run the pattern-preparation cell first")
    excluded_bias = {f"seq_{name}" for name in POPPER_EXCLUDED_ACTION_PREDS}
    offered_bias = {
        line[len("body_pred("):].split(",", 1)[0]
        for line in (TASK_DIR / "bias.pl").read_text(encoding="utf-8").splitlines()
        if line.startswith("body_pred(")
    }
    stale = sorted(excluded_bias & offered_bias)
    if stale:
        raise RuntimeError(
            "prepared Popper bias still offers excluded triangle predicates "
            f"{stale}; rerun the preparation stage"
        )
    if not os.environ.get("POPPER_DIR"):
        raise RuntimeError("POPPER_DIR must point to the Popper checkout")
    output, hard_timeout = _run_popper(TASK_DIR, popper_home, timeout)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(output, encoding="utf-8")
    clauses, provisional = _parse_solution(output)
    body_predicates = [
        line[len("body_pred("):].split(",", 1)[0]
        for line in (TASK_DIR / "bias.pl").read_text(encoding="utf-8").splitlines()
        if line.startswith("body_pred(")
    ]
    report = {
        **prepared,
        "method": "hybrid_sequential_full_tactical_vocabulary",
        "label_target": "negative outcome; non_negative is the control class",
        "training_sampling": "phase-2 context-matched examples",
        "body_predicates_given_to_popper": body_predicates,
        "learned_clauses": clauses,
        "provisional": provisional,
        "hard_timeout": hard_timeout,
        "training_output_is_not_heldout_quality_evidence": True,
        "popper_log": str(LOG_PATH),
        "task_dir": str(TASK_DIR),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nLearned sequential-pattern rules:")
    for clause in clauses:
        print("  negative_outcome(S) :- " + ", ".join(clause) + ".")
    print(f"\nDetailed training report -> {REPORT_PATH}")
    print("Training scores are discovery diagnostics; use run_phase5_sequential_cv.py for quality evaluation.")
    return report


def run(popper_home: Path, timeout: int, config: MiningConfig, force_inputs: bool, stage: str = "all") -> dict:
    """Run pattern preparation and/or Popper as independently selectable stages."""
    if stage not in {"all", "prepare", "popper"}:
        raise ValueError(f"unknown stage: {stage}")
    if stage == "popper":
        if not PREP_PATH.exists():
            raise FileNotFoundError(f"missing {PREP_PATH}; run --prepare-only first")
        prepared = json.loads(PREP_PATH.read_text(encoding="utf-8"))
    else:
        prepared = prepare_inputs(config, force_inputs)
    if stage == "prepare":
        return prepared
    return run_popper(popper_home, timeout, prepared)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=900, help="Popper search budget in seconds")
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
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true", help="mine patterns and write Popper inputs without running Popper")
    stage.add_argument("--popper-only", action="store_true", help="run Popper using inputs from --prepare-only")
    args = parser.parse_args()
    popper_home = os.environ.get("POPPER_DIR")
    if not args.prepare_only and not popper_home:
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
        run(Path(popper_home) if popper_home else Path("."), args.timeout, config, args.force_pattern_inputs,
            "prepare" if args.prepare_only else "popper" if args.popper_only else "all")
    except Exception as exc:
        print(f"SEQUENTIAL TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged("phase5_sequential", main))
