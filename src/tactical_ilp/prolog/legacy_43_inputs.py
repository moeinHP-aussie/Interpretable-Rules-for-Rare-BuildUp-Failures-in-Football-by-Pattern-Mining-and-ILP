"""
Phase 5 — build the Popper input files (exs.pl, bk.pl, bias.pl).

Popper's task, in this project: learn a rule for

    negative_outcome(Sequence)

that separates sequences labelled "negative" (the active outcome definition
is documented in docs/negative_outcome_definition.md) from "non_negative"
ones, using the tactical concepts
from facts_all.pl + facts_tactical.pl + bk.pl as background knowledge.

Why sequence-level, not action-level
-------------------------------------
Every ground-truth label in this project (sequence_label/2) is per-sequence,
so the head predicate must take a sequence, not an action.  Concepts that are
naturally per-action (hesitation, triangulation, ...) are lifted to
"occurred somewhere in this sequence" with in_sequence/2, which is exactly
what the sequence-level wrappers expose from the background knowledge.

Files produced (in data/prolog/popper/)
----------------------------------------
exs.pl    -- pos(negative_outcome(S)) / neg(negative_outcome(S))
bias.pl   -- head_pred/body_pred/type/direction declarations
bk.pl     -- a single consult of facts_all.pl + facts_tactical.pl + the
             hand-written bk.pl, all wrapped as body predicates Popper can
             call directly (Popper wants one bk.pl file, so this generates a
             thin wrapper rather than copying 120k lines).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tactical_ilp.config import PROLOG_DIR

from tactical_ilp.prolog.phase5_vocabulary import (
    COMPLEMENT_PREDS,
    LEAKY_PREDICATES,
    MUTUALLY_EXCLUSIVE_BODY_PAIRS,
    POPPER_DIR,
    POPPER_MAX_BODY,
    POPPER_MAX_CLAUSES,
    POPPER_MAX_VARS,
    POPPER_WRAPPED_ACTION_PREDS,
    OUTCOME_PROXIMAL_ACTION_PREDS,
    SEQUENCE_PREDS_1,
    WRAPPED_ACTION_PREDS,
    WRAPPED_PLAYER_SEQ_PREDS,
    build_bk_wrapper,
)
def build_legacy_43_bias(
    out_path: Path,
    max_vars: int = POPPER_MAX_VARS,
    max_body: int = POPPER_MAX_BODY,
    max_clauses: int = POPPER_MAX_CLAUSES,
) -> Path:
    """Write the unmasked legacy bias after excluding outcome-proximal features."""
    all_body_preds = (
        [f"seq_{p}" for p in POPPER_WRAPPED_ACTION_PREDS]
        + [f"seq_{p}" for p in WRAPPED_PLAYER_SEQ_PREDS]
        + SEQUENCE_PREDS_1
        + [f"not_{p}" for p in COMPLEMENT_PREDS]
    )
    if set(WRAPPED_ACTION_PREDS) & OUTCOME_PROXIMAL_ACTION_PREDS:
        raise ValueError("outcome-proximal action predicates must not be offered to Popper")
    if len(all_body_preds) != len(set(all_body_preds)):
        raise ValueError("body predicate list contains duplicates")

    # Fail closed if a future edit removes/renames a predicate but leaves a
    # stale mutex pair behind.  Every exclusion must name actual candidates.
    offered = set(all_body_preds)
    missing_mutex_preds = sorted(
        {pred for pair in MUTUALLY_EXCLUSIVE_BODY_PAIRS for pred in pair} - offered
    )
    if missing_mutex_preds:
        raise ValueError(
            "mutual-exclusion constraint references predicates not offered "
            f"to Popper: {missing_mutex_preds}"
        )

    # LEAKAGE GUARD: refuse to emit any predicate that depends on the target
    # label.  This is a hard failure, not a warning -- a leaking body_pred
    # silently produces a "perfect" but meaningless hypothesis.
    leaking = sorted(
        pred for pred in set(all_body_preds)
        if pred in LEAKY_PREDICATES or pred.removeprefix("seq_") in LEAKY_PREDICATES
    )
    if leaking:
        raise ValueError(
            "target leakage: these body predicates depend on sequence_label/2 "
            f"and must not be given to Popper: {leaking}"
        )

    lines = [
        "%% ===================================================================",
        "%%  AUTO-GENERATED - phase 5 Popper bias file",
        "%%",
        "%%  head: negative_outcome(Sequence) -- learn what marks a build-up",
        "%%  sequence as ending badly, from the tactical concepts as body preds.",
        "%%",
        "%%  Active full-tactical vocabulary; max_vars/body/clauses are deliberately",
        "%%  bounded to control Popper search cost. Validate discoveries with",
        "%%  match-grouped cross-validation, not training scores alone.",
        "%% ===================================================================",
        "",
        "head_pred(negative_outcome,1).",
        "type(negative_outcome,(sequence,)).",
        "direction(negative_outcome,(in,)).",
        "",
    ]
    for pred in all_body_preds:
        lines.append(f"body_pred({pred},1).")
        lines.append(f"type({pred},(sequence,)).")
        lines.append(f"direction({pred},(in,)).")
    lines.append("")

    lines.extend([
        "%% Definite same-clause mutual exclusions.",
        "%% Separate clauses may still use either member independently.",
    ])
    for left, right in MUTUALLY_EXCLUSIVE_BODY_PAIRS:
        if left not in offered or right not in offered:
            continue
        lines.append(
            f":- body_literal(C,{left},1,_), body_literal(C,{right},1,_)."
        )
    lines.append("")
    lines.append(f"max_vars({max_vars}).")
    lines.append(f"max_body({max_body}).")
    lines.append(f"max_clauses({max_clauses}).")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_examples(
    out_path: Path,
    use_matched_negatives: bool = True,
    holdout_frac: float = 0.25,
    seed: int = 202021,
) -> tuple[Path, int, int, dict]:
    """Write exs.pl, and a held-out evaluation set alongside it.

    Two corrections over the first version:

    1. ``use_matched_negatives`` (default True) draws negatives from
       ``ilp_examples.parquet`` -- the CONTEXT-MATCHED sample built in phase
       2.4 and documented in docs/negative_sampling_strategy.md.  The first
       version read labeled_sequences.parquet and used all 923 negatives,
       silently discarding that whole piece of work.  Matched sampling is the
       project's own defence against Popper learning *context* (deep, under
       pressure) instead of *outcome*.

    2. A held-out split.  Popper reports precision/recall on the examples it
       was given, so training on everything makes "Precision 1.00" mean
       nothing.  Sequences are split by MATCH (never within a match) so a
       sequence and its near-duplicates from the same game cannot straddle
       the split.
    """
    import random

    processed = PROLOG_DIR.parent / "processed"
    labeled = pd.read_parquet(processed / "labeled_sequences.parquet")

    if use_matched_negatives:
        matched = pd.read_parquet(processed / "ilp_examples.parquet")
        keep = set(matched["sequence_id"])
        ex = labeled[labeled.sequence_id.isin(keep)].copy()
    else:
        ex = labeled.copy()

    # split by match so no match appears in both train and holdout
    matches = sorted(ex["match_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(matches)
    n_hold = max(1, round(len(matches) * holdout_frac))
    holdout_matches = set(matches[:n_hold])

    train = ex[~ex.match_id.isin(holdout_matches)]
    hold = ex[ex.match_id.isin(holdout_matches)]

    def _emit(df: pd.DataFrame) -> str:
        lines = []
        for sid in sorted(df[df.label == "negative"]["sequence_id"]):
            lines.append(f"pos(negative_outcome({sid})).")
        for sid in sorted(df[df.label == "non_negative"]["sequence_id"]):
            lines.append(f"neg(negative_outcome({sid})).")
        return "\n".join(lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_emit(train), encoding="utf-8")
    (out_path.parent / "exs_holdout.pl").write_text(_emit(hold), encoding="utf-8")

    n_pos = int((train.label == "negative").sum())
    n_neg = int((train.label == "non_negative").sum())
    split = {
        "matched_negatives": use_matched_negatives,
        "train_pos": n_pos,
        "train_neg": n_neg,
        "holdout_pos": int((hold.label == "negative").sum()),
        "holdout_neg": int((hold.label == "non_negative").sum()),
        "holdout_matches": sorted(holdout_matches),
        "n_train_matches": len(matches) - n_hold,
    }
    return out_path, n_pos, n_neg, split


def generate_popper_inputs() -> dict:
    bk_path = build_bk_wrapper(POPPER_DIR / "bk.pl")
    bias_path = build_legacy_43_bias(POPPER_DIR / "bias.pl")
    exs_path, n_pos, n_neg, split = build_examples(POPPER_DIR / "exs.pl")
    return {
        "dir": POPPER_DIR,
        "bk": bk_path,
        "bias": bias_path,
        "exs": exs_path,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "split": split,
        "n_body_preds": (
            len(POPPER_WRAPPED_ACTION_PREDS) + len(WRAPPED_PLAYER_SEQ_PREDS)
            + len(SEQUENCE_PREDS_1) + len(COMPLEMENT_PREDS)
        ),
        "n_base_body_preds": len(POPPER_WRAPPED_ACTION_PREDS) + len(WRAPPED_PLAYER_SEQ_PREDS) + len(SEQUENCE_PREDS_1),
        "n_complement_preds": len(COMPLEMENT_PREDS),
        "max_vars": POPPER_MAX_VARS,
        "max_body": POPPER_MAX_BODY,
        "max_clauses": POPPER_MAX_CLAUSES,
    }


if __name__ == "__main__":
    info = generate_popper_inputs()
    print(f"Popper input dir: {info['dir']}")
    print(f"  exs.pl : {info['n_pos']} pos, {info['n_neg']} neg")
    print(f"  bias.pl: {info['n_body_preds']} body predicates")
    print()
    print("NOTE: each seq_<concept>/1 wrapper is a per-action existential, so")
    print("aggregate_all(count, seq_X(S), N) counts raw solutions (one per")
    print("qualifying action), not distinct sequences. Popper itself only ever")
    print("asks seq_X(S) for one fixed S at a time (true/false), so this does")
    print("not affect learning -- it only matters when reporting coverage by")
    print("hand.  Use distinct/2 (e.g. aggregate_all(count, distinct(S, seq_X(S)), N))")
    print("to report the number of distinct sequences instead.")
    print()
    print("run with:")
    print(f"  cd {info['dir']} && python /path/to/Popper/popper.py .")
