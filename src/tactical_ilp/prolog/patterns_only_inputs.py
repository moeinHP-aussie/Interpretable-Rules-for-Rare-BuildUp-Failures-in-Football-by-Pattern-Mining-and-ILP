"""Popper bias builders for pure-pattern diagnostics and the hybrid method."""

from __future__ import annotations

from pathlib import Path

from tactical_ilp.prolog.phase5_vocabulary import (
    COMPLEMENT_PREDS,
    MUTUALLY_EXCLUSIVE_BODY_PAIRS,
    POPPER_MAX_BODY,
    POPPER_MAX_CLAUSES,
    POPPER_MAX_VARS,
    POPPER_EXCLUDED_ACTION_PREDS,
    POPPER_WRAPPED_ACTION_PREDS,
    SEQUENCE_PREDS_1,
    WRAPPED_PLAYER_SEQ_PREDS,
)


def _tactical_predicate(item: str) -> str | None:
    """Return the base tactical predicate from plain or spatial item syntax."""
    if not item.startswith("tactical:"):
        return None
    return item.removeprefix("tactical:").split("@", maxsplit=1)[0]

# Each mined sequence is already a complete feature. Popper may select several
# features as separate clauses, but must not conjoin them into a new pattern.
PATTERN_ATOMIC_MAX_BODY = 1
PATTERN_MAX_VARS = 1


def build_patterns_only_bias(
    out_path: Path,
    pattern_predicates: tuple[str, ...],
    max_vars: int = PATTERN_MAX_VARS,
    max_body: int = PATTERN_ATOMIC_MAX_BODY,
    max_clauses: int = POPPER_MAX_CLAUSES,
) -> Path:
    """Write a bias where each mined ordered pattern is an atomic literal."""
    predicates = list(pattern_predicates)
    if not predicates:
        raise ValueError("patterns_only requires at least one selected pattern predicate")
    if len(predicates) != len(set(predicates)):
        raise ValueError("pattern predicate list contains duplicates")
    invalid = [pred for pred in predicates if not pred.startswith("seq_pattern_")]
    if invalid:
        raise ValueError(f"pattern predicates must use the seq_pattern_ prefix: {invalid}")
    if max_body != PATTERN_ATOMIC_MAX_BODY:
        raise ValueError("ordered-pattern predicates are atomic; max_body must be 1")
    if max_vars != PATTERN_MAX_VARS:
        raise ValueError("ordered-pattern predicates are unary over one sequence; max_vars must be 1")

    lines = [
        "%% AUTO-GENERATED: atomic ordered-pattern features; no raw legacy predicates.",
        "head_pred(negative_outcome,1).",
        "type(negative_outcome,(sequence,)).",
        "direction(negative_outcome,(in,)).",
        "",
    ]
    for pred in predicates:
        lines.extend((f"body_pred({pred},1).", f"type({pred},(sequence,)).", f"direction({pred},(in,))."))
    lines.extend(("", f"max_vars({max_vars}).", f"max_body({max_body}).", f"max_clauses({max_clauses}).", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_hybrid_bias(
    out_path: Path,
    pattern_features: list[dict],
    max_vars: int = POPPER_MAX_VARS,
    max_body: int = POPPER_MAX_BODY,
    max_clauses: int = POPPER_MAX_CLAUSES,
) -> dict[str, object]:
    """Offer mined patterns plus the complete revised tactical vocabulary.

    Pattern components remain available as independent atoms. Same-clause
    constraints prune only a pattern paired with one of its own components.
    """
    base = (
        [f"seq_{p}" for p in POPPER_WRAPPED_ACTION_PREDS]
        + [f"seq_{p}" for p in WRAPPED_PLAYER_SEQ_PREDS]
        + list(SEQUENCE_PREDS_1)
        + [f"not_{p}" for p in COMPLEMENT_PREDS]
    )
    consumed = set()
    for feature in pattern_features:
        for event in feature.get("pattern", []):
            for item in event:
                predicate = _tactical_predicate(item)
                if predicate:
                    consumed.add(f"seq_{predicate}")
    retained = list(base)
    patterns = [str(item["predicate"]) for item in pattern_features]
    predicates = retained + patterns
    excluded = {
        f"seq_{predicate}"
        for feature in pattern_features
        for event in feature.get("pattern", [])
        for item in event
        if (predicate := _tactical_predicate(item)) in POPPER_EXCLUDED_ACTION_PREDS
    }
    if excluded:
        raise ValueError(
            "pattern features contain triangle-derived predicates excluded "
            f"from Popper search: {sorted(excluded)}"
        )
    if len(predicates) != len(set(predicates)):
        raise ValueError("hybrid vocabulary contains duplicate body predicates")
    if any(not pred.startswith("seq_pattern_") for pred in patterns):
        raise ValueError("mined predicates must use the seq_pattern_ prefix")

    lines = [
        "%% AUTO-GENERATED: hybrid ordered-pattern and revised tactical vocabulary.",
        "%% Pattern components remain available; tautological pattern/component pairs are pruned.",
        "head_pred(negative_outcome,1).",
        "type(negative_outcome,(sequence,)).",
        "direction(negative_outcome,(in,)).",
        "",
    ]
    for pred in predicates:
        lines.extend((f"body_pred({pred},1).", f"type({pred},(sequence,)).", f"direction({pred},(in,))."))
    offered = set(predicates)
    lines.extend(("", "%% Definite same-clause mutual exclusions."))
    for left, right in MUTUALLY_EXCLUSIVE_BODY_PAIRS:
        if left in offered and right in offered:
            lines.append(f":- body_literal(C,{left},1,_), body_literal(C,{right},1,_).")
    lines.extend(("", "%% Anti-tautology: do not conjoin a pattern with its own components."))
    for feature in pattern_features:
        pattern = str(feature["predicate"])
        components = {
            f"seq_{predicate}"
            for event in feature.get("pattern", [])
            for item in event
            if (predicate := _tactical_predicate(item)) is not None
        }
        for component in sorted(components & offered):
            lines.append(
                f":- body_literal(C,{pattern},1,_), "
                f"body_literal(C,{component},1,_)."
            )
    lines.extend(("", f"max_vars({max_vars}).", f"max_body({max_body}).", f"max_clauses({max_clauses}).", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "path": out_path,
        "body_predicates": len(predicates),
        "base_predicates": len(base),
        "retained_base_predicates": len(retained),
        "pattern_predicates": len(patterns),
        "pattern_component_predicates": sorted(consumed),
        "max_vars": max_vars,
        "max_body": max_body,
        "max_clauses": max_clauses,
    }


def select_compressing_patterns(candidates: list, max_patterns: int) -> list:
    """Keep ranked candidates only when they reduce the hybrid vocabulary.

    Each accepted pattern adds one atomic feature and removes at least one
    previously available action predicate, so patterns alone cannot grow the
    body vocabulary beyond the current base-vocabulary size.
    """
    remaining = {f"seq_{name}" for name in POPPER_WRAPPED_ACTION_PREDS}
    selected = []
    for candidate in candidates:
        components = {
            f"seq_{_tactical_predicate(item)}"
            for event in candidate.pattern
            for item in event
            if _tactical_predicate(item) is not None
        }
        newly_consumed = components & remaining
        if not newly_consumed:
            continue
        selected.append(candidate)
        remaining.difference_update(newly_consumed)
        if len(selected) >= max_patterns:
            break
    return selected
