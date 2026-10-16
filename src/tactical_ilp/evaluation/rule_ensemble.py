"""Sparse logistic calibration over coach-approved symbolic rules.

This is an associative risk layer over the existing symbolic model. It does
not change Popper's rules or imply causal effects. The match split is inherited
from the Phase-5 examples so the holdout remains match-disjoint.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


L1_STRENGTH = 1.0  # sum-loss equivalent C=1; average-loss lambda is 1 / n_train
DEFAULT_RISK_THRESHOLD = 0.50


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _fit_l1_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, int, bool]:
    """Fit an intercept plus L1-penalized binary logistic model with FISTA."""
    n, p = x.shape
    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    intercept = math.log(prevalence / (1 - prevalence))
    weights = np.zeros(p, dtype=float)
    if p == 0:
        return intercept, weights, 0, True

    augmented = np.column_stack((np.ones(n), x))
    lipschitz = 0.25 * float(np.linalg.norm(augmented, ord=2) ** 2) / n
    lipschitz = max(lipschitz, 1e-8)
    penalty = L1_STRENGTH / n
    current = np.r_[intercept, weights]
    extrapolated = current.copy()
    momentum = 1.0
    converged = False

    for iteration in range(1, 10001):
        logits = extrapolated[0] + x @ extrapolated[1:]
        residual = _sigmoid(logits) - y
        gradient = augmented.T @ residual / n
        proposal = extrapolated - gradient / lipschitz
        proposal[1:] = np.sign(proposal[1:]) * np.maximum(
            np.abs(proposal[1:]) - penalty / lipschitz, 0.0
        )
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        next_extrapolated = proposal + ((momentum - 1.0) / next_momentum) * (proposal - current)
        if np.max(np.abs(proposal - current)) < 1e-8:
            current = proposal
            converged = True
            break
        current, extrapolated, momentum = proposal, next_extrapolated, next_momentum
    return float(current[0]), current[1:].copy(), iteration, converged


def _auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    sorted_scores = probability[order]
    ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    if len(y) == 0:
        return {"n": 0, "base_rate": None, "brier_score": None, "log_loss": None,
                "roc_auc": None, "threshold": threshold, "predicted_positive": 0,
                "precision_at_threshold": None, "recall_at_threshold": None}
    eps = 1e-12
    prediction = probability >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    positives = int(y.sum())
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(np.clip(probability, eps, 1))
                                   + (1 - y) * np.log(np.clip(1 - probability, eps, 1)))),
        "roc_auc": _auc(y, probability),
        "threshold": threshold,
        "predicted_positive": int(prediction.sum()),
        "precision_at_threshold": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_threshold": tp / positives if positives else None,
    }


def build_rule_ensemble(
    labeled: pd.DataFrame,
    approved_rules: list[dict],
    matches_by_rule: dict[str, set[str]],
    training_match_ids: set[int],
    holdout_match_ids: set[int],
    threshold: float = DEFAULT_RISK_THRESHOLD,
) -> dict:
    """Fit only approved-rule indicators on training matches; score all labels."""
    features = [r["name"] for r in approved_rules]
    if not features:
        return {"status": "no_approved_rules", "rules": [], "records": [],
                "risk_threshold": threshold}
    if not training_match_ids or not holdout_match_ids:
        return {"status": "missing_match_split", "rules": [], "records": [],
                "risk_threshold": threshold}

    frame = labeled.copy()
    frame["sequence_id"] = frame.sequence_id.astype(str)
    frame["label_binary"] = (frame.label == "negative").astype(int)
    frame["split"] = np.select(
        [frame.match_id.isin(training_match_ids), frame.match_id.isin(holdout_match_ids)],
        ["train", "holdout"], default="unassigned",
    )
    sequence_ids = frame.sequence_id.tolist()
    x_all = np.column_stack([
        np.fromiter((sid in matches_by_rule.get(name, set()) for sid in sequence_ids),
                    dtype=float, count=len(sequence_ids))
        for name in features
    ])
    y_all = frame.label_binary.to_numpy(dtype=float)
    train_mask = frame.split.to_numpy() == "train"
    holdout_mask = frame.split.to_numpy() == "holdout"
    if int(train_mask.sum()) < 2 or len(np.unique(y_all[train_mask])) < 2:
        return {"status": "insufficient_training_classes", "rules": [], "records": [],
                "risk_threshold": threshold, "training_sequences": int(train_mask.sum())}

    intercept, weights, iterations, converged = _fit_l1_logistic(x_all[train_mask], y_all[train_mask])
    probabilities = _sigmoid(intercept + x_all @ weights)
    weight_rows = []
    for name, weight in zip(features, weights):
        if abs(float(weight)) < 1e-3:
            tier = "pruned"
        elif weight > 1.0:
            tier = "critical_association"
        elif weight > 0.2:
            tier = "secondary_association"
        elif weight < -0.2:
            tier = "protective_association"
        else:
            tier = "weak_association"
        weight_rows.append({
            "name": name,
            "weight": float(weight),
            "odds_ratio": float(math.exp(float(np.clip(weight, -30, 30)))),
            "tier": tier,
            "training_prevalence": float(x_all[train_mask, features.index(name)].mean()),
        })

    records = []
    for i, row in frame.reset_index(drop=True).iterrows():
        contributions = [
            {"rule": name, "logit_contribution": float(weights[j])}
            for j, name in enumerate(features) if x_all[i, j] == 1 and abs(weights[j]) >= 1e-3
        ]
        records.append({
            "sequence_id": str(row.sequence_id),
            "match_id": int(row.match_id),
            "label": str(row.label),
            "split": str(row.split),
            "probability": float(probabilities[i]),
            "predicted_negative": bool(probabilities[i] >= threshold),
            "active_rules": [name for j, name in enumerate(features) if x_all[i, j] == 1],
            "logit_intercept": intercept,
            "rule_contributions": contributions,
        })

    return {
        "status": "fitted",
        "target": "negative outcome",
        "feature_names": features,
        "intercept": intercept,
        "l1_strength_sum_loss_equivalent": L1_STRENGTH,
        "risk_threshold": threshold,
        "threshold_note": "0.50 is a display threshold, not a validated coaching decision threshold.",
        "fit": {"training_sequences": int(train_mask.sum()),
                "training_matches": len(training_match_ids),
                "iterations": iterations, "converged": converged},
        "training_metrics": _metrics(y_all[train_mask], probabilities[train_mask], threshold),
        "holdout_metrics": _metrics(y_all[holdout_mask], probabilities[holdout_mask], threshold),
        "rule_weights": weight_rows,
        "records": records,
        "interpretation_warning": (
            "Odds ratios are conditional associations among approved rules, not causal effects. "
            "Rule discovery and coach approval used this project's data, so holdout diagnostics "
            "are informative but not a fully untouched external validation."
        ),
    }
