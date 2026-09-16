"""
Task 2.4 — Negative-sampling strategy for ``exs.pl``.

Risk #2 of the technical document ("invalid negative examples -> spurious
rules") is the single most dangerous failure mode of the whole ILP pipeline.
Choosing negatives uniformly at random from the dataset would let Popper learn
rules that merely separate *contexts* (e.g. "deep build-up under pressure")
instead of separating *outcomes* inside the same context.

Strategy implemented here — **context-matched stratified sampling**
------------------------------------------------------------------
Positive example  = a build-up sequence labelled ``negative`` (§ Task 2.2).
Negative example  = a sequence labelled ``non_negative`` drawn from the *same
context stratum* as a positive one.

A stratum is the triple

    (start_zone, pressure_bucket, length_bucket)

  start_zone       third|lane of the first action  (15 possible values)
  pressure_bucket  no_pressure / light_pressure / heavy_pressure of action #1
  length_bucket    short (2-3) | medium (4-6) | long (>=7) actions

Ratio: ``NEG_SAMPLES_PER_POSITIVE`` = 2 negatives per positive.  Rationale: 1:1
throws away most of the healthy build-up evidence and makes Confidence estimates
noisy, while ratios above ~1:3 push Popper towards the trivial "predict
non-negative" hypothesis and slow the solver down.  2:1 keeps the positive class
at ~33 % of the example file, which is comfortable for an MDL cost function.

Fallback ladder (recorded per drawn example in ``match_level``):
    3  exact stratum (zone + pressure + length)
    2  zone + length
    1  zone only
    0  global pool  (should stay rare; it is reported by the script)

Everything is deterministic: ``NEG_SAMPLING_RANDOM_SEED`` fixes the draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    NEG_SAMPLES_PER_POSITIVE,
    NEG_SAMPLING_RANDOM_SEED,
    PROCESSED_DIR,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.features.geometry import pitch_zone, pressure_level
from tactical_ilp.sequences.outcome_labeler import label_sequences
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

CONTEXT_PATH = PROCESSED_DIR / "sequence_context.parquet"
EXAMPLES_PATH = PROCESSED_DIR / "ilp_examples.parquet"


def _length_bucket(n: int) -> str:
    if n <= 3:
        return "short"
    if n <= 6:
        return "medium"
    return "long"


def build_context(force: bool = False) -> pd.DataFrame:
    """Labelled sequences enriched with the stratification keys."""
    if CONTEXT_PATH.exists() and not force:
        return pd.read_parquet(CONTEXT_PATH)

    labeled = label_sequences()
    _, members = build_sequences()
    merged = build_merged_all()

    first = members[members.order_in_sequence == 0]
    acts = merged.set_index(["match_id", "action_id"])
    cols = [
        "player_id",
        "type_name",
        "ff_opponents_3m",
        "ff_opponents_5m",
        "under_pressure",
        "start_x",
        "start_y",
    ]
    first = first.join(acts[cols], on=["match_id", "action_id"])

    first["pressure_bucket"] = [
        pressure_level(o3, o5, bool(flag))
        for o3, o5, flag in zip(
            first.ff_opponents_3m, first.ff_opponents_5m, first.under_pressure
        )
    ]
    first["start_zone"] = [pitch_zone(x, y) for x, y in zip(first.start_x, first.start_y)]
    first["first_player_id"] = first["player_id"]
    first["first_action_type"] = first["type_name"]

    ctx = labeled.merge(
        first[
            [
                "sequence_id",
                "pressure_bucket",
                "start_zone",
                "first_player_id",
                "first_action_type",
            ]
        ],
        on="sequence_id",
        how="left",
    )
    ctx["length_bucket"] = ctx["n_actions"].apply(_length_bucket)
    ctx["stratum"] = (
        ctx["start_zone"] + "#" + ctx["pressure_bucket"] + "#" + ctx["length_bucket"]
    )
    ctx.to_parquet(CONTEXT_PATH, index=False)
    return ctx


def sample_examples(force: bool = False) -> pd.DataFrame:
    """
    Build the ILP example set.

    Returns one row per selected sequence with ``example_class`` in
    {``pos``, ``neg``} and the ``match_level`` that produced it.
    """
    if EXAMPLES_PATH.exists() and not force:
        return pd.read_parquet(EXAMPLES_PATH)

    ctx = build_context()
    rng = np.random.default_rng(NEG_SAMPLING_RANDOM_SEED)

    positives = ctx[ctx.label == "negative"].copy()
    pool = ctx[ctx.label == "non_negative"].copy()

    used: set[str] = set()
    chosen: list[dict] = []

    def _draw(candidates: pd.DataFrame, k: int) -> list[str]:
        avail = [s for s in candidates.sequence_id if s not in used]
        if not avail:
            return []
        k = min(k, len(avail))
        picked = rng.choice(np.array(avail, dtype=object), size=k, replace=False)
        return list(picked)

    for _, p in positives.iterrows():
        need = NEG_SAMPLES_PER_POSITIVE
        ladders = [
            (3, pool[pool.stratum == p.stratum]),
            (
                2,
                pool[
                    (pool.start_zone == p.start_zone)
                    & (pool.length_bucket == p.length_bucket)
                ],
            ),
            (1, pool[pool.start_zone == p.start_zone]),
            (0, pool),
        ]
        for level, cand in ladders:
            if need <= 0:
                break
            picked = _draw(cand, need)
            for sid in picked:
                used.add(sid)
                chosen.append(
                    {"sequence_id": sid, "example_class": "neg", "match_level": level,
                     "matched_to": p.sequence_id}
                )
            need -= len(picked)

    neg_df = pd.DataFrame(chosen)
    pos_df = pd.DataFrame(
        {
            "sequence_id": positives.sequence_id,
            "example_class": "pos",
            "match_level": 3,
            "matched_to": positives.sequence_id,
        }
    )
    examples = pd.concat([pos_df, neg_df], ignore_index=True)
    examples = examples.merge(ctx, on="sequence_id", how="left")
    examples.to_parquet(EXAMPLES_PATH, index=False)
    return examples


def zone_distribution_overlap(pos: pd.DataFrame, neg: pd.DataFrame) -> float:
    """Total-variation style overlap of the two zone distributions (0..1)."""
    p = pos["zone"].value_counts(normalize=True)
    n = neg["zone"].value_counts(normalize=True)
    zones = set(p.index) | set(n.index)
    return float(sum(min(p.get(z, 0.0), n.get(z, 0.0)) for z in zones))


if __name__ == "__main__":
    ex = sample_examples(force=True)
    pos = ex[ex.example_class == "pos"]
    neg = ex[ex.example_class == "neg"]
    print(f"positive examples : {len(pos)}")
    print(f"negative examples : {len(neg)}   (ratio 1:{len(neg)/max(len(pos),1):.2f})")
    print()
    print("fallback level of drawn negatives (3 = exact stratum):")
    print(neg.match_level.value_counts().sort_index(ascending=False).to_string())
    print()
    overlap = zone_distribution_overlap(pos, neg)
    print(f"zone-distribution overlap : {overlap:.3f}   (acceptance: >= 0.40)")
    print()
    print("zone distribution (pos vs neg), top 8:")
    comp = pd.DataFrame(
        {
            "pos": pos.zone.value_counts(normalize=True),
            "neg": neg.zone.value_counts(normalize=True),
        }
    ).fillna(0).sort_values("pos", ascending=False).head(8)
    print(comp.round(3).to_string())
