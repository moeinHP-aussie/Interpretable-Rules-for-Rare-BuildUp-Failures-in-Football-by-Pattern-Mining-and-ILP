"""
Phase 2 — build-up sequences, outcome labels, player subsets, negative sampling.

    uv run python scripts/run_phase2.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH   # noqa: E402
from tactical_ilp.sequences.negative_sampling import (                     # noqa: E402
    sample_examples,
    zone_distribution_overlap,
)
from tactical_ilp.sequences.outcome_labeler import label_sequences         # noqa: E402
from tactical_ilp.sequences.player_subsets import build_subsets            # noqa: E402
from tactical_ilp.sequences.sequence_builder import build_all as build_seq # noqa: E402


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    banner("TASK 2.1 — segment build-up sequences")
    sequences, members = build_seq(force=args.force)
    print(f"sequences              : {len(sequences)}")
    print(f"matches                : {sequences.match_id.nunique()}")
    print(f"length min/median/max  : {sequences.n_actions.min()} / "
          f"{int(sequences.n_actions.median())} / {sequences.n_actions.max()}"
          f"   (acceptance: {MIN_SEQUENCE_LENGTH}..{MAX_SEQUENCE_LENGTH})")
    print(sequences.end_reason.value_counts().to_string())

    banner("TASK 2.2 — label with the operational negative-outcome definition")
    labeled = label_sequences(force=args.force)
    dist = labeled.label.value_counts(normalize=True)
    print(labeled.label.value_counts().to_string())
    print(f"minority class share   : {dist.min():.2%}   (acceptance: >= 10%)")

    banner("TASK 2.3 — extract 2/3-player interaction subsets")
    pairs, triples, with_subsets = build_subsets(force=args.force)
    cov = (with_subsets.n_pairs > 0).mean()
    print(f"pass links             : {len(pairs)}")
    print(f"three-player chains    : {len(triples)}")
    print(f"sequences with a pair  : {cov:.2%}   (acceptance: >= 50%)")

    banner("TASK 2.4 — context-matched negative sampling")
    ex = sample_examples(force=args.force)
    pos = ex[ex.example_class == "pos"]
    neg = ex[ex.example_class == "neg"]
    print(f"positive examples      : {len(pos)}")
    print(f"negative examples      : {len(neg)}")
    print(f"exact-stratum matches  : {(neg.match_level == 3).mean():.1%}")
    print(f"zone overlap pos/neg   : {zone_distribution_overlap(pos, neg):.3f}"
          f"   (acceptance: >= 0.40)")

    banner("PHASE 2 DONE")
    print("next: uv run python scripts/run_phase3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
