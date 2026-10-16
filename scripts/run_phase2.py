"""
Phase 2 — build-up sequences, outcome labels, player subsets, negative sampling.

    uv run python scripts/run_phase2.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import (                                            # noqa: E402
    MAX_GLOBAL_MATCH_SHARE,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
)
from tactical_ilp.sequences.negative_sampling import (                     # noqa: E402
    build_context,
    sample_examples,
    zone_distribution_overlap,
)
from tactical_ilp.sequences.outcome_labeler import label_sequences         # noqa: E402
from tactical_ilp.sequences.player_subsets import build_subsets            # noqa: E402
from tactical_ilp.sequences.sequence_builder import (                         # noqa: E402
    build_all as build_seq,
    sequence_cache_is_current,
)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Rebuild all downstream phase-2 products once when the segmentation
    # definition changes; otherwise preserve and reuse the packaged caches.
    rebuild = args.force or not sequence_cache_is_current()
    if rebuild and not args.force:
        print("Phase-2 cache uses an older segmentation/label definition; rebuilding its derived outputs.")

    banner("TASK 2.1 — segment build-up sequences")
    sequences, members = build_seq(force=rebuild)
    print(f"sequences              : {len(sequences)}")
    print(f"matches                : {sequences.match_id.nunique()}")
    print(f"length min/median/max  : {sequences.n_actions.min()} / "
          f"{int(sequences.n_actions.median())} / {sequences.n_actions.max()}"
          f"   (acceptance: {MIN_SEQUENCE_LENGTH}..{MAX_SEQUENCE_LENGTH})")
    print(sequences.end_reason.value_counts().to_string())

    banner("TASK 2.2 — label with the operational negative-outcome definition")
    labeled = label_sequences(force=rebuild)
    dist = labeled.label.value_counts(normalize=True)
    print(labeled.label.value_counts().to_string())
    print(f"minority class share   : {dist.min():.2%}   (acceptance: >= 10%)")
    print("negative reason counts:")
    print(
        labeled.loc[labeled.label == "negative", "negative_reason"]
        .value_counts(dropna=False)
        .to_string()
    )
    print("\nturnover zone x label:")
    print(pd.crosstab(labeled.turnover_zone, labeled.label).to_string())
    print("\nsequence end reason x label:")
    print(pd.crosstab(labeled.end_reason, labeled.label).to_string())
    print("\nlabel diagnostics:")
    print(f"regained within 5s             : {labeled.regained_within_5s.mean():.1%}")
    print(f"opponent shot after loss       : {labeled.opponent_shot_after.mean():.1%}")
    print(f"opponent entered defensive third: {labeled.opponent_entered_def_third.mean():.1%}")

    banner("TASK 2.3 — extract 2/3-player interaction subsets")
    pairs, triples, with_subsets = build_subsets(force=rebuild)
    cov = (with_subsets.n_pairs > 0).mean()
    print(f"pass links             : {len(pairs)}")
    print(f"three-player chains    : {len(triples)}")
    print(f"sequences with a pair  : {cov:.2%}   (acceptance: >= 50%)")

    banner("TASK 2.4 — context-matched negative sampling")
    build_context(force=rebuild)
    ex = sample_examples(force=rebuild)
    pos = ex[ex.example_class == "pos"]
    neg = ex[ex.example_class == "neg"]
    print(f"positive examples      : {len(pos)}")
    print(f"negative examples      : {len(neg)}")
    print(f"negative:positive ratio: 1:{len(neg) / max(len(pos), 1):.2f}")
    exact_share = float((neg.match_level == 3).mean()) if len(neg) else 0.0
    print(f"exact-stratum matches  : {exact_share:.1%}")
    global_share = float((neg.match_level == 0).mean()) if len(neg) else 0.0
    print(
        f"global fallback share  : {global_share:.1%}"
        f"   (maximum: {MAX_GLOBAL_MATCH_SHARE:.0%})"
    )
    print(f"zone overlap pos/neg   : {zone_distribution_overlap(pos, neg):.3f}"
          f"   (acceptance: >= 0.40)")

    banner("PHASE 2 DONE")
    print("next: uv run python scripts/run_phase3.py")
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged(2, main))
