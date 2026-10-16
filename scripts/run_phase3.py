"""
Phase 3 — rule-based feature extractor -> Prolog facts.

    uv run python scripts/run_phase3.py [--force]

Produces
    data/prolog/facts_match_<id>.pl   one sample match  (task 3.2)
    data/prolog/facts_all.pl          whole dataset     (task 3.3)
    data/prolog/facts_manifest.json   predicate counts
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import PROCESSED_DIR, PROLOG_DIR                    # noqa: E402
from tactical_ilp.data.load_raw_data import barca_match_ids                 # noqa: E402
from tactical_ilp.data.spadl_converter import build_all as build_spadl_all   # noqa: E402
from tactical_ilp.data.freeze_frames import build_all as build_frames_all    # noqa: E402
from tactical_ilp.sequences.sequence_builder import (                           # noqa: E402
    SEQUENCE_CACHE_VERSION,
    build_all as build_sequences,
    sequence_cache_is_current,
)
from tactical_ilp.sequences.outcome_labeler import label_sequences           # noqa: E402
from tactical_ilp.sequences.negative_sampling import build_context           # noqa: E402
from tactical_ilp.prolog.prolog_facts import (                              # noqa: E402
    FACTS_ALL_PATH,
    MANIFEST_PATH,
    PHASE3_FEATURE_VERSION,
    generate_facts,
)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def consult(path: Path) -> bool:
    if shutil.which("swipl") is None:
        print("  ! SWI-Prolog not on PATH — skipping the load check")
        return False
    r = subprocess.run(
        ["swipl", "-g", f"consult('{path.as_posix()}'), halt.", "-t", "halt(1)"],
        capture_output=True,
        text=True,
    )
    clean = r.returncode == 0 and "Warning" not in r.stderr and "ERROR" not in r.stderr
    print(f"  swipl consult: {'clean' if clean else 'PROBLEM'}  (exit {r.returncode})")
    if not clean and r.stderr:
        print(r.stderr[:1500])
    return clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    phase2_cache_was_stale = not sequence_cache_is_current()

    spadl_path = PROCESSED_DIR / "spadl_all_matches.parquet"
    if spadl_path.exists():
        ids = sorted(
            int(mid) for mid in pd.read_parquet(spadl_path)["match_id"].dropna().unique()
        )
    else:
        # Phase 3 may be run alone before phase 1 has populated its portable cache.
        ids = barca_match_ids()

    if args.force:
        banner("FORCE REBUILD — refreshing source and derived caches")
        build_spadl_all(force=True)
        build_frames_all(force=True)
        build_sequences(force=True)
        label_sequences(force=True)
        build_context(force=True)
    elif phase2_cache_was_stale:
        banner("REFRESH PHASE-2 CACHES — segmentation/label definition changed")
        build_sequences(force=True)
        label_sequences(force=True)
        build_context(force=True)

    sample_path = PROLOG_DIR / f"facts_match_{ids[0]}.pl"
    existing_manifest = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else {}
    )
    if (
        not args.force
        and not phase2_cache_was_stale
        and sample_path.exists()
        and FACTS_ALL_PATH.exists()
        and MANIFEST_PATH.exists()
        and existing_manifest.get("phase2_sequence_cache_version") == SEQUENCE_CACHE_VERSION
        and existing_manifest.get("phase3_feature_version") == PHASE3_FEATURE_VERSION
    ):
        banner("PHASE 3 — reuse packaged Prolog facts")
        print(f"sample facts          : {sample_path}")
        print(f"all facts             : {FACTS_ALL_PATH}")
        print(f"matches / sequences   : {existing_manifest.get('n_matches')} / {existing_manifest.get('n_sequences')}")
        print("Use --force to regenerate the facts from cached phase 1–2 data.")
        print("next: run scripts/run_phase4.py to generate tactical facts and validate the bundled bk.pl rules")
        return 0

    banner("TASK 3.2 — facts for one sample match")
    p1, counts1, ctx1 = generate_facts([ids[0]], sample_path)
    print(f"file                   : {p1}")
    print(f"sequences / facts      : {len(ctx1)} / {sum(counts1.values())}")
    consult(p1)

    banner("TASK 3.3 — facts for the whole dataset")
    p2, counts2, ctx2 = generate_facts(None, FACTS_ALL_PATH)
    print(f"file                   : {p2}")
    print(f"matches                : {ctx2.match_id.nunique()}")
    print(f"sequences              : {len(ctx2)}")
    print(f"total facts            : {sum(counts2.values())}")
    consult(p2)

    print()
    print("predicate counts:")
    for k, v in sorted(counts2.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:>7}")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "sample_match_id": int(ids[0]),
                "phase2_sequence_cache_version": SEQUENCE_CACHE_VERSION,
                "phase3_feature_version": PHASE3_FEATURE_VERSION,
                "sample_file": p1.name,
                "all_file": p2.name,
                "n_matches": int(ctx2.match_id.nunique()),
                "n_sequences": int(len(ctx2)),
                "total_facts": int(sum(counts2.values())),
                "predicate_counts": counts2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    banner("PHASE 3 DONE")
    print("next: run scripts/run_phase4.py to generate tactical facts and validate the bundled bk.pl rules")
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged(3, main))
