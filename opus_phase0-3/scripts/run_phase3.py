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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import PROLOG_DIR                                  # noqa: E402
from tactical_ilp.data.load_raw_data import barca_match_ids                 # noqa: E402
from tactical_ilp.prolog.prolog_facts import (                              # noqa: E402
    FACTS_ALL_PATH,
    MANIFEST_PATH,
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

    ids = barca_match_ids()

    banner("TASK 3.2 — facts for one sample match")
    sample_path = PROLOG_DIR / f"facts_match_{ids[0]}.pl"
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
    print("phase 4 (bk.pl) requires your tactical input — it is not automated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
