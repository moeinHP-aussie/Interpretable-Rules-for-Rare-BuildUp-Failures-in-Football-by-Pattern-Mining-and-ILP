"""
Phase 1 — download, convert and prepare the raw data.

    uv run python scripts/run_phase1.py            # uses the cache when present
    uv run python scripts/run_phase1.py --force    # rebuild everything

The first run downloads ~35 matches of event + 360 data from StatsBomb Open Data
and takes several minutes.  Every later run reads the local parquet cache and
takes a few seconds.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.data.freeze_frames import build_all as build_merged      # noqa: E402
from tactical_ilp.data.load_raw_data import barca_match_ids, download_all  # noqa: E402
from tactical_ilp.data.players_reference import build_players_reference    # noqa: E402
from tactical_ilp.data.spadl_converter import build_all as build_spadl     # noqa: E402


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    banner("TASK 1.1 — download and cache raw data")
    ids = barca_match_ids(force=args.force)
    print(f"Barcelona matches found: {len(ids)}")
    status = download_all(ids, force=args.force)
    print(f"matches with 360 data  : {int(status.has_360.sum())}/{len(status)}")

    banner("TASK 1.2 — convert to SPADL")
    spadl = build_spadl(ids, force=args.force)
    print(f"total SPADL actions    : {len(spadl)}")
    shots = spadl[spadl.type_name == "shot"]
    print("reference frame check (Barcelona must attack towards +x):")
    print(shots.groupby("is_barca")["start_x"].mean().round(1).to_string())

    banner("TASK 1.3 — link StatsBomb 360 freeze frames")
    merged = build_merged(ids, force=args.force)
    build_up = merged[merged.type_name.isin(["pass", "dribble"])]
    coverage = build_up["has_freeze_frame"].mean()
    print(f"build-up actions       : {len(build_up)}")
    print(f"360 coverage           : {coverage:.2%}   (acceptance: >= 70%)")

    banner("TASK 1.4 — player identity and role reference")
    ref = build_players_reference(force=args.force)
    core = ref[ref.is_core]
    print(f"players                : {len(ref)}  (core: {len(core)})")
    print(core[["player_atom", "display_name", "role", "appearances"]].head(15).to_string(index=False))

    banner("PHASE 1 DONE")
    print("next: uv run python scripts/run_phase2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
