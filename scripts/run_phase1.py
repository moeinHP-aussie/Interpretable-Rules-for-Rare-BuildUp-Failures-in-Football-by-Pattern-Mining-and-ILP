"""
Phase 1 — download, convert and prepare the raw data.

    uv run python scripts/run_phase1.py            # uses the cache when present
    uv run python scripts/run_phase1.py --force    # rebuild everything

    The first run downloads the selected team's season of event + 360 data from StatsBomb Open Data
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
from tactical_ilp.data.spadl_converter import read_if_current             # noqa: E402
from tactical_ilp.config import EXCLUDE_UNLINKED_360_MATCHES, PHASE1_VERSION, PROCESSED_DIR, TEAM_NAME    # noqa: E402
import pandas as pd                                                            # noqa: E402


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def _per_match_orientation(spadl: pd.DataFrame) -> None:
    """Hard per-match check (the season mean hides mirrored matches)."""
    shots = spadl[spadl.type_name.astype(str).str.startswith("shot")]
    pm = shots.groupby(["match_id", "is_barca"])["start_x"].mean().unstack()
    bad = pm[(pm.get(True, 100) <= 65) | (pm.get(False, 0) >= 40)]
    print(f"per-match orientation check: {len(pm) - len(bad)}/{len(pm)} matches OK")
    if len(bad):
        print(bad.round(1).to_string())
        raise RuntimeError(
            f"{TEAM_NAME} does not attack towards +x in matches {bad.index.tolist()}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    # A portable phase-1 snapshot may be shipped in data/processed without
    # the per-match raw download cache. Reuse it directly unless explicitly
    # asked to rebuild; this avoids even the initial match-list network call.
    cached_spadl = PROCESSED_DIR / "spadl_all_matches.parquet"
    cached_merged = PROCESSED_DIR / "spadl_with_360_all_matches.parquet"
    cached_players = PROCESSED_DIR / "players_reference.parquet"
    packaged_ok = False
    if not args.force and all(p.exists() for p in (cached_spadl, cached_merged, cached_players)):
        spadl = read_if_current(cached_spadl)
        merged = read_if_current(cached_merged)
        packaged_ok = spadl is not None and merged is not None
        if not packaged_ok:
            banner("packaged phase-1 data is STALE (written by older code)")
            print(f"current phase-1 version: {PHASE1_VERSION}")
            print("rebuilding SPADL + 360 tables from the raw cache / StatsBomb Open Data ...")
    if packaged_ok:
        banner("TASKS 1.1–1.4 — reuse packaged phase-1 data (no download)")
        selected_ids = barca_match_ids()
        frame_matches = merged.groupby("match_id")["has_freeze_frame"].any()
        missing_360 = [mid for mid in selected_ids if not bool(frame_matches.get(mid, False))]
        if missing_360 and not EXCLUDE_UNLINKED_360_MATCHES:
            raise RuntimeError(
                f"The selected team-season is not fully covered by 360 data; "
                f"missing matches: {missing_360}"
            )
        ref = pd.read_parquet(cached_players)
        print(f"SPADL actions          : {len(spadl)}")
        shots = spadl[spadl.type_name == "shot"]
        print(f"reference frame check ({TEAM_NAME} must attack towards +x):")
        print(shots.groupby("is_barca")["start_x"].mean().round(1).to_string())
        _per_match_orientation(spadl)
        build_up = merged[merged.type_name.isin(["pass", "dribble"])]
        coverage = build_up["has_freeze_frame"].mean()
        print(f"build-up actions       : {len(build_up)}")
        print(f"360 coverage           : {coverage:.2%}   (acceptance: >= 70%)")
        core = ref[ref.is_core]
        print(f"players                : {len(ref)}  (core: {len(core)})")
        print(core[["player_atom", "display_name", "role", "appearances"]].head(15).to_string(index=False))
        banner("PHASE 1 DONE — packaged processed data reused")
        print("Use --force to rebuild from raw StatsBomb data.")
        print("next: uv run python scripts/run_phase2.py")
        return 0

    banner("TASK 1.1 — download and cache raw data")
    ids = barca_match_ids(force=args.force)
    print(f"{TEAM_NAME} matches found: {len(ids)}")
    status = download_all(ids, force=args.force)
    print(f"matches with 360 data  : {int(status.has_360.sum())}/{len(status)}")
    if len(status) != len(ids):
        raise RuntimeError("download status does not cover every selected match")
    if not status.has_360.all():
        missing = status.loc[~status.has_360, "match_id"].astype(int).tolist()
        if not EXCLUDE_UNLINKED_360_MATCHES:
            raise RuntimeError(
                f"The selected team-season is not fully covered by 360 data; "
                f"missing/unlinked matches: {missing}"
            )
        print(f"!! {len(missing)} match(es) have no usable 360 data (missing or event ids "
              f"do not link) and will be EXCLUDED from the analysis: {missing}")

    banner("TASK 1.2 — convert to SPADL")
    spadl = build_spadl(ids, force=args.force)
    print(f"total SPADL actions    : {len(spadl)}")
    shots = spadl[spadl.type_name == "shot"]
    print(f"reference frame check ({TEAM_NAME} must attack towards +x):")
    print(shots.groupby("is_barca")["start_x"].mean().round(1).to_string())
    _per_match_orientation(spadl)

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
    from run_logging import run_logged
    raise SystemExit(run_logged(1, main))
