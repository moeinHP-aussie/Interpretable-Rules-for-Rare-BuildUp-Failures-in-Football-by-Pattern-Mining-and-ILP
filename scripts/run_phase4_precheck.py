"""
Phase 4 pre-check -- run the Prolog fact-volume/cost estimator over every
glossary concept that risks a large fact count, and save both a human-
readable report and a machine-readable JSON summary.

    uv run python scripts/run_phase4_precheck.py

Output:
    data/prolog/phase4_cost_report.txt   human-readable (same as stdout)
    data/prolog/phase4_cost_report.json  machine-readable, one row per concept
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactical_ilp.config import PROLOG_DIR  # noqa: E402
from tactical_ilp.estimation.cost_report import (  # noqa: E402
    print_report,
    run_all_estimates,
)


def main() -> int:
    results = run_all_estimates(n_sample_matches=8)

    txt_path = PROLOG_DIR / "phase4_cost_report.txt"
    json_path = PROLOG_DIR / "phase4_cost_report.json"

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results)
    text = buf.getvalue()
    print(text)
    txt_path.write_text(text, encoding="utf-8")

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nsaved -> {txt_path}")
    print(f"saved -> {json_path}")

    reds = [r for r in results if r.verdict == "RED"]
    return 1 if reds else 0


if __name__ == "__main__":
    raise SystemExit(main())
