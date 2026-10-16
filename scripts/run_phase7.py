"""Phase 7 entry point -- regenerate the coach report from the current data.

    python scripts/run_phase7.py            # rebuild data/reports/coach_report.html
    python scripts/run_phase7.py --check    # exit 1 if the file on disk is stale

Run this after ANY of: a new Popper run, an edit to
data/reports/phase6_coach_validation.yaml (approve / reject a rule), or new labels.
Nothing in the report is typed by hand -- see reporting/build_report.py.
"""
import sys

from tactical_ilp.reporting.build_report import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
