"""
Phase 5 evaluation — quick score on one match holdout.

Why this exists
---------------
Popper reports Precision/Recall on the examples it was given.  If those are
all the examples, the numbers measure memorisation, not learning.  Phase 5
now splits by match (scripts/run_phase5.py writes exs.pl + exs_holdout.pl),
and this script scores a hypothesis on matches excluded from that particular
fit. Once candidates have been repeatedly inspected on this split, it is no
longer an independent final test. Use run_phase5_cv.py for the primary
internal estimate.

Usage
-----
    # score the rule Popper printed (paste the body, comma separated)
    uv run python scripts/run_phase5_eval.py \\
        --rule "seq_dropped_between_lines,not_seq_first_time_play,seq_dead_end_support"

    # or score several clauses (a disjunction), one --rule per clause
    uv run python scripts/run_phase5_eval.py \\
        --rule "seq_dwelt_long,slow_sequence" \\
        --rule "seq_dropped_between_lines,seq_dead_end_support"

Always reports the base rate alongside precision, because precision on its
own is meaningless when the classes are imbalanced -- lift is the number
that says whether the rule found anything. Treat the one-split result as
diagnostic after trying multiple rules; use grouped CV before making a
coach-facing claim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactical_ilp.config import PROLOG_DIR  # noqa: E402

POPPER_DIR = PROLOG_DIR / "popper"


def build_goal(clauses: list[str], bk: Path, exs: Path) -> str:
    defs = []
    for i, body in enumerate(clauses):
        lits = ", ".join(x.strip() + "(S)" for x in body.split(",") if x.strip())
        defs.append(f"learned_{i}(S) :- {lits}.")
    disj = "; ".join(f"learned_{i}(S)" for i in range(len(clauses)))
    return f"""
consult('{bk.as_posix()}'),
consult('{exs.as_posix()}'),
{" ".join(f"assertz(({d[:-1]}))," for d in defs)}
assertz((learned(S) :- ({disj}))),
findall(P, pos(negative_outcome(P)), Pos), length(Pos, NP),
findall(N, neg(negative_outcome(N)), Neg), length(Neg, NN),
include(learned, Pos, TPs), length(TPs, TP),
include(learned, Neg, FPs), length(FPs, FP),
Base is NP / (NP + NN),
Prec is TP / max(1, TP + FP),
Rec is TP / max(1, NP),
Lift is Prec / Base,
format("examples   : ~w pos, ~w neg~n", [NP, NN]),
format("base rate  : ~4f~n", [Base]),
FN is NP - TP,
format("TP=~w FP=~w FN=~w~n", [TP, FP, FN]),
format("precision  : ~4f~n", [Prec]),
format("recall     : ~4f~n", [Rec]),
format("LIFT       : ~2f~n", [Lift]),
halt.
"""


def score(clauses: list[str], exs: Path, label: str) -> None:
    bk = POPPER_DIR / "bk.pl"
    goal = build_goal(clauses, bk, exs)
    r = subprocess.run(
        ["swipl", "-D", "encoding=utf8", "-g", goal],
        capture_output=True, text=True, timeout=900,
    )
    print(f"--- {label} ({exs.name}) ---")
    print(r.stdout.strip() or r.stderr.strip())
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", action="append", required=True,
                    help="comma-separated body predicates of one clause")
    args = ap.parse_args()

    train = POPPER_DIR / "exs.pl"
    hold = POPPER_DIR / "exs_holdout.pl"
    if not hold.exists():
        print("no exs_holdout.pl -- run scripts/run_phase5.py first")
        return 1

    print("hypothesis:")
    for c in args.rule:
        print(f"  negative_outcome(S) :- {', '.join(x.strip()+'(S)' for x in c.split(','))}.")
    print()
    score(args.rule, train, "TRAIN (what Popper optimised on)")
    score(args.rule, hold, "HOLDOUT (never seen)")
    print("A rule is only interesting if HOLDOUT lift is meaningfully above 1.00.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
