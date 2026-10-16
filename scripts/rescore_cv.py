#!/usr/bin/env python3
"""Rescore stored CV programs counting DISTINCT sequences (no Popper re-run).

Put this file next to the patched phase5_cv_common.py (the version using once/1).

OOF mode (re-evaluates the programs already stored in a CV report):
  python rescore_cv.py oof --report phase5_cross_validation.json \
      --task-pattern "/ABS/PATH/popper_cv_fold_{fold:02d}" --out phase5_cross_validation_rescored.json
  (task dir = the folder holding that fold's bk.pl and exs_holdout.pl)

Rules mode (train/holdout stats of individual rule bodies):
  python rescore_cv.py rules --bk bk.pl --exs-train exs.pl --exs-holdout exs_holdout.pl \
      --body seq_dead_end_support,fast_sequence --body seq_pattern_01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_cv_common import _score_program  # noqa: E402  (patched: once/1)

KEYS = ("n_positive", "n_negative", "support", "tp", "fp", "fn", "tn")


def _body(clause):
    return clause["body"] if isinstance(clause, dict) else clause


def oof(report: Path, pattern: str, out: Path) -> None:
    data = json.loads(report.read_text(encoding="utf-8"))
    total = {k: 0 for k in KEYS}
    for fr in data["fold_results"]:
        task = Path(pattern.format(fold=fr["fold"]))
        bk, ex = task / "bk.pl", task / "exs_holdout.pl"
        bodies = [_body(c) for c in fr["clauses"]]
        new = _score_program(bodies, bk, ex)
        old = fr["test_metrics"]
        print(f"fold {fr['fold']}: support {old['support']} -> {new['support']} | "
              f"TP {old['tp']} -> {new['tp']} | FP {old['fp']} -> {new['fp']} | "
              f"test sequences {fr.get('n_test_all_labelled_sequences')}")
        fr["test_metrics_original"] = old
        fr["test_metrics"] = new
        for clause in fr["clauses"]:
            if isinstance(clause, dict):
                clause["test_metrics_original"] = clause.get("test_metrics")
                clause["test_metrics"] = _score_program([clause["body"]], bk, ex)
        for k in KEYS:
            total[k] += int(new[k])
    n = total["n_positive"] + total["n_negative"]
    base = total["n_positive"] / n if n else 0.0
    prec = total["tp"] / total["support"] if total["support"] else 0.0
    rec = total["tp"] / total["n_positive"] if total["n_positive"] else 0.0
    total.update(base_rate=round(base, 6), precision=round(prec, 6),
                 recall=round(rec, 6), lift=round(prec / base, 6) if base else 0.0)
    data["out_of_fold_metrics_original"] = data.get("out_of_fold_metrics")
    data["out_of_fold_metrics"] = total
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOOF (rescored):", total)
    print("written:", out)


def rules(bk: Path, exs_train: Path, exs_holdout: Path, bodies: list[str]) -> None:
    for text in bodies:
        body = [x.strip() for x in text.split(",") if x.strip()]
        for name, ex in (("train", exs_train), ("holdout", exs_holdout)):
            m = _score_program([body], bk, ex)
            conf = m["tp"] / m["support"] if m["support"] else 0.0
            print(f"{text:70s} {name:8s} support={m['support']:4d} tp={m['tp']:3d} "
                  f"confidence={conf:.3f} lift={m['lift']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("oof")
    a.add_argument("--report", type=Path, required=True)
    a.add_argument("--task-pattern", required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("rules")
    b.add_argument("--bk", type=Path, required=True)
    b.add_argument("--exs-train", type=Path, required=True)
    b.add_argument("--exs-holdout", type=Path, required=True)
    b.add_argument("--body", action="append", required=True)
    args = ap.parse_args()
    if args.mode == "oof":
        oof(args.report, args.task_pattern, args.out)
    else:
        rules(args.bk, args.exs_train, args.exs_holdout, args.body)


if __name__ == "__main__":
    main()
