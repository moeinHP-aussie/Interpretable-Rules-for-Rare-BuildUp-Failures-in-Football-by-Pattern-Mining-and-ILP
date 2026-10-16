"""
Phase 4 — build the tactical concept facts and validate bk.pl.

    uv run python scripts/run_phase4.py

Phase 3 must have run first (facts_all.pl has to exist).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from tactical_ilp.config import PROLOG_DIR                                 # noqa: E402
from tactical_ilp.prolog.tactical_facts import (                           # noqa: E402
    PHASE4_FACTS_PATH,
    PHASE4_FEATURE_VERSION,
    PHASE4_MANIFEST_PATH,
    TEMPO_FAST_PERCENTILE,
    generate_tactical_facts,
)
from tactical_ilp.prolog.prolog_facts import MANIFEST_PATH, PHASE3_FEATURE_VERSION  # noqa: E402

BK_PATH = PROLOG_DIR / "bk.pl"
FACTS_ALL = PROLOG_DIR / "facts_all.pl"
BK_TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "tactical_ilp" / "prolog" / "bk.pl"

# Active rules used by the declared Popper vocabularies. Retired
# concepts remain documented in bk.pl but are not generated or validated here.
RULES = [
    ("1.1", "line_breaking_pass(_)"),
    ("1.2", "diagonal_pass(_)"),
    ("1.3", "line_pass(_)"),
    ("1.4a", "switch_to_right(_)"),
    ("1.4b", "switch_to_left(_)"),
    ("1.8", "aerial_pass(_)"),
    ("1.9", "long_pass(_)"),
    ("1.5", "pass_in_behind(_)"),
    ("1.6", "connecting_pass(_)"),
    ("2.3", "dropped_between_lines(_)"),
    ("3.4", "give_and_continue(_,_)"),
    ("3.6a", "triangulation(_)"),
    ("3.6b", "triangle_trap(_)"),
    ("3.6b", "rondo_cage(_)"),
    ("4.1", "supported_action(_)"),
    ("4.3", "dead_end_support(_)"),
    ("4.6", "opponent_visible_majority(_)"),
    ("5.1", "persistent_half_space_occupier(_,_)"),
    ("6.2", "fast_sequence(_)"),
    ("6.2", "slow_sequence(_)"),
    ("6.3", "first_time_play(_)"),
    ("6.3b", "quick_line_break(_)"),
    ("6.4", "dwelt_long(_)"),
    ("8.3", "counter_pressed(_)"),
    ("8.4", "press_resistance(_)"),
    ("12.1", "keeper_build_up_start(_)"),
    ("12.4", "keeper_played_open_play(_)"),
]

PROLOG_ERROR_DETAILS: list[str] = []


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def swipl_available() -> bool:
    try:
        return subprocess.run(["swipl", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def count_rule(goal: str) -> int | None:
    from tactical_ilp.config import PROLOG_DIR

    script = (
        f"consult('{FACTS_ALL}'), consult('{PHASE4_FACTS_PATH}'), "
        f"consult('{BK_PATH}'), aggregate_all(count, {goal}, N), write(N), nl, halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    if out.returncode != 0 or "ERROR:" in out.stderr:
        if not PROLOG_ERROR_DETAILS:
            detail = out.stderr.strip() or out.stdout.strip() or f"SWI-Prolog exit code {out.returncode}"
            PROLOG_ERROR_DETAILS.append(f"query: {goal}\n{detail}")
        return None
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def ensure_handwritten_bk() -> bool:
    """Seed the runtime Prolog directory with the bundled rule source if absent.

    ``data/`` is intentionally excluded from the clean Colab archive, while
    ``bk.pl`` is source code rather than a generated cache.  Keep its canonical
    copy under ``src/`` and materialize it only when the runtime file is absent,
    preserving any deliberate user edits already present in ``data/prolog``.
    """
    if not BK_TEMPLATE.exists():
        print(f"missing bundled BK source: {BK_TEMPLATE}")
        return False
    bundled_source = BK_TEMPLATE.read_text(encoding="utf-8")
    if BK_PATH.exists():
        runtime_source = BK_PATH.read_text(encoding="utf-8")

        # Synchronize only Phase-4 rules whose inputs/semantics are versioned
        # with the current tactical facts. Keep unrelated local BK rules.
        changed_heads = (
            "same_lane(A)",
            "dropped_between_lines(A)",
            "opponent_visible_majority(S)",
        )
        for head in changed_heads:
            rule_pattern = re.compile(
                rf"(?m)^{re.escape(head)} :-[\s\S]*?\.\r?\n"
            )
            bundled_rule = rule_pattern.search(bundled_source)
            runtime_rule = rule_pattern.search(runtime_source)
            if not bundled_rule:
                continue
            if runtime_rule and runtime_rule.group(0) == bundled_rule.group(0):
                continue
            if runtime_rule:
                runtime_source = (
                    runtime_source[: runtime_rule.start()]
                    + bundled_rule.group(0)
                    + runtime_source[runtime_rule.end() :]
                )
            else:
                runtime_source = runtime_source.rstrip() + "\n\n" + bundled_rule.group(0)

        cross_rule = "pass_like(A) :- action_type(A, cross)."
        if cross_rule in bundled_source and cross_rule not in runtime_source:
            runtime_source = runtime_source.rstrip() + "\n" + cross_rule + "\n"

        if runtime_source != BK_PATH.read_text(encoding="utf-8"):
            BK_PATH.write_text(runtime_source, encoding="utf-8")
            print("updated phase-4 rules in runtime bk.pl to match the current feature definitions")

        if "successful(A) :- action_result(A, success)." not in runtime_source:
            with BK_PATH.open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n%% Compact action-result aliases (phase-3 v3).\n"
                    "successful(A) :- action_result(A, success).\n"
                    "failed(A) :- action_result(A, Result), Result \\== success.\n"
                )
        return True
    BK_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BK_TEMPLATE, BK_PATH)
    print(f"installed bundled Prolog rules: {BK_PATH}")
    return True


def main() -> int:
    if not FACTS_ALL.exists():
        print(f"missing {FACTS_ALL} -- run scripts/run_phase3.py first")
        return 1
    if not MANIFEST_PATH.exists():
        print(f"missing {MANIFEST_PATH} -- regenerate phase-3 facts before phase 4")
        return 1
    import json

    phase3_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if phase3_manifest.get("phase3_feature_version") != PHASE3_FEATURE_VERSION:
        print("phase-3 facts use an older vocabulary; run scripts/run_phase3.py before phase 4")
        return 1
    if not ensure_handwritten_bk():
        return 1

    banner("TASK 4.1-4.5 — generate tactical concept facts")
    path, counts, ctx, fast_min = generate_tactical_facts()
    total = sum(counts.values())
    print(f"{path.name}: {total} facts over {len(ctx)} sequences")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v:>7}")

    import json

    PHASE4_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "file": path.name,
                "phase4_feature_version": PHASE4_FEATURE_VERSION,
                "tempo_fast_percentile": TEMPO_FAST_PERCENTILE,
                "tempo_fast_min_actions_per_second": fast_min,
                "n_sequences": int(len(ctx)),
                "n_matches": int(ctx.match_id.nunique()),
                "predicate_counts": counts,
                "total_facts": int(total),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    banner("TASK 4.6 — validate bk.pl rules against the facts")
    if not swipl_available():
        print("SWI-Prolog not on PATH — skipping rule validation")
        print("(install swi-prolog to run this step)")
        return 0
    if not BK_PATH.exists():
        print(f"missing {BK_PATH}")
        return 1

    print(f"{'concept':<10} {'rule':<40} {'solutions':>9}")
    print("-" * 60)
    empty = []
    errors = []
    for section, goal in RULES:
        n = count_rule(goal)
        flag = ""
        if n is None:
            flag = "  ERROR"
            errors.append(goal)
        elif n == 0:
            flag = "  <-- EMPTY"
            empty.append(goal)
        print(f"{section:<10} {goal:<40} {str(n):>9}{flag}")

    if errors:
        banner("PHASE 4 FAILED")
        print(f"ERROR: {len(errors)} rule checks failed in SWI-Prolog: {errors}")
        if PROLOG_ERROR_DETAILS:
            print("\nFirst underlying SWI-Prolog diagnostic:")
            print(PROLOG_ERROR_DETAILS[0])
        return 1
    banner("PHASE 4 DONE")
    if empty:
        print(f"WARNING: {len(empty)} rule(s) returned no solutions: {empty}")
        print("These predicates are retained for inspection; an empty count is data-dependent, not a Prolog load error.")
    else:
        print("all rules produce solutions")
    print("next: scripts/run_phase5.py")
    return 0


if __name__ == "__main__":
    from run_logging import run_logged
    raise SystemExit(run_logged(4, main))


