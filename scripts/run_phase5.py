"""
Phase 5 — generate Popper's input files and run Popper.

    uv run python scripts/run_phase5.py

Requires phase 4 to have run (facts_tactical.pl and bk.pl must exist).

Popper uses a separate uv-managed Python (>=3.14) environment with janus_swi.
The solver is invoked through that environment, not the project's Python
3.11/3.12 environment, and its full output is saved under data/reports/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import queue
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
POPPER_SEARCH_TIMEOUT_S = 900
# Keep a small outer margin for uv/process startup and writing the partial log.
POPPER_HARD_TIMEOUT_S = POPPER_SEARCH_TIMEOUT_S + 60


def read_max_clauses(bias_path: Path) -> int:
    """Read the one-clause-per-rule ceiling emitted in Popper's bias."""
    import re

    matches = re.findall(
        r"^\s*max_clauses\(\s*(\d+)\s*\)\s*\.\s*$",
        bias_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1 or int(matches[0]) < 1:
        raise ValueError(f"expected one positive max_clauses/1 in {bias_path}")
    return int(matches[0])


def enforce_noisy_clause_limit(max_clauses: int) -> None:
    """Make Popper's noisy combiner honor max_clauses/1 from bias.pl."""
    from bitarray.util import count_and, zeros
    from popper.combiner_mdl import CombinerMDL
    from popper.state import update_best_hypothesis

    original_build_mdl_model = CombinerMDL._build_mdl_model

    def build_bounded_mdl_model(self):
        result = original_build_mdl_model(self)
        model, rule_vars = result[0], result[1]
        model.Add(sum(rule_vars.values()) <= max_clauses)
        return result

    def bounded_greedy_upper_bound(self):
        covered_pos = zeros(self.tester.num_pos)
        covered_neg = zeros(self.tester.num_neg)
        total_size = 0
        candidates = list(self.saved_progs)
        selected = []

        while candidates and len(selected) < max_clauses:
            best = None
            best_delta = 0
            for hypothesis in candidates:
                new_tp = count_and(~covered_pos, self.coverage_pos[hypothesis])
                new_fp = count_and(~covered_neg, self.coverage_neg[hypothesis])
                size, _tp, _fp = self.scores[hypothesis]
                delta = size + new_fp - new_tp
                if best is None or delta < best_delta:
                    best = hypothesis
                    best_delta = delta
            if best is None or best_delta >= 0:
                break
            size, _tp, _fp = self.scores[best]
            total_size += size
            covered_pos |= self.coverage_pos[best]
            covered_neg |= self.coverage_neg[best]
            candidates.remove(best)
            selected.append(best)

        fn = self.tester.num_pos - covered_pos.count(1)
        fp = covered_neg.count(1)
        mdl = total_size + fp + fn
        current_best = self.state.best_hypothesis_mdl or float("inf")
        if mdl >= current_best:
            return False
        program = [rule for hypothesis in selected for rule in self.prog_lookup[hypothesis]]
        tp = self.tester.num_pos - fn
        tn = self.tester.num_neg - fp
        update_best_hypothesis(self.settings, self.state, program, total_size, (tp, fn, tn, fp))
        return True

    CombinerMDL._build_mdl_model = build_bounded_mdl_model
    CombinerMDL.greedy_upper_bound = bounded_greedy_upper_bound


def run_popper_worker() -> int:
    """Worker entry point, invoked by this file inside Popper's uv project."""
    popper_dir = os.environ.get("POPPER_DIR")
    if not popper_dir:
        raise RuntimeError("POPPER_DIR must point to the Popper checkout")
    sys.path.insert(0, str(Path(popper_dir).resolve()))

    # Remove the dispatch flag before Popper parses its normal CLI arguments.
    sys.argv.pop(1)
    from popper import stats
    from popper.loop import popper
    from popper.util import Settings, print_prog_score

    settings = Settings.from_args()
    max_clauses = read_max_clauses(Path(settings.bias_file))
    settings.max_rules = max_clauses
    enforce_noisy_clause_limit(max_clauses)
    print(f"Enforced maximum clauses per noisy hypothesis: {max_clauses}")
    program, score = popper(settings)
    if program:
        print_prog_score(program, score, settings, settings.noisy)
    else:
        print("NO SOLUTION")
    if settings.show_stats:
        stats.show()
    return 0 if program else 1


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


def swipl_version_ok() -> bool:
    """Popper's janus_swi needs SWI-Prolog >= 9.1.12."""
    try:
        out = subprocess.run(["swipl", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    import re

    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    if not m:
        return False
    major, minor, patch = map(int, m.groups())
    return (major, minor, patch) >= (9, 1, 12)


def validate_generated_files(paths: dict) -> bool:
    """Sanity-check the generated files parse and the wrappers fire, without
    needing Popper itself -- pure SWI-Prolog is enough for this."""
    if not swipl_available():
        print("SWI-Prolog not on PATH -- cannot even syntax-check the output")
        return False

    script = (
        f"consult('{paths['exs']}'), "
        f"aggregate_all(count, pos(_), NP), aggregate_all(count, neg(_), NN), "
        f"consult('{paths['bk']}'), "
        "aggregate_all(count, opponent_visible_majority(_), NMAJ), "
        "write(ok(NP,NN,NMAJ)), nl, halt."
    )
    out = subprocess.run(["swipl", "-q", "-g", script], capture_output=True, text=True)
    if out.returncode != 0 or "ERROR:" in out.stderr:
        print("VALIDATION FAILED:")
        print(out.stdout)
        print(out.stderr)
        return False
    print(f"exs.pl + bk.pl load cleanly: {out.stdout.strip()}")
    return True


def try_run_popper(popper_dir: Path, task_dir: Path) -> int | None:
    """Run Popper in its own uv-managed Python environment (Python >=3.14)."""
    from tactical_ilp.config import REPORTS_DIR

    popper_py = popper_dir / "popper.py"
    if not popper_py.exists():
        print(f"no popper.py at {popper_py} -- skipping actual run")
        return None
    uv = shutil.which("uv")
    if uv is None:
        print("uv not found -- cannot activate Popper's Python >=3.14 environment")
        return None

    command = [
        uv, "run", "--project", str(popper_dir), "python", "-u",
        str(Path(__file__).resolve()), "--_popper-worker",
        str(task_dir), "--noisy", "--timeout", str(POPPER_SEARCH_TIMEOUT_S),
    ]
    try:
        env = os.environ.copy()
        # Colab's project environment is active in the parent notebook.  Popper
        # intentionally uses its own Python/uv project, so don't let the
        # unrelated VIRTUAL_ENV setting confuse uv's environment selection.
        env.pop("VIRTUAL_ENV", None)
        env["POPPER_DIR"] = str(popper_dir.resolve())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / "phase5_popper_output.txt"
        print(f"Popper live output (also saved to {output_path}):", flush=True)
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def forward_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=forward_output, name="popper-output-reader", daemon=True)
        reader.start()
        captured: list[str] = []
        deadline = time.monotonic() + POPPER_HARD_TIMEOUT_S
        stream_closed = False
        timed_out = False
        with output_path.open("w", encoding="utf-8", buffering=1) as output_file:
            while not stream_closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = lines.get(timeout=min(0.5, remaining))
                except queue.Empty:
                    continue
                if line is None:
                    stream_closed = True
                else:
                    captured.append(line)
                    output_file.write(line)
                    print(line, end="", flush=True)

            if timed_out:
                proc.kill()
                proc.wait()
                reader.join(timeout=5)
                while True:
                    try:
                        line = lines.get_nowait()
                    except queue.Empty:
                        break
                    if line is not None:
                        captured.append(line)
                        output_file.write(line)
                        print(line, end="", flush=True)
                note = (
                    f"HARD TIMEOUT after {POPPER_HARD_TIMEOUT_S} seconds. "
                    "Any printed hypothesis is provisional; this run did not finish.\n"
                )
                output_file.write(note)
                print(note, end="", flush=True)
        combined = "".join(captured)
        # stdout can close a moment before the OS reports child termination;
        # collect the final status before interpreting worker success/failure.
        proc.wait()
        # A complete final SOLUTION is emitted only after Popper's search has
        # returned a program. Some solver/Janus builds can still exit nonzero
        # during shutdown after printing that result; preserve the completed
        # hypothesis and expose the status instead of discarding valid output.
        has_final_solution = bool(
            re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?\*+\s*SOLUTION\s*\*+\s*$", combined)
            and re.search(r"(?m)^\s*(?:\d+(?:\.\d+)?s\s+)?\*{10,}\s*$", combined)
        )
        if timed_out:
            return 1
        if proc.returncode != 0:
            print(f"Popper worker exited with status {proc.returncode}.")
            if has_final_solution:
                print("A complete final SOLUTION was emitted; preserving it as a completed search result.")
                return 0
            return proc.returncode
        if not has_final_solution:
            print("Popper finished without reporting a SOLUTION; no learned rule to review.")
            return 1
        return 0
    except Exception as exc:
        print(f"Popper invocation failed: {exc}")
        return 1


def main() -> int:
    import warnings
    import json

    warnings.filterwarnings("ignore")
    from tactical_ilp.config import PROLOG_DIR, REPORTS_DIR
    from tactical_ilp.prolog.legacy_43_inputs import generate_popper_inputs
    from tactical_ilp.prolog.tactical_facts import (
        PHASE4_FEATURE_VERSION,
        PHASE4_MANIFEST_PATH,
    )

    facts_tactical = PROLOG_DIR / "facts_tactical.pl"
    hand_bk = PROLOG_DIR / "bk.pl"
    missing = [str(path) for path in (facts_tactical, hand_bk) if not path.exists()]
    if missing:
        print("missing phase 4 output:")
        for path in missing:
            print(f"  - {path}")
        print("run scripts/run_phase4.py and check that it ends with PHASE 4 DONE")
        return 1
    if not PHASE4_MANIFEST_PATH.exists():
        print(f"missing {PHASE4_MANIFEST_PATH} -- regenerate phase-4 facts")
        return 1
    phase4_manifest = json.loads(PHASE4_MANIFEST_PATH.read_text(encoding="utf-8"))
    if phase4_manifest.get("phase4_feature_version") != PHASE4_FEATURE_VERSION:
        print("phase-4 facts use an older feature definition; rerun scripts/run_phase4.py")
        return 1

    banner("TASK 5.1 — generate Popper input files (exs.pl, bias.pl, bk.pl)")
    info = generate_popper_inputs()
    print(f"output dir : {info['dir']}")
    print(f"exs.pl     : {info['n_pos']} pos, {info['n_neg']} neg")
    split = info["split"]
    print(
        "match split: "
        f"train={split['n_train_matches']} matches "
        f"({split['train_pos']} pos/{split['train_neg']} neg), "
        f"holdout={len(split['holdout_matches'])} matches "
        f"({split['holdout_pos']} pos/{split['holdout_neg']} neg)"
    )
    print(f"matched negatives: {split['matched_negatives']}")
    print(
        f"bias.pl    : {info['n_body_preds']} body predicates "
        f"({info['n_base_body_preds']} base + {info['n_complement_preds']} complements), "
        f"max_vars={info['max_vars']}, max_body={info['max_body']}, "
        f"max_clauses={info['max_clauses']}"
    )
    print(
        f"Popper search timeout: {POPPER_SEARCH_TIMEOUT_S}s "
        f"(hard stop at {POPPER_HARD_TIMEOUT_S}s)"
    )
    print("Search trace below is Popper's emitted progress/incumbent output; internal solver decisions not emitted by Popper are not reconstructed.")

    banner("TASK 5.2 — validate the generated files")
    ok = validate_generated_files(info)
    if not ok:
        return 1

    banner("TASK 5.3 — run Popper in its dedicated environment")
    popper_dir = os.environ.get("POPPER_DIR")
    if not popper_dir:
        print("POPPER_DIR is required; set it to the Popper checkout prepared by the Colab setup.")
        return 1

    rc = try_run_popper(Path(popper_dir), info["dir"])
    if rc is None or rc != 0:
        banner("PHASE 5 FAILED")
        print("Popper did not complete successfully; inspect data/reports/phase5_popper_output.txt.")
        return 1
    banner("PHASE 5 DONE")
    print()
    print("Now score the learned rule on the held-out matches (exs_holdout.pl):")
    print('  uv run python scripts/run_phase5_eval.py --rule "<comma-separated body predicates from the printed rule>"')
    return rc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_popper-worker":
        raise SystemExit(run_popper_worker())
    from run_logging import run_logged
    raise SystemExit(run_logged(5, main))
