"""
Tasks 3.2 / 3.3 — Dump the rule-based feature extractor's output as **ground
Prolog facts**.

Architectural note (execution plan, phase 3 clarification)
----------------------------------------------------------
Every predicate produced here is a *pre-computed ground fact*, not a live Prolog
rule.  The geometry lives in Python (:mod:`tactical_ilp.features.geometry`) and
its result is frozen into the ``.pl`` file.  In phase 4 these facts sit next to
genuine Prolog rules (``line_breaking_pass/1``, ``supports/3``, ...) inside
``bk.pl``; the header of every generated file repeats this warning so the two
kinds never get confused.

Fact vocabulary for the active phase-5 search
---------------------------------------------
entities            sequence/1
sequence dimension  sequence_length/2
player dimension    has_role/2, plays_for/2
membership          involved_in/2, in_sequence/2, action_order/2, actor/2
action facts        action_type/2, action_result/2
geometry            at_third/2, at_lane/2, progressive/1
pressure            under_pressure/1, pressure_level/2
frame coverage      visible_player_count/2 (freeze-frame audit / quality gate)
pass shape          pass_direction/2, pass_length/2, pass_end_lane/2
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from tactical_ilp.config import (
    PRESSURE_RADIUS_M,
    PROLOG_DIR,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.data.players_reference import build_players_reference
from tactical_ilp.features.geometry import (
    is_progressive,
    lane_of,
    pass_direction,
    pass_length_class,
    pressure_level,
    third_of,
    under_pressure,
)
from tactical_ilp.sequences.negative_sampling import build_context
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

FACTS_ALL_PATH = PROLOG_DIR / "facts_all.pl"
MANIFEST_PATH = PROLOG_DIR / "facts_manifest.json"
PHASE3_FEATURE_VERSION = "active-popper-vocabulary-v7-prefix-only-final-action-facts"

PASS_LIKE = {
    "pass", "cross", "freekick_short", "freekick_crossed", "goalkick",
    "throw_in", "corner_short", "corner_crossed",
}

HEADER = """\
%% ===================================================================
%%  AUTO-GENERATED - DO NOT EDIT BY HAND
%%  Symbolic tactical analysis / phase 3 feature extractor
%%  Source : StatsBomb Open Data, FC Barcelona, La Liga 2020/21
%%           (competition_id=11, season_id=90)
%%
%%  !! ALL predicates in this file are PRE-COMPUTED GROUND FACTS.
%%  !! They are produced by Python geometry functions
%%  !! (tactical_ilp.features.geometry) and frozen here.
%%  !! They are NOT live Prolog rules and must not be confused with the
%%  !! tactical rules that phase 4 adds to bk.pl.
%%
%%  Reference frame: Barcelona always attacks towards +x.
%%    x in [0,105]  0 = own goal line, 105 = opponent goal line
%%    y in [0, 68]  0 = right touchline (attacking direction)
%%  Pressure radius: {radius} m
%% ===================================================================

"""


def _atom(value: str) -> str:
    """Make a safe lowercase Prolog atom."""
    s = re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "x_" + s
    return s


def _action_atom(match_id: int, action_id: int) -> str:
    return f"a_{int(match_id)}_{int(action_id):05d}"


def _player_atom(player_id) -> str:
    return f"p_{int(player_id)}"


class FactWriter:
    """Collects facts grouped by predicate so the output file needs no
    :- discontiguous declarations and stays human-readable."""

    def __init__(self) -> None:
        self.groups: dict[str, list[str]] = {}
        self.order: list[str] = []

    def add(self, predicate: str, *args) -> None:
        if predicate not in self.groups:
            self.groups[predicate] = []
            self.order.append(predicate)
        body = ",".join(str(a) for a in args)
        self.groups[predicate].append(f"{predicate}({body})." if args else f"{predicate}.")

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.groups.items()}

    def write(self, path: Path, title: str = "") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(HEADER.format(radius=PRESSURE_RADIUS_M))
            if title:
                fh.write(f"%% {title}\n\n")
            for pred in self.order:
                lines = self.groups[pred]
                arity = lines[0].count("(") and lines[0][:-1].split("(", 1)[1].count(",") + 1
                fh.write(f"%% ---- {pred}/{arity}  ({len(lines)} facts)\n")
                fh.write("\n".join(sorted(set(lines))))
                fh.write("\n\n")
        return path


# --------------------------------------------------------------------------
def generate_facts(match_ids: list[int] | None = None, out_path: Path | None = None):
    """Generate the fact file for the given matches (all matches by default)."""
    context = build_context()
    _, members = build_sequences()
    merged = build_merged_all()
    players = build_players_reference()

    if match_ids is not None:
        keep = set(int(m) for m in match_ids)
        context = context[context.match_id.isin(keep)]
        members = members[members.match_id.isin(keep)]
        merged = merged[merged.match_id.isin(keep)]

    seq_ids = set(context.sequence_id)
    members = members[members.sequence_id.isin(seq_ids)]

    acts = merged.set_index(["match_id", "action_id"])
    act_cols = [
        "player_id",
        "type_name",
        "result_name",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "under_pressure",
        "ff_opponents_3m",
        "ff_opponents_5m",
        "ff_n_visible",
    ]
    mem = members.join(acts[act_cols], on=["match_id", "action_id"])

    w = FactWriter()

    # ---------------- player dimension ------------------------------------
    # Names and standalone identity markers were never queried by active BK
    # rules. Keep role and team facts: goalkeeper/Barca rules depend on them.
    involved_players = set(int(p) for p in mem.player_id.dropna().unique())
    for r in players.itertuples():
        pid = int(r.player_id)
        if pid not in involved_players:
            continue
        pa = _player_atom(pid)
        w.add("has_role", pa, _atom(r.role))
        w.add("plays_for", pa, "barcelona")

    known = set(int(p) for p in players.player_id)
    for pid in sorted(involved_players - known):     # should be empty, kept for safety
        pa = _player_atom(pid)
        w.add("has_role", pa, "unknown")
        w.add("plays_for", pa, "barcelona")

    # ---------------- sequences -------------------------------------------
    for r in context.itertuples():
        sid = r.sequence_id
        w.add("sequence", sid)
        w.add("sequence_length", sid, int(r.n_actions))

    # Match-level possession share, nominal formation, role shifts, and
    # false-nine candidates were only consumed by descriptive BK concepts,
    # none of which is in the active phase-5 vocabularies. Their scans and
    # facts are retired; player roles below remain because active keeper rules
    # depend on has_role/2.

    # ---------------- membership ------------------------------------------
    # Player membership is prefix-only: the terminal actor must not create
    # a sequence-level feature because that action determines the label.
    prefix_mem = mem[mem.order_in_sequence < mem.groupby("sequence_id").order_in_sequence.transform("max")]
    inv = (
        prefix_mem.dropna(subset=["player_id"])[["player_id", "sequence_id"]]
        .drop_duplicates()
    )
    for r in inv.itertuples():
        w.add("involved_in", _player_atom(r.player_id), r.sequence_id)

    # ---------------- actions + geometry ----------------------------------
    last_order = mem.groupby("sequence_id")["order_in_sequence"].max().astype(int).to_dict()
    for r in mem.itertuples():
        aa = _action_atom(r.match_id, r.action_id)
        w.add("in_sequence", aa, r.sequence_id)
        w.add("action_order", aa, int(r.order_in_sequence))
        if int(r.order_in_sequence) < last_order[r.sequence_id]:
            w.add("in_prefix", aa, r.sequence_id)
        if int(r.order_in_sequence) == last_order[r.sequence_id]:
            # terminal action: the one that decides the outcome label. Body
            # predicates are evaluated on the PREFIX only (see phase5_vocabulary).
            w.add("final_action", aa)
        elif int(r.order_in_sequence) == last_order[r.sequence_id] - 1:
            # last retained action: look-ahead predicates cannot be evaluated on it
            w.add("last_retained_action", aa)
        if not pd.isna(r.player_id):
            w.add("actor", aa, _player_atom(r.player_id))
        w.add("action_type", aa, _atom(r.type_name))
        w.add("action_result", aa, _atom(r.result_name))
        # successful/1 and failed/1 are derived in bk.pl from this canonical
        # result fact; duplicating both as ground facts added one fact/action.
        # Keep the freeze-frame coverage count: it records whether the frame
        # meets the minimum-observation threshold used by the majority concept.
        if pd.notna(r.ff_n_visible):
            w.add("visible_player_count", aa, int(r.ff_n_visible))

        w.add("at_third", aa, _atom(third_of(r.start_x)))
        w.add("at_lane", aa, _atom(lane_of(r.start_y)))
        if is_progressive(r.start_x, r.start_y, r.end_x, r.end_y):
            w.add("progressive", aa)

        if under_pressure(r.ff_opponents_5m, bool(r.under_pressure)):
            w.add("under_pressure", aa)
        w.add(
            "pressure_level",
            aa,
            _atom(pressure_level(r.ff_opponents_3m, r.ff_opponents_5m, bool(r.under_pressure))),
        )

        if r.type_name in PASS_LIKE and all(
            pd.notna(value)
            for value in (r.start_x, r.start_y, r.end_x, r.end_y)
        ):
            w.add("pass_direction", aa, _atom(pass_direction(r.start_x, r.start_y, r.end_x, r.end_y)))
            w.add("pass_length", aa, _atom(pass_length_class(r.start_x, r.start_y, r.end_x, r.end_y)))
        if r.type_name in PASS_LIKE and pd.notna(r.end_y):
            w.add("pass_end_lane", aa, _atom(lane_of(r.end_y)))

    # Ordered pass-link and three-player chain facts are disabled for the
    # current phase-5 vocabulary. Their only consumers were give_and_go/3 and
    # third_man/4, neither of which is offered to Popper.

    path = out_path or FACTS_ALL_PATH
    title = (
        f"{len(context)} sequences, {len(mem)} actions, "
        f"{context.match_id.nunique()} matches"
    )
    w.write(path, title=title)
    return path, w.counts(), context


# --------------------------------------------------------------------------
# helpers used by the tests
# --------------------------------------------------------------------------
def count_predicate_occurrences(facts_file: str | Path, predicate: str) -> int:
    """Number of ground facts of the given predicate in a .pl file."""
    pattern = re.compile(rf"^{re.escape(predicate)}\s*\(", re.MULTILINE)
    text = Path(facts_file).read_text(encoding="utf-8")
    text = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))
    return len(pattern.findall(text))


def count_distinct_arg(facts_file: str | Path, predicate: str, position: int) -> int:
    """Distinct values appearing at ``position`` (0-based) of a predicate."""
    pattern = re.compile(rf"^{re.escape(predicate)}\s*\(([^)]*)\)\.", re.MULTILINE)
    text = Path(facts_file).read_text(encoding="utf-8")
    text = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))
    seen = set()
    for m in pattern.finditer(text):
        args = [a.strip() for a in m.group(1).split(",")]
        if position < len(args):
            seen.add(args[position])
    return len(seen)


def predicate_histogram(facts_file: str | Path) -> dict[str, int]:
    text = Path(facts_file).read_text(encoding="utf-8")
    hist: Counter[str] = Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*\(", line)
        if m:
            hist[m.group(1)] += 1
    return dict(hist)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from tactical_ilp.data.load_raw_data import barca_match_ids

    ids = barca_match_ids()

    # Task 3.2 — single sample match
    sample = [ids[0]]
    p1, c1, ctx1 = generate_facts(sample, PROLOG_DIR / f"facts_match_{ids[0]}.pl")
    print(f"[3.2] {p1.name}: {sum(c1.values())} facts, {len(ctx1)} sequences")

    # Task 3.3 — whole dataset
    p2, c2, ctx2 = generate_facts(None, FACTS_ALL_PATH)
    print(f"[3.3] {p2.name}: {sum(c2.values())} facts, {len(ctx2)} sequences")
    print()
    for k, v in sorted(c2.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>7}")

    manifest = {
        "phase3_feature_version": PHASE3_FEATURE_VERSION,
        "sample_match_id": int(ids[0]),
        "sample_file": str(p1.name),
        "all_file": str(p2.name),
        "n_sequences": int(len(ctx2)),
        "n_matches": int(ctx2.match_id.nunique()),
        "predicate_counts": c2,
        "total_facts": int(sum(c2.values())),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST_PATH}")
