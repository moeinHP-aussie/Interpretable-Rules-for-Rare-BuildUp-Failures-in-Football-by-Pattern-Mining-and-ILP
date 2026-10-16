"""
Phase 4 — dump the tactical concepts as ground Prolog facts.

This sits next to :mod:`tactical_ilp.prolog.prolog_facts` (phase 3) and follows
exactly the same contract: **everything written here is a pre-computed ground
fact**, never a live rule.  The rules live in ``bk.pl``.

Predicates produced
-------------------
concept 3.6a  forms_triangle/1
concept 3.6b  traps_opponent/1, caged_opponent/1
concept 1.1   breaks_defensive_line/1
concept 1.5   beyond_defensive_line/1
concept 6.1   hesitation/1
concept 6.2   tempo/2
concept 6.3   one_touch/1
concept 6.4   dwell_time/2  (discretised, not raw seconds)
concept 3.4   frequently_involved/2
concept 1.2   diagonal_pass_shape/1
concept 1.3   switch_shape/1
concept 4.1/4.3  has_open_support/1, no_open_support/1

Concept 0.1 (line clustering) is used by 1.1/1.5 but is deliberately **not**
dumped -- it is an internal helper of this stage, see the architectural note in
``features/geometry.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from tactical_ilp.config import (
    HALF_SPACE_MIN_TOUCHES,
    MIN_VISIBLE_PLAYERS_FOR_MAJORITY,
    MIN_VALID_FRAMES_FOR_SEQUENCE_MAJORITY,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PROLOG_DIR,
    TEMPO_FAST_PERCENTILE,
    TEMPO_SLOW_MAX,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.data.players_reference import build_players_reference
from tactical_ilp.features.frames import actor_point, frame_points, opponents
from tactical_ilp.features.geometry import (
    defensive_line_x,
    is_diagonal_pass,
    is_long_pass,
    is_switch_of_play,
    switch_direction,
    flank_dribble_features,
    action_movement_features,
    lane_of,
    midfield_line_x,
    pressure_level,
)
from tactical_ilp.features.tactical import (
    frequently_involved,
    is_hesitation,
    is_one_touch,
    line_summary,
    open_supporters,
    rest_defense,
    tempo_bucket,
    triangulation_summary,
)
from tactical_ilp.sequences.negative_sampling import build_context
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

PASS_LIKE = {
    "pass",
    "cross",
    "freekick_short",
    "freekick_crossed",
    "goalkick",
    "throw_in",
    "corner_short",
    "corner_crossed",
}

# Restart-origin sequences are removed in phase 2, except short/medium goal
# kicks retained as keeper build-up. Those retained starts are valid tempo
# examples; other restart actions remain excluded from the tempo calculation.
RESTART_TYPES = frozenset({
    "freekick_short", "freekick_crossed",
    "throw_in", "corner_short", "corner_crossed",
})

PHASE4_FACTS_PATH = PROLOG_DIR / "facts_tactical.pl"
PHASE4_MANIFEST_PATH = PROLOG_DIR / "facts_tactical_manifest.json"
PHASE4_FEATURE_VERSION = "phase4-v11-prefix-only-tempo-pass-exchange"

HEADER = """\
%% ===================================================================
%%  AUTO-GENERATED - DO NOT EDIT BY HAND
%%  Phase 4 - tactical concept facts
%%
%%  !! PRE-COMPUTED GROUND FACTS, not Prolog rules.
%%  !! Produced by tactical_ilp.features.tactical/geometry from SPADL
%%  !! coordinates and StatsBomb 360 freeze frames. Rules live in bk.pl.
%%
%%  Data limit that shapes every predicate here: non-actor players in a
%%  360 frame have NO player_id - only (x, y, teammate, keeper).  So all
%%  opponent-related concepts are positional proxies, never identities.
%%
%%  Reference frame: Barcelona always attacks towards +x.
%% ===================================================================

"""


def _atom(value: str) -> str:
    """Convert a feature label to a safe lowercase Prolog atom."""
    s = re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "x_" + s
    return s


def _action_atom(match_id: int, action_id: int) -> str:
    return f"a_{int(match_id)}_{int(action_id):05d}"


def _player_atom(player_id) -> str:
    return f"p_{int(player_id)}"


def _dwell_bucket(seconds: float | None) -> str | None:
    """Concept 6.4 -- time on the ball, discretised."""
    if seconds is None:
        return None
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return None
    if s != s or s < 0:
        return None
    if s <= 1.0:
        return "instant"
    if s <= 2.5:
        return "normal"
    return "long"


class FactWriter:
    """Same grouping writer as phase 3, kept local to avoid a circular import."""

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
            fh.write(HEADER)
            if title:
                fh.write(f"%% {title}\n\n")
            for pred in self.order:
                lines = sorted(set(self.groups[pred]))
                arity = lines[0].count("(") and lines[0][:-1].split("(", 1)[1].count(",") + 1
                fh.write(f"%% ---- {pred}/{arity}  ({len(lines)} facts)\n")
                fh.write("\n".join(lines))
                fh.write("\n\n")
        return path


def generate_tactical_facts(
    match_ids: list[int] | None = None, out_path: Path | None = None
):
    """Build the phase-4 fact file for the given matches (all by default)."""
    context = build_context()
    _, members = build_sequences()
    merged = build_merged_all()
    players = build_players_reference()

    if match_ids is not None:
        keep = {int(m) for m in match_ids}
        context = context[context.match_id.isin(keep)]
        members = members[members.match_id.isin(keep)]
        merged = merged[merged.match_id.isin(keep)]

    seq_ids = set(context.sequence_id)
    members = members[members.sequence_id.isin(seq_ids)]

    cols = [
        "player_id",
        "type_name",
        "result_name",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_height_name",
        "pass_recipient_id",
        "time_seconds",
        "time_on_ball_s",
        "original_event_id",
        "is_barca",
        "under_pressure",
        "counterpress",
        "ff_opponents_3m",
        "ff_opponents_5m",
        "has_freeze_frame",
    ]
    acts = merged.set_index(["match_id", "action_id"])
    mem = members.join(acts[cols], on=["match_id", "action_id"])
    player_roles = dict(zip(players.player_id.astype(int), players.role))

    # Empirical action-risk buckets and xG buckets were retired because no
    # active phase-5 body predicate consumes them.
    mem = mem.sort_values(["sequence_id", "order_in_sequence"])

    # Use the holding interval reconstructed from Ball Receipt -> release.
    # The gap between SPADL actions can include ball travel and event timing;
    # it is not a direct measure of how long the player held the ball.
    mem["dwell_s"] = pd.to_numeric(mem["time_on_ball_s"], errors="coerce")

    w = FactWriter()
    w.add("minimum_visible_majority_frames", MIN_VALID_FRAMES_FOR_SEQUENCE_MAJORITY)

    # ---------------- per-action concepts ---------------------------------
    for r in mem.itertuples():
        aa = _action_atom(r.match_id, r.action_id)
        if (
            r.type_name in PASS_LIKE
            and pd.notna(r.end_x)
            and float(r.end_x) > 2.0 * PITCH_LENGTH / 3.0
        ):
            w.add("final_third_destination", aa)
        if (
            r.type_name in PASS_LIKE
            and pd.notna(r.end_x)
            and pd.notna(r.end_y)
            and float(r.end_x) >= PITCH_LENGTH - PENALTY_AREA_DEPTH_M
            and (PITCH_WIDTH - PENALTY_AREA_WIDTH_M) / 2.0
            <= float(r.end_y)
            <= (PITCH_WIDTH + PENALTY_AREA_WIDTH_M) / 2.0
        ):
            w.add("penalty_area_destination", aa)
        # successful_forward_pass_10m was retired: neither Popper nor the
        # sequential miner consumes it in the current phase-5 vocabulary.
        if (
            r.type_name == "pass"
            and not pd.isna(r.player_id)
            and player_roles.get(int(r.player_id)) == "goalkeeper"
        ):
            w.add("keeper_open_play_pass", aa)
        pressure = pressure_level(
            r.ff_opponents_3m, r.ff_opponents_5m, bool(r.under_pressure)
        )

        # --- concept 6: rhythm / timing / decision making -----------------
        if is_hesitation(r.dwell_s, pressure):
            w.add("hesitation", aa)
        if is_one_touch(r.dwell_s):
            w.add("one_touch", aa)
        bucket = _dwell_bucket(r.dwell_s)
        if bucket:
            w.add("dwell_time", aa, bucket)

        # --- concept 1.2 / 1.3: two-axis pass shapes ----------------------
        if r.type_name in PASS_LIKE:
            if is_diagonal_pass(r.start_x, r.start_y, r.end_x, r.end_y):
                w.add("diagonal_pass_shape", aa)
            if is_switch_of_play(r.start_x, r.start_y, r.end_x, r.end_y):
                destination = switch_direction(r.start_x, r.start_y, r.end_x, r.end_y)
                if destination == "right":
                    w.add("switch_shape_to_right", aa)
                elif destination == "left":
                    w.add("switch_shape_to_left", aa)
            if r.pass_height_name == "High Pass":
                w.add("high_pass_shape", aa)
            if (
                all(pd.notna(v) for v in (r.start_x, r.start_y, r.end_x, r.end_y))
                and is_long_pass(r.start_x, r.start_y, r.end_x, r.end_y)
            ):
                w.add("long_pass_shape", aa)

        # Flank dribble/carry concepts use the project's fixed five-lane
        # coordinate system. These are ground facts over the observed SPADL
        # start/end locations, independent of 360-frame availability.
        for concept in flank_dribble_features(
            r.type_name, r.start_x, r.start_y, r.end_x, r.end_y
        ):
            w.add(concept, aa)
        for concept in action_movement_features(
            r.type_name, r.start_x, r.start_y, r.end_x, r.end_y
        ):
            w.add(concept, aa)

        # concept 8.3 — counter-press: StatsBomb's own flag, just dumped as a
        # fact.  Already computed upstream, never exposed before phase 4.
        if bool(getattr(r, "counterpress", False)):
            w.add("counterpress_action", aa)

        # --- frame-based concepts ----------------------------------------
        if not bool(r.has_freeze_frame) or not isinstance(r.original_event_id, str):
            continue
        points = frame_points(r.match_id, r.original_event_id, bool(r.is_barca))
        if not points:
            continue
        if (
            len(points) >= MIN_VISIBLE_PLAYERS_FOR_MAJORITY
            and actor_point(points) is not None
        ):
            w.add("opponent_visible_majority_eligible_action", aa)
            own_visible = sum(point.teammate for point in points)
            opposition_visible = len(points) - own_visible
            if opposition_visible > own_visible:
                w.add("opponent_visible_majority_action", aa)

        # concept 4.1 / 4.5 — support: open passing lanes, not distance.
        # A missing actor point is not evidence that the carrier has no
        # passing options.  Record the closed-world complement only when the
        # 360 frame actually identifies the carrier and the visible teammates.
        if actor_point(points) is not None:
            n_open = open_supporters(points)
            if n_open >= 1:
                w.add("has_open_support", aa)
            else:
                w.add("no_open_support", aa)
            if rest_defense(points) == "exposed":
                w.add("rest_defence_exposed", aa)
        # Passing-option counts and structural shape/cover/spacing summaries
        # are not in any active Popper vocabulary. The open-lane booleans above
        # are the support signals actually used by the current models.

        # concept 3.6a / 3.6b — triangulation
        tri = triangulation_summary(points)
        if tri["n_triangles"] > 0:
            w.add("forms_triangle", aa)
            if tri["n_trapping"] > 0:
                w.add("traps_opponent", aa)
            if tri["multi_trap"]:
                w.add("caged_opponent", aa)

        # concepts 1.1 / 1.5 — opponent lines (built on the 0.1 helper)
        ln = line_summary(points, r.start_x, r.end_x)
        if ln["has_line"]:
            if ln["breaks_defensive_line"]:
                w.add("breaks_defensive_line", aa)
            if ln["beyond_defensive_line"]:
                w.add("beyond_defensive_line", aa)

        # concept 2.3 — drop between the (opponent) lines: the actor himself
        # is positioned between the opponent defensive and midfield lines.
        # Also built on the 0.1 helper; only possible for the acting player
        # because only the actor carries a player_id in the frame.
        if bool(r.is_barca):
            opps_xs = [o.x for o in opponents(points, include_keeper=False)]
            carrier = actor_point(points)
            dline = defensive_line_x(opps_xs)
            mline = midfield_line_x(opps_xs)
            if carrier is not None and dline is not None and mline is not None:
                if mline < carrier.x < dline:
                    w.add("drop_between_lines", aa)

    # ---------------- per-sequence concepts -------------------------------
    # PREFIX-ONLY: every sequence ends with the action that decides its outcome
    # (a failed action for the vast majority of negatives). Sequence-level
    # descriptors are therefore computed on the actions BEFORE the final one, so
    # they cannot encode the terminal event that the label is derived from.
    grp = mem.groupby("sequence_id")
    valid_rates = []
    for _, sequence in grp:
        sequence = sequence.sort_values("order_in_sequence")
        if str(sequence.iloc[0].type_name) in RESTART_TYPES:
            continue
        sequence = sequence.iloc[:-1]
        if len(sequence) < 2:
            continue
        t0, t1 = sequence.time_seconds.min(), sequence.time_seconds.max()
        duration = (t1 - t0) if pd.notna(t1) and pd.notna(t0) else None
        if duration is not None and duration > 0:
            valid_rates.append(len(sequence) / float(duration))
    fast_min = (
        max(TEMPO_SLOW_MAX, float(pd.Series(valid_rates).quantile(TEMPO_FAST_PERCENTILE / 100)))
        if valid_rates else None
    )
    for sid, g in grp:
        ordered = g.sort_values("order_in_sequence")
        if str(ordered.iloc[0].type_name) in RESTART_TYPES:
            continue
        prefix = ordered.iloc[:-1]
        t0, t1 = prefix.time_seconds.min(), prefix.time_seconds.max()
        duration = (t1 - t0) if pd.notna(t1) and pd.notna(t0) else None
        bucket = tempo_bucket(len(prefix), duration, fast_min=fast_min) if len(prefix) >= 2 else None
        if bucket:
            w.add("tempo", sid, bucket)

        ordered_rows = list(prefix.itertuples())
        pass_rows = []
        for row in ordered_rows:
            actor_id = getattr(row, "player_id", None)
            recipient_id = getattr(row, "pass_recipient_id", None)
            if (
                row.type_name in PASS_LIKE
                and row.result_name == "success"
                and pd.notna(actor_id)
                and pd.notna(recipient_id)
            ):
                pass_rows.append((int(actor_id), int(recipient_id)))
        if any(
            first_actor != first_recipient
            and first_actor == second_recipient
            and first_recipient == second_actor
            for i, (first_actor, first_recipient) in enumerate(pass_rows)
            for second_actor, second_recipient in pass_rows[i + 1:]
        ):
            # Only the Boolean sequence feature is persisted. Player ids are
            # transient inputs to this check and are never written to Prolog.
            w.add("pass_exchange_sequence", sid)

        # pass_received_by_keeper was retired from keeper_played_open_play;
        # no active phase-5 vocabulary consumes this adjacency proxy.

        # Positional-rotation, sustained-press, outcome-transition,
        # immediate-exploitation, and rapid-combination summaries are retired
        # from current search vocabularies. In particular, outcome-derived
        # summaries must not be mixed into a negative-outcome predictor.

        # Player-level sequence summaries use only the observable prefix.
        prefix = ordered.iloc[:-1]

        # concept 3.4 — repeated involvement, boolean not raw count
        touches = prefix.dropna(subset=["player_id"]).groupby("player_id").size()
        for pid, n in touches.items():
            if frequently_involved(n):
                w.add("frequently_involved", _player_atom(pid), sid)

        # concept 5.1 — persistent half-space occupation.
        hs = prefix.dropna(subset=["player_id"]).copy()
        hs["lane"] = hs["start_y"].apply(lane_of)
        hs["is_half_space"] = hs["lane"].isin(["left_half_space", "right_half_space"])
        hs_counts = hs.groupby("player_id")["is_half_space"].sum()
        for pid, n in hs_counts.items():
            if n >= HALF_SPACE_MIN_TOUCHES:
                w.add("half_space_occupier", _player_atom(pid), sid)

    path = out_path or PHASE4_FACTS_PATH
    title = (
        f"{len(context)} sequences, {len(mem)} actions, "
        f"{context.match_id.nunique()} matches"
    )
    w.write(path, title=title)
    return path, w.counts(), context, fast_min


# --------------------------------------------------------------------------
if __name__ == "__main__":
    path, counts, ctx, fast_min = generate_tactical_facts()
    total = sum(counts.values())
    print(f"{path.name}: {total} facts, {len(ctx)} sequences")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v:>7}")
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
