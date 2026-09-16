"""
Task 2.2 — Operational definition of a **negative outcome**, and labelling of
every build-up sequence with it.

The prose version of the definition lives in
``docs/negative_outcome_definition.md``; the constants in
:mod:`tactical_ilp.config` are the machine-readable version.  Both must stay in
sync — the test ``test_negative_outcome_definition_documented`` checks that the
document really contains explicit numeric/spatial criteria and that they match
the code.

Definition (all three must hold)
--------------------------------
A1. **Turnover.** Barcelona loses the ball during the sequence, or within
    ``NEG_TURNOVER_WINDOW_S`` = 8 s after the sequence's last action.
A2. **Location.** The turnover happens at ``x <= 52.5`` — inside Barcelona's own
    half, in the project reference frame.
A3. **Consequence** (at least one of):
      C1  the opponent takes a shot within ``NEG_OPP_SHOT_WINDOW_S`` = 12 s of
          the turnover;
      C2  the opponent moves the ball into Barcelona's defensive third
          (``x <= 35``) within ``NEG_OPP_ENTRY_WINDOW_S`` = 12 s of the turnover.

Everything else is labelled ``non_negative``.  Requiring a consequence (A3) is
what implements technical document §2.7 — "target on the outcome, not on mere
repetition" — and prevents healthy, deliberate patterns from being mined as
dangerous ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    NEG_DANGER_ZONE_MAX_X,
    NEG_OPP_ENTRY_WINDOW_S,
    NEG_OPP_SHOT_WINDOW_S,
    NEG_TURNOVER_WINDOW_S,
    PROCESSED_DIR,
    BUILD_UP_EXIT_X,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

LABELED_PATH = PROCESSED_DIR / "labeled_sequences.parquet"

SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}

# machine-readable copy of the definition, used by the documentation test
DEFINITION = {
    "turnover_window_s": NEG_TURNOVER_WINDOW_S,
    "danger_zone_max_x": NEG_DANGER_ZONE_MAX_X,
    "opponent_shot_window_s": NEG_OPP_SHOT_WINDOW_S,
    "opponent_entry_window_s": NEG_OPP_ENTRY_WINDOW_S,
    "opponent_entry_x": BUILD_UP_EXIT_X,
}


def _label_one(seq: pd.Series, match_actions: pd.DataFrame) -> dict:
    """Look forward in the action stream and decide the outcome of one sequence."""
    period = seq["period_id"]
    t_end = seq["end_time"]
    after = match_actions[
        (match_actions["period_id"] == period)
        & (match_actions["row"] > seq["last_action_row"])
    ]

    # ---------- A1 + A2 : find the turnover -------------------------------
    lost = False
    turnover_x = np.nan
    turnover_t = np.nan

    window = after[after["time_seconds"] <= t_end + NEG_TURNOVER_WINDOW_S]
    opp = window[~window["is_barca"]]
    if len(opp):
        first_opp = opp.iloc[0]
        lost = True
        turnover_x = float(first_opp["start_x"])
        turnover_t = float(first_opp["time_seconds"])
    elif seq["end_reason"] in ("failed_action", "possession_end") and len(window) == 0:
        # ball went out of play / half ended right after the failed action
        lost = seq["end_reason"] == "failed_action"
        turnover_x = float(seq["end_x"])
        turnover_t = float(t_end)

    in_own_half = bool(lost and turnover_x <= NEG_DANGER_ZONE_MAX_X)

    # ---------- A3 : consequence ------------------------------------------
    opp_shot = False
    opp_entry = False
    if lost and not np.isnan(turnover_t):
        follow = after[
            after["time_seconds"]
            <= turnover_t + max(NEG_OPP_SHOT_WINDOW_S, NEG_OPP_ENTRY_WINDOW_S)
        ]
        follow_opp = follow[~follow["is_barca"]]
        if len(follow_opp):
            shots = follow_opp[
                follow_opp["type_name"].isin(SHOT_TYPES)
                & (follow_opp["time_seconds"] <= turnover_t + NEG_OPP_SHOT_WINDOW_S)
            ]
            opp_shot = len(shots) > 0
            entries = follow_opp[
                (follow_opp["time_seconds"] <= turnover_t + NEG_OPP_ENTRY_WINDOW_S)
                & (
                    (follow_opp["end_x"] <= BUILD_UP_EXIT_X)
                    | (follow_opp["start_x"] <= BUILD_UP_EXIT_X)
                )
            ]
            opp_entry = len(entries) > 0

    is_negative = bool(in_own_half and (opp_shot or opp_entry))

    return {
        "sequence_id": seq["sequence_id"],
        "lost_possession": lost,
        "turnover_x": turnover_x,
        "turnover_time": turnover_t,
        "turnover_in_own_half": in_own_half,
        "opponent_shot_after": opp_shot,
        "opponent_entered_def_third": opp_entry,
        "label": "negative" if is_negative else "non_negative",
    }


def label_sequences(force: bool = False) -> pd.DataFrame:
    if LABELED_PATH.exists() and not force:
        return pd.read_parquet(LABELED_PATH)

    sequences, _ = build_sequences()
    merged = build_merged_all()
    merged = merged.sort_values(["match_id", "period_id", "time_seconds"])
    merged["row"] = merged.groupby("match_id").cumcount()

    by_match = {mid: grp for mid, grp in merged.groupby("match_id")}

    rows = [
        _label_one(seq, by_match[seq["match_id"]])
        for _, seq in sequences.iterrows()
    ]
    labels = pd.DataFrame(rows)
    out = sequences.merge(labels, on="sequence_id", how="left")

    # zone of the sequence start (used by the negative-sampling comparability test)
    from tactical_ilp.features.geometry import lane_of, third_of

    out["zone"] = [
        f"{third_of(x)}|{lane_of(y)}" for x, y in zip(out.start_x, out.start_y)
    ]
    out.to_parquet(LABELED_PATH, index=False)
    return out


if __name__ == "__main__":
    df = label_sequences(force=True)
    print(f"sequences: {len(df)}")
    print()
    print("label balance:")
    print(df.label.value_counts().to_string())
    print(df.label.value_counts(normalize=True).round(4).to_string())
    print()
    print("diagnostics:")
    for col in [
        "lost_possession",
        "turnover_in_own_half",
        "opponent_shot_after",
        "opponent_entered_def_third",
    ]:
        print(f"  {col:<28} {df[col].mean():.1%}")
    print()
    print("label x end_reason:")
    print(pd.crosstab(df.end_reason, df.label).to_string())
