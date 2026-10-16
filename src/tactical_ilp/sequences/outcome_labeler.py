"""Label build-up sequences by unrecovered possession loss in the own half."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    DEFENSIVE_THIRD_MAX_X,
    MIDFIELD_REGAIN_WINDOW_S,
    OWN_HALF_MAX_X,
    BUILD_UP_ZONE_ENTRY_X,
    PROCESSED_DIR,
)
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

LABELED_PATH = PROCESSED_DIR / "labeled_sequences.parquet"
LABEL_CACHE_VERSION = "own-half-loss-no-regain-5s-no-opponent-foul-v6"
LABEL_CACHE_VERSION_PATH = PROCESSED_DIR / "label_cache_version.txt"

# Machine-readable definition consumed by docs_sync.
DEFINITION = {
    "own_half_max_x": OWN_HALF_MAX_X,
    "regain_window_s": MIDFIELD_REGAIN_WINDOW_S,
    "exclude_opponent_foul_stoppage": True,
}

FAILED_RESULTS = {"fail", "owngoal"}
SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}
TRANSITION_CONTEXT_WINDOW_S = 12.0
CONTROLLED_REGAIN_TYPES = {
    "pass",
    "cross",
    "dribble",
    "take_on",
    "carry",
    "goalkick",
    "freekick_short",
    "freekick_crossed",
    "throw_in",
    "corner_short",
    "corner_crossed",
    "keeper_pick_up",
    "keeper_claim",
}


def _is_success(action: pd.Series) -> bool:
    return str(action.get("result_name", "")) == "success"


def _is_controlled_regain(action: pd.Series) -> bool:
    """Whether a Barcelona action evidences retained, controlled possession."""
    return (
        _is_success(action)
        and str(action.get("type_name", "")) in CONTROLLED_REGAIN_TYPES
    )


def _after_sequence(seq: pd.Series, match_actions: pd.DataFrame) -> pd.DataFrame:
    return match_actions[
        (match_actions["period_id"] == seq["period_id"])
        & (match_actions["row"] > int(seq["last_action_row"]))
    ].sort_values("row")


def _regained_within_window(
    after: pd.DataFrame, turnover_row: int, turnover_time: float
) -> bool:
    """Whether the selected team establishes controlled possession within 5 s."""
    later = after[
        (after["row"] >= turnover_row)
        & (after["time_seconds"] <= turnover_time + MIDFIELD_REGAIN_WINDOW_S)
    ].sort_values("row")
    for _, action in later.iterrows():
        elapsed = float(action["time_seconds"]) - turnover_time
        if elapsed < 0:
            continue
        if bool(action["is_barca"]) and _is_controlled_regain(action):
            return True
    return False


def _label_one(seq: pd.Series, match_actions: pd.DataFrame) -> dict:
    after = _after_sequence(seq, match_actions)
    last_action = match_actions[
        match_actions["row"] == int(seq["last_action_row"])
    ].iloc[0]

    # A failed final Barcelona action is itself direct evidence of a loss.
    final_action_failed = (
        bool(last_action["is_barca"])
        and str(last_action.get("result_name", "")) in FAILED_RESULTS
    )
    build_up_completed = str(seq.get("end_reason", "")) in {
        "shot",
        "attacking_zone_entry",
    }

    # Otherwise possession must pass immediately to the opponent. An own-team
    # action after the segment proves that this segment did not lose the ball.
    first_opponent = None
    if len(after) and not bool(after.iloc[0]["is_barca"]):
        first_opponent = after.iloc[0]

    # A foul as the first opposing action is a stoppage, not reliable evidence
    # that Barcelona lost possession. It can also follow an apparent failed
    # action, so give the foul event precedence for this boundary decision.
    opponent_foul_stoppage = bool(
        first_opponent is not None
        and str(first_opponent.get("type_name", "")) == "foul"
    )

    lost_possession = (
        final_action_failed or first_opponent is not None
    ) and not opponent_foul_stoppage
    if opponent_foul_stoppage:
        turnover_x = np.nan
        turnover_time = np.nan
        turnover_row = -1
    elif final_action_failed:
        turnover_x = float(last_action["end_x"])
        turnover_time = float(last_action["time_seconds"])
        turnover_row = int(last_action["row"])
    elif first_opponent is not None:
        turnover_x = float(first_opponent["start_x"])
        turnover_time = float(first_opponent["time_seconds"])
        turnover_row = int(first_opponent["row"])
    else:
        turnover_x = np.nan
        turnover_time = np.nan
        turnover_row = -1

    defensive_third_loss = bool(
        lost_possession
        and not np.isnan(turnover_x)
        and turnover_x <= DEFENSIVE_THIRD_MAX_X
    )
    own_midfield_loss = bool(
        lost_possession
        and not np.isnan(turnover_x)
        and DEFENSIVE_THIRD_MAX_X < turnover_x <= OWN_HALF_MAX_X
    )

    # These observations remain available to tactical predicates and reports,
    # but do not participate in the negative-label decision.
    context = (
        after[
            (after["row"] > turnover_row)
            & (after["time_seconds"] <= turnover_time + TRANSITION_CONTEXT_WINDOW_S)
            & (~after["is_barca"])
        ]
        if lost_possession
        else after.iloc[0:0]
    )
    opponent_shot_after = bool(context["type_name"].isin(SHOT_TYPES).any())
    opponent_entered_def_third = bool(
        (
            (context["start_x"] <= DEFENSIVE_THIRD_MAX_X)
            | (context["end_x"] <= DEFENSIVE_THIRD_MAX_X)
        ).any()
    )

    regained_within_5s = bool(
        lost_possession
        and _regained_within_window(after, turnover_row, turnover_time)
    )
    advanced_opponent_middle_loss = bool(
        lost_possession
        and pd.notna(turnover_x)
        and OWN_HALF_MAX_X < turnover_x < BUILD_UP_ZONE_ENTRY_X
    )
    is_negative = bool(
        lost_possession
        and pd.notna(turnover_x)
        and turnover_x <= OWN_HALF_MAX_X
        and not regained_within_5s
    )

    if is_negative:
        reason = "own_half_loss_not_regained_within_5s"
    elif opponent_foul_stoppage:
        reason = "opponent_foul_stoppage"
    elif regained_within_5s:
        reason = "loss_regained_within_5s"
    elif lost_possession and pd.notna(turnover_x) and turnover_x > OWN_HALF_MAX_X:
        reason = "loss_outside_own_half"
    elif lost_possession:
        reason = "loss_location_unknown"
    else:
        reason = "no_possession_loss"

    return {
        "sequence_id": seq["sequence_id"],
        "lost_possession": bool(lost_possession),
        "turnover_x": turnover_x,
        "turnover_time": turnover_time,
        "turnover_row": turnover_row,
        "turnover_zone": (
            "defensive_third" if defensive_third_loss else
            "own_half_midfield" if own_midfield_loss else
            "advanced_opponent_middle" if advanced_opponent_middle_loss else
            "outside_target_zones" if lost_possession else "none"
        ),
        "final_action_failed": bool(final_action_failed),
        "build_up_completed": bool(build_up_completed),
        "opponent_shot_after": opponent_shot_after,
        "opponent_entered_def_third": opponent_entered_def_third,
        "regained_within_5s": bool(regained_within_5s),
        "negative_reason": reason,
        "label": "negative" if is_negative else "non_negative",
    }


def label_sequences(force: bool = False) -> pd.DataFrame:
    cache_current = (
        LABEL_CACHE_VERSION_PATH.exists()
        and LABEL_CACHE_VERSION_PATH.read_text(encoding="utf-8").strip()
        == LABEL_CACHE_VERSION
    )
    if LABELED_PATH.exists() and not force and cache_current:
        return pd.read_parquet(LABELED_PATH)

    sequences, _ = build_sequences()
    merged = build_merged_all()
    merged = merged.sort_values(
        ["match_id", "period_id", "time_seconds", "action_id"]
    )
    merged["row"] = merged.groupby("match_id").cumcount()
    by_match = {mid: grp for mid, grp in merged.groupby("match_id")}

    labels = pd.DataFrame(
        [
            _label_one(seq, by_match[seq["match_id"]])
            for _, seq in sequences.iterrows()
        ]
    )
    out = sequences.merge(labels, on="sequence_id", how="left")

    from tactical_ilp.features.geometry import lane_of, third_of

    out["zone"] = [
        f"{third_of(x)}|{lane_of(y)}" for x, y in zip(out.start_x, out.start_y)
    ]
    out.to_parquet(LABELED_PATH, index=False)
    LABEL_CACHE_VERSION_PATH.write_text(LABEL_CACHE_VERSION + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    df = label_sequences(force=True)
    print(f"sequences: {len(df)}")
    print("\nlabel balance:")
    print(df.label.value_counts().to_string())
    print(df.label.value_counts(normalize=True).round(4).to_string())
    print("\nturnover zones:")
    print(pd.crosstab(df.turnover_zone, df.label).to_string())
    print("\nnegative reasons:")
    print(df.loc[df.label == "negative", "negative_reason"].value_counts().to_string())
    print("\nlabel x end_reason:")
    print(pd.crosstab(df.end_reason, df.label).to_string())
