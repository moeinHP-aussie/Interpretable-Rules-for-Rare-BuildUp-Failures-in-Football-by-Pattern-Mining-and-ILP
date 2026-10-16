"""Describe terminal actions for approved rules over the complete labelled corpus."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.features.frames import frame_points
from tactical_ilp.features.geometry import pass_direction, pass_length_class
from tactical_ilp.features.tactical import line_summary

PASS_TYPES = {"pass", "cross", "freekick_short", "goalkick", "throw_in", "corner_short"}
ACTION_GROUPS = {
    "pass": "pass", "cross": "pass", "freekick_short": "pass",
    "goalkick": "pass", "throw_in": "pass", "corner_short": "pass",
    "dribble": "dribble", "take_on": "take_on", "bad_touch": "bad_touch",
}
ACTION_LABELS = {
    "pass": "پاس", "dribble": "دریبل", "take_on": "Take-on",
    "bad_touch": "لمس ناموفق", "other": "سایر اکشن‌ها", "unknown": "نامشخص",
}
RESULT_LABELS = {
    "failed_own_action": "خطای مستقیم اکشن خودی",
    "opponent_acquisition:tackle": "بازیابی حریف با تکل",
    "opponent_acquisition:interception": "بازیابی حریف با قطع پاس",
    "opponent_acquisition:other": "بازیابی حریف با اکشن دیگر",
    "no_qualifying_turnover": "بدون گردش مالکیتِ واجد شرایط",
    "inconclusive": "نامشخص / خارج از دامنهٔ برچسب‌گذاری",
    "unknown": "رخداد پایانی ناموجود",
}
DIRECTIONS = {"forward": "روبه‌جلو", "sideways": "عرضی", "backward": "روبه‌عقب", "unknown": "نامشخص"}
LENGTHS = {"short": "کوتاه", "medium": "متوسط", "long": "بلند", "unknown": "نامشخص"}
LINE_BREAK = {"yes": "بله (proxy هندسی)", "no": "خیر (proxy هندسی)", "unknown": "نامشخص"}


def _event_group(type_name: object) -> str:
    raw = "" if pd.isna(type_name) else str(type_name).strip().lower()
    return ACTION_GROUPS.get(raw, "other")


def _row_action(match_actions: pd.DataFrame, row_no: int) -> pd.Series | None:
    rows = match_actions[match_actions.row == row_no]
    return None if rows.empty else rows.iloc[0]


def _terminal_pair(match_actions: pd.DataFrame, label: pd.Series) -> tuple[pd.Series | None, str]:
    """Return the own action associated with the sequence end and its result class."""
    turnover_row = int(label.get("turnover_row", -1))
    failed_own = bool(label.get("final_action_failed", False))
    last_action_row = int(label.get("last_action_row", -1))

    if failed_own:
        event = _row_action(match_actions, turnover_row if turnover_row >= 0 else last_action_row)
        return event, "failed_own_action" if event is not None else "unknown"

    turnover = _row_action(match_actions, turnover_row) if turnover_row >= 0 else None
    if turnover is not None and not bool(turnover.get("is_barca", False)):
        event_type = str(turnover.get("type_name", "")).lower()
        result = (
            f"opponent_acquisition:{event_type}"
            if event_type in {"tackle", "interception"}
            else "opponent_acquisition:other"
        )
        period = int(label.get("period_id", -1))
        prior = match_actions[
            (match_actions.period_id == period)
            & (match_actions.row < turnover_row)
            & (match_actions.row >= last_action_row)
            & match_actions.is_barca.astype(bool)
        ]
        if prior.empty:
            prior = match_actions[
                (match_actions.period_id == period)
                & (match_actions.row < turnover_row)
                & match_actions.is_barca.astype(bool)
            ]
        own_action = prior.sort_values("row").iloc[-1] if not prior.empty else None
        return own_action, result if own_action is not None else "unknown"

    if str(label.get("label", "")) == "inconclusive":
        result = "inconclusive"
    else:
        result = "no_qualifying_turnover"
    return _row_action(match_actions, last_action_row), result


def _pass_attributes(action: pd.Series) -> tuple[str, str, str]:
    coords = [action.get(key) for key in ("start_x", "start_y", "end_x", "end_y")]
    if not all(pd.notna(value) for value in coords):
        return "unknown", "unknown", "unknown"
    sx, sy, ex, ey = map(float, coords)
    direction = pass_direction(sx, sy, ex, ey)
    length = pass_length_class(sx, sy, ex, ey)

    has_frame = bool(action.get("has_freeze_frame", False)) if pd.notna(action.get("has_freeze_frame", False)) else False
    event_id = action.get("original_event_id")
    attempt = "unknown"
    if has_frame and isinstance(event_id, str):
        points = frame_points(int(action.match_id), event_id, bool(action.get("is_barca", True)))
        if points:
            attempt = "yes" if line_summary(points, sx, ex).get("breaks_defensive_line") else "no"
    return direction, length, attempt


def build_global_terminal_action_summaries(
    labeled: pd.DataFrame,
    matched_by_rule: dict[str, set[str]],
) -> dict[str, dict]:
    """Aggregate matched sequences from the full labelled corpus, not CV folds."""
    actions = build_merged_all().sort_values(
        ["match_id", "period_id", "time_seconds", "action_id"]
    ).copy()
    actions["row"] = actions.groupby("match_id").cumcount()
    by_match = {int(match_id): frame for match_id, frame in actions.groupby("match_id")}
    labels = labeled.copy()
    labels["sequence_id"] = labels.sequence_id.astype(str)
    labels = labels.set_index("sequence_id", drop=False)

    summaries: dict[str, dict] = {}
    for rule_name, sequence_ids in matched_by_rule.items():
        own_counts = Counter({key: 0 for key in ACTION_LABELS})
        result_counts = Counter({key: 0 for key in RESULT_LABELS})
        direction_counts = Counter({key: 0 for key in DIRECTIONS})
        length_counts = Counter({key: 0 for key in LENGTHS})
        line_counts = Counter({key: 0 for key in LINE_BREAK})
        n_passes = 0

        for sequence_id in sorted(sequence_ids):
            if sequence_id not in labels.index:
                own_counts["unknown"] += 1
                result_counts["unknown"] += 1
                continue
            label = labels.loc[sequence_id]
            match_actions = by_match.get(int(label.match_id))
            if match_actions is None:
                own_counts["unknown"] += 1
                result_counts["unknown"] += 1
                continue
            action, result = _terminal_pair(match_actions, label)
            result_counts[result if result in result_counts else "unknown"] += 1
            if action is None:
                own_counts["unknown"] += 1
                continue

            group = _event_group(action.get("type_name"))
            own_counts[group] += 1
            if group == "pass":
                n_passes += 1
                direction, length, line_attempt = _pass_attributes(action)
                direction_counts[direction] += 1
                length_counts[length] += 1
                line_counts[line_attempt] += 1

        summaries[rule_name] = {
            "n_covered_sequences": len(sequence_ids),
            "own_action_counts": dict(own_counts),
            "terminal_result_counts": dict(result_counts),
            "pass_count": n_passes,
            "pass_direction_counts": dict(direction_counts),
            "pass_length_counts": dict(length_counts),
            "line_break_attempt_counts": dict(line_counts),
        }
    return summaries
