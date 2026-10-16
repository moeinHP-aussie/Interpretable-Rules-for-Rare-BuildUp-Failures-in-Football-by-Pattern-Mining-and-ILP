"""
Task 1.2 — Convert raw StatsBomb events into the standard SPADL action table.

One row = one on-the-ball action with a fixed schema
(player_id, team_id, type_name, start_x, start_y, end_x, end_y, result_name, ...).

Direction of play
-----------------
``convert_to_actions(events, home_team_id)`` already calls
``_fix_direction_of_play``, so by passing the selected team's id we obtain a
single, stable reference frame for the whole project:

    * the selected team always attacks towards +x (x = 105 is its opponent goal)
    * x = 0 is the selected team's own goal, also for opponent actions
    * y = 0 is the right touchline seen from the selected team's attack direction

Every geometric predicate in phase 3 relies on this convention.

IMPORTANT: no additional rotation must be applied here. StatsBomb coordinates
are already normalised per acting team (each team attacks towards x=120) and
``_fix_direction_of_play`` flips every action whose team is not
``home_team_id``. With ``home_team_id=TEAM_ID`` the selected team therefore
already attacks towards +x in EVERY match, home or away. (An earlier version
rotated the selected team's actions again in away matches, which put it at -x
in those matches; ``assert_orientation`` below guards against any regression.)
"""

from __future__ import annotations

import warnings
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from tactical_ilp.config import (
    PHASE1_VERSION, PROCESSED_DIR, TEAM_ID, cache_path,
)
from tactical_ilp.data.load_raw_data import barca_match_ids, load_events

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "player_id",
    "team_id",
    "type_name",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result_name",
}

SPADL_ALL_PATH = PROCESSED_DIR / "spadl_all_matches.parquet"

# Event-level timing signals used to reconstruct a player's holding interval.
RELEASE_EVENT_TYPES = frozenset({"Pass", "Shot", "Clearance"})
BALL_TAKING_TYPES = frozenset(
    {"Pass", "Shot", "Carry", "Clearance", "Interception", "Ball Recovery"}
)
DISRUPTIVE_TYPES = frozenset(
    {
        "Dispossessed",
        "Miscontrol",
        "Foul Committed",
        "Foul Won",
        "Error",
        "Ball Out",
        "Offside",
        "Duel",
        "Bad Behaviour",
    }
)
MAX_HOLD_S = 10.0


def _timestamp_to_seconds(ts: pd.Series) -> pd.Series:
    """Period-relative StatsBomb ``timestamp`` as float seconds (ms precision).

    Handles the representations produced by the different socceraction
    versions / the parquet cache: datetime64 (date part ignored), timedelta64
    and plain ``HH:MM:SS.mmm`` strings.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        return (ts - ts.dt.normalize()).dt.total_seconds()
    if pd.api.types.is_timedelta64_dtype(ts):
        return ts.dt.total_seconds()
    return pd.to_timedelta(ts.astype(str)).dt.total_seconds()


def event_clock_seconds(events: pd.DataFrame) -> pd.Series:
    """Period-relative seconds with millisecond precision, indexed like ``events``.

    socceraction's ``time_seconds`` is the period-relative clock (it subtracts the
    45/90/105-minute period starts from ``60 * minute + second``), i.e. the
    integer floor of the StatsBomb ``timestamp``. This returns the timestamp
    itself, keeping the sub-second part that tempo, one-touch and the 5 s
    regain windows need, on exactly the same scale and period origin.
    """
    return _timestamp_to_seconds(events["timestamp"]).astype(float)


def compute_time_on_ball(events: pd.DataFrame) -> pd.Series:
    """Holding seconds from a completed receipt to the player's next release.

    This is event-time-derived and only emitted when the preceding receipt is
    unambiguous. It is not a substitute for tracking-based possession time.
    """
    intervals = _compute_holding_intervals(events)
    if intervals.empty:
        return pd.Series(dtype=float, name="time_on_ball_s")
    return intervals.set_index("release_event_id")["duration_s"].rename("time_on_ball_s")


def _compute_holding_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Return unambiguous receipt-to-release intervals, including their bounds."""
    ev = events.sort_values(["period_id", "index"], kind="mergesort")
    # period-relative seconds: same scale as SPADL ``time_seconds``.
    t = event_clock_seconds(ev).to_numpy()
    extras = ev["extra"] if "extra" in ev.columns else pd.Series([None] * len(ev))

    out: list[dict] = []
    holder: dict | None = None
    for i, (eid, period, typ, pid, team, poss, extra) in enumerate(
        zip(
            ev["event_id"],
            ev["period_id"],
            ev["type_name"],
            ev["player_id"],
            ev["team_id"],
            ev["possession"],
            extras,
        )
    ):
        if holder is not None and (
            period != holder["period"] or poss != holder["possession"]
        ):
            holder = None

        if typ == "Ball Receipt*":
            extra_data = extra if isinstance(extra, Mapping) else {}
            receipt = extra_data.get("ball_receipt") or {}
            outcome = receipt.get("outcome") or {}
            if outcome.get("name") == "Incomplete":
                holder = None
            else:
                holder = {
                    "period": period,
                    "player": pid,
                    "team": team,
                    "possession": poss,
                    "t": t[i],
                }
            continue

        if holder is None:
            continue
        if typ in DISRUPTIVE_TYPES:
            holder = None
            continue
        if pid == holder["player"]:
            if typ in RELEASE_EVENT_TYPES:
                dt = t[i] - holder["t"]
                if 0.0 <= dt <= MAX_HOLD_S:
                    out.append(
                        {
                            "release_event_id": eid,
                            "period_id": period,
                            "player_id": pid,
                            "team_id": team,
                            "possession": poss,
                            "start_seconds": holder["t"],
                            "end_seconds": t[i],
                            "duration_s": float(dt),
                        }
                    )
                holder = None
            # A Carry by the current holder is evidence of continued control,
            # not a new player taking the ball. Keep the receipt timer running.
        elif typ in BALL_TAKING_TYPES:
            holder = None
    return pd.DataFrame(
        out,
        columns=[
            "release_event_id", "period_id", "player_id", "team_id", "possession",
            "start_seconds", "end_seconds", "duration_s",
        ],
    )


def _propagate_time_on_ball_to_dribbles(
    actions: pd.DataFrame, intervals: pd.DataFrame
) -> pd.Series:
    """Attribute a holding interval to synthetic dribbles inside that interval.

    SPADL synthetic dribbles have no source event id. Their midpoint timestamp,
    player, team, period and possession are used to connect them to an already
    established receipt-to-release interval. The complete interval remains an
    event-time proxy; it is not a per-dribble tracking measurement.
    """
    values = pd.Series(np.nan, index=actions.index, dtype=float)
    if intervals.empty or "action_source" not in actions:
        return values

    synthetic = actions["action_source"].eq("synthetic_dribble")
    for interval in intervals.itertuples(index=False):
        if pd.isna(interval.player_id) or pd.isna(interval.team_id) or pd.isna(interval.possession):
            continue
        mask = (
            synthetic
            & actions["period_id"].eq(interval.period_id).fillna(False)
            & actions["player_id"].eq(interval.player_id).fillna(False)
            & actions["team_id"].eq(interval.team_id).fillna(False)
            & actions["possession"].eq(interval.possession).fillna(False)
            & actions["time_seconds"].between(interval.start_seconds, interval.end_seconds)
        )
        values.loc[mask] = interval.duration_s
    return values


def _pass_height_name(extra) -> str | None:
    """Read the explicit StatsBomb pass height retained in event ``extra``."""
    if not isinstance(extra, Mapping):
        return None
    pass_data = extra.get("pass") or {}
    height = pass_data.get("height") or {}
    name = height.get("name") if isinstance(height, Mapping) else None
    return str(name) if name is not None else None


def _pass_recipient_id(extra):
    """Return an optional StatsBomb pass recipient id for sequence analysis."""
    if not isinstance(extra, Mapping):
        return None
    pass_data = extra.get("pass") or {}
    recipient = pass_data.get("recipient") if isinstance(pass_data, Mapping) else None
    if isinstance(recipient, Mapping):
        return recipient.get("id")
    return None


LOCATIONLESS_NON_ACTION_TYPES = frozenset(
    {
        "Starting XI",
        "Half Start",
        "Half End",
        "Injury Stoppage",
        "Player Off",
        "Player On",
        "Substitution",
        "Tactical Shift",
        "Bad Behaviour",
    }
)


def drop_locationless_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows without a location before SPADL conversion.

    Returns the cleaned events and missing-location rows that are not known
    administrative event types, so callers can log the affected candidates.
    """
    def has_location(value) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return False
        try:
            return bool(np.isfinite(float(value[0])) and np.isfinite(float(value[1])))
        except (TypeError, ValueError):
            return False

    missing = ~events["location"].map(has_location)
    dropped = events[missing & ~events["type_name"].isin(LOCATIONLESS_NON_ACTION_TYPES)]
    return events[~missing].copy(), dropped


def order_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Order equal-time actions by StatsBomb event index, keeping synthetic
    dribbles immediately after their preceding event-backed action.
    """
    out = actions.copy()
    idx = out["event_index"].astype(float)
    out["_order_key"] = idx.fillna(idx.ffill() + 0.5)
    return out.sort_values(
        ["period_id", "time_seconds", "_order_key"], kind="mergesort"
    ).drop(columns="_order_key").reset_index(drop=True)


def _is_current(df: pd.DataFrame) -> bool:
    return (
        "phase1_version" in df.columns
        and not df.empty
        and df["phase1_version"].eq(PHASE1_VERSION).all()
    )


def read_if_current(path, match_ids: list[int] | None = None) -> pd.DataFrame | None:
    """Return only a derived table built by this phase-1 schema/version."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not _is_current(df):
        return None
    if match_ids is not None:
        if "match_id" not in df.columns:
            return None
        if set(df["match_id"].unique()) != {int(m) for m in match_ids}:
            return None
    return df


def _refine_time_seconds(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Replace socceraction's integer-second ``time_seconds`` by ms-precision time.

    * event-backed actions: StatsBomb ``timestamp`` (period-relative, as ``time_seconds``);
    * synthetic dribbles: midpoint of the neighbouring actions (as in SPADL);
    * the clock is forced to be non-decreasing in StatsBomb event order inside
      each period, so sorting by time never contradicts the event order.
    Rows without a usable timestamp keep their integer time.
    """
    out = actions.copy()
    clock = event_clock_seconds(events)
    clock.index = events["event_id"].astype(str).to_numpy()
    clock = clock[~clock.index.duplicated(keep="first")]

    old = out["time_seconds"].astype(float)
    new = out["original_event_id"].astype("string").map(clock).astype(float)

    # a refined time must stay within the same second as the legacy value
    inconsistent = new.notna() & (((new - old) < -1e-9) | ((new - old) >= 1.0 + 1e-9))
    if inconsistent.any():
        log.warning(
            "%d action(s) have a timestamp inconsistent with minute/second; "
            "keeping their integer time",
            int(inconsistent.sum()),
        )
        new[inconsistent] = np.nan

    synthetic = out["original_event_id"].isna()
    # neighbours in the converter's own row order
    mid = (new.shift(1) + new.shift(-1)) / 2.0
    new = new.where(~synthetic, mid)
    out["time_seconds"] = new.fillna(old)

    idx = out["event_index"].astype(float)
    key = idx.fillna(idx.ffill() + 0.5)
    out = out.assign(_k=key).sort_values(["period_id", "_k"], kind="mergesort")
    out["time_seconds"] = out.groupby("period_id")["time_seconds"].cummax()
    return out.drop(columns="_k").sort_index()


class OrientationError(RuntimeError):
    """The selected team is not attacking towards +x in some match."""


def orientation_summary(actions: pd.DataFrame) -> dict:
    """Mean x of shots (fallback: keeper actions) by side, for one match."""
    shot = actions["type_name"].astype(str).str.startswith("shot")
    own = actions["team_id"] == TEAM_ID
    return {
        "n_shots_selected": int((shot & own).sum()),
        "n_shots_opponent": int((shot & ~own).sum()),
        "shot_x_selected": float(actions.loc[shot & own, "start_x"].mean()),
        "shot_x_opponent": float(actions.loc[shot & ~own, "start_x"].mean()),
        "keeper_x_selected": float(
            actions.loc[own & actions["type_name"].astype(str).str.startswith("keeper"), "start_x"].mean()
        ),
    }


def assert_orientation(actions: pd.DataFrame, match_id: int) -> dict:
    """Fail loudly if the selected team does not attack towards +x in this match.

    Checked PER MATCH (an aggregate mean hides the problem: with half of the
    matches mirrored the season mean looks like a plausible ~55).
    """
    r = orientation_summary(actions)
    if r["n_shots_selected"] >= 2 and r["n_shots_opponent"] >= 2:
        ok = (
            r["shot_x_selected"] > 65.0
            and r["shot_x_opponent"] < 40.0
            and r["shot_x_selected"] - r["shot_x_opponent"] > 30.0
        )
    elif r["n_shots_selected"] >= 2:
        ok = r["shot_x_selected"] > 65.0
    elif r["n_shots_opponent"] >= 2:
        ok = r["shot_x_opponent"] < 40.0
    elif not np.isnan(r["keeper_x_selected"]):
        ok = r["keeper_x_selected"] < 35.0
    else:
        log.warning("match %s: not enough shots/keeper actions to verify orientation", match_id)
        return r
    if not ok:
        raise OrientationError(
            f"match {match_id}: selected team does not attack towards +x "
            f"(mean shot x selected={r['shot_x_selected']:.1f}, "
            f"opponent={r['shot_x_opponent']:.1f}; expected ~90 / ~15)"
        )
    return r


def build_spadl(match_id: int, force: bool = False) -> pd.DataFrame:
    """SPADL action table for one match, oriented in the selected team's frame."""
    path = cache_path("spadl", match_id)
    if not force:
        cached = read_if_current(path)
        if cached is not None:
            return cached

    import socceraction.spadl as spadl

    events = load_events(match_id)
    clean_events, dropped = drop_locationless_events(events)
    if len(dropped):
        log.warning(
            "match %s: dropped %d event(s) without a location that could become actions: %s",
            match_id,
            len(dropped),
            dropped["type_name"].value_counts().to_dict(),
        )
    actions = spadl.statsbomb.convert_to_actions(clean_events, home_team_id=TEAM_ID)
    actions = spadl.add_names(actions)

    # NOTE: no further rotation. ``convert_to_actions(home_team_id=TEAM_ID)``
    # already puts the selected team at +x in every match (see module docstring).

    # SPADL does not retain this StatsBomb attribute in its standard action
    # table. Preserve the explicit source label for phase-4 tactical facts;
    # synthetic actions have no original event id and remain unknown.
    event_height = events.set_index(events.event_id.astype(str))["extra"].map(_pass_height_name)
    actions["pass_height_name"] = actions.original_event_id.astype("string").map(event_height)
    event_recipient = (
        events.set_index(events.event_id.astype(str))["pass_recipient_id"]
        if "pass_recipient_id" in events.columns
        else events.set_index(events.event_id.astype(str))["extra"].map(_pass_recipient_id)
    )
    actions["pass_recipient_id"] = actions.original_event_id.astype("string").map(event_recipient)

    # carry over StatsBomb flags needed later:
    #   under_pressure / counterpress -> pressure model (phase 3)
    #   possession / possession_team_id -> sequence segmentation (phase 2)
    flag_columns = [
        "under_pressure",
        "counterpress",
        "possession",
        "possession_team_id",
        "index",
        "duration",
        "play_pattern_name",
    ]
    if "shot_statsbomb_xg" in events.columns:
        flag_columns.append("shot_statsbomb_xg")
    flags = events.set_index("event_id")[flag_columns]
    actions = actions.join(flags, on="original_event_id")
    actions = actions.rename(
        columns={"index": "event_index", "duration": "event_duration"}
    )
    actions["under_pressure"] = actions["under_pressure"].fillna(False).astype(bool)
    actions["counterpress"] = actions["counterpress"].fillna(False).astype(bool)
    actions["possession"] = actions["possession"].ffill().bfill().astype("Int64")
    actions["possession_team_id"] = (
        actions["possession_team_id"].ffill().bfill().astype("Int64")
    )
    actions["match_id"] = match_id

    # Synthetic dribbles have no original event id; duplicated source ids are
    # the two SPADL rows created when one StatsBomb event is split.
    shared = actions["original_event_id"].notna() & actions.duplicated(
        "original_event_id", keep=False
    )
    actions["action_source"] = np.select(
        [actions["original_event_id"].isna(), shared],
        ["synthetic_dribble", "interception_split"],
        default="event",
    )

    actions = _refine_time_seconds(actions, events)

    holding_intervals = _compute_holding_intervals(events)
    tob = (
        holding_intervals.set_index("release_event_id")["duration_s"]
        if not holding_intervals.empty
        else pd.Series(dtype=float)
    )
    actions["time_on_ball_s"] = actions["original_event_id"].map(tob)
    synthetic_dwell = _propagate_time_on_ball_to_dribbles(actions, holding_intervals)
    synthetic_mask = actions["action_source"].eq("synthetic_dribble")
    actions.loc[synthetic_mask, "time_on_ball_s"] = synthetic_dwell.loc[synthetic_mask]
    actions.loc[actions["type_name"] == "interception", "time_on_ball_s"] = np.nan

    # Historical schema name; this boolean means "belongs to the configured
    # analysis team", including Leverkusen runs.
    actions["is_barca"] = actions["team_id"] == TEAM_ID
    actions = order_actions(actions)
    actions["action_id"] = range(len(actions))
    actions["phase1_version"] = PHASE1_VERSION

    assert_orientation(actions, match_id)

    actions.to_parquet(path, index=False)
    return actions


def build_all(match_ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Concatenated SPADL table for every Barcelona match."""
    ids = match_ids if match_ids is not None else barca_match_ids()
    if not force:
        cached = read_if_current(SPADL_ALL_PATH, ids)
        if cached is not None:
            return cached
    frames = []
    for i, mid in enumerate(ids, 1):
        df = build_spadl(mid, force=force)
        frames.append(df)
        print(f"[{i:>2}/{len(ids)}] match {mid}: {len(df):>5} SPADL actions")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(SPADL_ALL_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_all()
    print()
    print(f"total actions: {len(df)}")
    print(df["type_name"].value_counts().head(12).to_string())
    barca = df[df.is_barca]
    print(f"\nBarcelona actions: {len(barca)}  ({len(barca)/len(df):.1%})")
    shots = df[df.type_name == "shot"]
    print("\nsanity check — mean shot x by side (selected team must be high, opponent low):")
    print(shots.groupby("is_barca")["start_x"].mean().to_string())
    per_match = shots.groupby(["match_id", "is_barca"])["start_x"].mean().unstack()
    bad = per_match[(per_match.get(True, 0) <= 65) | (per_match.get(False, 100) >= 40)]
    print(f"matches failing the per-match orientation check: {len(bad)}")
