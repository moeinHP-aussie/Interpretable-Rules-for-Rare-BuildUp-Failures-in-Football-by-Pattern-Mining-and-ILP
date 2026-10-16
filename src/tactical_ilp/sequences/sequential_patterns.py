"""Leakage-aware, gap-constrained sequential pattern features.

This module turns the existing build-up action membership table into a small
itemset sequence, mines bounded closed patterns on training matches only, and
projects the selected patterns back to sequences as unary Prolog facts.

The miner is a local prefix-projection implementation. It honors a maximum
number of intervening actions and computes closure by equal sequence-support
sets within the explicitly configured search bounds. It is intentionally
called a bounded gap-constrained closed miner in reports; it does not claim to
be the canonical Gap-BIDE implementation.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from tactical_ilp.config import PROCESSED_DIR, PROLOG_DIR
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.features.geometry import (
    is_diagonal_pass,
    is_progressive,
    is_switch_of_play,
    lane_of,
    pass_direction,
    pressure_level,
    third_of,
)
from tactical_ilp.features.tactical import is_hesitation, is_one_touch
from tactical_ilp.prolog.phase5_vocabulary import LOOKAHEAD_ACTION_PREDS, POPPER_WRAPPED_ACTION_PREDS
from tactical_ilp.sequences.outcome_labeler import label_sequences
from tactical_ilp.sequences.negative_sampling import build_context
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

PATTERN_INPUT_PATH = PROCESSED_DIR / "sequential_pattern_sequences.parquet"
PATTERN_REPORT_PATH = PROLOG_DIR.parent / "reports" / "sequential_pattern_mining.json"
PATTERN_INPUT_VERSION = 16  # 16: own-half loss labels and prefix-only terminal-action exclusion

# Twenty interpretable item values, grouped into mutually exclusive families
# where appropriate. These draw on the phase-3/4 definitions and do not use
# labels, turnover events, or opponent actions after the sequence boundary.
ACTION_FAMILY_MAP: dict[str, str] = {
    "pass": "pass",
    "cross": "cross",
    "carry": "carry",
    "dribble": "carry",
    "take_on": "carry",
    "tackle": "defensive",
    "interception": "defensive",
    "clearance": "defensive",
    "shot": "shot",
    "shot_freekick": "shot",
    "shot_penalty": "shot",
    "freekick_short": "restart",
    "freekick_crossed": "restart",
    "goalkick": "restart",
    "throw_in": "restart",
    "corner_short": "restart",
    "corner_crossed": "restart",
}
CORE_ITEM_VOCABULARY = tuple(
    [f"family:{x}" for x in ("pass", "cross", "carry", "defensive", "shot", "restart", "other")]
    + ["result:success", "result:unsuccessful"]
    + [f"pressure:{x}" for x in ("no_pressure", "light_pressure", "heavy_pressure")]
    + [f"direction:{x}" for x in ("forward", "sideways", "backward")]
    + ["progressive", "shape:diagonal", "shape:switch", "timing:hesitation", "timing:one_touch"]
)
assert len(CORE_ITEM_VOCABULARY) == 20

# Keep the original low-level representation available as a diagnostic arm,
# but mine the production sequential arm from the exact action-level concepts
# already offered to Popper. The sequence- and player-level predicates remain
# directly available in the sequential Popper bias;
# complements do not describe positive events and are intentionally omitted.
TACTICAL_ACTION_VOCABULARY = tuple(f"tactical:{name}" for name in POPPER_WRAPPED_ACTION_PREDS)
assert len(TACTICAL_ACTION_VOCABULARY) == len(set(TACTICAL_ACTION_VOCABULARY))
ITEM_VOCABULARY = CORE_ITEM_VOCABULARY  # backward-compatible name for core tests/tools
VOCABULARY_MODES = ("core", "tactical", "combined")
# LOOKAHEAD_ACTION_PREDS (imported above from phase5_vocabulary): predicates that
# inspect the next action or derive their result from the dwell interval to it.
# They cannot be evaluated on the last retained action when the labelled suffix
# is hidden.

PASS_LIKE = {
    "pass", "cross", "freekick_short", "freekick_crossed", "goalkick",
    "throw_in", "corner_short", "corner_crossed"
}
SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}
@dataclass(frozen=True)
class MiningConfig:
    max_gap: int = 2  # number of intervening action itemsets; distance <= 3
    min_support: int = 5
    min_target_support: int = 5
    min_growth_rate: float = 1.5
    smoothing: float = 0.5
    min_pattern_events: int = 2
    max_pattern_events: int = 3
    max_items_per_event: int = 1
    max_selected_patterns: int = 8
    excluded_suffix_actions: int = 1
    max_candidates: int = 50_000

    def __post_init__(self) -> None:
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")
        if self.min_support < 1 or self.min_target_support < 1:
            raise ValueError("support thresholds must be positive")
        if self.min_pattern_events < 1 or self.max_pattern_events < self.min_pattern_events:
            raise ValueError("pattern event bounds must satisfy 1 <= min_pattern_events <= max_pattern_events")
        if self.max_items_per_event < 1 or self.max_selected_patterns < 1:
            raise ValueError("pattern size and selected count must be positive")
        if self.excluded_suffix_actions < 0:
            raise ValueError("excluded_suffix_actions must be >= 0")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be > 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")


@dataclass(frozen=True)
class PatternCandidate:
    pattern: tuple[tuple[str, ...], ...]
    support_ids: frozenset[str]
    target_support: int
    control_support: int
    target_rate: float
    control_rate: float
    growth_rate: float
    log2_growth: float

    @property
    def key(self) -> str:
        return pattern_key(self.pattern)


def _is_missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _action_items(row, dwell_s: float | None) -> frozenset[str]:
    """Return phase-3/4-consistent Boolean symbols for one SPADL action."""
    typ = str(row.type_name)
    family = ACTION_FAMILY_MAP.get(typ, "other")
    items = {f"family:{family}"}
    items.add("result:success" if str(row.result_name) == "success" else "result:unsuccessful")

    items.add(
        "pressure:" + pressure_level(
            None if _is_missing(row.ff_opponents_3m) else row.ff_opponents_3m,
            None if _is_missing(row.ff_opponents_5m) else row.ff_opponents_5m,
            bool(row.under_pressure) if not _is_missing(row.under_pressure) else False,
        )
    )

    coords = (row.start_x, row.start_y, row.end_x, row.end_y)
    if all(not _is_missing(value) for value in coords):
        sx, sy, ex, ey = map(float, coords)
        if typ in PASS_LIKE or typ in {"dribble", "carry", "take_on"}:
            items.add("direction:" + pass_direction(sx, sy, ex, ey))
        if is_progressive(sx, sy, ex, ey):
            items.add("progressive")
        if typ in PASS_LIKE:
            if is_diagonal_pass(sx, sy, ex, ey):
                items.add("shape:diagonal")
            if is_switch_of_play(sx, sy, ex, ey):
                items.add("shape:switch")

    pressure = next((x.split(":", 1)[1] for x in items if x.startswith("pressure:")), "no_pressure")
    if is_hesitation(dwell_s, pressure):
        items.add("timing:hesitation")
    if is_one_touch(dwell_s):
        items.add("timing:one_touch")

    unexpected = items.difference(CORE_ITEM_VOCABULARY)
    if unexpected:
        raise ValueError(f"action item outside the fixed vocabulary: {sorted(unexpected)}")
    return frozenset(items)


def _prolog_quote(path: Path) -> str:
    """Quote an absolute path as a Prolog atom."""
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def parse_action_predicate_export(output: str) -> dict[str, frozenset[str]]:
    """Parse `action|predicate` rows emitted by the SWI-Prolog exporter."""
    by_action: dict[str, set[str]] = {}
    allowed = set(POPPER_WRAPPED_ACTION_PREDS)
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            action, predicate = line.strip().split("|", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"malformed action-predicate export row: {line!r}") from exc
        if predicate not in allowed:
            raise ValueError(f"unexpected action predicate in Prolog export: {predicate!r}")
        by_action.setdefault(action, set()).add(f"tactical:{predicate}")
    return {action: frozenset(items) for action, items in by_action.items()}


def _spatial_tactical_item(predicate_item: str, action) -> str:
    """Attach SPADL type, coarse start channel and pitch third to one event.

    The five base lanes are pooled into three miner-only channels: either wing
    and its adjacent half-space map to the same side channel. Example:
    ``tactical:line_pass@pass@left_channel@middle_third``. Keeping this as one
    atomic symbol preserves the miner's one-item-per-event bound while making
    each discovered motif spatially interpretable.
    """
    predicate = predicate_item.removeprefix("tactical:")
    typ = str(action.type_name).lower().replace(" ", "_")
    base_lane = "unknown" if _is_missing(action.start_y) else lane_of(action.start_y)
    lane = {
        "left_wing": "left_channel",
        "left_half_space": "left_channel",
        "central_channel": "central_channel",
        "right_half_space": "right_channel",
        "right_wing": "right_channel",
    }.get(base_lane, base_lane)
    third = "unknown" if _is_missing(action.start_x) else third_of(action.start_x)
    return f"tactical:{predicate}@{typ}@{lane}@{third}"


def suppress_boundary_lookahead(items: Iterable[str], action_index: int, kept_action_count: int) -> frozenset[str]:
    """Remove predicates that inspect an excluded suffix action at the boundary."""
    result = set(items)
    if action_index == kept_action_count - 1:
        result.difference_update(f"tactical:{name}" for name in LOOKAHEAD_ACTION_PREDS)
    return frozenset(result)


def export_action_predicate_items(timeout_seconds: int = 900) -> dict[str, frozenset[str]]:
    """Evaluate the active Popper action predicates over the Prolog BK.

    This avoids reimplementing phase-3/4 tactical definitions in Python and
    guarantees that mined events use the same rules as Popper's current bias.
    """
    executable = shutil.which("swipl")
    required = (PROLOG_DIR / "facts_all.pl", PROLOG_DIR / "facts_tactical.pl", PROLOG_DIR / "bk.pl")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Prolog facts required for tactical pattern inputs: " + ", ".join(map(str, missing)))
    if executable is None:
        raise RuntimeError("SWI-Prolog (`swipl`) is required to export the existing action-level tactical predicates")

    predicates = ",".join(POPPER_WRAPPED_ACTION_PREDS)
    prolog = "\n".join((
        f":- consult({_prolog_quote(required[0])}).",
        f":- consult({_prolog_quote(required[1])}).",
        f":- consult({_prolog_quote(required[2])}).",
        "run_export :-",
        f"    Predicates = [{predicates}],",
        "    findall(Action-Predicate,",
        "        ( member(Predicate, Predicates),",
        "          Goal =.. [Predicate, Action],",
        "          call(Goal),",
        "          \\+ set_piece_action(Action),",
        "          in_sequence(Action, _)",
        "        ), Raw),",
        "    sort(Raw, Pairs),",
        "    forall(member(Action-Predicate, Pairs), format('~w|~w~n', [Action, Predicate])),",
        "    halt.",
        ":- initialization(run_export).",
        "",
    ))
    with tempfile.NamedTemporaryFile("w", suffix=".pl", encoding="utf-8", delete=False) as handle:
        script_path = Path(handle.name)
        handle.write(prolog)
    try:
        result = subprocess.run(
            [executable, "-q", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"SWI-Prolog action-predicate export exceeded {timeout_seconds}s") from exc
    finally:
        script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SWI-Prolog action-predicate export failed: {detail}")
    return parse_action_predicate_export(result.stdout)


def build_pattern_sequences(
    force: bool = False,
    excluded_suffix_actions: int = 1,
    vocabulary: str = "tactical",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Build and cache ordered itemsets for all current build-up sequences.

    The suffix is excluded before deriving look-ahead-dependent dwell items,
    so the last retained action receives no dwell-derived symbol.
    """
    if excluded_suffix_actions < 0:
        raise ValueError("excluded_suffix_actions must be >= 0")
    if vocabulary not in VOCABULARY_MODES:
        raise ValueError(f"vocabulary must be one of {VOCABULARY_MODES}")
    output_path = cache_path or PATTERN_INPUT_PATH
    if output_path.exists() and not force:
        cached = pd.read_parquet(output_path)
        if (
            "excluded_suffix_actions" in cached.columns
            and cached.excluded_suffix_actions.nunique() == 1
            and int(cached.excluded_suffix_actions.iloc[0]) == excluded_suffix_actions
            and "feature_vocabulary" in cached.columns
            and cached.feature_vocabulary.nunique() == 1
            and str(cached.feature_vocabulary.iloc[0]) == vocabulary
            and {"start_zone", "pressure_bucket", "length_bucket"}.issubset(cached.columns)
            and "pattern_input_version" in cached.columns
            and cached.pattern_input_version.nunique() == 1
            and int(cached.pattern_input_version.iloc[0]) == PATTERN_INPUT_VERSION
        ):
            return cached

    sequences, members = build_sequences()
    labelled = label_sequences()
    merged = build_merged_all()
    metadata_cols = ["sequence_id", "match_id", "label"]
    meta = labelled[metadata_cols].copy()
    context = build_context()
    meta = meta.merge(
        context[["sequence_id", "start_zone", "pressure_bucket", "length_bucket"]],
        on="sequence_id", how="left", validate="one_to_one",
    )
    action_rows = members.merge(
        merged,
        on=["match_id", "action_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_raw"),
    )
    if action_rows.type_name.isna().any():
        raise ValueError("sequence_actions contains action IDs missing from SPADL data")
    action_rows = action_rows.sort_values(["sequence_id", "order_in_sequence"])
    tactical_by_action = export_action_predicate_items() if vocabulary in {"tactical", "combined"} else {}

    groups = {sid: g.reset_index(drop=True) for sid, g in action_rows.groupby("sequence_id", sort=False)}
    output: list[dict] = []
    suffix = excluded_suffix_actions
    for row in sequences.itertuples(index=False):
        sid = str(row.sequence_id)
        group = groups[sid]
        keep_count = max(0, len(group) - suffix)
        prefix = group.iloc[:keep_count]
        itemsets: list[list[str]] = []
        for index, action in prefix.iterrows():
            dwell = None
            if index + 1 < keep_count:
                # This is the event-derived holding interval from receipt to
                # release, not the gap to the next SPADL action (which can be
                # dominated by pass flight time). Keep the boundary mask on
                # the final retained action for conservative leakage control.
                dwell = action.get("time_on_ball_s")
            core_items = _action_items(action, dwell) if vocabulary in {"core", "combined"} else frozenset()
            action_key = f"a_{int(action.match_id)}_{int(action.action_id):05d}"
            tactical_items = set(tactical_by_action.get(action_key, ()))
            # These BK concepts inspect the following action. Do not let an
            # excluded suffix action leak back into the last retained event.
            tactical_items = suppress_boundary_lookahead(tactical_items, index, keep_count)
            tactical_items = {
                _spatial_tactical_item(item, action) for item in tactical_items
            }
            if vocabulary == "tactical":
                items = frozenset(tactical_items)
            elif vocabulary == "combined":
                items = frozenset(set(core_items) | tactical_items)
            else:
                items = core_items
            itemsets.append(sorted(items))
        label_row = meta.loc[meta.sequence_id == sid, "label"]
        if len(label_row) != 1:
            raise ValueError(f"expected one outcome label for sequence {sid}")
        output.append({
            "sequence_id": sid,
            "match_id": str(row.match_id),
            "label": str(label_row.iloc[0]),
            "start_zone": str(meta.loc[meta.sequence_id == sid, "start_zone"].iloc[0]),
            "pressure_bucket": str(meta.loc[meta.sequence_id == sid, "pressure_bucket"].iloc[0]),
            "length_bucket": str(meta.loc[meta.sequence_id == sid, "length_bucket"].iloc[0]),
            "n_actions": int(row.n_actions),
            "excluded_suffix_actions": suffix,
            "feature_vocabulary": vocabulary,
            "pattern_input_version": PATTERN_INPUT_VERSION,
            "itemsets_json": json.dumps(itemsets, separators=(",", ":")),
        })
    frame = pd.DataFrame(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def decode_itemsets(value: str) -> tuple[frozenset[str], ...]:
    decoded = json.loads(value) if isinstance(value, str) else value
    return tuple(frozenset(map(str, itemset)) for itemset in decoded)


def pattern_key(pattern: Sequence[Sequence[str]]) -> str:
    return " -> ".join("{" + " & ".join(items) + "}" for items in pattern)


class GapConstrainedClosedMiner:
    """Bounded prefix-projection miner with exact support-set closure checks."""

    def __init__(self, config: MiningConfig):
        self.config = config
        self.visited = 0
        self.supports: dict[tuple[tuple[str, ...], ...], frozenset[str]] = {}

    def mine(self, sequences: Mapping[str, Sequence[frozenset[str]]]) -> list[tuple[tuple[tuple[str, ...], ...], frozenset[str]]]:
        singleton: dict[str, dict[str, set[tuple[int, ...]]]] = {}
        for sid, events in sequences.items():
            for position, event in enumerate(events):
                for item in event:
                    singleton.setdefault(item, {}).setdefault(str(sid), set()).add((position,))

        for item in sorted(singleton):
            embeddings = {sid: frozenset(pos) for sid, pos in singleton[item].items()}
            if len(embeddings) >= self.config.min_support:
                self._grow(((item,),), embeddings, sequences)

        # A right-extension-only closure check misses equal-support extensions
        # that add an item to an earlier event (for example A -> B versus
        # {A,X} -> B). Group by exact sequence support, then retain only
        # patterns that are not a structural subsequence of a larger member.
        by_support: dict[frozenset[str], list[tuple[tuple[str, ...], ...]]] = {}
        for pattern, support in self.supports.items():
            by_support.setdefault(support, []).append(pattern)
        result = []
        for support, patterns in by_support.items():
            ordered = sorted(patterns, key=lambda p: (-_pattern_size(p), p))
            for pattern in ordered:
                size = _pattern_size(pattern)
                closed = not any(
                    _pattern_size(superpattern) > size
                    and _is_structural_subpattern(pattern, superpattern)
                    for superpattern in patterns
                )
                if closed:
                    result.append((pattern, support))
        return result

    def _grow(
        self,
        pattern: tuple[tuple[str, ...], ...],
        embeddings: Mapping[str, frozenset[tuple[int, ...]]],
        sequences: Mapping[str, Sequence[frozenset[str]]],
    ) -> None:
        self.visited += 1
        if self.visited > self.config.max_candidates:
            raise RuntimeError(
                f"pattern search exceeded max_candidates={self.config.max_candidates} "
                f"after visiting {self.visited:,} candidates; "
                "increase min_support or reduce max_pattern_events/max_items_per_event"
            )
        support = frozenset(embeddings)
        if len(support) < self.config.min_support:
            return
        self.supports[pattern] = support

        last_items = pattern[-1]
        if len(last_items) < self.config.max_items_per_event:
            after_item = last_items[-1]
            i_extensions: dict[str, dict[str, set[tuple[int, ...]]]] = {}
            for sid, occurrences in embeddings.items():
                events = sequences[sid]
                for occurrence in occurrences:
                    end = occurrence[-1]
                    for item in events[end]:
                        if item > after_item and item not in last_items:
                            i_extensions.setdefault(item, {}).setdefault(sid, set()).add(occurrence)
            for item in sorted(i_extensions):
                ext_pattern = (*pattern[:-1], tuple(sorted((*last_items, item))))
                ext_emb = {sid: frozenset(pos) for sid, pos in i_extensions[item].items()}
                if len(ext_emb) >= self.config.min_support:
                    self._grow(ext_pattern, ext_emb, sequences)

        if len(pattern) >= self.config.max_pattern_events:
            return
        s_extensions: dict[str, dict[str, set[tuple[int, ...]]]] = {}
        max_distance = self.config.max_gap + 1
        for sid, occurrences in embeddings.items():
            events = sequences[sid]
            for occurrence in occurrences:
                last_position = occurrence[-1]
                stop = min(len(events), last_position + max_distance + 1)
                for position in range(last_position + 1, stop):
                    for item in events[position]:
                        s_extensions.setdefault(item, {}).setdefault(sid, set()).add((*occurrence, position))
        for item in sorted(s_extensions):
            ext_pattern = (*pattern, (item,))
            ext_emb = {sid: frozenset(pos) for sid, pos in s_extensions[item].items()}
            if len(ext_emb) >= self.config.min_support:
                self._grow(ext_pattern, ext_emb, sequences)


def _pattern_size(pattern: Sequence[Sequence[str]]) -> int:
    return sum(len(itemset) for itemset in pattern)


def _is_structural_subpattern(
    pattern: Sequence[Sequence[str]], superpattern: Sequence[Sequence[str]]
) -> bool:
    """Whether each pattern itemset embeds in a later superset itemset."""
    if len(pattern) > len(superpattern) or _pattern_size(pattern) >= _pattern_size(superpattern):
        return False
    position = 0
    for itemset in pattern:
        required = set(itemset)
        found = False
        while position < len(superpattern):
            candidate = set(superpattern[position])
            position += 1
            if required.issubset(candidate):
                found = True
                break
        if not found:
            return False
    return True


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_greater(target_hits: int, target_n: int, control_hits: int, control_n: int) -> float:
    """One-sided Fisher exact p-value for enrichment in the target class."""
    population = target_n + control_n
    total_hits = target_hits + control_hits
    low = max(0, target_n - (population - total_hits))
    high = min(target_n, total_hits)
    observed = min(max(target_hits, low), high)
    log_denominator = _log_choose(population, target_n)
    log_probs = [
        _log_choose(total_hits, k)
        + _log_choose(population - total_hits, target_n - k)
        - log_denominator
        for k in range(observed, high + 1)
    ]
    if not log_probs:
        return 1.0
    peak = max(log_probs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(value - peak) for value in log_probs))


def benjamini_yekutieli(p_values: Sequence[float]) -> list[float]:
    """FDR q-values valid under arbitrary dependence among mined patterns."""
    count = len(p_values)
    if not count:
        return []
    order = sorted(range(count), key=lambda i: p_values[i])
    harmonic = math.fsum(1.0 / rank for rank in range(1, count + 1))
    adjusted = [1.0] * count
    running = 1.0
    for position in range(count - 1, -1, -1):
        original_index = order[position]
        rank = position + 1
        running = min(running, p_values[original_index] * count * harmonic / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def score_patterns(
    mined: Iterable[tuple[tuple[tuple[str, ...], ...], frozenset[str]]],
    target_ids: set[str],
    control_ids: set[str],
    config: MiningConfig,
    strata_by_id: Mapping[str, tuple[str, ...]] | None = None,
) -> list[PatternCandidate]:
    """Score prevalence, standardizing over shared context strata when supplied."""
    target_n, control_n = len(target_ids), len(control_ids)
    if not target_n or not control_n:
        raise ValueError("both target and control classes need training examples")
    candidates = []
    alpha = config.smoothing
    for pattern, support_ids in mined:
        # This arm studies ordered combinations; singleton event labels are
        # deliberately not offered to Popper as sequential-pattern features.
        if not config.min_pattern_events <= len(pattern) <= config.max_pattern_events:
            continue
        tp = len(support_ids & target_ids)
        cp = len(support_ids & control_ids)
        if tp < config.min_target_support:
            continue
        if strata_by_id:
            target_strata = {strata_by_id[sid] for sid in target_ids if sid in strata_by_id}
            control_strata = {strata_by_id[sid] for sid in control_ids if sid in strata_by_id}
            common = target_strata & control_strata
            common_target_n = sum(
                1 for sid in target_ids if strata_by_id.get(sid) in common
            )
            if not common or not common_target_n:
                continue
            tr = cr = 0.0
            for stratum in common:
                t_ids = {sid for sid in target_ids if strata_by_id.get(sid) == stratum}
                c_ids = {sid for sid in control_ids if strata_by_id.get(sid) == stratum}
                if not t_ids or not c_ids:
                    continue
                weight = len(t_ids) / common_target_n
                tr += weight * ((len(support_ids & t_ids) + alpha) / (len(t_ids) + 2 * alpha))
                cr += weight * ((len(support_ids & c_ids) + alpha) / (len(c_ids) + 2 * alpha))
        else:
            tr = (tp + alpha) / (target_n + 2 * alpha)
            cr = (cp + alpha) / (control_n + 2 * alpha)
        growth = tr / cr
        if growth < config.min_growth_rate:
            continue
        candidates.append(PatternCandidate(
            pattern=pattern,
            support_ids=support_ids,
            target_support=tp,
            control_support=cp,
            target_rate=tr,
            control_rate=cr,
            growth_rate=growth,
            log2_growth=math.log2(growth),
        ))
    return sorted(
        candidates,
        key=lambda item: (-item.growth_rate, -item.target_support, len(item.pattern), item.key),
    )[: config.max_selected_patterns]


def fit_patterns(
    frame: pd.DataFrame,
    train_match_ids: Iterable[str],
    config: MiningConfig,
    diagnostics: dict[str, int | float] | None = None,
) -> list[PatternCandidate]:
    """Fit the pattern vocabulary on training matches only."""
    allowed = {str(mid) for mid in train_match_ids}
    train = frame[frame.match_id.astype(str).isin(allowed)]
    database = {
        str(row.sequence_id): decode_itemsets(row.itemsets_json)
        for row in train.itertuples(index=False)
    }
    target_ids = set(train.loc[train.label == "negative", "sequence_id"].astype(str))
    control_ids = set(train.loc[train.label == "non_negative", "sequence_id"].astype(str))
    context_columns = ("start_zone", "pressure_bucket", "length_bucket")
    strata_by_id = None
    if set(context_columns).issubset(train.columns):
        strata_by_id = {
            str(row.sequence_id): tuple(str(getattr(row, key)) for key in context_columns)
            for row in train.itertuples(index=False)
        }
    if not target_ids or not control_ids:
        raise ValueError("both target and control classes need training examples")
    miner = GapConstrainedClosedMiner(config)
    # Mine target-class-closed patterns: this avoids spending the candidate
    # budget on control-only motifs and makes closure reflect retained target
    # coverage. Then project the candidates onto controls for contrast scores.
    target_database = {sid: database[sid] for sid in target_ids}
    mined_target = miner.mine(target_database)
    mined_contrast = []
    for pattern, target_support in mined_target:
        control_support = {
            sid for sid in control_ids
            if pattern_matches(database[sid], pattern, config.max_gap)
        }
        mined_contrast.append((pattern, frozenset(target_support | control_support)))
    candidates = score_patterns(mined_contrast, target_ids, control_ids, config, strata_by_id)
    if diagnostics is not None:
        diagnostics.update({
            "training_target_sequences": len(target_ids),
            "training_control_sequences": len(control_ids),
            "patterns_visited": miner.visited,
            "target_closed_patterns": len(mined_target),
            "patterns_selected": len(candidates),
            "growth_rate_context_adjusted": strata_by_id is not None,
            "shared_context_strata": len(
                {strata_by_id[sid] for sid in target_ids if sid in strata_by_id}
                & {strata_by_id[sid] for sid in control_ids if sid in strata_by_id}
            ) if strata_by_id else 0,
        })
    return candidates


def pattern_matches(events: Sequence[frozenset[str]], pattern: Sequence[Sequence[str]], max_gap: int) -> bool:
    """Whether itemsets occur in order with no more than max_gap items between."""
    return next(pattern_embeddings(events, pattern, max_gap), None) is not None


def pattern_embeddings(
    events: Sequence[frozenset[str]],
    pattern: Sequence[Sequence[str]],
    max_gap: int,
) -> Iterable[tuple[int, ...]]:
    """Yield ordered event-index embeddings admitted by the configured gap."""
    if not pattern:
        return

    def extend(step: int, positions: tuple[int, ...]):
        if step == len(pattern):
            yield positions
            return
        previous = positions[-1]
        stop = min(len(events), previous + max_gap + 2)
        required = set(pattern[step])
        for position in range(previous + 1, stop):
            if required.issubset(events[position]):
                yield from extend(step + 1, (*positions, position))

    first_items = set(pattern[0])
    for position, event in enumerate(events):
        if first_items.issubset(event):
            yield from extend(1, (position,))


def _pattern_item_metadata(item: str) -> dict[str, str | None]:
    """Decode a mined symbol into reviewed concept and spatial attributes."""
    if not item.startswith("tactical:"):
        return {"symbol": item, "predicate": None, "action_type": None, "lane": None, "third": None}
    values = item.removeprefix("tactical:").split("@")
    values += ["unknown"] * (4 - len(values))
    return {
        "symbol": item,
        "predicate": values[0],
        "action_type": values[1],
        "lane": values[2],
        "third": values[3],
    }


def write_pattern_facts(
    frame: pd.DataFrame,
    selected: Sequence[PatternCandidate],
    out_path: Path,
    config: MiningConfig,
) -> dict:
    """Write fold-local unary pattern facts over all sequences in ``frame``."""
    lines = [
        "% AUTO-GENERATED, fold-specific sequential-pattern features.",
        f"% max_gap={config.max_gap} intervening actions; suffix={config.excluded_suffix_actions} action(s) excluded.",
        "% Pattern selection used only training matches; facts below are projected to all rows.",
        "",
    ]
    feature_meta = []
    for rank, candidate in enumerate(selected, start=1):
        pred = f"seq_pattern_{rank:02d}"
        hits = []
        step_distance_counts: Counter[tuple[int, int]] = Counter()
        for row in frame.itertuples(index=False):
            events = decode_itemsets(row.itemsets_json)
            embeddings = list(pattern_embeddings(events, candidate.pattern, config.max_gap))
            if embeddings:
                hits.append(str(row.sequence_id))
                # Distance-distribution metadata is computed only over the
                # training support IDs selected during fit, never over CV
                # held-out matches. Pattern projection coverage remains a
                # separate descriptive total over the supplied frame.
                if str(row.sequence_id) in candidate.support_ids:
                    for embedding in embeddings:
                        for step, (previous, current) in enumerate(zip(embedding, embedding[1:]), start=1):
                            step_distance_counts[(step, current - previous)] += 1
        lines.extend(f"{pred}({sid})." for sid in sorted(hits))
        pattern_chain = [
            {
                "step": index,
                "items": [_pattern_item_metadata(item) for item in items],
            }
            for index, items in enumerate(candidate.pattern, start=1)
        ]
        feature_meta.append({
            "predicate": pred,
            "pattern": [list(items) for items in candidate.pattern],
            "pattern_chain": pattern_chain,
            "step_distance_constraints": [
                {
                    "from_step": step,
                    "to_step": step + 1,
                    "min_action_index_distance": 1,
                    "max_action_index_distance": config.max_gap + 1,
                    "max_intervening_actions": config.max_gap,
                }
                for step in range(1, len(candidate.pattern))
            ],
            "observed_step_distance_counts": [
                {
                    "from_step": step,
                    "to_step": step + 1,
                    "action_index_distance": distance,
                    "intervening_actions": distance - 1,
                    "embedding_count": count,
                }
                for (step, distance), count in sorted(step_distance_counts.items())
            ],
            "pattern_text": candidate.key,
            "training_target_support": candidate.target_support,
            "training_control_support": candidate.control_support,
            "target_rate": candidate.target_rate,
            "control_rate": candidate.control_rate,
            "growth_rate": candidate.growth_rate,
            "log2_growth": candidate.log2_growth,
            "growth_smoothing_alpha": config.smoothing,
            "all_sequence_coverage": len(hits),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(out_path), "features": feature_meta, "fact_count": sum(x["all_sequence_coverage"] for x in feature_meta)}


def search_space_proxy(n_predicates: int, max_body: int = 1, max_clauses: int = 3) -> dict:
    """Count unordered unary body combinations as a reproducible search proxy."""
    from math import comb

    clause_bodies = sum(comb(n_predicates, size) for size in range(1, min(max_body, n_predicates) + 1))
    programs = sum(comb(clause_bodies, n) for n in range(1, min(max_clauses, clause_bodies) + 1))
    return {
        "body_predicates": n_predicates,
        "max_body": max_body,
        "max_clauses": max_clauses,
        "unordered_clause_body_proxy": clause_bodies,
        "unordered_program_proxy": programs,
    }


def save_mining_report(report: dict, path: Path = PATTERN_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
