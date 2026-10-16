"""Shared tactical predicate vocabulary and Popper BK wrapper."""

from __future__ import annotations

from pathlib import Path

from tactical_ilp.config import PROLOG_DIR

POPPER_DIR = PROLOG_DIR / "popper"
POPPER_MAX_VARS = 2
POPPER_MAX_BODY = 3
POPPER_MAX_CLAUSES = 4

# ---------------------------------------------------------------------------
# body predicates offered to Popper, grouped by arity.
#
# Active discovery vocabulary: expose all reviewed, non-target-derived
# tactical concepts. costly_hesitation remains diagnostic-only because its
# definition explicitly joins hesitation to failure of the next action.
#
# DESIGN CONSTRAINT (phase 4 pre-check + glossary v5): every body predicate
# here is boolean or small-domain.  None of them exposes a raw numeric value.
# Predicates that are naturally per-action are wrapped with an
# "occurred in this sequence" existential (see WRAPPED_ACTION_PREDS) rather
# than handed to Popper directly, so Popper never has to reach through
# in_sequence/2 itself -- that would add a variable and blow up the search
# space exactly as the phase-3 report warned about chain/4.
# ---------------------------------------------------------------------------

# sequence-level predicates already defined in bk.pl / facts_all.pl -- usable
# as-is, arity 1 (Sequence).
SEQUENCE_PREDS_1 = [
    "opponent_visible_majority",
    "slow_sequence",
    "fast_sequence",
    "keeper_played_open_play",
    "seq_pass_exchange",
    "seq_flank_to_center",
]

# Known aliases/features that expose sequence_label/2, the target Popper is
# asked to predict. They must never appear as body predicates. The guard below
# checks both direct names and seq_<name> wrappers.
LEAKY_PREDICATES = {
    "hesitant_negative_sequence",
    "negative_sequence",
    "positive_sequence",
}

# This action feature is intentionally retained in bk.pl for retrospective
# diagnosis, but excluded from every discovery vocabulary: it looks one action
# ahead for a failed action, which overlaps the operational outcome label.
OUTCOME_PROXIMAL_ACTION_PREDS = {"costly_hesitation"}

# action-level predicates from bk.pl that we lift to "some action in this
# sequence satisfies P" via a generated wrapper predicate seq_P/1.
WRAPPED_ACTION_PREDS = [
    "line_breaking_pass",
    "diagonal_pass",
    "switch_to_right",
    "switch_to_left",
    "aerial_pass",
    "long_pass",
    "pass_in_behind",
    "line_pass",
    "connecting_pass",
    "dropped_between_lines",
    "first_time_play",
    "dwelt_long",
    "supported_action",
    "dead_end_support",
    "triangulation_opponent_trapped",
    "rondo_cage",
    "press_resistance",
    "counter_pressed",
    "rest_defence_exposed",
    "under_pressure",
    # Flank dynamics (SPADL lane transitions)
    "flank_line_carry",
    "flank_cut_inside",
    "flank_dribble",
    "central_long_carry",
    "central_take_on",
    "central_backward_pass",
    "flank_backward_carry",
    "flank_to_center_progression",
    # Central circulation (consecutive-pass spatial proxies)
    "central_switch",
    "recycle_to_same_flank",
    "center_cycling",
]

# These predicates inspect the NEXT action (or a dwell interval ending at it).
# On the last retained action of the prefix the next action is the hidden final
# action, so they must not fire there.
LOOKAHEAD_ACTION_PREDS = frozenset({
    "line_pass", "first_time_play", "dwelt_long", "quick_line_break",
})

# Aerial-pass and geometric triangulation/caging signals remain in the BK for
# diagnostics, but are excluded from Popper search. Restart deliveries were
# too easily treated as build-up long/high passes and generated spurious rules.
POPPER_EXCLUDED_ACTION_PREDS = frozenset({
    "aerial_pass",
    "triangulation",
    "triangle_trap",
    "triangulation_opponent_trapped",
    "rondo_cage",
})

# This action predicate remains available to the sequential miner, but Popper
# receives only its symmetric alias to avoid offering duplicate body atoms.
POPPER_WRAPPED_ACTION_PREDS = [
    pred for pred in WRAPPED_ACTION_PREDS
    if pred != "flank_to_center_progression" and pred not in POPPER_EXCLUDED_ACTION_PREDS
]

# player x sequence predicates, lifted to "some player satisfies P in this
# sequence" -- e.g. persistent_half_space_occupier(Player, Seq).
WRAPPED_PLAYER_SEQ_PREDS = [
    "persistent_half_space_occupier",
]

# Candidate complements are explicit sequence-level not_<pred>/1 wrappers.
# These are search candidates, not validated rules: match-grouped CV decides
# whether they generalize. The open-support complement is handled specially
# below so missing 360 frames do not count as evidence of no support.
COMPLEMENT_PREDS = [
    # Positive concepts only in the current reviewed vocabulary.
]

# Same-clause anti-redundancy pairs are listed here. Constraints apply to one
# Popper clause only; they do not prevent separate rules from using either
# feature. Some pairs are logically nested and some are intentionally pruned
# despite possible witnesses on different actions in a sequence.
MUTUALLY_EXCLUSIVE_BODY_PAIRS: list[tuple[str, str]] = [
    ("seq_flank_dribble", "seq_flank_line_carry"),
    ("seq_flank_dribble", "seq_flank_cut_inside"),
]

BK_WRAPPER_HEADER = """\
%% ===================================================================
%%  AUTO-GENERATED - phase 5 Popper background knowledge wrapper
%%
%%  Loads the phase 3/4 facts and hand-written bk.pl, then defines one
%%  seq_<concept>/1 wrapper per action-level or player-level concept so
%%  Popper only ever reasons over Sequence, never over Action or Player
%%  variables directly (keeps the search space small).
%%
%%  PATHS ARE RELATIVE to this file, resolved at load time via
%%  prolog_load_context/2.  An earlier version baked in absolute paths from
%%  the machine that generated it, which broke the moment the project was
%%  unzipped anywhere else (Colab, Windows, a different user).
%% ===================================================================

%%  SWI-Prolog resolves a relative path in consult/1 against the directory
%%  of the file containing the directive, so these stay correct wherever the
%%  project is unpacked.
:- consult('../facts_all.pl').
:- consult('../facts_tactical.pl').
:- consult('../bk.pl').

"""


def _wrapper_action(pred: str) -> str:
    """Existential lift over the PREFIX of the sequence (final action excluded).

    The final action is the one that decides the outcome label, so concepts
    evaluated on it would partly re-encode the label. Look-ahead concepts are
    additionally blocked on the last retained action, whose "next action" is
    the hidden final one.
    """
    guard = "    in_prefix(A, S),\n"
    if pred in LOOKAHEAD_ACTION_PREDS:
        guard += "    \\+ last_retained_action(A),\n"
    return f"seq_{pred}(S) :-\n{guard}    {pred}(A).\n"


def _wrapper_player_seq(pred: str) -> str:
    return f"seq_{pred}(S) :-\n    in_prefix(A, S), actor(A, P), {pred}(P, S).\n"


def build_bk_wrapper(out_path: Path) -> Path:
    """Write the Popper bk.pl wrapper.

    Uses relative consults (see BK_WRAPPER_HEADER) so the generated file
    stays valid after the project is moved, zipped, or unpacked on Colab.
    """
    lines = [BK_WRAPPER_HEADER]
    for pred in dict.fromkeys(WRAPPED_ACTION_PREDS):
        lines.append(_wrapper_action(pred))
    for pred in dict.fromkeys(WRAPPED_PLAYER_SEQ_PREDS):
        lines.append(_wrapper_player_seq(pred))

    # Clearer football-facing alias; this is exactly the same sequence signal,
    # not a second feature with a different underlying definition.
    lines.append("seq_open_support(S) :-\n    seq_supported_action(S).\n")
    lines.append("seq_flank_to_center(S) :-\n    seq_flank_to_center_progression(S).\n")
    lines.append(
        "seq_no_open_support(S) :-\n"
        "    in_prefix(A, S),\n"

        "    no_open_support(A).\n"
    )

    # Complement predicates are explicit Boolean features, not raw Prolog
    # negation in Popper's hypothesis language. Because each base seq_* is an
    # existential wrappers make each complement mean that the event/concept
    # was absent from every action in the sequence. They are candidates only;
    # cross-validation decides whether they generalize.
    for pred in dict.fromkeys(COMPLEMENT_PREDS):
        if pred == "seq_open_support":
            lines.append(
                "not_seq_open_support(S) :-\n"
                "    seq_no_open_support(S),\n"
                "    \\+ seq_supported_action(S).\n"
            )
        else:
            lines.append(f"not_{pred}(S) :-\n    sequence(S),\n    \\+ {pred}(S).\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
