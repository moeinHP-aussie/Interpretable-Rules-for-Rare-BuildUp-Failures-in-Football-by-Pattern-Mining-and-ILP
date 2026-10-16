"""
Phase 7 -- figures.  Returned as PNG bytes so the report can embed them as
base64 (a single self-contained HTML file, nothing to copy alongside it).

DESIGN RULE: no Persian text inside any figure.  matplotlib does not shape or
reorder Arabic-script text, so Persian labels come out as disconnected,
reversed letters.  Figures carry only numbers, symbols and short Latin
legends; every Persian explanation lives in the HTML, where the browser does
the shaping.  This also removes any font dependency.
"""

from __future__ import annotations

import io
import os

# Colab may export its inline notebook backend through MPLBACKEND. This
# subprocess/report renderer needs a normal non-interactive backend instead.
os.environ.pop("MPLBACKEND", None)

import matplotlib
from matplotlib.colors import to_hex

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from tactical_ilp.config import (
    DEFENSIVE_THIRD_MAX_X,
    OPPONENT_THIRD_START_X,
    OWN_HALF_MAX_X,
)

# SPADL frame here is 105 x 68 m with Barcelona attacking +x.
PITCH_L, PITCH_W = 105.0, 68.0

C_NEG = "#c0392b"      # sequence ended in a negative outcome
C_OK = "#2c7fb8"       # sequence did not
C_WIT = "#f39c12"      # the action that satisfied a literal
C_INK = "#1b2733"
UNKNOWN_PLAYER_COLOR = "#7f8c8d"
PLAYER_PALETTE = plt.get_cmap("tab20")


def actor_color_map(actions: pd.DataFrame) -> dict[object, str]:
    """Give each actor in one sequence a distinct, repeatable display color."""
    if "player_id" not in actions:
        return {}
    actor_ids = []
    for player_id in actions["player_id"].tolist():
        if pd.notna(player_id) and player_id not in actor_ids:
            actor_ids.append(player_id)
    return {player_id: to_hex(PLAYER_PALETTE(i % 20))
            for i, player_id in enumerate(actor_ids)}


def _overlap_label_offsets(actions: pd.DataFrame, threshold_m: float = 3.0) -> dict[int, tuple[int, int]]:
    """Offset order labels whose action start points would otherwise overlap."""
    rows = list(actions.itertuples())
    unseen = set(range(len(rows)))
    groups = []
    while unseen:
        seed = unseen.pop()
        group, frontier = {seed}, [seed]
        while frontier:
            i = frontier.pop()
            ri = rows[i]
            near = [j for j in unseen
                    if np.hypot(float(rows[j].start_x) - float(ri.start_x),
                                float(rows[j].start_y) - float(ri.start_y)) < threshold_m]
            group.update(near)
            unseen.difference_update(near)
            frontier.extend(near)
        if len(group) > 1:
            groups.append(sorted(group, key=lambda i: int(rows[i].order)))

    ring = [(-10, 9), (10, 9), (-10, -9), (10, -9), (0, 15), (0, -15),
            (-16, 0), (16, 0), (-14, 14), (14, 14), (-14, -14), (14, -14)]
    offsets = {}
    for group in groups:
        for i, row_index in enumerate(group):
            offsets[int(rows[row_index].order)] = ring[i % len(ring)]
    return offsets


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _pitch(ax=None, half: bool = False):
    """Full pitch, always.  ``half`` crops the axis to the OWN half (x <= 52.5).

    mplsoccer's own ``half=True`` draws the attacking half (x >= 52.5). The
    report now uses full-pitch views because build-up sequences span both halves.
    """
    p = Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W,
              line_color="#9aa5b1", pitch_color="white", linewidth=1.0,
              half=False, line_zorder=1)
    if ax is None:
        fig, ax = p.draw(figsize=(5.4, 4.6 if half else 3.6))
    else:
        p.draw(ax=ax)
        fig = ax.figure
    if half:
        ax.set_xlim(-2, PITCH_L / 2 + 2)
    return p, fig, ax


def location_map(w: pd.DataFrame, title_latin: str) -> bytes:
    """Where on the pitch did the witnessing actions start?

    Drawn on the full pitch because build-up sequences can now start anywhere
    in the own half and continue into the opponent's final third.
    """
    p, fig, ax = _pitch(half=False)
    ax.set_title(title_latin, fontsize=10, color=C_INK)
    if not w.empty:
        p.scatter(w["start_x"], w["start_y"], ax=ax, s=26, color=C_WIT,
                  edgecolor=C_INK, linewidth=0.4, alpha=0.75, zorder=3)
    ax.plot([OPPONENT_THIRD_START_X] * 2, [0, PITCH_W], color=C_NEG, ls="--", lw=1.1, zorder=2)
    ax.text(OPPONENT_THIRD_START_X + 0.8, PITCH_W - 3,
            f"x={OPPONENT_THIRD_START_X:g}", color=C_NEG, fontsize=8, zorder=4)
    ax.plot([DEFENSIVE_THIRD_MAX_X] * 2, [0, PITCH_W], color="#7d3c98",
            ls=":", lw=1.0, zorder=2)
    ax.text(DEFENSIVE_THIRD_MAX_X + 0.8, PITCH_W - 7,
            f"defensive x={DEFENSIVE_THIRD_MAX_X:g}", color="#7d3c98", fontsize=7, zorder=4)
    ax.plot([OWN_HALF_MAX_X] * 2, [0, PITCH_W], color="#7d3c98",
            ls=":", lw=1.0, zorder=2)
    ax.text(OWN_HALF_MAX_X + 0.8, PITCH_W - 7,
            f"own-half x={OWN_HALF_MAX_X:g}", color="#7d3c98", fontsize=7, zorder=4)
    return _png(fig)


def sequence_example(actions: pd.DataFrame, witness_orders: dict[str, set[int]],
                     negative: bool, title_latin: str,
                     shared_witness_orders: set[int] | None = None) -> bytes:
    """Plot actions by actor color; actions satisfying literals are ringed.

    ``actions``: rows of one sequence with start/end coords and ``order``.
    ``witness_orders``: literal-short-name -> set of orders that satisfied it.
    """
    p, fig, ax = _pitch(half=False)
    a = actions.sort_values("order")
    colours = actor_color_map(a)
    offsets = _overlap_label_offsets(a)
    ax.set_title(title_latin, fontsize=9, color=C_NEG if negative else C_OK)
    for _, r in a.iterrows():
        player_id = getattr(r, "player_id", None)
        colour = colours.get(player_id, UNKNOWN_PLAYER_COLOR)
        p.arrows(r.start_x, r.start_y, r.end_x, r.end_y, ax=ax, color=colour,
                 width=1.0, headwidth=3.2, headlength=3.2, alpha=0.68, zorder=3)
        dx, dy = offsets.get(int(r.order), (0, 0))
        ax.annotate(
            str(int(r.order)), xy=(r.start_x, r.start_y), xytext=(dx, dy),
            textcoords="offset points", ha="center", va="center", fontsize=7,
            color="white", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.2", fc=colour, ec="white", lw=0.7),
            arrowprops=(dict(arrowstyle="-", color="#7f8c8d", lw=0.6, alpha=0.75)
                        if (dx, dy) != (0, 0) else None),
        )
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    for i, (lit, orders) in enumerate(witness_orders.items()):
        sub = a[a["order"].isin(orders)]
        if len(sub):
            p.scatter(sub.start_x, sub.start_y, ax=ax, s=170, facecolor="none",
                      edgecolor=[C_WIT, "#27ae60", "#8e44ad", "#16a085"][i % 4],
                      linewidth=2.2, marker=markers[i % 4], zorder=4, label=lit)
    if shared_witness_orders:
        sub = a[a["order"].isin(shared_witness_orders)]
        if len(sub):
            p.scatter(sub.start_x, sub.start_y, ax=ax, s=260, facecolor="none",
                      edgecolor=C_INK, linewidth=1.5, marker="*", zorder=7,
                      label="both conditions on same action")
    if witness_orders:
        ax.legend(loc="lower left", fontsize=7, frameon=True)
    return _png(fig)


def rate_bars(rows: list[dict]) -> bytes:
    """Baseline vs each approved rule: share of matching sequences that are negative.

    rows: [{"label": "R1", "conf": float, "n": int}, ...]  plus a first baseline row.
    """
    fig, ax = plt.subplots(figsize=(5.4, 0.55 * len(rows) + 1.0))
    labels = [r["label"] for r in rows]
    vals = [r["conf"] * 100 for r in rows]
    colours = ["#95a5a6"] + [C_NEG] * (len(rows) - 1)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colours, height=0.6)
    ax.set_title("Negative-outcome rate by rule", fontsize=10, color=C_INK)
    for yi, r, v in zip(y, rows, vals):
        ax.text(v + 0.6, yi, f"{v:.1f}%   n={r['n']}", va="center", fontsize=8, color=C_INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.35 if vals else 1)
    ax.set_xlabel("negative-outcome rate (%)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def train_vs_holdout(rows: list[dict]) -> bytes:
    """Confidence on train vs holdout for every rule (approved AND rejected).

    rows: [{"label", "train", "holdout", "n_train", "n_holdout", "rejected": bool}]
    This is the picture that shows overfitting: a rejected rule's bar collapses.
    """
    fig, ax = plt.subplots(figsize=(max(7.5, 0.82 * len(rows)), 4.1))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r["train"] * 100 for r in rows], w, color="#34495e", label="train")
    hb = ax.bar(x + w / 2, [r["holdout"] * 100 for r in rows], w, color="#e67e22", label="holdout")
    ax.set_title("Rule confidence on training and holdout data", fontsize=10, color=C_INK)
    for xi, r in zip(x, rows):
        ax.text(xi - w / 2, r["train"] * 100 + 1.5, f"n={r['n_train']}", ha="center", fontsize=7)
        ax.text(xi + w / 2, r["holdout"] * 100 + 1.5, f"n={r['n_holdout']}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8,
                       rotation=55, ha="right", rotation_mode="anchor")
    for tick, row in zip(ax.get_xticklabels(), rows):
        if row["rejected"]:
            tick.set_color(C_NEG)
    ax.tick_params(axis="x", pad=5)
    ax.set_ylabel("confidence (%)", fontsize=8)
    ax.set_ylim(-16, 115)
    ax.axhline(0, color="#7f8c8d", lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.28)
    return _png(fig)
