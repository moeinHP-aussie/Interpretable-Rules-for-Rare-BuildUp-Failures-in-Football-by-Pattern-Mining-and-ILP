"""
Task 2.3 — Extract the 2- and 3-player subsets that actually interacted inside
each build-up sequence.

Technical document §2.5 wants Popper to be able to learn relational patterns
(``pattern(P1, P2, Seq)``), not only single-player habits.  Feeding Popper every
possible pair of the 11 players would explode the search space, so we only keep
pairs with a **demonstrated direct interaction** inside the sequence.

Interaction model used here
---------------------------
``passes_to(P1, P2, Seq)``
    action *i* is a successful pass/cross by P1 and action *i+1* of the same
    sequence is performed by a different Barcelona player P2.

``receives_from`` is simply the inverse and is not stored separately.

``triple``
    a chain P1 -> P2 -> P3 built from two consecutive ``passes_to`` links, with
    three distinct players (the classic "third-man" shape).

Support/cover relations (which need the 360 freeze frame rather than the ball
chain) belong to phase 4 — they are tactical knowledge, not raw extraction.
"""

from __future__ import annotations

import pandas as pd

from tactical_ilp.config import PROCESSED_DIR
from tactical_ilp.data.freeze_frames import build_all as build_merged_all
from tactical_ilp.sequences.sequence_builder import build_all as build_sequences

PAIRS_PATH = PROCESSED_DIR / "sequence_pairs.parquet"
TRIPLES_PATH = PROCESSED_DIR / "sequence_triples.parquet"
SEQ_WITH_SUBSETS_PATH = PROCESSED_DIR / "sequences_with_subsets.parquet"

PASS_TYPES = {"pass", "cross", "freekick_short", "goalkick", "throw_in", "corner_short"}


def build_subsets(force: bool = False):
    """Return (pairs, triples, sequences_with_subsets)."""
    if (
        PAIRS_PATH.exists()
        and TRIPLES_PATH.exists()
        and SEQ_WITH_SUBSETS_PATH.exists()
        and not force
    ):
        return (
            pd.read_parquet(PAIRS_PATH),
            pd.read_parquet(TRIPLES_PATH),
            pd.read_parquet(SEQ_WITH_SUBSETS_PATH),
        )

    sequences, members = build_sequences()
    merged = build_merged_all()

    acts = merged.set_index(["match_id", "action_id"])
    cols = ["player_id", "type_name", "result_name", "start_x", "start_y", "end_x", "end_y"]
    joined = members.join(acts[cols], on=["match_id", "action_id"])
    joined = joined.sort_values(["sequence_id", "order_in_sequence"])

    pair_rows: list[dict] = []
    for sid, grp in joined.groupby("sequence_id", sort=False):
        rows = grp.to_dict("records")
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if a["type_name"] not in PASS_TYPES:
                continue
            if a["result_name"] != "success":
                continue
            if pd.isna(a["player_id"]) or pd.isna(b["player_id"]):
                continue
            p1, p2 = int(a["player_id"]), int(b["player_id"])
            if p1 == p2:
                continue
            pair_rows.append(
                {
                    "sequence_id": sid,
                    "match_id": a["match_id"],
                    "order_in_sequence": a["order_in_sequence"],
                    "passer_id": p1,
                    "receiver_id": p2,
                    "pass_type": a["type_name"],
                    "start_x": a["start_x"],
                    "start_y": a["start_y"],
                    "end_x": a["end_x"],
                    "end_y": a["end_y"],
                }
            )

    pairs = pd.DataFrame(pair_rows)

    # ---- triples: two chained passes with three distinct players ----------
    triple_rows: list[dict] = []
    if len(pairs):
        for sid, grp in pairs.groupby("sequence_id", sort=False):
            links = grp.sort_values("order_in_sequence").to_dict("records")
            for i in range(len(links) - 1):
                a, b = links[i], links[i + 1]
                if a["receiver_id"] != b["passer_id"]:
                    continue
                trio = {a["passer_id"], b["passer_id"], b["receiver_id"]}
                if len(trio) != 3:
                    continue
                triple_rows.append(
                    {
                        "sequence_id": sid,
                        "match_id": a["match_id"],
                        "p1": a["passer_id"],
                        "p2": b["passer_id"],
                        "p3": b["receiver_id"],
                        "order_in_sequence": a["order_in_sequence"],
                    }
                )
    triples = pd.DataFrame(
        triple_rows, columns=["sequence_id", "match_id", "p1", "p2", "p3", "order_in_sequence"]
    )

    # ---- per-sequence summary --------------------------------------------
    if len(pairs):
        by_seq = pairs.groupby("sequence_id").apply(
            lambda g: sorted({(int(r.passer_id), int(r.receiver_id)) for r in g.itertuples()}),
            include_groups=False,
        )
    else:
        by_seq = pd.Series(dtype=object)

    seq = sequences.copy()
    seq["player_pairs"] = seq["sequence_id"].map(by_seq)
    seq["player_pairs"] = seq["player_pairs"].apply(lambda v: v if isinstance(v, list) else [])
    seq["n_pairs"] = seq["player_pairs"].apply(len)
    if len(triples):
        tri_by_seq = triples.groupby("sequence_id").size()
    else:
        tri_by_seq = pd.Series(dtype=int)
    seq["n_triples"] = seq["sequence_id"].map(tri_by_seq).fillna(0).astype(int)

    # parquet cannot store tuples -> store as "p1>p2" strings
    seq["player_pairs"] = seq["player_pairs"].apply(
        lambda lst: [f"{a}>{b}" for a, b in lst]
    )

    pairs.to_parquet(PAIRS_PATH, index=False)
    triples.to_parquet(TRIPLES_PATH, index=False)
    seq.to_parquet(SEQ_WITH_SUBSETS_PATH, index=False)
    return pairs, triples, seq


if __name__ == "__main__":
    pairs, triples, seq = build_subsets(force=True)
    print(f"sequences        : {len(seq)}")
    print(f"pass links       : {len(pairs)}")
    print(f"distinct pairs   : {pairs.groupby(['passer_id','receiver_id']).ngroups}")
    print(f"triples          : {len(triples)}")
    print()
    cov = (seq["n_pairs"] > 0).mean()
    print(f"sequences with >= 1 pair : {cov:.2%}   (acceptance: >= 50%)")
    print(f"sequences with >= 1 triple: {(seq['n_triples'] > 0).mean():.2%}")
    print()
    print("most frequent pass links:")
    top = (
        pairs.groupby(["passer_id", "receiver_id"]).size().sort_values(ascending=False).head(10)
    )
    from tactical_ilp.data.players_reference import build_players_reference

    ref = build_players_reference().set_index("player_id")["display_name"]
    for (a, b), n in top.items():
        print(f"  {ref.get(a, a):<22} -> {ref.get(b, b):<22} {n}")
