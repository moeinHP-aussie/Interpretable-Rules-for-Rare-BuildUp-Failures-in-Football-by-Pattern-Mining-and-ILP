"""Synthetic correctness checks for the leakage-aware pattern feature path."""

import json

import pandas as pd

from tactical_ilp.sequences.sequential_patterns import (
    CORE_ITEM_VOCABULARY,
    benjamini_yekutieli,
    fisher_exact_greater,
    GapConstrainedClosedMiner,
    ITEM_VOCABULARY,
    MiningConfig,
    PatternCandidate,
    _spatial_tactical_item,
    TACTICAL_ACTION_VOCABULARY,
    parse_action_predicate_export,
    fit_patterns,
    pattern_matches,
    score_patterns,
    search_space_proxy,
    suppress_boundary_lookahead,
    write_pattern_facts,
)
from tactical_ilp.prolog.phase5_vocabulary import POPPER_WRAPPED_ACTION_PREDS
from tactical_ilp.prolog.patterns_only_inputs import build_patterns_only_bias, build_hybrid_bias
from tactical_ilp.prolog.patterns_only_inputs import select_compressing_patterns


def test_core_diagnostic_vocabulary_has_twenty_symbols():
    assert ITEM_VOCABULARY == CORE_ITEM_VOCABULARY
    assert len(CORE_ITEM_VOCABULARY) == 20
    assert len(set(CORE_ITEM_VOCABULARY)) == 20


def test_tactical_pattern_vocabulary_matches_existing_action_bias():
    assert len(TACTICAL_ACTION_VOCABULARY) == len(POPPER_WRAPPED_ACTION_PREDS)
    assert TACTICAL_ACTION_VOCABULARY == tuple(
        f"tactical:{name}" for name in POPPER_WRAPPED_ACTION_PREDS
    )
    assert set(TACTICAL_ACTION_VOCABULARY) == {f"tactical:{name}" for name in POPPER_WRAPPED_ACTION_PREDS}


def test_tactical_default_bounds_fit_under_candidate_cap():
    config = MiningConfig()
    assert config.max_pattern_events == 3
    assert config.max_items_per_event == 1
    assert config.max_selected_patterns == 8
    theoretical_bound = sum(len(TACTICAL_ACTION_VOCABULARY) ** n for n in range(1, config.max_pattern_events + 1))
    assert theoretical_bound <= config.max_candidates


def test_prolog_export_parser_and_suffix_lookahead_guard():
    exported = parse_action_predicate_export(
        "a_123_00001|line_breaking_pass\na_123_00001|press_resistance\n"
    )
    assert exported["a_123_00001"] == frozenset({
        "tactical:line_breaking_pass", "tactical:press_resistance"
    })
    boundary = suppress_boundary_lookahead(
        {"tactical:line_pass", "tactical:first_time_play", "tactical:press_resistance"},
        action_index=2,
        kept_action_count=3,
    )
    assert boundary == {"tactical:press_resistance"}


def test_max_gap_counts_intervening_actions():
    events = (frozenset({"a"}), frozenset({"x"}), frozenset({"y"}), frozenset({"b"}))
    assert pattern_matches(events, (("a",), ("b",)), max_gap=2)
    assert not pattern_matches(events, (("a",), ("b",)), max_gap=1)


def test_closed_miner_keeps_cooccurrence_and_ordered_events():
    db = {
        f"s{i}": (frozenset({"a", "x"}), frozenset({"b"}))
        for i in range(3)
    }
    miner = GapConstrainedClosedMiner(
        MiningConfig(max_gap=0, min_support=2, max_pattern_events=2, max_items_per_event=2)
    )
    patterns = dict(miner.mine(db))
    assert (("a", "x"), ("b",)) in patterns
    assert patterns[(("a", "x"), ("b",))] == frozenset({"s0", "s1", "s2"})
    assert (("a",), ("b",)) not in patterns  # same-support superpattern closes it


def test_smoothed_growth_filter_selects_target_enriched_patterns():
    ids = frozenset(f"t{i}" for i in range(5))
    scored = score_patterns(
        [((("a",), ("b",)), ids)],
        target_ids={f"t{i}" for i in range(100)},
        control_ids={f"c{i}" for i in range(100)},
        config=MiningConfig(min_support=1, min_target_support=5, min_growth_rate=1.5),
    )
    assert len(scored) == 1
    assert scored[0].target_support == 5
    assert scored[0].control_support == 0
    assert scored[0].growth_rate > 1.5
    assert scored[0].log2_growth > 0
    assert fisher_exact_greater(5, 100, 0, 100) < 0.05
    assert benjamini_yekutieli([0.01, 0.02]) == [0.03, 0.03]


def test_search_proxy_exposes_reduced_bias():
    baseline = search_space_proxy(43)
    reduced = search_space_proxy(13)
    assert reduced["unordered_clause_body_proxy"] < baseline["unordered_clause_body_proxy"]


def test_patterns_only_bias_contains_only_mined_pattern_predicates(tmp_path):
    path = build_patterns_only_bias(
        tmp_path / "bias.pl", ("seq_pattern_01", "seq_pattern_02")
    )
    text = path.read_text(encoding="utf-8")
    body = [line for line in text.splitlines() if line.startswith("body_pred(")]
    assert len(body) == 2
    assert "body_pred(seq_pattern_01,1)." in body
    assert body == ["body_pred(seq_pattern_01,1).", "body_pred(seq_pattern_02,1)."]


def test_hybrid_bias_keeps_unconsumed_full_tactical_features(tmp_path):
    info = build_hybrid_bias(
        tmp_path / "hybrid_bias.pl",
        [{"predicate": "seq_pattern_01", "pattern": [["tactical:line_pass@pass@left_channel@middle_third"]]}],
    )
    text = info["path"].read_text(encoding="utf-8")
    assert "body_pred(seq_pattern_01,1)." in text
    assert "body_pred(seq_line_pass,1)." in text
    assert "body_pred(seq_press_resistance,1)." in text
    assert "body_pred(opponent_visible_majority,1)." in text
    assert info["body_predicates"] <= 35


def test_tactical_pattern_items_include_type_and_spatial_location():
    row = type("Action", (), {
        "type_name": "pass", "start_x": 50.0, "start_y": 60.0,
    })()
    assert _spatial_tactical_item("tactical:line_pass", row) == (
        "tactical:line_pass@pass@left_channel@middle_third"
    )


def test_pattern_spatial_mapping_pools_five_base_lanes_into_three_channels():
    expected = {
        5.0: "right_channel", 20.0: "right_channel", 34.0: "central_channel",
        48.0: "left_channel", 63.0: "left_channel",
    }
    for y, channel in expected.items():
        row = type("Action", (), {"type_name": "pass", "start_x": 50.0, "start_y": y})()
        assert _spatial_tactical_item("tactical:line_pass", row).split("@")[2] == channel


def test_spatial_pattern_compression_matches_base_tactical_predicate():
    candidate = PatternCandidate(
        pattern=(("tactical:line_pass@pass@left_channel@middle_third",),
                 ("tactical:triangulation@pass@central_channel@middle_third",)),
        support_ids=frozenset({"s1"}), target_support=1,
        control_support=0, target_rate=0.75, control_rate=0.25,
        risk_difference=0.5, fisher_p_value=0.01, fdr_q_value=0.02,
    )
    selected = select_compressing_patterns([candidate], max_patterns=8)
    assert selected == [candidate]


def test_pattern_facts_metadata_contains_chain_and_step_distances(tmp_path):
    pattern = (("tactical:line_pass@pass@left_channel@middle_third",),
               ("tactical:long_pass@cross@right_channel@final_third",))
    candidate = PatternCandidate(
        pattern=pattern, support_ids=frozenset({"s1"}), target_support=1,
        control_support=0, target_rate=0.75, control_rate=0.25,
        risk_difference=0.5, fisher_p_value=0.01, fdr_q_value=0.02,
    )
    frame = pd.DataFrame([{
        "sequence_id": "s1", "itemsets_json": json.dumps([
            [pattern[0][0]], ["other"], [pattern[1][0]],
        ]),
    }])
    info = write_pattern_facts(
        frame, [candidate], tmp_path / "facts_patterns.pl", MiningConfig(max_gap=2)
    )
    feature = info["features"][0]
    assert feature["pattern_chain"][0]["items"][0] == {
        "symbol": pattern[0][0], "predicate": "line_pass", "action_type": "pass",
        "lane": "left_channel", "third": "middle_third",
    }
    assert feature["step_distance_constraints"][0]["max_intervening_actions"] == 2
    assert feature["observed_step_distance_counts"] == [{
        "from_step": 1, "to_step": 2, "action_index_distance": 2,
        "intervening_actions": 1, "embedding_count": 1,
    }]


def test_fold_fit_ignores_held_out_match_labels_and_sequences():
    rows = []
    for i in range(5):
        rows.append({"sequence_id": f"t{i}", "match_id": "train_target", "label": "negative",
                     "itemsets_json": json.dumps([["a"], ["b"]])})
        rows.append({"sequence_id": f"c{i}", "match_id": "train_control", "label": "non_negative",
                     "itemsets_json": json.dumps([["x"], ["y"]])})
    config = MiningConfig(min_support=2, fdr_q_threshold=0.05,
                          max_pattern_events=2, max_items_per_event=1)
    original = pd.DataFrame(rows)
    held_out = pd.DataFrame([
        {"sequence_id": f"h{i}", "match_id": "held_out", "label": "negative",
         "itemsets_json": json.dumps([["a"], ["b"]])}
        for i in range(20)
    ])
    fitted = fit_patterns(original, {"train_target", "train_control"}, config)
    fitted_with_test_rows_present = fit_patterns(
        pd.concat([original, held_out], ignore_index=True),
        {"train_target", "train_control"},
        config,
    )
    assert [(x.pattern, x.support_ids, x.fdr_q_value) for x in fitted] == [
        (x.pattern, x.support_ids, x.fdr_q_value) for x in fitted_with_test_rows_present
    ]
