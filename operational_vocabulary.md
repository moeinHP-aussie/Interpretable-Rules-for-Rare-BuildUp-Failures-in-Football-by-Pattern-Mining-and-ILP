# Operational Vocabulary and Conceptual Relations

This list is derived from the Phase 5 executable vocabulary in `src/tactical_ilp/prolog/phase5_vocabulary.py` and the rules in `src/tactical_ilp/prolog/bk.pl`. "Active for Popper" means the predicate is offered in the bias as a body feature. "Miner" refers only to predicates that are active in `POPPER_WRAPPED_ACTION_PREDS`; a definition in the BK alone does not mean it is consumed by the miner.

## Action-Centric Concepts and Miner Labels

27 predicates from this list are active in both Popper and the miner in the existential form `seq_<predicate>(S)`. `aerial_pass` and the two triangle/cage concepts are excluded from both searches. `flank_to_center_progression` reaches Popper only through the sequence alias and is not in the miner.

| # | Concept / base predicate | Level and status in Popper | Current operational definition |
|---:|---|---|---|
| 1 | `line_breaking_pass` | Action; active (`seq_line_breaking_pass`) | A successful pass that starts behind the estimated opponent defensive line and crosses it. The line is estimated from 360 freeze-frame coordinates. |
| 2 | `diagonal_pass` | Action; active (`seq_diagonal_pass`) | Pass-like action with a longitudinal advance of at least 5 m, a Euclidean length of at least 12 m, and an angle of 20 to 70 degrees relative to the longitudinal axis. |
| 3 | `switch_to_right` | Action; active (`seq_switch_to_right`) | A successful pass with a lateral displacement of at least 25 m that reaches the right half-space or right wing. |
| 4 | `switch_to_left` | Action; active (`seq_switch_to_left`) | Same as the previous item, with the destination in the left half-space or left wing. |
| 5 | `aerial_pass` | BK/facts for auditing only; excluded from Popper and miner | Pass-like action carrying the height label `High Pass`; this label alone does not establish an aerial-pass tactic. |
| 6 | `long_pass` | Action; open play only (`seq_long_pass`) | Pass-like action of at least 30 m; corners, free kicks, throw-ins, and goal kicks are excluded. |
| 7 | `pass_in_behind` | Action; active (`seq_pass_in_behind`) | A successful pass that ends behind the estimated defensive line and in the attacking third or penalty area; the origin is not necessarily behind the line. |
| 8 | `line_pass` | Action; active (`seq_line_pass`) | A successful pass in one of the non-central lanes that starts and ends in the same lane and is not `sideways`. |
| 9 | `connecting_pass` | Action; active (`seq_connecting_pass`) | A short successful pass (`<10m`) that does not carry the `progressive` attribute; the label does not establish a subjective intent to retain possession. |
| 10 | `dropped_between_lines` | Action; active (`seq_dropped_between_lines`) | The ball carrier is positioned between the midfield line and the estimated opponent defensive line; both lines are built from observable coordinates. |
| 11 | `first_time_play` | Action; active (`seq_first_time_play`) | Action hold time of at most 0.5 seconds; an approximation of a one-touch play, since the number of touches is not directly recorded. |
| 12 | `dwelt_long` | Action; active (`seq_dwelt_long`) | Ball hold time of more than 2.5 seconds. |
| 13 | `supported_action` | Action; active (`seq_supported_action`) | In a valid frame, at least one teammate has a direct path to the ball carrier with no opponent within 2 m of that path. |
| 14 | `dead_end_support` | Action; active (`seq_dead_end_support`) | In a valid frame, no visible teammate has an open passing path; the absence of a frame alone does not mean "unsupported". |
| 15 | `triangulation_opponent_trapped` | Action; BK/diagnostic, excluded from Popper and miner | At least one meaningful triangle between the carrier and two teammates encloses a defender. |
| 16 | `rondo_cage` | Action; BK/diagnostic, excluded from Popper and miner | At least two triangles enclosing a defender in one frame/action. |
| 17 | `press_resistance` | Action; active (`seq_press_resistance`) | A successful action under heavy pressure; heavy pressure is estimated from at least two opponents within a 3 m radius. |
| 18 | `counter_pressed` | Action; active (`seq_counter_pressed`) | The StatsBomb source's counterpress flag is recorded for the action. |
| 19 | `rest_defence_exposed` | Action; active (`seq_rest_defence_exposed`) | In the frame, fewer than two non-goalkeeper teammates are at least 5 m behind the ball. |
| 20 | `under_pressure` | Action; active (`seq_under_pressure`) | Binary pressure is determined from the 360 frame, and from the StatsBomb flag when the frame is unavailable; its intensity definition is not the same as that of `press_resistance`. |
| 21 | `flank_line_carry` | Action; active (`seq_flank_line_carry`) | A dribble from a side wing, ending in the same wing, with a longitudinal advance of at least 5 m toward the goal. |
| 22 | `flank_cut_inside` | Action; active (`seq_flank_cut_inside`) | A dribble started from a side wing that ends in the half-space or the central channel. |
| 23 | `flank_dribble` | Action; active (`seq_flank_dribble`) | A dribble from either of the two side wings, regardless of destination. |
| 24 | `central_long_carry` | Action; active (`seq_central_long_carry`) | A carry/dribble originating in any non-wing lane with a Euclidean length of at least 3.5 m; direction is irrelevant. |
| 25 | `central_take_on` | Action; active (`seq_central_take_on`) | A `take_on` action originating in one of the three non-wing lanes; winning the duel is not required. |
| 26 | `central_backward_pass` | Action; active (`seq_central_backward_pass`) | Pass-like action with a non-wing origin, a Euclidean length of at least 7 m, and `Δx ≤ −5m`. |
| 27 | `flank_backward_carry` | Action; active (`seq_flank_backward_carry`) | A carry/dribble from the wing that ends in the same wing with `Δx ≤ −4m`. |
| 28 | `flank_to_center_progression` | Action; sequence alias `seq_flank_to_center` in Popper; excluded from miner | Pass-like action from the wing to the half-space or center, with a length of at least 10 m and `Δx ≥ 5m`. The predicate is defined in the BK; Popper sees it only through the sequence alias, and the current miner does not consume it. |
| 29 | `central_switch` | Sequence-derived action; active (`seq_central_switch`) | Two consecutive successful passes in the pass stream: from one of the wings to the center, then from the center to the opposite wing. A non-pass action between the two is allowed; receiver/player continuity is not verified. |
| 30 | `recycle_to_same_flank` | Sequence-derived action; active (`seq_recycle_to_same_flank`) | Two consecutive successful passes in the pass stream: from the wing to the center, and from the center back to the same wing. |
| 31 | `center_cycling` | Sequence-derived action; active (`seq_center_cycling`) | Two consecutive successful passes in the pass stream, both starting in the center and ending in the center. |

## Sequence-Level and Player-Sequence-Level Features

| # | Predicate in Popper | Level | Current operational definition |
|---:|---|---|---|
| 1 | `opponent_visible_majority` | Sequence; active | The sequence contains at least two eligible 360 frames, and in more than half of them the number of visible opponents exceeds the number of visible teammates; each frame requires at least 7 visible players. |
| 2 | `slow_sequence` | Sequence; active | The action rate (actions per second) is at most 0.5. |
| 3 | `fast_sequence` | Sequence; active | The action rate is at or above the 70th percentile of the distribution of valid sequence rates in the same data/extraction (deciles 8 to 10). |
| 4 | `keeper_played_open_play` | Sequence; active | The goalkeeper himself executes at least one successful `pass`-type pass of `short` or `medium` length in the sequence; the goalkeeper receiving the ball and long passes are not counted. |
| 5 | `seq_pass_exchange` | Sequence; active | Among the sequence's successful passes, player A passes to B and then B passes to A; identifiers are used only for the computation, and only the Boolean value is stored in the final feature. |
| 6 | `seq_flank_to_center` | Sequence; active / generic name | The symmetric Popper name for the existence of `flank_to_center_progression` in the sequence; left and right are not distinguished. |
| 7 | `persistent_half_space_occupier` (wrapper: `seq_persistent_half_space_occupier`) | Player-sequence; active | At least one player in the same sequence has at least two actions originating in the half-space; the player's identity is hidden as a variable in the Popper body. |

### Helper Names and Dynamic Patterns

`seq_open_support` is an alias of the `seq_supported_action` signal, and `seq_no_open_support` is an alias of the `no_open_support` feature; neither alias is offered in the active Popper bias. The `seq_pattern_XX` predicates are defined from the miner output of each run or fold and are not fixed vocabulary concepts.

## Conceptual Overlap Relations

A definitive relation follows from the predicate definitions; a structural relation means that two features can occur together in one action or sequence without either implying the other.

| Pair / group of concepts | Relation type | Explanation and practical effect |
|---|---|---|
| `line_breaking_pass` and `pass_in_behind` | Partial overlap, not complete | Both are successful, pass-like actions whose destination is behind the estimated line; in addition, `pass_in_behind` must end in the attacking third or penalty area. Hence a line-breaking pass outside the attacking third satisfies only the first condition, and a pass from in front of the line to behind it within the attacking third can satisfy only the second. |
| `rondo_cage` ⊂ `triangulation_opponent_trapped` | Definitive within the same action | A cage requires at least two enclosing triangles; the trapped-triangle predicate requires at least one. The relation also holds after wrapping. |
| `flank_line_carry` ⊂ `flank_dribble` | Definitive within the same action | The line condition further restricts a dribble started from the wing; every line dribble is also a wing dribble. |
| `flank_cut_inside` ⊂ `flank_dribble` | Definitive within the same action | Both derive from a dribble started in the wing; cut-inside only restricts the destination further. |
| `fast_sequence` and `slow_sequence` | Mutually incompatible within one sequence | Each sequence receives one tempo bucket; the slow rate is at most 0.5 and fast is at the 70th percentile. However, the absence of a valid tempo can make both false. |
| `switch_to_left` and `switch_to_right` | Incompatible for a single pass; combinable within a sequence | A single destination cannot be both left and right; two different passes in a sequence can activate both existential features. |
| `first_time_play` and `dwelt_long` | Incompatible for a single action; combinable within a sequence | The ranges ≤0.5 s and >2.5 s are disjoint, but different actions of one sequence can have both. |
| `supported_action` and `dead_end_support` | Complementary only in a valid frame/action; sequence-level co-occurrence possible | In a single valid frame with an identifiable carrier, the existence of an open option versus its absence are disjoint; two existential wrappers over different actions may both be true. |
| `line_pass` and `connecting_pass` | Probable overlap | Both are successful passes; a short, same-lane, non-sideways pass that is not progressive can activate both. The definitions are not duplicates: one measures shape/path and the other measures shortness/progression. |
| `central_backward_pass` with `line_pass`, `diagonal_pass`, `long_pass`, or `switch_*` | Probable overlap | A single pass can simultaneously have several geometric descriptors; these features measure different axes (origin, direction, distance, angle, and lateral displacement). |
| `flank_to_center_progression` with `diagonal_pass`, `long_pass`, `line_breaking_pass`, and `pass_in_behind` | Probable overlap, generally composable | A wing-to-center pass may also satisfy any of the length/angle/defensive-line constraints. In addition to crossing the defensive line, the definition of `pass_in_behind` also requires a destination in the attacking third or the area. |
| Bias constraints: `rondo_cage` + `triangulation_opponent_trapped`; `flank_dribble` + any subset | Combination pruning within one clause | `rondo_cage` and `triangulation_opponent_trapped` have been removed from the Popper bias. The active co-occurrence constraint only forbids combining `seq_flank_dribble` with `seq_flank_line_carry` or `seq_flank_cut_inside` in one clause. |
| `central_switch`, `recycle_to_same_flank`, `center_cycling` | Overlap within a sequence, not for a single pass pair | For a specific pair of passes, the destination/origin lane yields different paths; but several distinct pairs in one sequence can activate several predicates. These are spatial proxies and do not establish receiver-identity linkage. |
| `seq_pass_exchange` with spatial patterns and actions | Probable co-occurrence, independent definition | The back-and-forth exchange is verified with receiver identifiers; spatial features rely on path and lane. Neither implies the other. |
