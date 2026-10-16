%% ===================================================================
%%  bk.pl - BACKGROUND KNOWLEDGE (phase 4)
%%
%%  HAND-WRITTEN Prolog RULES.  Unlike facts_all.pl and facts_tactical.pl
%%  (which are auto-generated ground facts), everything in this file is a
%%  live rule that derives new tactical concepts from those facts.
%%
%%  Load order:
%%      :- consult('facts_all.pl').        % phase 3 facts
%%      :- consult('facts_tactical.pl').   % phase 4 facts
%%      :- consult('bk.pl').               % this file
%%
%%  Naming convention: every rule head here is a concept from
%%  docs/tactical_concepts_glossary.md and carries its section number.
%%
%%  DESIGN CONSTRAINT (phase 4 pre-check + glossary v5):
%%  Rules stay low-arity and avoid introducing new entities.  Predicates
%%  intended for Popper's bias must be boolean or small-domain; raw numeric
%%  values are never exposed as rule arguments.
%% ===================================================================


%% ===================================================================
%% 1. PASS SHAPES  (glossary section 1)
%% ===================================================================

%% Pass-exchange evidence is stored under an internal helper name in
%% facts_tactical.pl. The public sequence predicate lives here exactly once;
%% current_predicate/1 keeps it safely false in folds with no exchange facts.
seq_pass_exchange(Sequence) :-
    current_predicate(pass_exchange_sequence/1),
    pass_exchange_sequence(Sequence).

%% PREFIX-ONLY CONVENTION: final_action/1 (generated in facts_all.pl) marks the
%% terminal action of each sequence -- the action that decides the outcome
%% label. Sequence-level concepts below and the seq_<concept> wrappers
%% (phase5_vocabulary.py) ignore it. last_retained_action/1 marks the action
%% just before it (look-ahead concepts are blocked there).

%% 1.1 line-breaking pass -- the ball travels from behind an opponent line
%% to beyond it.  The line itself comes from the 0.1 positional proxy and is
%% already resolved in facts_tactical.pl, so this rule only combines it with
%% success and pass-hood.
line_breaking_pass(A) :-
    breaks_defensive_line(A),
    successful(A),
    pass_like(A).

%% RETIRED FROM THE ACTIVE TACTICAL VOCABULARY (research decision, 2026-10):
%% midfield_line_breaking_pass(A) :-
%%     breaks_midfield_line(A),
%%     successful(A),
%%     pass_like(A).

%% 1.2 diagonal pass -- meaningful movement on BOTH axes.
%%
%% NOTE (bug found during phase 4 validation): the first version of this rule
%% matched pass_direction(A, forward_left) and friends.  Those symbols do not
%% exist -- pass_direction/2 only ever takes forward | sideways | backward, so
%% the rule silently returned zero solutions.  The two-axis test now lives in
%% Python (geometry.is_diagonal_pass) and arrives as diagonal_pass_shape/1.
diagonal_pass(A) :-
    diagonal_pass_shape(A),
    pass_like(A).

successful_diagonal_pass(A) :-
    diagonal_pass(A),
    successful(A).

%% 1.3 directional switch of play -- lane change plus >=25m lateral travel;
%% destination side is determined in Barcelona's fixed five-lane frame.
switch_to_right(A) :-
    switch_shape_to_right(A), pass_like(A), successful(A).
switch_to_left(A) :-
    switch_shape_to_left(A), pass_like(A), successful(A).

%% Height and distance are independent pass properties, sourced from StatsBomb
%% metadata and Euclidean SPADL geometry respectively.
%% A corner delivery may carry the StatsBomb High Pass tag, but it is a
%% restart rather than an aerial pass in open-play build-up. Keep it out of
%% this tactical predicate so it cannot enter the ordered-pattern vocabulary.
aerial_pass(A) :- high_pass_shape(A), pass_like(A), \+ set_piece_action(A).
long_pass(A) :- long_pass_shape(A), pass_like(A), \+ set_piece_action(A).

%% 1.5 pass in behind -- ball ends beyond the whole defensive line.
%% A pass-in-behind must finish in Barcelona's attacking final third or
%% penalty area. This distinguishes it from line-breaking passes ending
%% beyond the estimated line earlier in the pitch.
pass_in_behind(A) :-
    beyond_defensive_line(A),
    (final_third_destination(A); penalty_area_destination(A)),
    successful(A),
    pass_like(A).


%% 1.3 line pass -- stays in the same lane, that lane is a wide/half-space
%% corridor (not the central channel), and travels mainly along x.
%% NOTE: glossary numbering reused "1.3" for both this and switch_of_play in
%% earlier drafts; this project keeps "1.3 line pass" and "1.4 switch" as the
%% two distinct concepts the glossary itself separates in its final text.
line_pass(A) :-
    pass_like(A),
    successful(A),
    at_lane(A, L),
    wide_lane(L),
    same_lane(A),
    \+ sideways_only(A).

wide_lane(right_wing).
wide_lane(right_half_space).
wide_lane(left_half_space).
wide_lane(left_wing).

%% Compare the pass's own start/end lanes. Do not infer its recipient from
%% action_order(A)+1: synthetic dribbles and other intervening actions make
%% that next row an unreliable proxy for the pass endpoint.
same_lane(A) :-
    at_lane(A, L),
    pass_end_lane(A, L).

sideways_only(A) :- pass_direction(A, sideways).

%% switch_to_left/1 and switch_to_right/1 are mutually exclusive for one pass;
%% both differ from central_switch/1, which requires a two-pass route through
%% the central channel.

%% 1.6 connecting pass -- short, low-risk, not progressive: the opposite of a
%% line-breaking or progressive pass.
connecting_pass(A) :-
    pass_like(A),
    pass_length(A, short),
    \+ progressive(A),
    successful(A).

%% Phase-5 flank dynamics. dribble coordinates are SPADL locations; lane
%% names come from the five-lane geometry in features/geometry.py.
%% The carrying facts themselves are precomputed in facts_tactical.pl.
%% flank_line_carry/1, flank_cut_inside/1 and flank_dribble/1 are ground
%% facts emitted directly by tactical_facts.py.

%% Consecutive-pass central circulation. These spatial proxies do not prove
%% that the recipient of the first pass is the actor of the next pass: the
%% active SPADL representation has no general pass-recipient identifier.
%% A transition therefore means consecutive successful pass actions whose
%% endpoint/start lanes provide the stated shape, not verified one-player play.
left_flank_lane(left_wing).
left_flank_lane(left_half_space).
right_flank_lane(right_wing).
right_flank_lane(right_half_space).
%% Consecutive in the ordered stream of pass actions; intervening carries or
%% other non-pass actions are allowed, but an intervening pass is not.
next_pass_action(Prev, A, S) :-
    in_sequence(Prev, S), in_sequence(A, S),
    action_order(Prev, N), action_order(A, M), N < M,
    pass_like(Prev), pass_like(A),
    \+ (in_sequence(B, S), action_order(B, K), N < K, K < M, pass_like(B)).

central_switch(A) :-
    pass_like(A), successful(A),
    in_sequence(A, S), next_pass_action(Prev, A, S),
    pass_like(Prev), successful(Prev),
    pass_end_lane(Prev, central_channel), at_lane(A, central_channel),
    at_lane(Prev, From), pass_end_lane(A, To),
    left_flank_lane(From), right_flank_lane(To).
central_switch(A) :-
    pass_like(A), successful(A),
    in_sequence(A, S), next_pass_action(Prev, A, S),
    pass_like(Prev), successful(Prev),
    pass_end_lane(Prev, central_channel), at_lane(A, central_channel),
    at_lane(Prev, From), pass_end_lane(A, To),
    right_flank_lane(From), left_flank_lane(To).

recycle_to_same_flank(A) :-
    pass_like(A), successful(A),
    in_sequence(A, S), next_pass_action(Prev, A, S),
    pass_like(Prev), successful(Prev),
    pass_end_lane(Prev, central_channel), at_lane(A, central_channel),
    at_lane(Prev, From), pass_end_lane(A, To),
    left_flank_lane(From), left_flank_lane(To).
recycle_to_same_flank(A) :-
    pass_like(A), successful(A),
    in_sequence(A, S), next_pass_action(Prev, A, S),
    pass_like(Prev), successful(Prev),
    pass_end_lane(Prev, central_channel), at_lane(A, central_channel),
    at_lane(Prev, From), pass_end_lane(A, To),
    right_flank_lane(From), right_flank_lane(To).

center_cycling(A) :-
    pass_like(A), successful(A), at_lane(A, central_channel),
    pass_end_lane(A, central_channel),
    in_sequence(A, S), next_pass_action(Prev, A, S),
    pass_like(Prev), successful(Prev), at_lane(Prev, central_channel),
    pass_end_lane(Prev, central_channel).

%% ===================================================================
%% 2. OFF-BALL MOVEMENT AND ATTACKING POSITIONING  (glossary section 2)
%% ===================================================================

%% 2.3 drop between the (opponent) lines -- the Barcelona player carrying the
%% ball is himself positioned between the opponent defensive and midfield
%% lines (0.1 helper, positional proxy for both lines).
%%
%% NOTE on scope: only the actor can be checked this way.  A non-actor
%% teammate in a 360 frame carries no player_id at all (not even "teammate,
%% unnamed" -- see glossary 2.3b), so this rule cannot ask "is our defensive
%% midfielder between the lines", only "is whoever has the ball right now".
dropped_between_lines(A) :-
    drop_between_lines(A).


%% ===================================================================
%% 3. COMBINATIONS  (glossary section 3)
%% ===================================================================

%% RETIRED FROM ACTIVE PHASE 5: rapid_combination/1 is no longer generated.
%% rapid_combination_sequence(S) :- rapid_combination(S).

%% RETIRED FROM ACTIVE PHASE 5: ordered pass-link and chain facts are no
%% longer generated because give_and_go/3 and third_man/4 are not in any
%% current Popper vocabulary.
%% give_and_go(P1, P2, S) :-
%%     pass_link(P1, P2, S, N),
%%     N1 is N + 1,
%%     pass_link(P2, P1, S, N1),
%%     P1 \== P2.
%% third_man(P1, P2, P3, S) :-
%%     chain(P1, P2, P3, S),
%%     P1 \== P3.

%% 3.4 give-and-continue -- a player keeps reappearing in the sequence.
%% Boolean by construction (frequently_involved/2), never a raw count.
give_and_continue(P, S) :-
    frequently_involved(P, S),
    involved_in(P, S).

%% 3.6a triangulation -- the carrier and two nearby teammates form a real
%% triangle.  Reported independently of whether it traps anyone, so that the
%% event stays separable from its consequence (glossary 3.6 / 6.1 pattern).
triangulation(A) :-
    forms_triangle(A).

%% 3.6b trapped opponent -- at least one opponent stands inside a triangle.
triangle_trap(A) :-
    traps_opponent(A).

%% Retained vocabulary name: a passing triangle has trapped an opponent.
triangulation_opponent_trapped(A) :-
    traps_opponent(A).

%% several overlapping trapping triangles: a single triangle is often
%% accidental, a cage is deliberate.
rondo_cage(A) :-
    caged_opponent(A).

%% ===================================================================
%% 4. SUPPORT AND ATTACKING COVER  (glossary section 4)
%% ===================================================================

%% 4.1 support -- OPEN PASSING LANE, not distance and not passes_to/3.
%%
%% NOTE (explicit correction, glossary v3): an earlier version of this
%% predicate derived "support" from passes_to/3, which only reflects who
%% happened to receive the ball, not real geometric support.  has_open_support/1
%% comes from a proper point-to-segment distance test (features/tactical.py:
%% open_supporters) against every visible opponent.
supported_action(A) :- has_open_support(A).

%% 4.4 rest_defence_exposed/1 is emitted as a ground fact by tactical_facts.py
%% from the visible 360 frame. It is a proxy, not the full team's shape.

%% 4.3 dead-end support -- the carrier has no visible teammate with an open
%% passing lane in this action's valid 360 frame.  This is deliberately
%% separate from progressive/1: a sequence can fail to move forward even
%% when an open forward or diagonal option exists and is not used.
dead_end_support(A) :- no_open_support(A).

%% 4.5 passing options -- discretised count of open supporters.
%% RETIRED: the current vocabulary uses the observed open-lane booleans above;
%% discretised option-count facts are no longer produced.
%% few_passing_options(A) :- passing_options(A, few).
%% some_passing_options(A) :- passing_options(A, some).
%% many_passing_options(A) :- passing_options(A, many).


%% ===================================================================
%% 5. SPATIAL STRUCTURE AND PITCH OCCUPATION  (glossary section 5)
%% ===================================================================

%% 5.1 half-space occupation -- a player repeatedly touching the ball from a
%% half-space lane within one sequence (positional proxy, sequence-level).
persistent_half_space_occupier(P, S) :- half_space_occupier(P, S).

%% 5.2 / 5.3 visible-frame proxies; counts and shape omit unseen players.
%% RETIRED: overload, visible team shape, second-defender cover, and spacing
%% summaries are outside all current Popper vocabularies.
%% overload(A, Z) :- overload_zone(A, Z).
%% wide_team_shape(A) :- visible_team_shape(A, wide, _).
%% narrow_team_shape(A) :- visible_team_shape(A, narrow, _).
%% compact_team_shape(A) :- visible_team_shape(A, _, compact).
%% stretched_team_shape(A) :- visible_team_shape(A, _, stretched).

%% 2.3p local structural spacing, independent of role-based concepts.
%% Buckets summarize only the visible non-keeper footprint for each side.
%% structural_spacing(A, Side, Dimension, Bucket) :-
%%     structural_spacing_fact(A, Side, Dimension, Bucket).

%% ===================================================================
%% 7. DEFENSIVE PHASE  (glossary section 7)
%% ===================================================================
%% NOTE: this section is explicitly scoped to actions that already occur
%% INSIDE a build-up sequence (a tackle or interception that starts the
%% sequence), not a general defensive-phase model.  The glossary itself flags
%% section 7 as written for future phases beyond the current build-up focus.

%% 7.1 / 7.2 successful defensive actions that open a build-up sequence.
%% These already exist as action_type/2 ground facts; naming them explicitly
%% makes the tactical concept queryable on its own rather than hidden inside
%% a generic type check.
%% RETIRED: the defensive-action aliases are not generated for active Popper
%% search; sequence outcomes are analyzed from the current build-up features.
%% regains_possession(A) :- successful_tackle(A).
%% regains_possession(A) :- successful_interception(A).


%% ===================================================================
%% 8. PRESSING AND COUNTER-PRESSING  (glossary section 8)
%% ===================================================================

%% 8.3 counter-press -- StatsBomb's own flag, dumped as-is.
counter_pressed(A) :- counterpress_action(A).

%% 8.1 repeated heavy pressure on early build-up actions, as a team-press proxy.
%% RETIRED: sustained_opponent_pressure/1 is no longer generated.
%% sustained_high_press(S) :- sustained_opponent_pressure(S).

%% 8.4 press resistance -- a successful action taken under heavy pressure.
%% Outcome-based, exactly as the glossary defines it.
press_resistance(A) :-
    pressure_level(A, heavy_pressure),
    successful(A).


%% ===================================================================
%% 9. TRANSITION  (glossary section 9)
%% ===================================================================

%% 9.1 counter-attack -- numeric superiority in the acting team's opponent half.
%% 9.1b counter-attack potential -- the same superiority in its own half.
%% Half labels are precomputed relative to the acting team, not inferred from
%% Barcelona's defensive-third bucket.  Both concepts remain coverage-limited
%% proxies because a 360 frame does not always show every outfield player.
%% CAVEAT (glossary-flagged): the 360 frame does not always show every
%% outfield player, so this is a coverage-limited proxy, not an exact count.
%%
%% Counts from the previous defensive-third implementation do not apply to
%% these team-relative half rules; regenerate phase-4 facts before measuring.
%% RETIRED: relative-half and numeric-superiority facts are not emitted in the
%% active build-up representation.
%% counter_attack(A) :- numeric_superiority(A), opponent_half(A).
%% counter_attack_potential(A) :- numeric_superiority(A), own_half(A).


%% ===================================================================
%% 12. GOALKEEPER CONCEPTS  (glossary section 12)
%% ===================================================================

%% 12.1 playing out with the keeper -- event-data proxy: a goalkeeper's first
%% action is a successful short/medium goal kick.  This does not prove the
%% keeper is a sweeper or describe their position outside the goal area.
keeper_build_up_start(A) :-
    action_order(A, 0),
    actor(A, P),
    has_role(P, goalkeeper),
    action_type(A, goalkick),
    successful(A),
    non_long_goal_kick(A).

%% A keeper claim means the goalkeeper caught/collected the ball; it does not
%% show that they played out with their feet.  A short or medium goal kick is
%% the available event-data proxy for a deliberate short build-up.
non_long_goal_kick(A) :- pass_length(A, short).
non_long_goal_kick(A) :- pass_length(A, medium).

short_or_medium_pass(A) :- pass_length(A, short).
short_or_medium_pass(A) :- pass_length(A, medium).


%% ===================================================================
%% 7.6 / 10.1 / 10.2 / 10.4 — recorded events and restart types
%% ===================================================================

%% 7.5 opponent defensive-line height, estimated from visible defenders.
%% >= one third of the pitch from its own goal is high; closer is low.
%% Freeze-frame coverage is incomplete, so this is a positional proxy.
%% RETIRED: line-height and line-alignment outputs are not produced by the
%% current phase-4 extractor.
%% high_defensive_line(A) :- opponent_line_height(A, high).
%% low_defensive_line(A) :- opponent_line_height(A, low).
%% coordinated_offside_line(A) :- offside_line_alignment(A, aligned).
%% broken_offside_line(A) :- offside_line_alignment(A, stretched).

%% 7.4 local cover proxy: a second visible defender is goal-side of the first.
%% second_defender_in_cover(A) :- second_defender_cover(A).

%% The event result proves an offside was called; it does not prove why the
%% defensive line produced it, so this is intentionally not named an
%% "offside trap" rule.
offside_event(A) :- action_result(A, offside).

corner_action(A) :- action_type(A, corner_short).
corner_action(A) :- action_type(A, corner_crossed).

free_kick_action(A) :- action_type(A, freekick_short).
free_kick_action(A) :- action_type(A, freekick_crossed).
free_kick_action(A) :- action_type(A, shot_freekick).

throw_in_action(A) :- action_type(A, throw_in).

keeper_save(A) :- action_type(A, keeper_save).
keeper_save_successful(A) :- keeper_save(A), successful(A).
keeper_save_beaten(A) :- keeper_save(A), failed(A).

%% 13.2 empirical, prior-smoothed risk buckets; no raw probabilities in ILP.
%% RETIRED: action-risk buckets are not computed for the active vocabulary.
%% low_risk_action(A) :- action_risk(A, low).
%% medium_risk_action(A) :- action_risk(A, medium).
%% high_risk_action(A) :- action_risk(A, high).

%% 13.3 discretised StatsBomb xG (only available on shot events).
%% Build-up sequences in the current dataset contain no terminal shot actions,
%% so facts_tactical.pl may not define shot_quality/2 at all. Guard the
%% optional fact predicate so these concepts return zero instead of raising
%% an unknown-procedure error. Future datasets with shot facts use them as usual.
%% RETIRED: xG buckets are not computed for the current build-up vocabulary.
%% shot_quality_fact(A, Bucket) :- current_predicate(shot_quality/2), shot_quality(A, Bucket).
%% low_quality_shot(A) :- shot_quality_fact(A, low).
%% medium_quality_shot(A) :- shot_quality_fact(A, medium).
%% high_quality_shot(A) :- shot_quality_fact(A, high).
%% very_high_quality_shot(A) :- shot_quality_fact(A, very_high).

%% 11.x match/team structure; sequence links are available for ILP queries.
%% RETIRED: match formation, role-shift, false-nine, possession-share, and
%% positional-rotation metadata are not used by the active sequence models.
%% match_formation(S, F) :- sequence_match(S, M), nominal_formation(M, F).
%% formation_role_shift(S, R, D) :- sequence_match(S, M), role_group_shift(M, R, D).
%% false_nine(S, P) :- sequence_match(S, M), false_nine_candidate(M, P).
%% possession_share_bucket(M, B) :- possession_share(M, B).
%% positional_rotation(S) :- positional_rotation_proxy(S).

%% Descriptive sequence-level facts are emitted by the phase-4 extractor:
%% sustained_opponent_pressure/1, defensive_transition/1,
%% defensive_transition_danger/1 and immediate_exploitation/1.

set_piece_action(A) :- corner_action(A).
set_piece_action(A) :- free_kick_action(A).
set_piece_action(A) :- throw_in_action(A).
set_piece_action(A) :- action_type(A, goalkick).


%% ===================================================================
%% 6. RHYTHM, TIMING, DECISION MAKING  (glossary section 6)
%% ===================================================================

%% 6.1 hesitated/1 was retired from the phase-5 validation list because the
%% active Popper and sequential-mining vocabularies do not consume this alias.

%% hesitation that actually cost possession -- kept as a SEPARATE predicate
%% rather than folded into the definition above.
%%
%% NOTE (found during phase 4 validation): the obvious version,
%%     costly_hesitation(A) :- hesitation(A), failed(A).
%% is structurally impossible in this dataset and always returns zero.  All
%% 198 failed actions are the LAST action of their sequence (a sequence ends
%% when possession is lost), and dwell time is measured as the gap to the NEXT
%% action -- so a failed action never has a dwell time, and therefore never
%% carries hesitation/1.
%%
%% The meaningful formulation is one step earlier: the player hesitated, and
%% the sequence died on the very next action.
costly_hesitation(A) :-
    hesitation(A),
    in_sequence(A, S),
    action_order(A, N),
    N1 is N + 1,
    in_sequence(B, S),
    action_order(B, N1),
    failed(B).

%% 6.2 tempo -- sequence-level, discretised.
fast_sequence(S) :- tempo(S, fast).
slow_sequence(S) :- tempo(S, slow).

%% RETIRED FROM ACTIVE SEARCH: the former tempo_break(S) combined a slow
%% sequence bucket with any one-touch action, not a measured tempo transition.

%% 6.3 one-touch play
first_time_play(A) :-
    one_touch(A).

%% 6.3b a one-touch action that also breaks a line: the quick combination
%% that actually hurts.
quick_line_break(A) :-
    one_touch(A),
    breaks_defensive_line(A),
    successful(A).

%% 6.4 time on the ball
dwelt_long(A) :- dwell_time(A, long).
released_instantly(A) :- dwell_time(A, instant).

%% RETIRED FROM ACTIVE SEARCH: pressured_dwell duplicated the long-dwell
%% concept with an extra approximate 360-pressure condition.

%% RETIRED FROM ACTIVE PHASE-5 VOCABULARY: no_successful_forward_pass_10m/1
%% was neither offered to Popper nor consumed by the sequential miner.
%% no_successful_forward_pass_10m(S) :-
%%     sequence(S),
%%     \+ (in_sequence(A, S), successful_forward_pass_10m(A)).

%% Sequence-level opponent majority: more than half of its eligible 360 frames
%% must show an opponent majority, with at least two eligible frames. Do not
%% existentially lift one snapshot.
opponent_visible_majority(S) :-
    sequence(S),
    aggregate_all(count,
        (in_prefix(A, S),
         opponent_visible_majority_eligible_action(A)), Eligible),
    aggregate_all(count,
        (in_prefix(A, S),
         opponent_visible_majority_action(A)), Majority),
    minimum_visible_majority_frames(Minimum),
    Eligible >= Minimum,
    Majority * 2 > Eligible.

keeper_played_open_play(S) :-
    in_prefix(A, S),
    keeper_open_play_pass(A),
    successful(A),
    short_or_medium_pass(A).


%% ===================================================================
%% CROSS-CUTTING HELPERS
%% ===================================================================

%% Result aliases are live rules over the single canonical action_result/2
%% fact instead of two extra ground facts per action in facts_all.pl.
successful(A) :- action_result(A, success).
failed(A) :- action_result(A, Result), Result \== success.

pass_like(A) :- action_type(A, pass).
pass_like(A) :- action_type(A, cross).
pass_like(A) :- action_type(A, freekick_short).
pass_like(A) :- action_type(A, freekick_crossed).
pass_like(A) :- action_type(A, goalkick).
pass_like(A) :- action_type(A, throw_in).
pass_like(A) :- action_type(A, corner_short).
pass_like(A) :- action_type(A, corner_crossed).
