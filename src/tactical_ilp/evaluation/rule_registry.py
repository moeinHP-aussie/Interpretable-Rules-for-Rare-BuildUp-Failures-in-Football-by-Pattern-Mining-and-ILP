"""Current phase-6 registry.

Archived discoveries depend on retired features and are kept in historical
reports only. ``run_phase6.py`` reads the current Popper SOLUTION directly, so
no stale hypothesis is rescored on the revised vocabulary.
"""

from tactical_ilp.evaluation.rule_metrics import Rule

ALL_RULES: list[Rule] = []
