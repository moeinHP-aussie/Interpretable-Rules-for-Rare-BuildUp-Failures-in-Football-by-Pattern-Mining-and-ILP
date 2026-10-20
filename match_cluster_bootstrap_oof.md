# Match-cluster bootstrap of archived out-of-fold results

This report reconstructs sequence-level OOF predictions from saved fold-specific models and Prolog facts. The reconstructed confusion counts exactly match the archived reports for both teams and all four procedures.

We ran 20,000 percentile-bootstrap replicates (seed 20261019 for Barcelona and 20261020 for Leverkusen). Matches were sampled with replacement within their original test fold, and all sequences from each sampled match were retained. The same match draws were used for every method to calculate paired lift differences. The fitted fold models and split were fixed; intervals are conditional on those fits and do not include model-refitting uncertainty.

| Team | Procedure | Lift | 95% match-cluster bootstrap CI | Matches flagged / evaluated | Largest share from one match |
|---|---|---:|---:|---:|---:|
| Barcelona 2020/21 | hybrid | 1.89 | [1.48, 2.33] | 28 / 35 | 10.1% |
| Barcelona 2020/21 | miner only | 2.03 | [1.26, 2.84] | 25 / 35 | 8.5% |
| Barcelona 2020/21 | popper no patterns | 1.64 | [1.20, 2.13] | 23 / 35 | 11.0% |
| Barcelona 2020/21 | cart | 1.44 | [0.95, 1.93] | 24 / 35 | 8.3% |

### Paired lift differences for Barcelona 2020/21

Difference is hybrid lift minus comparator lift; the same match bootstrap draws were used for both methods.

- Hybrid minus miner only: 95% CI [-0.75, 0.45].
- Hybrid minus popper no patterns: 95% CI [-0.28, 0.78].
- Hybrid minus cart: 95% CI [-0.10, 1.05].

### Bayer Leverkusen 2023/24

| Procedure | Lift | 95% match-cluster bootstrap CI | Matches flagged / evaluated | Largest share from one match |
|---|---:|---:|---:|---:|
| hybrid | 1.26 | [0.63, 1.90] | 26 / 31 | 9.6% |
| miner only | 1.36 | [0.72, 2.10] | 22 / 31 | 8.9% |
| popper no patterns | 1.73 | [1.09, 2.32] | 31 / 31 | 8.3% |
| cart | 1.56 | [0.72, 2.35] | 22 / 31 | 10.9% |

### Paired lift differences for Bayer Leverkusen 2023/24

Difference is hybrid lift minus comparator lift; the same match bootstrap draws were used for both methods.

- Hybrid minus miner only: 95% CI [-0.96, 0.74].
- Hybrid minus popper no patterns: 95% CI [-0.93, -0.003].
- Hybrid minus cart: 95% CI [-0.80, 0.21].

## Interpretation

For Barcelona, the hybrid and miner-only lift intervals are above 1; the CART interval includes 1. For Leverkusen, the hybrid interval includes 1. Paired intervals show no consistent hybrid advantage over the miner-only and CART procedures across both team-seasons. The Leverkusen hybrid-minus-pattern-free-Popper interval is slightly below zero; this is conditional on one fixed split and is not a general method ranking.

The number of matches with positive predictions and the maximum contribution from one match are descriptive summaries of the reconstructed OOF predictions. They show that the hybrid predictions were not concentrated in a single match in these runs.
