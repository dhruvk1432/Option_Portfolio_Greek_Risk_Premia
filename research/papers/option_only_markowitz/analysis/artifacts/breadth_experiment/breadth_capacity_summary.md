# Breadth Capacity Experiment

**Superseded diagnostic.** This file records the first baseline capacity probe before the
estimator and net-liquidity-cap fixes. It is useful as the failure anchor, but it is not
the final breadth answer. Use `../breadth_solutions/README.md` and
`../breadth_solutions/p3_decision_table.md` for the current regularized+capped result.

New names present in panel: 48/48

| Universe | Underlyings | Contracts | $1M net Sharpe | $1M mean capacity cost | $1M max capacity ratio | Smallest AUM with net Sharpe > 0 | Highest AUM with net Sharpe > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n8 | 8 | 49 | -1.443 | 0.709629 | 184.870 | $100,000 | $250,000 |
| nAll | 55 | 262 | -2.055 | 1.508795 | 587.954 | n/a | n/a |

## Interpretation

Under this superseded baseline-only setup, nAll does not reduce the $1M mean monthly
capacity cost versus n8. nAll lowers survival versus n8 because nAll has no positive net
Sharpe in the sweep. The current breadth-solution run fixes the estimator and net
liquidity-cap formulation; with VIX, the selected 57-group E1 row now reaches gross Sharpe
1.915 and net Sharpe 1.499 at $1M, while hard full deployment remains infeasible above
small-account scale.

AUM sweep: $100,000, $250,000, $500,000, $1,000,000, $2,000,000.
