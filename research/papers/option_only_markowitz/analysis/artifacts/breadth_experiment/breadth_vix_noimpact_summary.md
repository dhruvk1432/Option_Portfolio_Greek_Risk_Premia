# Breadth/VIX No-Impact Experiment

**Superseded diagnostic.** This file records the first baseline breadth probe with the
paper estimator and no liquidity-cap fix. It is useful as the failure anchor, but it is not
the final breadth answer. Use `../breadth_solutions/README.md` and
`../breadth_solutions/p3_decision_table.md` for the current regularized+capped result.

New breadth names present in panel: 48/48

| Config | With VIX | Underlyings | Contracts | Gross Sharpe | Net Sharpe (impact removed) | Mean capacity cost |
|---|---:|---:|---:|---:|---:|---:|
| orig+VIX | True | 9 | 54 | 1.374 | 0.698 | 0.00000000 |
| larger+VIX | True | 57 | 267 | 0.765 | -0.804 | 0.00000000 |
| orig | False | 8 | 49 | 0.842 | 0.578 | 0.00000000 |
| larger | False | 56 | 262 | 0.456 | -0.923 | 0.00000000 |

## Read

Under this superseded baseline-only setup, breadth does not consistently improve both gross
and impact-free net Sharpe across the VIX and no-VIX settings: gross breadth deltas are
-0.610 with VIX and -0.386 without VIX, while impact-free net deltas are -1.502 and
-1.501. That negative result is the failure anchor. The current breadth-solution run fixes
the estimator and net liquidity-cap formulation; with VIX, the selected 57-group E1 row now
reaches gross Sharpe 1.915 and net Sharpe 1.499 at $1M.
