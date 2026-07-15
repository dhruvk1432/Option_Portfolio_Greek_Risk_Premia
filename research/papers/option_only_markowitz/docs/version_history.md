# Model and result version history

All figures below are retrospective development evidence. Results from different rows are
not directly comparable unless the evaluation window and endpoint also match.

## Legacy E1 (repository baseline before the repair)

E1 used the Greek loading matrix but omitted factor/residual covariance cross terms,
optimized maximum Sharpe without costs inside the objective, deducted costs during scoring,
and used a split-leg gross normalization that did not uniquely determine economically
deployed scale. E1 was selected after post-2020 results had been inspected.

For the 60-month January 2021--April 2026 rolling full-cost evaluation, the reported
annualized arithmetic returns were 19.7% (orig), 8.0% (larger), 87.2% (orig+VIX), and
112.2% (larger+VIX). Maximum drawdowns were -86.3%, -90.7%, -77.0%, and -63.5%,
respectively. The separate headline full-sample net Sharpes were 0.778, 0.587, 1.383,
and 1.628 in that universe order. The VIX-enabled CPCV wealth paths were absorbed at zero
in March 2020, so those books fail the repaired survival standard regardless of Sharpe.

## R1 repaired net utility (current principal frozen specification)

R1 added the complete joint Greek factor/residual covariance, cost-aware net utility,
cash, a 15% predicted-volatility target, CVaR, stress, margin, collateral, assignment,
liquidity, and whole-contract checks. It retained direct truncation toward cash and treated
an infeasible integer conversion as a survival failure. No such integer failure occurred
in R1's 60-month January 2021--April 2026 replay.

| Universe | Annualized geometric return | Terminal wealth | Maximum drawdown |
|---|---:|---:|---:|
| orig | 4.06% | 1.220 | -28.54% |
| larger | 3.37% | 1.180 | -25.00% |
| orig+VIX | 22.99% | 2.814 | -6.59% |
| larger+VIX | 19.14% | 2.401 | -7.17% |

R1 and its freeze manifest remain unchanged.

## R1.1 initial 25% direct-truncation version (superseded)

R1.1 raised the predicted-volatility ceiling to 25%, tested rather than forced a 50%
deployment target, added the VIX-40 execution design, and added the gated EGARCH diagnostic.
It used direct whole-contract truncation and held cash when the integer book was infeasible,
but still classified those cash months as integer failures. The artifact also overwrote the
rejected book's constraint values with zeros from the selected cash book.

For the January 2018--April 2026 replay, non-VIX results were 11.77% (orig) and 12.28%
(larger) annualized geometrically. The VIX results were approximately 23.00% (orig+VIX,
terminal wealth 4.976) and 21.95% (larger+VIX, terminal wealth 4.654). The VIX books were
marked as failing the survival gate because direct conversion was infeasible in roughly
16--24 months per arm. The return path held cash in those months, creating the apparent
contradiction of high reported returns and a failed execution verdict.

## R1.1 five-method integer-repair version (rejected)

This development version compared direct truncation, risk-leg removal, an additional VIX
contract, iterative constraint-guided reduction, and a sign-restricted mixed-integer conic
solve. It selected the feasible candidate with the highest checked net utility. This fixed
the hidden-diagnostic problem and produced feasible integer books, but it traded portfolios
different from the continuous target. The user rejected that behavior as economically
suboptimal relative to abstaining.

| Universe | Annualized geometric return | Terminal wealth | Maximum drawdown |
|---|---:|---:|---:|
| orig | 11.42% | 2.313 | -40.44% |
| larger | 12.41% | 2.475 | -38.91% |
| orig+VIX | 19.17% | 3.892 | -31.38% |
| larger+VIX | 18.06% | 3.621 | -23.23% |

Direct truncation was feasible in only 327 of 744 base/EGARCH decisions under that
version's candidate eligibility test. The mixed-integer candidate was selected 385 times.
These results are retained only as rejected development history and are not current.

## R1.1 direct-or-abstain interim bug (aborted before publication)

The first restoration attempt correctly removed substitute portfolios but mistakenly
treated any retained leg with negative standalone expected return after costs as a hard
infeasibility. That was wrong: a negative standalone-return leg may be an optimal portfolio
hedge. The partial replay produced 9.87% for orig with four abstentions and 4.09% for
orig+VIX with 64 abstentions. The larger universes were stopped before completion. These
numbers were never accepted as a model result.

## R1.1 corrected direct-or-abstain version (current development extension)

The current policy truncates the continuous target toward zero and trades that book only
when the complete portfolio satisfies volatility, CVaR, stress, margin, collateral,
liquidity, assignment, Greek, and concentration constraints. Standalone negative-edge hedge
legs are allowed. If the complete integer book is infeasible, the strategy holds cash for
that period. Cash is a valid decision, not a failure, and no alternative risky portfolio is
substituted. Separate columns preserve every rejected book's original diagnostics.

| Universe | Annualized geometric return | Terminal wealth | Maximum drawdown | Cash abstentions |
|---|---:|---:|---:|---:|
| orig | 11.77% | 2.368 | -38.11% | 0 / 93 |
| larger | 12.28% | 2.454 | -39.40% | 0 / 93 |
| orig+VIX | 16.01% | 3.160 | -17.05% | 32 / 93 |
| larger+VIX | 14.23% | 2.804 | -12.13% | 34 / 93 |

Across base and EGARCH arms, direct conversion is selected in 621 of 744 decisions and
cash in 123. Integer failures are zero. The risk-off arm remains unscored because matched
licensed event-date OPRA quotes are incomplete.

## 2026-07-15 paper restructure

This publication pass cut the paper from 44 pages to 20 pages. It added mechanism names
and a figure-first presentation that leads with the result. R2 was removed from the paper
only. All R2 pipelines, Make targets, regeneration instructions, and artifacts remain.

The verifier deliberately changed the paper page gate from 25-45 to 18-30. The R1
frozen-source check now exempts the two rewritten prose files,
`sections/short_paper.tex` and `sections/short_appendix.tex`, from the unchanged-hash
requirement. It still enforces exact frozen hashes on the executable policy sources. The
verification report records each prose exemption explicitly.

All reported numbers are unchanged. The freeze manifests were not modified.

## Git versioning convention

Each accepted model-policy change must be committed and pushed before another model-policy
change begins. Branches use `codex/option-paper-<version>-YYYYMMDD`; the commit message and
this file must identify the evaluation window, execution policy, artifact directory, and
verification status. Rejected experiments remain in the research-trial registry and must
not silently replace an accepted branch.
