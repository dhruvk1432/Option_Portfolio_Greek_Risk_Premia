# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

Self-contained publication bundle for the paper
[`option_only_portfolio_optimization_dhruv_kohli.pdf`](option_only_portfolio_optimization_dhruv_kohli.pdf).

Paper display names map to internal artifact codes. **Sharpe Prototype** is E1,
**Survival Allocator** is R1, and **High Ceiling Allocator** is R1.1.

## What this is

The paper leads with one retrospective development result. The High Ceiling Allocator on
the eight-name plus VIX universe returns 16.0% annualized under frozen modeled costs. Its
Sortino ratio is 2.275 under those costs and 2.052 at observed touch fills with 94.1% entry
coverage in the licensed-quote audit. The result is not a live track record or a
confirmatory claim.

The research path begins with the Sharpe Prototype, repairs it through the Survival
Allocator, and ends with the High Ceiling Allocator. The repaired framework uses premium
weights and a complete joint covariance of Greek factors and Greek residuals, including
their cross terms. It allocates with cost-aware mean-variance utility, permits cash, and
enforces CVaR, stress, margin, collateral, assignment, whole-contract, and point-in-time
liquidity constraints before a position is accepted.

The empirical layer takes that clean theory to an **exchange-realistic book** through three
data-driven layers: a full transaction-cost stack sourced from historical market-hours
CBBO quotes, **volume-aware pre-trade liquidity caps** (each contract sized to 5% of its
displayed training-window volume), and **whole-contract integer execution** at the standard
100 multiplier. Every headline number is reported after all three are imposed.

All existing evidence is a retrospective development backtest, not an untouched holdout or
live track record. Confirmatory claims require 36 future untouched monthly observations.

The separately versioned R1.1 development arm raises the predicted-volatility ceiling to
25%, tests a sign-restricted 50% deployment target, and adds a close-to-next-open VIX-40
risk-off rule plus a gated EGARCH diagnostic. In the checked replay the deployment target
is never feasible. Whole-contract execution uses the direct conversion when feasible and
otherwise records a cash/no-trade month; no alternative portfolio is substituted. The
rejected conversion's diagnostics remain visible, EGARCH is not promoted, and the risk-off return
remains unscored because the required event-date licensed CBBO files are absent.

R2 is a separate diagnostic extension that combines 36-month and expanding complete joint
covariances, residual multiple imputation, a training-only HAR/EWMA volatility overlay,
robust zero-target Sortino direction, and net-log-growth scaling. It retains direct-or-cash
integer execution and all R1.1 survival limits. R2 becomes active only if every registered
historical, paired-bootstrap, repriced-state, and refit Monte Carlo gate passes.

## Legacy E1 development results (not R1 headline evidence)

| Universe | Net Sharpe | Gross Sharpe | Rolling OOS | vs. capped naive | Verdict |
|---|---|---|---|---|---|
| `orig` (8 names) | 0.778 | 0.975 | 0.585 | +0.41 | capacity-infeasible diagnostic |
| `orig+VIX` (8 + VIX) | 1.383 | 1.675 | 1.033 | +2.27 (p<0.001) | pass |
| `larger` (56 names) | 0.587 | 0.847 | 0.480 | −0.01 | mixed |
| `larger+VIX` (56 + VIX) | 1.628 | 2.010 | 1.273 | +1.34 (p<0.001) | pass |

These values document the specification-search path. E1 was selected after inspection of
the post-2020 window, omitted costs from allocation, and has VIX CPCV wealth paths absorbed
at zero in March 2020. Under R1's hard survival gate that is a failed verdict regardless of
positive arithmetic Sharpe.

## Repository layout

```
option_only_portfolio_optimization_dhruv_kohli.tex   LaTeX root (compiled paper)
option_only_portfolio_optimization_dhruv_kohli.pdf   compiled paper
sections/short_paper.tex                             main theory and model progression
sections/short_execution_audit.tex                   execution, robustness, and claim boundary
sections/short_appendix.tex                          technical appendix and relocated evidence
tables/, figures/                                    generated exhibits (never hand-edited)
references.bib                                        bibliography (@article/@book only)
analysis/                                            empirical pipeline (see REPRODUCIBILITY.md)
analysis/artifacts/breadth_solutions/robustness/     locked headline artifacts + scoreboard
artifacts/                                            core empirical ledgers
verification/                                         independent verifier + latest report
docs/                                                 release notes and data source ledger
REPRODUCIBILITY.md                                    exact commands to regenerate + verify
leakage.md                                            point-in-time integrity audit
```

`src/` (production/shadow layer), `data_pull/`, `data_ingestion/`, `tests/`, and the
`Makefile` live at the repository root.

## Quickstart

Run from the repository root after copying `.env.example` to `.env` and installing
`requirements.txt` into `.venv`.

```bash
make verify   # regenerate core artifacts, recompile the PDF, run the 435 audit checks
make test     # focused publication unit tests
make r1-repaired  # multi-hour monthly R1 development regeneration
make r11-higher-risk  # full 2018--2026 R1.1 replay and diagnostic artifacts
make r2-robust-sortino  # full R2 replay, locked simulation suites, and promotion gate
make r11-event-cbbo-plan  # cost estimate only; no licensed download
make execution-audit  # regenerate aggregate licensed-quote execution evidence
```

`make verify` is the authoritative end-to-end check. It passes only when data lineage,
point-in-time timing, optimizer constraints, settlement coverage, cost ledgers, every table
value (independently recomputed against the artifacts), the bibliography, and the compiled
PDF all agree. The latest report is in
[`verification/verification_report.md`](verification/verification_report.md).

To regenerate the headline breadth results from scratch (hours), see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Claim boundary

The durable claim is a disciplined option-allocation framework and its validation layer,
evaluated as a whole-contract backtest under realistic costs and volume-aware caps. The
paper does **not** claim live alpha, broker-executed fills, live margin parity, or
production tradability. Broad-universe and VIX spread rows that lack matched historical CBBO
use a point-in-time inferred proxy and are calibrated execution-sensitivity inputs, not
final execution truth. Production deployment would require broker-grade quotes, fills,
margin previews, assignment/exercise handling, borrow, capacity, routing, and
reconciliation. The `src/option_portfolio_production` verifier intentionally fails until
those ledgers are supplied.
