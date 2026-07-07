# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

Self-contained publication bundle for the paper
[`option_only_portfolio_optimization_dhruv_kohli.pdf`](option_only_portfolio_optimization_dhruv_kohli.pdf).

## What this is

A premium-weighted Markowitz framework for books composed **entirely of listed options**.
Options are treated as funded, expiring, state-contingent cashflows rather than
asset-return columns: premium weights replace the unit budget, covariance is induced by a
Greek/state map plus regularized residual risk, expected returns are built from structural
risk-premium channels rather than historical contract means, and the allocation is a
robust conic program that carries costs, Greek/stress budgets, margin, and pre-trade
liquidity caps inside the feasible set.

The empirical layer takes that clean theory to an **exchange-realistic book** through three
data-driven layers: a full transaction-cost stack sourced from historical market-hours
CBBO quotes, **volume-aware pre-trade liquidity caps** (each contract sized to 5% of its
displayed training-window volume), and **whole-contract integer execution** at the standard
100 multiplier. Every headline number is reported after all three are imposed.

The evidence is a historical backtest, not a live track record; that is treated as a reason
to test the specification harder, not to state the supported conclusions more weakly.

## Headline results (four universes, \$1M NAV, full-cost net)

| Universe | Net Sharpe | Gross Sharpe | Rolling OOS | vs. capped naive | Verdict |
|---|---|---|---|---|---|
| `orig` (8 names) | 0.778 | 0.975 | 0.585 | +0.41 | capacity-infeasible diagnostic |
| `orig+VIX` (8 + VIX) | 1.383 | 1.675 | 1.033 | +2.27 (p<0.001) | pass |
| `larger` (56 names) | 0.587 | 0.847 | 0.480 | −0.01 | mixed |
| `larger+VIX` (56 + VIX) | 1.628 | 2.010 | 1.273 | +1.34 (p<0.001) | pass |

The two VIX-enabled books beat capped-naive option diversification with statistical
significance and exceed a stock-Markowitz baseline in point estimate (the stock gap is not
statistically resolvable on 60 months, and the paper says so). Capacity is a binding
result: displayed option depth supports these books at roughly \$1M NAV and refuses them at
\$5M.

## Repository layout

```
option_only_portfolio_optimization_dhruv_kohli.tex   LaTeX root (compiled paper)
option_only_portfolio_optimization_dhruv_kohli.pdf   compiled paper
sections/short_paper.tex, short_appendix.tex         the entire paper body + appendix
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
make verify   # regenerate core artifacts, recompile the PDF, run the 386 audit checks
make test     # focused publication unit tests
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
