# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

Standalone publication repository for the paper:

```text
research/papers/option_only_markowitz/option_only_portfolio_optimization_dhruv_kohli.pdf
```

This project asks how an allocator should choose among listed calls and puts when every instrument is a funded, expiring, state-contingent cashflow rather than a standard asset-return column. The paper is theory-first: it develops the option-allocation mathematics up front, tests four locked E1 universes with artifact-backed figures, compact tables, real-world costs, liquidity caps, and robustness checks, and keeps detailed solver/proof mechanics in a compact technical appendix.

The contribution is a disciplined option allocation and validation framework. The empirical evidence is a point-in-time, pre-production research simulation. It does **not** claim live alpha, broker-executed performance, live margin parity, production tradability, or deployable option trading performance.

## Key Ideas

- **Options are cashflows, not tickers.** Positions are scaled by portfolio NAV and option premium; cash is collateral/numeraire, not an optimized risky asset.
- **Risk is induced through Greeks.** Option covariance is built from delta, gamma, vega, VIX-forward exposure, residual risk, and the covariance of underlying state factors.
- **Expected returns are conditional premia.** The research implementation separates carry/theta, variance-risk premium, skew/tail premia, VIX-regime effects, and relative-value ingredients.
- **VIX options are volatility instruments.** VIX option Greeks are anchored to VX futures/forwards, and expiry P&L is headline-grade only when exact VRO/SOQ settlement supports the relevant rows.
- **Validation gates matter.** Gross Sharpe is treated as a diagnostic; post-cost, settlement, beta/Greek attribution, liquidity/capacity, assignment-risk, inference, and no-lookahead checks determine what can be claimed.

## Headline Results

The paper evaluates one locked specification (E1: structural-only means, diagonal residual covariance, N-scaled covariance shrinkage, 75% shrink-to-zero, and pre-trade net liquidity caps at 5% of displayed training-window volume) on four universes at `$1M` NAV. Every book is rounded to whole option contracts before any performance number is computed. Full-cost net results over the 60-month test window:

| Universe | Gross → Net Sharpe | Net Sortino | Verdict |
|---|---|---|---|
| `orig` (8 equity names) | 0.975 → 0.778 | 1.783 | capacity diagnostic (cap budget below one NAV) |
| `orig+VIX` | 1.675 → 1.383 | 3.551 | pass |
| `larger` (56 names) | 0.847 → 0.587 | 1.298 | mixed (ties capped-naive options) |
| `larger+VIX` | 2.010 → 1.628 | 4.377 | pass |

Three findings frame the numbers:

- **The VIX-enabled books beat both baselines** — ordinary stock Markowitz on the same underlyings and capped naive option books under identical costs and caps. The optimizer edge over capped naive is `+2.273` and `+1.335` annualized Sharpe points (`p < 0.001`, Jobson–Korkie–Memmel with paired block bootstrap). The edge over stock Markowitz is positive in point estimate but not statistically resolvable on 60 monthly observations, and the paper says so directly.
- **The edge is carry, not volatility timing.** Channel ablations attribute the VIX-book performance to option carry disciplined by stress and vega budgets rather than directional volatility forecasts.
- **Capacity is a first-class result.** Displayed month-end option depth supports the strategy at roughly `$1M` NAV and refuses it at `$5M` and above, so this is a small-account allocation claim, not a scalable-fund claim.

## Robustness Layer

The locked candidates are validated in `analysis/artifacts/breadth_solutions/robustness/` with 12 blocked chronological groups and 66 CPCV splits run in two designs — a liquid-era (post-2018) design and a claim-window (post-2020) design, each with a one-month purge and embargo whose realized train/test gap the verifier measures on both sides of every test block — plus PBO, Monte Carlo resampled histories, refit-stability draws, repriced synthetic option universes, drawdown-breach and volatility-path simulations, reality-check and deflated-Sharpe inference, and true rolling 36-month out-of-sample refits (rolling net Sharpe `1.273` for `larger+VIX`, `1.033` for `orig+VIX`). Both CPCV designs are positive for all four books; the binding stress is the adversarial repriced no-premium overlay, and a zero-cost 56-equity stock-Markowitz baseline fails the same screens the option books pass. Run `make final-results` to regenerate the scoreboard, distribution, heatmap, capacity, and walk-forward exhibits from these ledgers.

Spread sourcing is separated by evidentiary weight: the eight baseline equity names use exact historical market-hours panel CBBO, while VIX rows and added broad names use a point-in-time inferred CBBO proxy calibrated from the liquid CBBO surface (about four-fifths of broad cost rows). The proxy rows are calibrated execution-sensitivity inputs, not final execution proof; exact market-hours NBBO/CBBO matched to every decision row remains the main execution-data caveat.

## Forward Shadow Layer

The free production-readiness bridge is broker-neutral shadow logging. Export the locked E1 target contract file, then combine it with market-hours NBBO/CBBO, displayed size, account NAV, optional broker margin previews, and optional rejection notes:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.export_shadow_targets \
  --config larger+VIX \
  --out /tmp/larger_vix_shadow_targets.csv

.venv/bin/python -m src.option_portfolio_production.shadow \
  --targets /tmp/larger_vix_shadow_targets.csv \
  --quotes /path/to/market_hours_quotes.csv \
  --nav 1000000 \
  --decision-time 2026-07-06T19:45:00Z \
  --out-dir /tmp/option_shadow_run
```

Shadow fills are labeled `shadow_nbbo_displayed_size_cross`. They are forward-validation evidence only and intentionally do not satisfy the strict production verifier.

## What Is Included

```text
Option_Portfolio_Greek_Risk_Premia/
├── data/                         # Public VRO/SOQ outputs plus licensed-data placeholders
├── data_ingestion/market_data/    # Public and licensed market-data pull helpers
├── data_pull/                     # Publication-facing data-pull CLI
├── research/papers/option_only_markowitz/
│   ├── option_only_portfolio_optimization_dhruv_kohli.pdf
│   ├── option_only_portfolio_optimization_dhruv_kohli.tex
│   ├── analysis/                  # Empirical runner, costs, inference, VIX panel, simulation
│   ├── artifacts/                 # Machine-readable generated outputs
│   ├── docs/                      # Source ledger and release notes
│   ├── figures/                   # Generated paper figures
│   ├── sections/                  # Manuscript sections
│   ├── tables/                    # Generated LaTeX tables and empirical summary
│   └── verification/              # Independent audit harness and outputs
├── src/                           # Reusable option model and pre-production primitives
└── tests/                         # Focused reproducibility and publication tests
```

The included generated artifacts make the paper inspectable without redistributing licensed OPRA/Databento data. A full empirical rebuild requires licensed local inputs documented in [`data/README.md`](data/README.md).

## Quickstart

Create an environment and inspect the data plan:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m data_pull.pull --preset option-paper
```

The default data-pull command is a dry run. It writes a manifest showing expected inputs, selected jobs, and credential-presence booleans. It does not download data, run paid jobs, or expose credential values.

Validate the included code and generated publication artifacts:

```bash
make test
.venv/bin/python -m research.papers.option_only_markowitz.verification.verify --skip-regenerate --skip-compile --skip-render
```

After licensed local data are available, rebuild the empirical outputs and paper:

```bash
make execution-audit
make paper
make verify
make test
```

## Reproducing Results

There are two practical reproducibility levels.

**Artifact-level verification** works with this repository as published. It checks the included paper outputs, ledgers, claim boundaries, and generated summaries:

```bash
make test
.venv/bin/python -m research.papers.option_only_markowitz.verification.verify --skip-regenerate --skip-compile --skip-render
```

**Full empirical regeneration** requires the licensed market-data files in `data/README.md` or the paid Databento jobs run under your own credentials:

```bash
make data-public      # public Cboe/VIX inputs, including VRO/SOQ settlement
make data-paid        # paid OPRA/Databento jobs; requires credentials
make paper            # regenerate artifacts and compile the PDF
make verify           # run the independent verifier
```

The exact replication command sequence and data-availability note are in:

```text
research/papers/option_only_markowitz/REPRODUCIBILITY.md
```

## Data Requirements And Rights

Raw OPRA/Databento files are not redistributed. The repository includes only generated research artifacts, code, documentation, placeholders, and normalized public Cboe VRO/SOQ settlement outputs:

```text
data/public/cboe/vro_soq/vro_soq_settlements.csv
data/public/cboe/vro_soq/vro_soq_download_audit.csv
data/public/cboe/vro_soq/vro_soq_manifest.json
```

For a full rebuild, provide or regenerate:

```text
data/feature_store/option_greek_proxy_panel.parquet
data/feature_store/opra_surface_panel.parquet
data/feature_store/option_greek_quality.csv
data/universe/multi_raw_close.csv
data/universe/vx_futures_daily.parquet
data/universe/vix_complex.parquet
data/databento_cache/opra_vix_chain_*.parquet
```

Do not commit raw licensed market data or `.env` files.

## Main Commands

```bash
make data-plan       # dry-run the full option-paper data plan
make data-validate   # check expected local input paths; no network calls
make data-public     # run public/free data pulls
make data-paid       # run paid Databento jobs with explicit credentials
make execution-audit # regenerate aggregate licensed-quote execution evidence
make paper           # regenerate artifacts and compile the paper
make verify          # run the independent paper verifier
make test            # run focused reproducibility and publication tests
make clean           # remove local Python/LaTeX intermediates
```

## Verification Standard

The full verifier (`make verify`) regenerates the fast empirical core, recompiles the PDF, and runs 435 independent checks covering generated outputs, point-in-time timing (including the two-sided CPCV purge/embargo gap), optimizer constraints, settlement coverage, cost ledgers, figure visibility, inference outputs, bibliography scope, PDF availability, and claim boundaries; `make test` runs 155 focused unit tests. In the public standalone package, raw licensed data checks are documented and replaced by artifact-level verification unless the licensed inputs are present locally.

The latest included verifier outputs are under:

```text
research/papers/option_only_markowitz/verification/
```

## Claim Boundary

The durable claim is that option-only portfolio construction can be framed as a premium-weighted Markowitz problem with explicit cashflow accounting, Greek-induced covariance, conditional option-risk-premium forecasts, and audit gates. The empirical results support framework usefulness and pre-production diagnostics in this sample and under these conventions.

Production deployment would require richer executable quote histories, broker-grade fills, margin previews, assignment and exercise handling, borrow, capacity, order routing, live data reconciliation, and broker position reconciliation.

## License

No open-source license has been assigned in this extraction. Add a license before accepting external reuse or contributions.
