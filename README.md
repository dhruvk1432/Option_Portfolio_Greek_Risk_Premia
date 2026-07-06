# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

Standalone publication repository for the paper:

```text
research/papers/option_only_markowitz/option_only_portfolio_optimization_dhruv_kohli.pdf
```

This project asks how an allocator should choose among listed calls and puts when every instrument is a funded, expiring, state-contingent cashflow rather than a standard asset-return column. It builds a premium-weighted option-only Markowitz framework that maps each option into NAV-normalized cashflows, payoff and settlement conventions, Greeks, conditional option-risk-premium forecasts, Greek-induced covariance, implementation screens, and portfolio constraints.

The contribution is a disciplined option allocation and validation framework. The empirical evidence is a point-in-time, pre-production research simulation. It does **not** claim live alpha, broker-executed performance, live margin parity, production tradability, or deployable option trading performance.

## Key Ideas

- **Options are cashflows, not tickers.** Positions are scaled by portfolio NAV and option premium; cash is collateral/numeraire, not an optimized risky asset.
- **Risk is induced through Greeks.** Option covariance is built from delta, gamma, vega, VIX-forward exposure, residual risk, and the covariance of underlying state factors.
- **Expected returns are conditional premia.** The research implementation separates carry/theta, variance-risk premium, skew/tail premia, VIX-regime effects, and relative-value ingredients.
- **VIX options are volatility instruments.** VIX option Greeks are anchored to VX futures/forwards, and expiry P&L is headline-grade only when exact VRO/SOQ settlement supports the relevant rows.
- **Validation gates matter.** Gross Sharpe is treated as a diagnostic; post-cost, settlement, beta/Greek attribution, liquidity/capacity, assignment-risk, inference, and no-lookahead checks determine what can be claimed.

## Latest Breadth/Capacity Diagnostic

The newest diagnostic asks whether expanding from the eight-name equity-option universe to 56 names fixes the post-cost result. The baseline eight equity names use historical panel CBBO for spread inputs: the `orig` source-coverage ledger has 5,777 panel-CBBO cost rows across 49 asset IDs and all eight baseline underlyings. Missing added-name and VIX spread rows now use a point-in-time inferred CBBO proxy calibrated from the historical liquid equity/ETF CBBO surface, not stale off-hours Cboe snapshots or blanket 10%/15% class defaults.

The updated result is that breadth pays with VIX at `$1M` when the mean estimator is fixed and liquidity caps bind net contract positions. The 57-group E1 book (diagonal residual covariance, N-scaled shrinkage, structural-only mean) reaches gross Sharpe `1.915` and net Sharpe `1.499`; the old paper configuration remains negative at `-1.837` net and the best capped-naive benchmark reaches only `0.266`. Without VIX, the 56-name optimizer is positive but does not establish edge (`0.551` net versus `0.550` for capped equal-risk). The same run is also the capacity warning: above roughly `$2M`, representative month-end contracts do not support full deployment under the tested participation caps, and positive relaxed rows deploy only a trivial fraction of NAV. The machine-readable audit is under `research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/`.

The breadth robustness layer is now under `analysis/artifacts/breadth_solutions/robustness/`. It validates the locked E1 capped candidates for `orig`, `orig+VIX`, `larger`, and `larger+VIX` with 12 blocked chronological groups, 66 CPCV splits, purge/embargo of one month, 1,000 resampled paths, 200 refit paths, 1,000 repriced paths, path simulations, drawdown breach rates, reality-check inference, and true rolling 36-month monthly refits. The spread audit passes with zero `current_cboe_liquid_quote` and zero `default` rows. The static full-run rows classify `orig` as diagnostic capacity-infeasible, `larger` as mixed, and the VIX-enabled E1 books as passes; `larger+VIX` has net Sharpe `1.499`, net Sortino `4.004`, rolling net Sharpe `1.217`, MC refit net Sharpe p05 `1.262`, and MC resampled net Sharpe p05 `0.989`.

The paper conclusion now uses visual-first final scoreboards. Run `make final-results` to regenerate `figures/final_breadth_validation_distributions.pdf`, `figures/final_baseline_comparison.pdf`, and the underlying `final_result_scoreboard.csv`. The baseline chart compares the locked E1 option books against two simple benchmarks: ordinary underlying Markowitz and the best capped naive option allocation. The VIX-enabled books beat both baselines; the no-VIX books do not clear that stronger bar.

The broad-name spread rows are not final execution proof. Exact historical market-hours OPRA/NBBO or broker CBBO matched to every decision row remains the main execution-data caveat. The inferred broad-name and VIX spread rows are calibrated execution-sensitivity inputs; they are explicitly separated from the baseline eight-name historical panel-CBBO rows.

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
│   ├── docs/                      # Source ledger, replication package, release notes
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
research/papers/option_only_markowitz/docs/replication_package.md
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
make paper           # regenerate artifacts and compile the paper
make verify          # run the independent paper verifier
make test            # run focused reproducibility and publication tests
make clean           # remove local Python/LaTeX intermediates
```

## Verification Standard

The verifier checks generated outputs, point-in-time timing, optimizer constraints, settlement coverage, cost ledgers, figure visibility, inference outputs, bibliography scope, PDF availability, and claim boundaries. In the public standalone package, raw licensed data checks are documented and replaced by artifact-level verification unless the licensed inputs are present locally.

The latest included verifier outputs are under:

```text
research/papers/option_only_markowitz/verification/
```

## Claim Boundary

The durable claim is that option-only portfolio construction can be framed as a premium-weighted Markowitz problem with explicit cashflow accounting, Greek-induced covariance, conditional option-risk-premium forecasts, and audit gates. The empirical results support framework usefulness and pre-production diagnostics in this sample and under these conventions.

Production deployment would require richer executable quote histories, broker-grade fills, margin previews, assignment and exercise handling, borrow, capacity, order routing, live data reconciliation, and broker position reconciliation.

## License

No open-source license has been assigned in this extraction. Add a license before accepting external reuse or contributions.
