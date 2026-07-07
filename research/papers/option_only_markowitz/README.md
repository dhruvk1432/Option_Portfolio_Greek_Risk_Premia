# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

This folder is the self-contained publication bundle for the paper:

```text
option_only_portfolio_optimization_dhruv_kohli.pdf
```

The paper is now a mid-length theory-first version of the premium-weighted option-only Markowitz framework for listed calls and puts. Options are treated as funded, expiring, state-contingent cashflows rather than ordinary asset-return columns. The empirical layer tests the four locked E1 universes with exact/proxy spread-source labels, full costs, liquidity caps, walk-forward return paths, visual robustness summaries, and a production-readiness boundary. A compact technical appendix keeps the no-free-exposure, monotonicity, conic-solver, net-cap, and PSD-estimator details from the longer version without restoring the old table-heavy appendix. It is not broker-executed or live-trading evidence.

## What Is Included

- `option_only_portfolio_optimization_dhruv_kohli.tex`: LaTeX root.
- `option_only_portfolio_optimization_dhruv_kohli.pdf`: final compiled paper.
- `sections/`: the canonical paper body, compact technical appendix, and retained legacy long-form section files for audit/reference.
- `analysis/`: empirical pipeline, VIX panel construction, cost scenarios, inference, breadth/capacity diagnostics, and publication utilities.
- `tables/`, `figures/`, `artifacts/`: generated paper outputs and machine-readable ledgers.
- `verification/`: independent paper verifier and pass/fail reports.
- `docs/`: source ledger, replication package, and release notes.
- `REPRODUCIBILITY.md`: compact reproducibility note.

## Quickstart

Run from the repository root after creating `.env` from `.env.example` and installing `requirements.txt` into `.venv`.

```bash
# Dry-run the paper data plan; no downloads and no paid data pulls.
.venv/bin/python -m data_pull.pull --preset option-paper

# Optional: pull credential-free public inputs, including Cboe VRO/SOQ settlement.
make data-public

# Regenerate paper artifacts and compile the PDF.
make paper

# Run the independent verifier.
make verify

# Run focused publication tests.
make test
```

The canonical PDF path is:

```text
research/papers/option_only_markowitz/option_only_portfolio_optimization_dhruv_kohli.pdf
```

## Data Requirements

The paper is reproducible for a licensed local user, but raw OPRA/Databento data are not redistributed. A reproducer must supply or regenerate the expected local files:

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/option_greek_quality.csv`
- `data/universe/multi_raw_close.csv`
- `data/databento_cache/opra_vix_chain_*.parquet`
- `data/universe/vx_futures_daily.parquet`
- `data/universe/vix_complex.parquet`

Exact VRO/SOQ settlement can be downloaded from public Cboe inputs with `make data-public` or supplied through `OPTION_MARKOWITZ_VRO_FILE` / `OPTION_MARKOWITZ_VRO_DIR`. VIX option results are headline-grade only when every required VIX expiry row uses `vro_soq_exact` settlement.

## Verification Standard

The paper is considered release-ready only when these commands pass:

```bash
make paper
make verify
make test
```

The verifier records the current check count in `verification/verification_summary.json` and covers data lineage, point-in-time timing, optimizer constraints, settlement coverage, cost ledgers, figures, bibliography scope, PDF rendering, and claim boundaries, including independent recomputation of the reported performance metrics and ordering checks on every bootstrap confidence interval. The latest verification report is in `verification/verification_report.md`.

## New Research Diagnostics

The pipeline also emits repaired execution-sensitivity scenarios, a cost-aware Sortino variant, a CBBO spread cost surface, and VIX chain state diagnostics. The repaired scenarios use `_repaired` labels and are execution-sensitivity analysis only; they are deliberately excluded from headline growth tables and the reality-check family. The `Cost-aware Sortino + VIX` strategy uses train-window-only entry-cost estimates and is diagnostic rather than a headline or simulation strategy, although its gross and scenario columns expand the reality-check family. `make cbbo-surface` builds `data/feature_store/cbbo_spread_surface.parquet` from the local OPRA full-day CBBO cache when available, refining rows that otherwise fall back to class-default spreads. VIX chain features and vol-of-vol regime tables are diagnostics only and do not feed expected returns. All of these outputs remain research simulation evidence, not live tradability evidence.

### Breadth and Capacity Stage

The latest breadth stage writes `analysis/artifacts/breadth_solutions/`. It tests 8-name, 9-name-with-VIX, 56-name, and 57-name-with-VIX universes across estimator regularization, pre-trade liquidity caps, capped-naive benchmarks, inferred spread costs, and AUM scale. The 8-name no-VIX baseline is exact on equity-option spreads: `p3_spread_source_coverage.csv` reports 5,777 `panel_cbbo` cost rows across 49 asset IDs and all eight baseline underlyings. The 9-name-with-VIX baseline uses the same exact equity-option rows, while VIX option spread costs use the inferred liquid-option CBBO proxy.

Large-universe net rows are source-audited rather than blanket-adjusted. `p3_spread_source_coverage.csv` reports 23,740 added-name equity-option cost rows across 213 asset IDs and 47 underlyings using `inferred_cbbo_proxy`, with median relative spread `1.97%`; VIX proxy rows have median relative spread `2.53%`. At `$1M`, the 57-group E1 book reaches gross Sharpe `1.915` and net Sharpe `1.499`, versus `-1.837` net for the uncapped paper estimator and `0.266` for the best capped-naive benchmark. Without VIX, the 56-name optimizer is positive but essentially tied with capped equal-risk naive (`0.551` versus `0.550` net Sharpe). Full deployment remains capacity-infeasible above small-account scale. The old Cboe delayed-chain assumption file is retained only as an audit/rebuild utility; off-hours snapshots are rejected and are not used in the regenerated P1/P2/P3 breadth tables.

The breadth robustness runner writes `analysis/artifacts/breadth_solutions/robustness/` and locks the E1 capped candidate rather than reselecting knobs in test folds. It uses 12 chronological groups, 66 CPCV splits, 78 total CV splits per config, one-month purge/embargo, 1,000 resampled historical paths, 200 refit paths, 1,000 repriced synthetic paths, circular-block and GARCH-style path simulations, drawdown breach rates, reality-check inference, and rolling 36-month monthly OOS refits. The spread-source audit passes with zero current-Cboe rows and zero default rows. Static full-sample E1 status is: `orig` diagnostic capacity-infeasible, `larger` mixed, `orig+VIX` pass, and `larger+VIX` pass. The `larger+VIX` E1 book keeps net Sharpe `1.499`, net Sortino `4.004`, MC refit p05 net Sharpe `1.262`, MC resampled p05 net Sharpe `0.989`, and rolling net Sharpe `1.217`. Repriced synthetic net paths use a realized full-cost overlay, not synthetic NBBO/CBBO.

Run `make final-results` from the repository root after the breadth robustness artifacts exist. This builds the paper exhibit set: `figures/short_theory_flow.pdf`, `figures/short_four_variant_scoreboard.pdf`, `figures/short_walk_forward_return_paths.pdf`, `figures/short_validation_distributions.pdf`, `figures/short_robustness_heatmap.pdf`, and `figures/short_capacity_spread_panel.pdf`. The walk-forward figure compounds the rolling OOS full-cost returns for the four E1 variants, matched capped-naive option baselines, and underlying Markowitz. Those figures make the final claim boundary explicit: the two VIX-enabled E1 books beat both baselines, while `larger` no-VIX is mixed and `orig` no-VIX is capacity-infeasible. The older `final_*` charts are still emitted for compatibility.

The broad-name and VIX inferred-spread rows are calibrated execution-sensitivity inputs. They should be replaced by matched market-hours OPRA/NBBO or broker CBBO history before any live-trading claim. The paper now cites OPRA, Cboe DataShop, and IBKR operational references for this boundary.

### Forward Shadow Targets

The locked E1 target exporter and broker-neutral shadow runner are the free next step toward production readiness:

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

The shadow runner writes `shadow_target_ledger.csv`, `shadow_quote_ledger.csv`,
`shadow_execution_ledger.csv`, `shadow_fill_ledger.csv`, `shadow_margin_ledger.csv`,
`shadow_rejected_order_ledger.csv`, `shadow_reconciliation_ledger.csv`, and a shadow
summary. Shadow fills use `shadow_nbbo_displayed_size_cross`; they are not production
fills and do not satisfy `src.option_portfolio_production.verification`.

### Distributional Robustness Stage

Run `make robustness` to build the distributional-robustness layer. The stage takes roughly
35-40 minutes on the publication machine, uses fixed seeds recorded in
`tables/distributional_robustness_summary.json`, and writes `artifacts/cv_*.csv`,
`artifacts/mc_*.csv`, `tables/cv_*.tex`, `tables/mc_*.tex`, and the summary JSON. CPCV in
this layer is intentionally non-PIT: it is an overfitting and path-distribution diagnostic,
not a tradable out-of-sample claim.

## Claim Boundary

The durable claim is a disciplined option allocation framework and validation layer. The paper does not claim live alpha, production tradability, broker-executed fills, live margin parity, or deployable option trading performance. Production deployment would require broker-grade quotes, fills, margin previews, assignment/exercise handling, borrow, capacity, order routing, and reconciliation.
