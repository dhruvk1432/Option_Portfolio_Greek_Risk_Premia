# Reproducibility Note

Paper: **Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia**.

## Commands


Before running the commands below on a fresh machine, copy `.env.example` to `.env` and inspect the data acquisition plan:

```bash
.venv/bin/python -m data_pull.pull --preset option-paper
```

This dry run records expected data paths and selected vendor jobs without downloading or exposing credentials.

From the repository root:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage all
cd research/papers/option_only_markowitz
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
bibtex option_only_portfolio_optimization_dhruv_kohli
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
```

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_option_only_markowitz_model.py -q
```

## Data

The empirical pipeline uses local, offline files:

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/option_greek_quality.csv`
- `data/feature_store/option_greek_assumptions.md`
- `data/universe/multi_raw_close.csv`
- `data/databento_cache/opra_vix_chain_*.parquet`
- `data/universe/vx_futures_daily.parquet`
- `data/universe/vix_complex.parquet`

Gross option returns are computed from selected-option listed-expiry payoff returns times
signed premium/NAV weights, so option premium cost is included. The publication run also
writes a conservative post-cost research simulation with spread crossing, explicit fees,
slippage, borrow, capacity penalties, simulated margin funding, and short-option
assignment-risk flags. This is not a live broker fill model. The empirical pipeline uses prior-date
option selections, raw daily underlying closes on listed expiry dates, split factors that
convert terminal equity closes back into decision-date contract units, and VX-forward
Black-76 Greeks for VIX options. Exact VRO/SOQ settlement is pulled from public Cboe
settlement endpoints by `python -m data_pull.pull --preset validate --jobs public-vro-soq
--execute`, or supplied through `OPTION_MARKOWITZ_VRO_DIR` / `OPTION_MARKOWITZ_VRO_FILE`.
When exact settlement is absent or incomplete, VIX option expiry rows are labeled as
`vix_close_settlement_proxy` and VIX results are diagnostic rather than headline-grade. No
live API key is read or used by the empirical pipeline.

## Generated Outputs

- Core tables: `tables/data_summary.tex`, `tables/timing_diagnostics.tex`,
  `tables/trading_data_audit.tex`, `tables/portfolio_performance.tex`,
  `tables/portfolio_performance_diagnostics.tex`, `tables/risk_calibration.tex`,
  `tables/approximation_diagnostics.tex`
- Equity-drift and premium-accounting diagnostics: `tables/exposure_summary.tex`,
  `tables/greek_exposure_summary.tex`, `tables/factor_regression.tex`,
  `tables/pnl_attribution.tex`, `tables/regime_performance.tex`,
  `tables/leave_one_out.tex`
- Claim audit: `tables/claim_strength_summary.tex`, `tables/claim_audit.tex`
- Figures: `figures/portfolio_growth.pdf`, `figures/portfolio_growth_all_strategies.pdf`,
  `figures/random_sharpe_histogram.pdf`, `figures/risk_calibration.pdf`,
  `figures/regime_sharpes.pdf`, `figures/leave_one_out_sharpe.pdf`
- Machine-readable summary: `tables/empirical_summary.json`
- Gross returns: `artifacts/strategy_returns.csv`
- Post-cost returns: `artifacts/strategy_returns_post_cost.csv`
- Cost, capacity, margin, assignment, settlement, and inference ledgers:
  `artifacts/cost_ledger.csv`, `capacity_ledger.csv`, `research_margin_ledger.csv`,
  `assignment_risk_ledger.csv`, `vix_settlement_coverage.csv`, `inference_summary.csv`
- Pre-production cost and viability ledgers:
  `artifacts/net_strategy_returns_by_cost_scenario.csv`,
  `artifacts/required_capital_returns.csv`, `artifacts/cost_scenario_ledger.csv`,
  `artifacts/rejected_trade_ledger.csv`, `artifacts/hurdle_selection_ledger.csv`,
  `artifacts/no_trade_periods.csv`, `artifacts/liquidity_tier_performance.csv`,
  `artifacts/forecast_ablation_performance.csv`, `artifacts/post_cost_survival.csv`,
  and `artifacts/reality_check_inference.csv`
- Weights, timing diagnostics, trading-data audit, split adjustments, regressions,
  attribution, regimes, and leave-one-out diagnostics: `artifacts/*.csv`
- Environment and hashes: `environment_lock.json`, `artifact_hash_manifest.csv`

The paper bibliography cites only actual papers or books. Operational
data-source URLs and local provenance files are documented here for
reproducibility, not used as scholarly references.

As of the current run, the filtered equity panel has 160,315 option rows; the VIX option
filter keeps 103,650 raw rows before monthly representative selection.  The combined
monthly universe has 54 rolling option-bucket assets, 129 monthly snapshots, and a 69/60
train/test monthly split.  VIX headline rows use exact public Cboe VRO/SOQ settlement when
the verifier marks them eligible.  Gross, legacy post-cost, and pre-production executable
cost scenarios are separated.  The new cost-scenario artifacts report midpoint,
half-spread, and full-spread returns, required-capital returns, rejected/no-fill rows,
liquidity-tier diagnostics, forecast ablations, post-cost survival, and PSR/DSR
reality-check inference.  These outputs are still research simulations; they are not live
broker fills, broker margin previews, or order-routing evidence.

## Production-Promotion Harness

The production-promotion layer is separate from the research backtest:

```bash
.venv/bin/python -m pytest tests/test_option_portfolio_production.py -q
.venv/bin/python -m src.option_portfolio_production.verification \
  --paper-root research/papers/option_only_markowitz
```

The second command is expected to fail on the current research artifacts until production
ledgers are supplied.  Critical production inputs are exact `vro_soq_exact` VIX settlement,
order and fill ledgers, explicit fees/slippage, margin previews, assignment/corporate-action
ledgers, vendor quote reconciliation, and broker position reconciliation.  This failure is
intentional: the production verifier prevents the paper's proxy-settlement, pre-cost
research run from being mislabeled as executable trading evidence.

## Known Limitations

- Rolling buckets are not permanent listed contracts.
- Expiry payoff uses raw daily underlying closes, not same-contract
  option liquidation quotes.
- Expected returns are estimated from training means and are noisy.
- The Greek factor approximation leaves material residual option-return
variation.
- The exercise validates the option-only Markowitz covariance machinery;
it is not an execution-ready trading strategy.

## Publication Replication Package

See `docs/replication_package.md` for exact commands, exact VRO/SOQ configuration, artifact
hashes, environment lockfile details, and the Databento/OPRA data-availability note. Raw
licensed OPRA data are not redistributed; a licensed local user can reproduce by placing
files under the documented local paths.
