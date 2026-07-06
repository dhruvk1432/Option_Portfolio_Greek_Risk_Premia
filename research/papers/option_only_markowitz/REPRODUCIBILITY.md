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
# Optional but recommended when the OPRA full-day CBBO cache is available.
make cbbo-surface

.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage all
.venv/bin/python -m research.papers.option_only_markowitz.analysis.regenerate_from_artifacts
cd research/papers/option_only_markowitz
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
bibtex option_only_portfolio_optimization_dhruv_kohli
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
```

To rebuild the derived statistics tables from the shipped return ledgers without re-running
the full data pipeline, use the artifact-level entry point:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.regenerate_from_artifacts
```

This regenerates the inference, reality-check, simulation, and diagnostics tables from
`artifacts/strategy_returns*.csv`. The current revision regenerated
`tables/reality_check_inference.tex`, `tables/inference_summary.tex`,
`tables/portfolio_performance_diagnostics.tex`,
`tables/portfolio_performance_net_diagnostics.tex`, `tables/simulation_summary.tex`,
`tables/simulation_assumptions.tex`, and `tables/drawdown_breach_rates.tex` with corrected
statistics: PSR/DSR are computed in per-month units, the reality check is a centered
max-statistic block bootstrap, wealth-path simulations absorb defaulted paths at zero and
report the defaulted-path share, and the bootstrap confidence-interval columns are
correctly ordered.

`make cbbo-surface` builds `data/feature_store/cbbo_spread_surface.parquet` from the local
`data/databento_cache/opra_surface_full_day_cbbo` directory. In this checkout that path is
expected to be a symlink to the sibling OPRA cache. If the symlink or cache is absent, the
surface parquet is absent and the baseline empirical pipeline transparently falls back to
class-default spreads for rows without panel CBBO spreads. The breadth-solution reruns use
this same derived surface to infer missing large-name and VIX spreads point-in-time; to
reproduce the checked-in breadth net cells, the surface must be present.

Distributional robustness is a separate long-running stage:

```bash
make robustness
```

It runs blocked k-fold/CPCV, resampled-universe, refit, and repriced synthetic-universe
diagnostics. Expected runtime is roughly 35-40 minutes. The stage is deterministic under
the fixed seeds written to `tables/distributional_robustness_summary.json`; determinism
was verified by double-run artifact comparison during the robustness-layer audit. The
outputs are listed in `docs/replication_package.md` and include `artifacts/cv_*.csv`,
`artifacts/mc_*.csv`, `tables/cv_*.tex`, `tables/mc_*.tex`, and the robustness summary
JSON.

The breadth/capacity diagnostic is a separate exploratory stage. Run the phases in order
because P3 reads the P1 regularization ledger:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p2_liquidity_experiment --include-no-vix
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p3_combined_experiment
```

Outputs are written to `research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/`.
The stage reuses the local OPRA-derived paper inputs and adds no new raw-data requirement.
Its large-universe net Sharpe rows are source-audited: `p3_spread_source_coverage.csv`
identifies rows where missing historical CBBO is filled from `inferred_cbbo_proxy`, a
point-in-time proxy calibrated from historical liquid equity/ETF CBBO buckets. The old
Cboe delayed-chain builder is retained only as an optional audit/rebuild utility; it must
be run during Cboe regular option-market hours, rejects weekend, holiday, and after-hours
snapshots, and refuses partial requested symbol coverage by default. Rebuild those
optional assumptions with:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.build_current_option_spread_assumptions
```

The regenerated P1/P2/P3 breadth tables do not consume the stale off-hours Cboe file.
Current Cboe snapshots remain useful as a live market-hours reasonableness check, but the
checked-in net cells are based on the offline inferred-CBBO proxy.

This does not affect the no-VIX 8-name equity baseline: the `orig` source-coverage row is
entirely `panel_cbbo` for equity-option spread inputs, with 5,777 rows, 49 asset IDs, and
all eight baseline underlyings covered. The `orig+VIX` baseline is exact for those eight
equity names; VIX option spread costs use the same inferred liquid-option CBBO proxy as
the large-universe missing rows.

The regenerated `$1M` decision table is the reference for the current breadth claim:
`larger+VIX` moves from `-1.837` net Sharpe in the uncapped paper configuration to
`+1.499` net Sharpe and gross `1.915` in the E1 regularized/capped row, beating the best
capped-naive book by `1.232`. The no-VIX `larger` row is positive but essentially tied
with capped equal-risk naive (`0.551` versus `0.550` net Sharpe).

The production-candidate breadth validation is a second long-running stage that locks the
E1 capped row and does not reselect knobs inside test folds:

```bash
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_robustness_experiment --configs all --out-dir research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/robustness
```

The checked run uses `nav=1_000_000`, `participation=0.05`, 12 chronological groups,
66 CPCV splits, 78 total CV/PBO splits per config, purge/embargo of one month, 1,000
resampled paths, 200 refit paths, 1,000 repriced paths, 1,000 path simulations, and
rolling 36-month monthly OOS refits. It writes
`breadth_validation_summary.{csv,json,md}`, `breadth_cv_*`, `breadth_mc_*`,
`breadth_simulation_summary.csv`, `breadth_drawdown_breach_rates.csv`,
`breadth_reality_check_inference.csv`, `breadth_rolling_oos*.csv`, and
`tables/breadth_robustness_*.tex`. The spread-source audit passes with zero
`current_cboe_liquid_quote` rows and zero `default` spread rows. Repriced synthetic net
paths subtract a circular-block sample of realized full-cost drag; they are a cost overlay,
not synthetic NBBO/CBBO.

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_option_only_markowitz_model.py -q
.venv/bin/python -m pytest tests/test_cap_constrained_model.py -q
.venv/bin/python -m pytest tests/test_breadth_robustness_experiment.py -q
```

Forward shadow trading is separate from both the research backtest and the production
verifier. Export locked E1 targets, supply market-hours NBBO/CBBO snapshots and optional
broker margin/rejection files, then run the broker-neutral shadow ledger:

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

The resulting `shadow_*` ledgers are forward-validation artifacts only. Fill rows must use
`shadow_nbbo_displayed_size_cross`, and the production verifier intentionally continues to
fail unless real settlement, order, fill, margin, assignment, quote-reconciliation, and
broker-position ledgers are supplied.

## Data

The empirical pipeline uses local, offline files:

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/cbbo_spread_surface.parquet` (optional derived CBBO spread surface)
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
- New diagnostic tables: `tables/execution_repair_diagnostics.tex`,
  `tables/execution_repair_comparison.tex`, `tables/sortino_objective_diagnostics.tex`,
  `tables/cost_input_spread_source_coverage.tex`, and
  `tables/vol_of_vol_regime_performance.tex`
- Figures: `figures/portfolio_growth.pdf`, `figures/portfolio_growth_all_strategies.pdf`,
  `figures/random_sharpe_histogram.pdf`, `figures/risk_calibration.pdf`,
  `figures/regime_sharpes.pdf`, `figures/vix_regime_sharpes.pdf`,
  `figures/leave_one_out_sharpe.pdf` (the regime figures clip Sharpe values below
  -8 with an annotated marker; `plot_regime_sharpes` derives regime labels from
  the data, fixing a previously empty VIX-regime panel)
- Machine-readable summary: `tables/empirical_summary.json`
- Gross returns: `artifacts/strategy_returns.csv`
- Post-cost returns: `artifacts/strategy_returns_post_cost.csv`
- Cost, capacity, margin, assignment, settlement, and inference ledgers:
  `artifacts/cost_ledger.csv`, `capacity_ledger.csv`, `research_margin_ledger.csv`,
  `assignment_risk_ledger.csv`, `vix_settlement_coverage.csv`, `inference_summary.csv`
- Pre-production cost and viability ledgers:
  `artifacts/net_strategy_returns_by_cost_scenario.csv`,
  `artifacts/net_strategy_returns_by_cost_scenario_repaired.csv`,
  `artifacts/required_capital_returns.csv`, `artifacts/cost_scenario_ledger.csv`,
  `artifacts/cost_scenario_ledger_repaired.csv`, `artifacts/rejected_trade_ledger.csv`,
  `artifacts/rejected_trade_ledger_repaired.csv`,
  `artifacts/required_capital_ledger_repaired.csv`,
  `artifacts/repaired_trade_ledger.csv`, `artifacts/execution_repair_diagnostics.csv`,
  `artifacts/execution_repair_comparison.csv`, `artifacts/hurdle_selection_ledger.csv`,
  `artifacts/no_trade_periods.csv`, `artifacts/liquidity_tier_performance.csv`,
  `artifacts/forecast_ablation_performance.csv`, `artifacts/post_cost_survival.csv`,
  and `artifacts/reality_check_inference.csv`
- Cost-aware Sortino and spread-source diagnostics:
  `artifacts/sortino_entry_costs.csv`, `artifacts/sortino_objective_diagnostics.csv`,
  and `artifacts/cost_input_spread_source_coverage.csv`
- VIX chain and data-extension diagnostics:
  `artifacts/vix_chain_state_features.csv`,
  `artifacts/vol_of_vol_regime_performance.csv`, and
  `artifacts/data_extension_manifest.csv`
- Distributional robustness diagnostics: `artifacts/cv_fold_schedule.csv`,
  `artifacts/cv_fold_ledger.csv`, `artifacts/cv_split_is_oos.csv`,
  `artifacts/cv_cpcv_path_metrics.csv`, `artifacts/cv_cpcv_path_month_returns.csv`,
  `artifacts/cv_pbo_summary.csv`, `artifacts/cv_regime_performance.csv`,
  `artifacts/cv_runtime_log.csv`, `artifacts/cv_context_consistency.csv`,
  `artifacts/mc_resampled_fixed_paths.csv`, `artifacts/mc_resampled_summary.csv`,
  `artifacts/mc_refit_paths.csv`, `artifacts/mc_refit_summary.csv`,
  `artifacts/mc_resampled_assumptions.csv`, `artifacts/mc_repriced_paths.csv`,
  `artifacts/mc_repriced_summary.csv`, `artifacts/mc_repriced_paths_gauss_copula.csv`,
  `artifacts/mc_repriced_summary_gauss_copula.csv`,
  `artifacts/mc_repriced_assumptions.csv`, `artifacts/mc_universe_comparison.csv`,
  and `tables/distributional_robustness_summary.json`
- Breadth/capacity diagnostic artifacts:
  `analysis/artifacts/breadth_solutions/README.md`,
  `analysis/artifacts/breadth_solutions/p1_regularization_results.csv`,
  `p1_regularization_results.json`, `p1_summary.md`,
  `p2_liquidity_results.csv`, `p2_liquidity_results.json`,
  `p2_caps_detail.csv`, `p2_summary.md`,
  `p3_combined_results.csv`, `p3_combined_results.json`,
  `p3_spread_source_coverage.csv`, `p3_decision_table.md`,
  `current_option_spread_assumptions.csv`, and
  `current_option_spread_fetch_audit.csv`
- Breadth robustness validation artifacts:
  `analysis/artifacts/breadth_solutions/robustness/breadth_validation_summary.csv`,
  `breadth_validation_summary.json`, `breadth_validation_summary.md`,
  `breadth_spread_source_coverage.csv`, `breadth_realized_candidate_summary.csv`,
  `breadth_cv_fold_schedule.csv`, `breadth_cv_fold_ledger.csv`,
  `breadth_cv_cpcv_path_metrics.csv`, `breadth_cv_pbo_summary.csv`,
  `breadth_mc_resampled_summary.csv`, `breadth_mc_refit_summary.csv`,
  `breadth_mc_repriced_summary.csv`, `breadth_mc_repriced_assumptions.csv`,
  `breadth_simulation_summary.csv`, `breadth_drawdown_breach_rates.csv`,
  `breadth_reality_check_inference.csv`, `breadth_rolling_oos.csv`,
  `breadth_rolling_oos_summary.csv`, and
  `tables/breadth_robustness_summary.tex`,
  `tables/breadth_robustness_cpcv.tex`, `tables/breadth_robustness_pbo.tex`,
  `tables/breadth_robustness_mc_resampled.tex`,
  `tables/breadth_robustness_simulation.tex`, and
  `tables/breadth_robustness_rolling_oos.tex`
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

Two numeric differences are expected relative to pre-extension runs. First, the
reality-check family expands because the cost-aware Sortino variant contributes gross and
cost-scenario columns, so PSR/DSR and family-wise reality-check values can move for every
variant. Second, when the CBBO spread surface is present, rows that previously used
class-default spreads may use measured surface spreads, so post-cost and cost-scenario
numbers can change even when gross returns are unchanged.

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
intentional: the production verifier prevents the paper's exact-settlement,
research-cost simulation run from being mislabeled as executable trading evidence.

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
