# Publication Replication Package


## Data pull preflight

From the repository root, copy `.env.example` to `.env`, fill local keys/paths, and dry-run the acquisition plan:

```bash
cp .env.example .env
.venv/bin/python -m data_pull.pull --preset option-paper
```

Execute credential-free public inputs with:

```bash
.venv/bin/python -m data_pull.pull --preset public --execute
```

Execute paid Databento OPRA jobs only after reviewing the dry-run manifest:

```bash
.venv/bin/python -m data_pull.pull --preset option-paper --execute --allow-paid
```

Raw OPRA data are not redistributed. The command records file-status and credential-presence booleans in `research/reports/pipeline_reports/data_pull_manifest.json`, but never writes credential values.

## Exact commands

Run from the repository root.

```bash
# 1. Pull public Cboe VRO/SOQ settlement for the paper's VIX expiries.
.venv/bin/python -m data_pull.pull --preset validate --jobs public-vro-soq --execute

# Optional override when using a separately versioned exact-settlement file.
export OPTION_MARKOWITZ_VRO_FILE=data/public/cboe/vro_soq/vro_soq_settlements.csv

# 2. Optional but recommended: build the derived CBBO spread surface.
# Requires data/databento_cache/opra_surface_full_day_cbbo, usually a symlink
# to the sibling OPRA cache. If absent, the baseline empirical run falls back to
# class-default spreads where panel CBBO spreads are missing; the checked-in
# breadth net cells require this surface for the inferred CBBO proxy.
make cbbo-surface

# 3. Regenerate all empirical artifacts, gross and post-cost tables, inference, and hashes.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage all

# 4. Optional long-running distributional-robustness diagnostics.
# This is included in make option-paper and writes cv_*/mc_* artifacts.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage robustness

# 5. Optional breadth/capacity diagnostic.
# P3 reads the P1 ledger, so run these in order.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p2_liquidity_experiment --include-no-vix
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_p3_combined_experiment

# 6. Optional long-running breadth robustness validation for the four production candidates.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.breadth_robustness_experiment --configs all --out-dir research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/robustness

# 7. Compile the paper.
cd research/papers/option_only_markowitz
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
bibtex option_only_portfolio_optimization_dhruv_kohli
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
cd ../../../..

# 8. Run the independent verifier.
.venv/bin/python -m research.papers.option_only_markowitz.verification.verify
```

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_option_only_markowitz_model.py tests/test_option_only_markowitz_verification.py tests/test_option_only_publication_upgrade.py tests/test_cap_constrained_model.py -q -p no:cacheprovider
```

## Generated publication artifacts

- `tables/empirical_summary.json` contains every headline number used by the paper.
- `artifacts/strategy_returns.csv` contains gross research returns.
- `artifacts/strategy_returns_post_cost.csv` contains conservative post-cost research returns.
- `artifacts/vix_settlement_coverage.csv` determines whether VIX-enabled results are headline-grade or diagnostic.
- `artifacts/vix_settlement_audit.csv` records exact/proxy settlement coverage by VIX expiry.
- `artifacts/inference_summary.csv` contains bootstrap confidence intervals.
- `artifacts/cost_ledger.csv`, `capacity_ledger.csv`, `research_margin_ledger.csv`, and `assignment_risk_ledger.csv` contain implementation diagnostics.
- Repaired execution-sensitivity outputs include `artifacts/net_strategy_returns_by_cost_scenario_repaired.csv`, `cost_scenario_ledger_repaired.csv`, `rejected_trade_ledger_repaired.csv`, `required_capital_ledger_repaired.csv`, `repaired_trade_ledger.csv`, `execution_repair_diagnostics.csv`, and `execution_repair_comparison.csv`.
- Cost-aware Sortino and spread-source diagnostics include `artifacts/sortino_entry_costs.csv`, `artifacts/sortino_objective_diagnostics.csv`, and `artifacts/cost_input_spread_source_coverage.csv`.
- VIX chain and data-extension diagnostics include `artifacts/vix_chain_state_features.csv`, `artifacts/vol_of_vol_regime_performance.csv`, and `artifacts/data_extension_manifest.csv`.
- Matching diagnostic tables include `tables/execution_repair_diagnostics.tex`, `tables/execution_repair_comparison.tex`, `tables/sortino_objective_diagnostics.tex`, `tables/cost_input_spread_source_coverage.tex`, and `tables/vol_of_vol_regime_performance.tex`.
- Distributional-robustness artifacts include `artifacts/cv_fold_schedule.csv`,
  `cv_fold_ledger.csv`, `cv_split_is_oos.csv`, `cv_cpcv_path_metrics.csv`,
  `cv_cpcv_path_month_returns.csv`, `cv_pbo_summary.csv`, `cv_regime_performance.csv`,
  `cv_runtime_log.csv`, `cv_context_consistency.csv`, `mc_resampled_fixed_paths.csv`,
  `mc_resampled_summary.csv`, `mc_refit_paths.csv`, `mc_refit_summary.csv`,
  `mc_resampled_assumptions.csv`, `mc_repriced_paths.csv`, `mc_repriced_summary.csv`,
  `mc_repriced_paths_gauss_copula.csv`, `mc_repriced_summary_gauss_copula.csv`,
  `mc_repriced_assumptions.csv`, and `mc_universe_comparison.csv`.
- Distributional-robustness tables include `tables/cv_fold_performance.tex`,
  `cv_cpcv_distribution.tex`, `cv_regime_performance.tex`,
  `mc_resampled_universes.tex`, `mc_refit_stability.tex`,
  `mc_repriced_universes.tex`, `mc_universe_comparison.tex`, and
  `mc_repriced_assumptions.tex`, plus `tables/distributional_robustness_summary.json`.
- Breadth/capacity diagnostics include
  `analysis/artifacts/breadth_solutions/README.md`,
  `analysis/artifacts/breadth_solutions/p1_regularization_results.csv`,
  `p1_regularization_results.json`, `p1_summary.md`,
  `p2_liquidity_results.csv`, `p2_liquidity_results.json`,
  `p2_caps_detail.csv`, `p2_summary.md`,
  `p3_combined_results.csv`, `p3_combined_results.json`,
  `p3_spread_source_coverage.csv`, `p3_decision_table.md`,
  `current_option_spread_assumptions.csv`, and
  `current_option_spread_fetch_audit.csv`.
  Large-universe net cells use `inferred_cbbo_proxy` where historical CBBO is missing;
  the proxy is calibrated point-in-time from the historical liquid equity/ETF CBBO
  surface. The Cboe delayed-chain files are retained as optional audit/rebuild artifacts
  and are not consumed by the regenerated breadth tables. Gross and cap-budget diagnostics
  are the cleaner large-universe signals. The current `$1M` decision cell is `larger+VIX`
  E1 regularized/capped: gross Sharpe `1.915`, net Sharpe `1.499`, versus `-1.837` for
  the uncapped paper estimator and `0.266` for the best capped-naive benchmark. The
  no-VIX 56-name optimizer is positive but tied with capped equal-risk naive (`0.551`
  versus `0.550` net Sharpe).
- Breadth robustness validation artifacts include
  `analysis/artifacts/breadth_solutions/robustness/breadth_validation_summary.csv`,
  `breadth_validation_summary.json`, `breadth_validation_summary.md`,
  `breadth_spread_source_coverage.csv`, `breadth_realized_candidate_summary.csv`,
  `breadth_cv_fold_schedule.csv`, `breadth_cv_fold_ledger.csv`,
  `breadth_cv_cpcv_path_metrics.csv`, `breadth_cv_pbo_summary.csv`,
  `breadth_mc_resampled_summary.csv`, `breadth_mc_refit_summary.csv`,
  `breadth_mc_repriced_summary.csv`, `breadth_mc_repriced_assumptions.csv`,
  `breadth_simulation_summary.csv`, `breadth_drawdown_breach_rates.csv`,
  `breadth_reality_check_inference.csv`, `breadth_rolling_oos.csv`, and
  `breadth_rolling_oos_summary.csv`, plus compact paper tables
  `tables/breadth_robustness_summary.tex`, `breadth_robustness_cpcv.tex`,
  `breadth_robustness_pbo.tex`, `breadth_robustness_mc_resampled.tex`,
  `breadth_robustness_simulation.tex`, and `breadth_robustness_rolling_oos.tex`.
  The checked run uses 12 chronological groups, 66 CPCV splits, 78 total CV/PBO splits
  per config, one-month purge/embargo, and the corrected spread policy
  (`use_current_spread_assumptions=False`, `use_inferred_spread_proxy=True`). It reports
  zero current-Cboe rows and zero default-spread rows. Repriced synthetic net rows are
  historical full-cost overlays, not synthetic NBBO/CBBO quotes.
- `environment_lock.json` records Python and package versions.
- `artifact_hash_manifest.csv` records SHA-256 hashes for generated tables, figures, artifacts, and paper metadata.
- Forward shadow-trading utilities are code-only until a user supplies market-hours quote
  exports. `analysis/export_shadow_targets.py` writes locked E1 target-contract CSVs, and
  `src.option_portfolio_production.shadow` consumes target, quote, NAV, margin-preview,
  and rejection CSVs to write `shadow_*` ledgers. These ledgers are intentionally separate
  from production verification and cannot certify live tradability.

Expected numeric diffs versus pre-extension runs are limited to three channels: the
cost-aware Sortino variant expands the reality-check family, a present CBBO spread surface
can replace class-default spread assumptions for rows without panel CBBO spreads, and the
breadth-solution stage can infer missing large-name/VIX spreads from historical liquid
CBBO buckets. These changes can move reality-check and post-cost numbers without changing
gross option returns.

## Data availability and licensing

The code is reproducible for a licensed local user, but raw Databento/OPRA options data must not be redistributed through the replication package. The package provides schemas, code, generated output hashes, and local path conventions. A replicating user must obtain OPRA/Databento data under their own license and place it in the expected local data directories.

Exact VRO/SOQ files are downloaded from public Cboe settlement endpoints by `public-vro-soq`, or may be supplied locally through `OPTION_MARKOWITZ_VRO_DIR` or `OPTION_MARKOWITZ_VRO_FILE`. If exact settlement is absent or incomplete, VIX option rows remain proxy-labeled and VIX-enabled results are not headline-grade. Raw Databento/OPRA option data remains licensed and is not redistributed.
