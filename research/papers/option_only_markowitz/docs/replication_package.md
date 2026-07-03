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
# to the sibling OPRA cache. If absent, the empirical run falls back to
# class-default spreads where panel CBBO spreads are missing.
make cbbo-surface

# 3. Regenerate all empirical artifacts, gross and post-cost tables, inference, and hashes.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage all

# 4. Optional long-running distributional-robustness diagnostics.
# This is included in make option-paper and writes cv_*/mc_* artifacts.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage robustness

# 5. Compile the paper.
cd research/papers/option_only_markowitz
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
bibtex option_only_portfolio_optimization_dhruv_kohli
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
cd ../../../..

# 6. Run the independent verifier.
.venv/bin/python -m research.papers.option_only_markowitz.verification.verify
```

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_option_only_markowitz_model.py tests/test_option_only_markowitz_verification.py tests/test_option_only_publication_upgrade.py -q -p no:cacheprovider
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
- `environment_lock.json` records Python and package versions.
- `artifact_hash_manifest.csv` records SHA-256 hashes for generated tables, figures, artifacts, and paper metadata.

Expected numeric diffs versus pre-extension runs are limited to two channels: the cost-aware
Sortino variant expands the reality-check family, and a present CBBO spread surface can
replace class-default spread assumptions for rows without panel CBBO spreads. These changes
can move reality-check and post-cost numbers without changing gross option returns.

## Data availability and licensing

The code is reproducible for a licensed local user, but raw Databento/OPRA options data must not be redistributed through the replication package. The package provides schemas, code, generated output hashes, and local path conventions. A replicating user must obtain OPRA/Databento data under their own license and place it in the expected local data directories.

Exact VRO/SOQ files are downloaded from public Cboe settlement endpoints by `public-vro-soq`, or may be supplied locally through `OPTION_MARKOWITZ_VRO_DIR` or `OPTION_MARKOWITZ_VRO_FILE`. If exact settlement is absent or incomplete, VIX option rows remain proxy-labeled and VIX-enabled results are not headline-grade. Raw Databento/OPRA option data remains licensed and is not redistributed.
