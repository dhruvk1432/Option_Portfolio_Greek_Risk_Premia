# Source Ledger

This ledger maps the paper's main claim families to formal references and
machine-readable research outputs.

| Claim family | Evidence |
|---|---|
| Markowitz mean-variance and maximum-Sharpe framing | Markowitz (1952) |
| Local option pricing, replication, and Greeks | Black and Scholes (1973); Merton (1973); Black (1976); Shreve (2004) |
| Factor risk models, covariance shrinkage, and optimizer sensitivity | Ledoit and Wolf (2004); Grinold and Kahn (2000) |
| Performance ratios and backtest overfitting caution | Sharpe (1966); Sortino and Price (1994); Keating and Shadwick (2002); Magdon-Ismail and Atiya (2004); Grinold and Kahn (2000); Bailey et al. (2017); Lopez de Prado (2018) |
| OPRA/Databento data venue and public market-data provenance | OPRA overview; Cboe DataShop option EOD/NBBO summary; `REPRODUCIBILITY.md`; `data/README.md` |
| Greek panel provenance and quality | `data/feature_store/option_greek_assumptions.md`; `data/feature_store/option_greek_quality.csv` |
| Empirical tables and figures | `research/papers/option_only_markowitz/analysis/run_empirics.py`; `tables/empirical_summary.json`; `artifacts/*.csv` |
| Premium/NAV accounting and performance metrics | `src/portfolio/option_only_markowitz_model.py`; `tables/portfolio_performance.tex`; `tables/portfolio_performance_diagnostics.tex`; `tables/exposure_summary.tex`; `tables/greek_exposure_summary.tex`; `tables/empirical_summary.json` |
| Point-in-time timing and split-adjusted listed-expiry return construction | `tables/timing_diagnostics.tex`; `tables/trading_data_audit.tex`; `artifacts/holding_return_detail.csv`; `artifacts/timing_diagnostics.csv`; `artifacts/trading_data_audit.csv`; `artifacts/split_adjustments.csv` |
| Equity-drift diagnostics, factor betas, P&L attribution, regimes, and leave-one-out robustness | `tables/empirical_summary.json`; `artifacts/factor_regression.csv`; `artifacts/pnl_attribution.csv`; `artifacts/regime_performance.csv`; `artifacts/vix_regime_performance.csv`; `artifacts/leave_one_out.csv` |
| Conditional option expected returns, volatility risk premia, option premia, skew/tail premia | Coval and Shumway (2001); Bakshi and Kapadia (2003); Carr and Wu (2009); Bollerslev, Tauchen, and Zhou (2009); Broadie, Chernov, and Johannes (2009) |
| VIX option data construction and settlement status | `analysis/vix_option_panel.py`; `artifacts/vix_holding_return_detail.csv`; `artifacts/vix_data_audit.csv`; `tables/trading_data_audit.tex`; Cboe product/settlement pages documented operationally, not in bibliography |
| Conditional premia implementation | `analysis/conditional_premia.py`; `artifacts/conditional_premia_components.csv`; `tables/empirical_summary.json` |
| Breadth, estimator regularization, net liquidity-cap diagnostics, capped-naive benchmarks, spread-source audit, locked-candidate robustness validation, and final visual scoreboards | `analysis/breadth_solutions_lib.py`; `analysis/breadth_p1_regularization_experiment.py`; `analysis/breadth_p2_liquidity_experiment.py`; `analysis/breadth_p3_combined_experiment.py`; `analysis/breadth_robustness_experiment.py`; `analysis/build_final_results_summary.py`; `analysis/publication_costs.py`; `analysis/build_current_option_spread_assumptions.py`; `analysis/artifacts/breadth_solutions/*.csv`; `analysis/artifacts/breadth_solutions/*.json`; `analysis/artifacts/breadth_solutions/*.md`; `analysis/artifacts/breadth_solutions/p3_spread_source_coverage.csv`; `analysis/artifacts/breadth_solutions/robustness/breadth_validation_summary.csv`; `analysis/artifacts/breadth_solutions/robustness/breadth_cv_pbo_summary.csv`; `analysis/artifacts/breadth_solutions/robustness/breadth_rolling_oos_summary.csv`; `analysis/artifacts/breadth_solutions/robustness/final_result_scoreboard.csv`; `analysis/artifacts/breadth_solutions/robustness/final_validation_distribution_summary.csv`; `figures/final_breadth_validation_distributions.pdf`; `figures/final_baseline_comparison.pdf`; `tables/breadth_robustness_*.tex`; `tables/breadth_spread_source_methodology.tex`; `tables/final_result_scoreboard.tex`; `analysis/artifacts/breadth_solutions/current_option_spread_assumptions.csv`; `analysis/artifacts/breadth_solutions/current_option_spread_fetch_audit.csv`; `tests/test_cap_constrained_model.py`; `tests/test_breadth_robustness_experiment.py` |
| Rolling OOS and figure visibility audits | `tables/rolling_oos.tex`; `artifacts/rolling_oos.csv`; `artifacts/figure_visibility_audit.csv` |
| Forward shadow and production promotion layer: settlement, executable quotes, fills, margin, assignment, broker adapter, shadow verifier, and production verifier | `src/option_portfolio_production/*`; `src/option_portfolio_production/shadow.py`; `analysis/export_shadow_targets.py`; `tests/test_option_portfolio_production.py`; `tests/test_option_portfolio_shadow.py`; operational anchors: OPRA overview, Cboe DataShop option EOD/NBBO summary, IBKR free-trial delayed-data disclosure, Cboe VIX/VRO/SOQ pages, Databento OPRA timestamp/symbology docs, OCC ODD and margin resources |

| Claim-audit status | `tables/claim_audit.tex`; `artifacts/claim_audit.csv` |

Operational data-source URLs are documented in `REPRODUCIBILITY.md`, not in the scholarly
bibliography.

| Publication replication package, artifact hashes, environment lock, and OPRA redistribution boundary | `docs/replication_package.md`; `artifact_hash_manifest.csv`; `environment_lock.json`; operational provenance only, not scholarly bibliography |
| Exact VRO/SOQ public ingestion and VIX headline gating | `data_pull/cboe_vro_soq.py`; `analysis/vix_option_panel.py`; `artifacts/vix_settlement_coverage.csv`; `artifacts/vix_settlement_audit.csv`; Cboe VRO/SOQ operational convention recorded here, not in bibliography |
| Pre-production executable-cost scenarios, required capital, liquidity tiers, forecast ablations, and reality-check inference | `analysis/execution_cost_scenarios.py`; `analysis/publication_costs.py`; `analysis/inference.py`; `artifacts/net_strategy_returns_by_cost_scenario.csv`; `artifacts/required_capital_returns.csv`; `artifacts/post_cost_survival.csv` when generated through summary tables; `artifacts/reality_check_inference.csv`; operational cost/capacity route from the local knowledge base, not cited as a scholarly reference |
