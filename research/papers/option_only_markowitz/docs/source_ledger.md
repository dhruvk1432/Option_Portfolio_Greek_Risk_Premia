# Source Ledger

This ledger maps the paper's main claim families to formal references and
machine-readable research outputs.

| Claim family | Evidence |
|---|---|
| Markowitz mean-variance and maximum-Sharpe framing | Markowitz (1952) |
| Local option pricing, replication, and Greeks | Black and Scholes (1973); Merton (1973); Black (1976); Shreve (2004) |
| Factor risk models, covariance shrinkage, and optimizer sensitivity | Ledoit and Wolf (2004); Grinold and Kahn (2000) |
| Performance ratios and backtest overfitting caution | Sharpe (1966); Sortino and Price (1994); Keating and Shadwick (2002); Magdon-Ismail and Atiya (2004); Grinold and Kahn (2000); Bailey et al. (2017); Lopez de Prado (2018) |
| OPRA/Databento data venue and public market-data provenance | Recorded in `REPRODUCIBILITY.md`; not cited as scholarly references |
| Greek panel provenance and quality | `data/feature_store/option_greek_assumptions.md`; `data/feature_store/option_greek_quality.csv` |
| Empirical tables and figures | `research/papers/option_only_markowitz/analysis/run_empirics.py`; `tables/empirical_summary.json`; `artifacts/*.csv` |
| Premium/NAV accounting and performance metrics | `src/portfolio/option_only_markowitz_model.py`; `tables/portfolio_performance.tex`; `tables/portfolio_performance_diagnostics.tex`; `tables/exposure_summary.tex`; `tables/greek_exposure_summary.tex`; `tables/empirical_summary.json` |
| Point-in-time timing and split-adjusted listed-expiry return construction | `tables/timing_diagnostics.tex`; `tables/trading_data_audit.tex`; `artifacts/holding_return_detail.csv`; `artifacts/timing_diagnostics.csv`; `artifacts/trading_data_audit.csv`; `artifacts/split_adjustments.csv` |
| Equity-drift diagnostics, factor betas, P&L attribution, regimes, and leave-one-out robustness | `tables/empirical_summary.json`; `artifacts/factor_regression.csv`; `artifacts/pnl_attribution.csv`; `artifacts/regime_performance.csv`; `artifacts/vix_regime_performance.csv`; `artifacts/leave_one_out.csv` |
| Conditional option expected returns, volatility risk premia, option premia, skew/tail premia | Coval and Shumway (2001); Bakshi and Kapadia (2003); Carr and Wu (2009); Bollerslev, Tauchen, and Zhou (2009); Broadie, Chernov, and Johannes (2009) |
| VIX option data construction and settlement status | `analysis/vix_option_panel.py`; `artifacts/vix_holding_return_detail.csv`; `artifacts/vix_data_audit.csv`; `tables/trading_data_audit.tex`; Cboe product/settlement pages documented operationally, not in bibliography |
| Conditional premia implementation | `analysis/conditional_premia.py`; `artifacts/conditional_premia_components.csv`; `tables/empirical_summary.json` |
| Rolling OOS and figure visibility audits | `tables/rolling_oos.tex`; `artifacts/rolling_oos.csv`; `artifacts/figure_visibility_audit.csv` |
| Production promotion layer: settlement, executable quotes, fills, margin, assignment, broker adapter, and production verifier | `src/option_portfolio_production/*`; `tests/test_option_portfolio_production.py`; operational anchors: Cboe VIX/VRO/SOQ pages, Databento OPRA timestamp/symbology docs, OCC ODD and margin resources; not cited as scholarly references |

| Claim-audit status | `tables/claim_audit.tex`; `artifacts/claim_audit.csv` |

Operational data-source URLs are documented in `REPRODUCIBILITY.md`, not in the scholarly
bibliography.

| Publication replication package, artifact hashes, environment lock, and OPRA redistribution boundary | `docs/replication_package.md`; `artifact_hash_manifest.csv`; `environment_lock.json`; operational provenance only, not scholarly bibliography |
| Exact VRO/SOQ public ingestion and VIX headline gating | `data_pull/cboe_vro_soq.py`; `analysis/vix_option_panel.py`; `artifacts/vix_settlement_coverage.csv`; `artifacts/vix_settlement_audit.csv`; Cboe VRO/SOQ operational convention recorded here, not in bibliography |
| Pre-production executable-cost scenarios, required capital, liquidity tiers, forecast ablations, and reality-check inference | `analysis/execution_cost_scenarios.py`; `analysis/publication_costs.py`; `analysis/inference.py`; `artifacts/net_strategy_returns_by_cost_scenario.csv`; `artifacts/required_capital_returns.csv`; `artifacts/post_cost_survival.csv` when generated through summary tables; `artifacts/reality_check_inference.csv`; operational cost/capacity route from the local knowledge base, not cited as a scholarly reference |
