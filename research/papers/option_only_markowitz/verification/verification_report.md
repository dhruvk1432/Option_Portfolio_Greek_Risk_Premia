# Option-Only Markowitz Verification Report

Status: **PASS**
Critical failures: `0`
Total checks: `176`
Hash manifest rows: `125`

## Category Summary

| Category | Passed | Failed |
|---|---:|---:|
| artifacts | 4 | 0 |
| claims | 13 | 0 |
| data | 14 | 0 |
| empirical | 105 | 0 |
| math | 9 | 0 |
| optimizer | 7 | 0 |
| paper | 12 | 0 |
| pit | 11 | 0 |
| producer | 1 | 0 |

## Failed Checks

No failed checks.

## Passed Critical Evidence

- `artifacts` / `required generated outputs exist`: []
- `artifacts` / `empirical summary exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/research/papers/option_only_markowitz/tables/empirical_summary.json
- `artifacts` / `empirical summary schema`: ["approximation", "claim_audit", "claim_strength", "cost_diagnostics", "cost_scenario_diagnostics", "data", "drawdown_breach_rates", "exposure", "factor_regression", "figure_visibility", "forecast_ablation_components", "forecast_ablation_performance", "hurdle_summary", "inference", "leave_one_out", "liquidity_tier_diagnostics", "liquidity_tier_performance", "performance", "performance_gross_only", "performance_post_cost", "pnl_attribution", "post_cost_survival", "random_feasible", "reality_check_inference", "regime_performance", "risk_calibration", "rolling_oos", "simulation_assumptions", "simulation_summary", "split_adjustments", "timing_diagnostics", "trading_data_audit", "vix_regime_performance", "vix_required_settlement_download_audit", "vix_settlement_audit", "vix_settlement_coverage"]
- `data` / `input exists or is externally licensed: equity option feature store`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/feature_store/option_greek_proxy_panel.parquet
- `data` / `input exists or is externally licensed: Greek quality summary`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/feature_store/option_greek_quality.csv
- `data` / `input exists or is externally licensed: raw close panel`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/universe/multi_raw_close.csv
- `data` / `input exists or is externally licensed: VIX complex`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/universe/vix_complex.parquet
- `data` / `input exists or is externally licensed: VX futures curve`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/universe/vx_futures_daily.parquet
- `data` / `VIX raw monthly shards present or externally licensed`: 0
- `data` / `standalone package uses generated data artifacts when licensed raw inputs are absent`: raw OPRA/Databento inputs omitted
- `data` / `VIX OSI parser preserves terms`: ["VIX", "2026-06-17T00:00:00", "call", 30.0]
- `data` / `VIX detail settlement source complete`: {"vro_soq_exact": 536}
- `data` / `VIX headline rows use exact VRO/SOQ`: {"vro_soq_exact": 536}
- `data` / `VIX settlement audit artifact exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/research/papers/option_only_markowitz/artifacts/vix_settlement_audit.csv
- `data` / `VIX Greek model is Black-76 VX-forward`: {"black76_vx_forward": 536}
- `data` / `VIX underlying is VX forward`: {"VX_FRONT": 536}
- `data` / `VIX long-option return lower bound`: -1.0
- `pit` / `ledger has rows`: 6313
- `pit` / `decisions precede payoff`: decision_date < payoff_date
- `pit` / `equity payoff no later than return date`: equity rows=5777
- `pit` / `VIX proxy payoff timing is after decision`: VIX rows=536
- `pit` / `state snapshot observable by decision`: state_snapshot_date <= decision_date
- `pit` / `OOS forecast train end before return`: OOS rows=3088
- `pit` / `OOS decision dates after frozen train split`: 2020-12-31 00:00:00
- `pit` / `all option returns finite`: 6313
- `pit` / `no long premium return below -100 percent`: -1.0
- `pit` / `timing diagnostic train/test split recorded`: [{"Diagnostic": "Return construction", "Value": "Prior-date option selection, split-adjusted listed-expiry payoff"}, {"Diagnostic": "Minimum option mark filter", "Value": "0.25"}, {"Diagnostic": "Max train decision date", "Value": "2020-11-30"}]
- `pit` / `trading audit pass/proxy only`: ["yes"]
- `math` / `BSM finite-difference delta`: 0.6128357071576026
- `math` / `BSM finite-difference gamma`: 0.02497447913712423
- `math` / `BSM finite-difference vega`: 24.457407521069207
- `math` / `Black-76 finite-difference delta`: 0.6987191589614669
- `math` / `Black-76 finite-difference gamma`: 0.06777879544668074
- `math` / `Black-76 finite-difference vega`: 2.359259167534462
- `math` / `Greek covariance construction identity`: 1.1102230246251565e-16
- `math` / `Greek covariance PSD`: 2.64191038264704
- `math` / `closed-form tangency formula`: {"a": 0.48486490671208493, "b": 0.5151350932879151}
- `optimizer` / `constraint slack nonnegative: Equity-option Greek Markowitz`: {"Beta SPY slack": 2.149391775674303e-13, "Gross NAV slack": 1.0000000055798708e-06, "Max underlying gross slack": 0.09035474290243295, "Per-contract slack": 0.045282650870974384, "Short gross slack": 1.0000000041365809e-06, "Strategy": "Equity-option Greek Markowitz", "Stress slack": 0.14342202014501054, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Greek Markowitz + VIX`: {"Beta SPY slack": 0.657913093255635, "Gross NAV slack": 1.0000000018051125e-06, "Max underlying gross slack": 0.15000100000000183, "Per-contract slack": 0.020904982890004697, "Short gross slack": 1.0000000060794711e-06, "Strategy": "Greek Markowitz + VIX", "Stress slack": 1.2045919817182948e-14, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Beta/delta-neutral + VIX`: {"Beta SPY slack": 0.1045402351164757, "Gross NAV slack": 1.0000000333354464e-06, "Max underlying gross slack": 0.12888969580132412, "Per-contract slack": 0.0507831389769525, "Short gross slack": 1.0000000097154516e-06, "Strategy": "Beta/delta-neutral + VIX", "Stress slack": 0.33988169220585873, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Equal premium`: {"Beta SPY slack": null, "Gross NAV slack": 1.000000000916934e-06, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "Equal premium", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Equal risk`: {"Beta SPY slack": null, "Gross NAV slack": 1.0000000024712463e-06, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "Equal risk", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: VIX hedge sleeve`: {"Beta SPY slack": null, "Gross NAV slack": 9.999999999177334e-07, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "VIX hedge sleeve", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint audit rows written`: 6
- `empirical` / `post-cost returns no greater than gross for option strategies on average`: {"Beta/delta-neutral + VIX": -0.42234176802488166, "Equal premium": -0.2591236114652083, "Equal risk": -0.11505201556888449, "Equity-option Greek Markowitz": -0.7328719186936924, "Greek Markowitz + VIX": -0.6104286502961673, "VIX hedge sleeve": -0.4232794511500628}
- `empirical` / `post-cost return schema matches gross`: ["Beta/delta-neutral + VIX", "Delta-matched equities", "Equal premium", "Equal risk", "Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Underlying Markowitz", "VIX hedge sleeve"]
- `empirical` / `all executable cost scenarios present`: ["full_spread", "half_spread", "mid"]
- `empirical` / `strategy has all cost scenarios: Equity-option Greek Markowitz`: ["Equity-option Greek Markowitz::full_spread", "Equity-option Greek Markowitz::half_spread", "Equity-option Greek Markowitz::mid"]
- `empirical` / `strategy has all cost scenarios: Greek Markowitz + VIX`: ["Greek Markowitz + VIX::full_spread", "Greek Markowitz + VIX::half_spread", "Greek Markowitz + VIX::mid"]
- `empirical` / `strategy has all cost scenarios: Beta/delta-neutral + VIX`: ["Beta/delta-neutral + VIX::full_spread", "Beta/delta-neutral + VIX::half_spread", "Beta/delta-neutral + VIX::mid"]
- `empirical` / `strategy has all cost scenarios: Equal premium`: ["Equal premium::full_spread", "Equal premium::half_spread", "Equal premium::mid"]
- `empirical` / `strategy has all cost scenarios: Equal risk`: ["Equal risk::full_spread", "Equal risk::half_spread", "Equal risk::mid"]
- `empirical` / `strategy has all cost scenarios: VIX hedge sleeve`: ["VIX hedge sleeve::full_spread", "VIX hedge sleeve::half_spread", "VIX hedge sleeve::mid"]
- `empirical` / `required-capital returns artifact exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/research/papers/option_only_markowitz/artifacts/required_capital_returns.csv
- `empirical` / `rejected/no-fill ledger is auditable`: ["return_date", "strategy", "scenario", "asset_id", "reject_reason"]
- `empirical` / `hurdle_selection_ledger.csv schema`: ["hurdle", "asset_id", "expected_return", "expected_cost", "risk_estimate", "passed"]
- `empirical` / `liquidity_tier_performance.csv schema`: ["Liquidity tier", "Strategy", "Ann. return", "Ann. vol", "Sharpe", "Calmar", "Omega"]
- `empirical` / `forecast_ablation_performance.csv schema`: ["Ablation", "Ann. return", "Ann. vol", "Sharpe", "Calmar", "Omega", "Active assets"]
- `empirical` / `reality_check_inference.csv schema`: ["Variant", "Sharpe", "Probabilistic Sharpe", "Deflated Sharpe", "Bootstrap max Sharpe p95", "N", "Block size", "Seed"]
- `empirical` / `capacity_market_impact_diagnostics.csv schema`: ["Strategy", "Scenario", "Avg contracts traded", "Avg quoted spread", "Avg monthly cost", "Max capacity used", "Rejected trades", "Mean margin/NAV", "Mean stress capital/NAV"]
- `empirical` / `simulation_summary.csv schema`: ["Return basis", "Strategy", "Requested method", "Status", "Simulation", "Reason", "N paths", "Ann. return p05", "Ann. return p50", "Ann. return p95", "Sortino p05", "Sortino p50", "Sortino p95", "Max DD p05", "Max DD p50", "Max DD p95", "Terminal wealth p05", "Terminal wealth p50", "Terminal wealth p95"]
- `empirical` / `simulation_assumptions.csv schema`: ["Strategy", "Return basis", "Method", "Status", "N obs", "Source start", "Source end", "Periods/year", "Period mean", "Period volatility", "Skewness", "Excess kurtosis", "Lag1 autocorr", "Block length", "Path count", "Interpretation", "Reason"]
- `empirical` / `drawdown_breach_rates.csv schema`: ["Return basis", "Strategy", "Requested method", "Simulation", "Breach 10%", "Breach 25%", "Breach 50%", "Breach 75%", "Breach 90%"]
- `empirical` / `factor-control raw inputs absent; generated regression artifacts used`: /Users/dhruvkohli/Desktop/Github Repos/Option_Only_Markowitz_Cashflow_Engineering/data/feature_store/option_greek_proxy_panel.parquet
- `empirical` / `information ratio recorded from generated summary: Equity-option Greek Markowitz`: 0.5982743778565097
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Ann. return`: 0.894164572585072
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Ann. vol`: 1.0618025335815389
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Sharpe`: 0.8421194565895302
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Sortino`: 1.7700811621300996
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Calmar`: 1.0204897200736815
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Omega`: 1.887746709782267
- `empirical` / `worst month matches summary: Equity-option Greek Markowitz`: -0.4548043633961028
- `empirical` / `information ratio recorded from generated summary: Greek Markowitz + VIX`: 1.1998006679456976
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Ann. return`: 1.2463192285080005
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Ann. vol`: 0.9068172193093107
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Sharpe`: 1.3743885779509966
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Sortino`: 3.3852591193963497
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Calmar`: 2.033918612174948
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Omega`: 2.7843631987725814
- `empirical` / `worst month matches summary: Greek Markowitz + VIX`: -0.3706500401498496
- `empirical` / `information ratio recorded from generated summary: Beta/delta-neutral + VIX`: 1.1470654479222597
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Ann. return`: 1.1720618697652665
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Ann. vol`: 0.8291774074138495
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Sharpe`: 1.4135236431740843
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Sortino`: 3.4101063613903224
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Calmar`: 2.0573537885252158
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Omega`: 2.947600334500002
- `empirical` / `worst month matches summary: Beta/delta-neutral + VIX`: -0.3541833244318854
- `empirical` / `information ratio recorded from generated summary: Equal premium`: -0.11981879207294609
- `empirical` / `performance metric matches summary: Equal premium / Ann. return`: -0.0555215986452688
- `empirical` / `performance metric matches summary: Equal premium / Ann. vol`: 1.878473564887012
- `empirical` / `performance metric matches summary: Equal premium / Sharpe`: -0.029556763365263736
- `empirical` / `performance metric matches summary: Equal premium / Sortino`: -0.05531007631445422
- `empirical` / `performance metric matches summary: Equal premium / Calmar`: -0.05553271091792234
- `empirical` / `performance metric matches summary: Equal premium / Omega`: 0.977176547323733
- `empirical` / `worst month matches summary: Equal premium`: -0.6623626401921293
- `empirical` / `information ratio recorded from generated summary: Equal risk`: -0.057746568749952626
- `empirical` / `performance metric matches summary: Equal risk / Ann. return`: 0.10434553413679806
- `empirical` / `performance metric matches summary: Equal risk / Ann. vol`: 1.2751366252116842
- `empirical` / `performance metric matches summary: Equal risk / Sharpe`: 0.08183086586464863
- `empirical` / `performance metric matches summary: Equal risk / Sortino`: 0.14638589036582483
- `empirical` / `performance metric matches summary: Equal risk / Calmar`: 0.10693488072634065
- `empirical` / `performance metric matches summary: Equal risk / Omega`: 1.0641387065829715
- `empirical` / `worst month matches summary: Equal risk`: -0.5155029411966354
- `empirical` / `information ratio recorded from generated summary: VIX hedge sleeve`: -6.13214961326551
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Ann. return`: -9.28491238522107
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Ann. vol`: 1.4441818674927205
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Sharpe`: -6.429184989935397
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Sortino`: -3.0892753979009937
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Calmar`: nan
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Omega`: 0.040425690950258435
- `empirical` / `worst month matches summary: VIX hedge sleeve`: -1.0
- `empirical` / `information ratio recorded from generated summary: Delta-matched equities`: 1.0407484610659437
- `empirical` / `performance metric matches summary: Delta-matched equities / Ann. return`: 1.0954040285123476
- `empirical` / `performance metric matches summary: Delta-matched equities / Ann. vol`: 0.9311728351232496
- `empirical` / `performance metric matches summary: Delta-matched equities / Sharpe`: 1.1763702582317712
- `empirical` / `performance metric matches summary: Delta-matched equities / Sortino`: 2.31736748845411
- `empirical` / `performance metric matches summary: Delta-matched equities / Calmar`: 1.2251171795483853
- `empirical` / `performance metric matches summary: Delta-matched equities / Omega`: 2.366610085977659
- `empirical` / `worst month matches summary: Delta-matched equities`: -0.4540048157737572
- `empirical` / `information ratio recorded from generated summary: Underlying Markowitz`: 0.07169818618150268
- `empirical` / `performance metric matches summary: Underlying Markowitz / Ann. return`: 0.1785517211015295
- `empirical` / `performance metric matches summary: Underlying Markowitz / Ann. vol`: 0.19645175039322108
- `empirical` / `performance metric matches summary: Underlying Markowitz / Sharpe`: 0.9088833301008387
- `empirical` / `performance metric matches summary: Underlying Markowitz / Sortino`: 1.8560338426832323
- `empirical` / `performance metric matches summary: Underlying Markowitz / Calmar`: 0.5902286749234769
- `empirical` / `performance metric matches summary: Underlying Markowitz / Omega`: 1.9287455618155838
- `empirical` / `worst month matches summary: Underlying Markowitz`: -0.0732331521875972
- `empirical` / `random feasible p95 matches summary`: 1.1449656633511185
- `empirical` / `random feasible seed output count`: 250
- `empirical` / `factor regression artifact exists for standalone verification`: 8
- `empirical` / `factor regression artifact strategies match summary`: ["Beta/delta-neutral + VIX", "Delta-matched equities", "Equal premium", "Equal risk", "Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Underlying Markowitz", "VIX hedge sleeve"]
- `empirical` / `P&L attribution reconciles: Equity-option Greek Markowitz`: 0.8941645725850716
- `empirical` / `P&L attribution reconciles: Greek Markowitz + VIX`: 1.2463192285080005
- `empirical` / `P&L attribution reconciles: Beta/delta-neutral + VIX`: 1.1720618697652667
- `empirical` / `P&L attribution reconciles: Equal premium`: -0.05552159864526729
- `empirical` / `P&L attribution reconciles: Equal risk`: 0.10434553413679737
- `empirical` / `P&L attribution reconciles: VIX hedge sleeve`: -9.28491238522107
- `empirical` / `simulation covers central strategy/basis pairs`: [["Full-spread post-cost", "Beta/delta-neutral + VIX"], ["Full-spread post-cost", "Delta-matched equities"], ["Full-spread post-cost", "Equal premium"], ["Full-spread post-cost", "Equal risk"], ["Full-spread post-cost", "Equity-option Greek Markowitz"], ["Full-spread post-cost", "Greek Markowitz + VIX"], ["Full-spread post-cost", "Underlying Markowitz"], ["Full-spread post-cost", "VIX hedge sleeve"], ["Gross before costs", "Beta/delta-neutral + VIX"], ["Gross before costs", "Delta-matched equities"], ["Gross before costs", "Equal premium"], ["Gross before costs", "Equal risk"], ["Gross before costs", "Equity-option Greek Markowitz"], ["Gross before costs", "Greek Markowitz + VIX"], ["Gross before costs", "Underlying Markowitz"], ["Gross before costs", "VIX hedge sleeve"]]
- `empirical` / `simulation methods include block and volatility clustered`: ["circular_block_bootstrap", "egarch_or_ewma"]
- `empirical` / `simulation inputs are OOS only`: 2021-01-29T00:00:00
- `empirical` / `simulation source length matches OOS returns`: [{"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Equity-option Greek Markowitz"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Equity-option Greek Markowitz"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Greek Markowitz + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Greek Markowitz + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Beta/delta-neutral + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Beta/delta-neutral + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Delta-matched equities"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Delta-matched equities"}]
- `empirical` / `simulation breach probabilities bounded`: {"max": 1.0, "min": 0.0}
- `empirical` / `VIX regime table complete`: 24
- `empirical` / `leave-one-out row present: No META`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No NVDA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No TSLA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No META/NVDA/TSLA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `rolling 36M OOS recorded`: 20.0
- `claims` / `VIX exact settlement claim gated by source coverage`: {"Claim": "VIX option expiry P\\&L is exact listed settlement P\\&L", "Evidence": "All VIX expiry rows use exact Cboe VRO/SOQ settlement", "Status": "Supported", "Type": "Generated empirical"}
- `claims` / `post-cost research claim is generated`: {"Claim": "Post-cost research returns include implementation frictions", "Evidence": "Generated mid, half-spread, full-spread, fee, capacity, required-capital, and assignment-risk ledgers", "Status": "Implemented as conservative research simulation", "Type": "Generated empirical"}
- `claims` / `broker-executed evidence overclaim rejected`: {"Claim": "Pre-production results are broker-executed live evidence", "Evidence": "No live fills, order routing, broker margin preview, or broker reconciliation", "Status": "Not claimed", "Type": "Rejected overclaim"}
- `claims` / `production tradability not claimed`: {"Claim": "Strategy is production tradable after costs", "Evidence": "No live fills, live broker margin preview, order routing, or broker reconciliation", "Status": "Not claimed", "Type": "Production claim"}
- `claims` / `alpha-independence overclaim downgraded`: {"Claim": "Result is not only long-call equity drift", "Evidence": "exposure, regression, regimes, equity benchmarks, leave-one-out", "Status": "Not supported as an alpha-independence claim", "Type": "Generated diagnostic"}
- `claims` / `claim audit has theorem and empirical claim types`: {"Accounting identity": 1, "Conditional empirical": 1, "Generated diagnostic": 3, "Generated empirical": 3, "Modeling convention": 1, "Production claim": 1, "Rejected overclaim": 2, "Theorem": 2}
- `claims` / `bibliography entry types papers/books only`: []
- `claims` / `operational sources absent from bibliography`: []
- `claims` / `source ledger records cboe`: cboe
- `claims` / `source ledger records databento`: databento
- `claims` / `source ledger records vix option`: vix option
- `claims` / `source ledger records artifact`: artifact
- `paper` / `compiled PDF exists`: 415393
- `paper` / `PDF page count plausible`: 24
- `paper` / `PDF caveat text includes not claimed`: not claimed
- `paper` / `PDF caveat text includes premium weights`: premium weights
- `paper` / `PDF text includes exact VRO/SOQ`: VRO/SOQ
- `paper` / `PDF caveat text includes transaction costs`: transaction costs
- `paper` / `PDF caveat text includes slippage`: slippage
- `paper` / `PDF text includes tail-path simulation caveat`: tail-path simulation diagnostics
- `paper` / `references include option-risk-premium papers`: reference text
- `artifacts` / `hash manifest covers outputs`: 125
