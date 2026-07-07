# Option-Only Markowitz Verification Report

Status: **PASS**
Critical failures: `0`
Total checks: `386`
Hash manifest rows: `201`

## Category Summary

| Category | Passed | Failed |
|---|---:|---:|
| artifacts | 4 | 0 |
| claims | 13 | 0 |
| data | 27 | 0 |
| empirical | 187 | 0 |
| inference | 83 | 0 |
| math | 9 | 0 |
| optimizer | 8 | 0 |
| paper | 15 | 0 |
| pit | 11 | 0 |
| producer | 2 | 0 |
| robustness | 27 | 0 |

## Failed Checks

No failed checks.

## Passed Critical Evidence

- `producer` / `empirical runner exits cleanly`: exit=0
- `paper` / `latex bibtex compile pipeline`: pe1/public/amsfonts/cm/cmsy10.
pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy8.pf
b></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/symbols/msbm1
0.pfb>
Output written on option_only_portfolio_optimization_dhruv_kohli.pdf (34 pages,
 534134 bytes).
Transcript written on option_only_portfolio_optimization_dhruv_kohli.log.


$ /Library/TeX/texbin/lualatex -interaction=nonstopmode option_only_portfolio_optimization_dhruv_kohli.tex
exit=0
opentype/public/lm/lmmono10-regular.otf></System/Library/Fonts
/Supplemental/Times New Roman.ttf></System/Library/Fonts/Supplemental/Times New
 Roman Bold.ttf></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts
/cm/cmex10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/
cmextra/cmex8.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfon
ts/cm/cmmi10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfont
s/cm/cmmi6.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/
cm/cmmi8.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm
/cmr10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/c
mr6.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmr8
.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy10.
pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy8.pf
b></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/symbols/msbm1
0.pfb>
Output written on option_only_portfolio_optimization_dhruv_kohli.pdf (34 pages,
 534134 bytes).
Transcript written on option_only_portfolio_optimization_dhruv_kohli.log.


$ /Library/TeX/texbin/lualatex -interaction=nonstopmode option_only_portfolio_optimization_dhruv_kohli.tex
exit=0
opentype/public/lm/lmmono10-regular.otf></System/Library/Fonts
/Supplemental/Times New Roman.ttf></System/Library/Fonts/Supplemental/Times New
 Roman Bold.ttf></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts
/cm/cmex10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/
cmextra/cmex8.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfon
ts/cm/cmmi10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfont
s/cm/cmmi6.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/
cm/cmmi8.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm
/cmr10.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/c
mr6.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmr8
.pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy10.
pfb></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy8.pf
b></usr/local/texlive/2025/texmf-dist/fonts/type1/public/amsfonts/symbols/msbm1
0.pfb>
Output written on option_only_portfolio_optimization_dhruv_kohli.pdf (34 pages,
 534134 bytes).
Transcript written on option_only_portfolio_optimization_dhruv_kohli.log.

- `artifacts` / `required generated outputs exist`: []
- `artifacts` / `empirical summary exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/research/papers/option_only_markowitz/tables/empirical_summary.json
- `artifacts` / `empirical summary schema`: ["approximation", "claim_audit", "claim_strength", "cost_diagnostics", "cost_input_spread_sources", "cost_scenario_diagnostics", "data", "data_extension_manifest", "drawdown_breach_rates", "execution_repair_comparison", "execution_repair_diagnostics", "exposure", "factor_regression", "figure_visibility", "forecast_ablation_components", "forecast_ablation_performance", "hurdle_summary", "inference", "leave_one_out", "liquidity_tier_diagnostics", "liquidity_tier_performance", "performance", "performance_gross_only", "performance_post_cost", "pnl_attribution", "post_cost_survival", "random_feasible", "reality_check_inference", "regime_performance", "repair_config", "risk_calibration", "rolling_oos", "simulation_assumptions", "simulation_summary", "sortino_diagnostics", "sortino_entry_cost_summary", "split_adjustments", "timing_diagnostics", "trading_data_audit", "vix_chain_feature_summary", "vix_regime_performance", "vix_required_settlement_download_audit", "vix_settlement_audit", "vix_settlement_coverage", "vol_of_vol_regime_performance", "zero_imputation_diagnostics"]
- `data` / `input exists or is externally licensed: equity option feature store`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/data/feature_store/option_greek_proxy_panel.parquet
- `data` / `input exists or is externally licensed: Greek quality summary`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/data/feature_store/option_greek_quality.csv
- `data` / `input exists or is externally licensed: raw close panel`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/data/universe/multi_raw_close.csv
- `data` / `input exists or is externally licensed: VIX complex`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/data/universe/vix_complex.parquet
- `data` / `input exists or is externally licensed: VX futures curve`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/data/universe/vx_futures_daily.parquet
- `data` / `VIX raw monthly shards present or externally licensed`: 137
- `data` / `filtered equity panel row count`: 160315
- `data` / `equity representative choices`: 5825
- `data` / `equity return cells finite`: 5777
- `data` / `equity return lower bound`: -1.0
- `data` / `primary underlyings present`: ["AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA"]
- `data` / `equity marks and Greeks finite`: ["close", "spot", "strike", "delta", "gamma", "vega", "theta", "iv_proxy"]
- `data` / `equity option expiry after snapshot`: True
- `data` / `summary equity row count matches recompute`: 160315
- `data` / `primary Greek coverage valid_delta_share`: 0.9514013487595124
- `data` / `primary Greek coverage valid_gamma_share`: 0.9514013487595124
- `data` / `primary Greek coverage valid_vega_share`: 0.9514013487595124
- `data` / `VIX stack nonempty`: 935561
- `data` / `VIX dedupe key is date-symbol`: 0
- `data` / `VIX OSI parser preserves terms`: ["VIX", "2026-06-17T00:00:00", "call", 30.0]
- `data` / `summary VIX filtered rows plausible`: 103650
- `data` / `VIX detail settlement source complete`: {"vro_soq_exact": 536}
- `data` / `VIX headline rows use exact VRO/SOQ`: {"vro_soq_exact": 536}
- `data` / `VIX settlement audit artifact exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/research/papers/option_only_markowitz/artifacts/vix_settlement_audit.csv
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
- `optimizer` / `constraint slack nonnegative: Greek Markowitz + VIX`: {"Beta SPY slack": 0.6579132188280075, "Gross NAV slack": 1.000000002693291e-06, "Max underlying gross slack": 0.15000100000000194, "Per-contract slack": 0.020904979921186506, "Short gross slack": 1.0000000066900938e-06, "Strategy": "Greek Markowitz + VIX", "Stress slack": 2.6756374893466273e-14, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Beta/delta-neutral + VIX`: {"Beta SPY slack": 0.10452852610385399, "Gross NAV slack": 1.0000000232324169e-06, "Max underlying gross slack": 0.1288900735899968, "Per-contract slack": 0.050782840117196804, "Short gross slack": 1.0000000130461206e-06, "Strategy": "Beta/delta-neutral + VIX", "Stress slack": 0.3398816609556664, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Cost-aware Sortino + VIX`: {"Beta SPY slack": null, "Gross NAV slack": 9.999999999177334e-07, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "Cost-aware Sortino + VIX", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Equal premium`: {"Beta SPY slack": null, "Gross NAV slack": 1.000000000916934e-06, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "Equal premium", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: Equal risk`: {"Beta SPY slack": null, "Gross NAV slack": 1.000000002693291e-06, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "Equal risk", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint slack nonnegative: VIX hedge sleeve`: {"Beta SPY slack": null, "Gross NAV slack": 9.999999999177334e-07, "Max underlying gross slack": null, "Per-contract slack": null, "Short gross slack": null, "Strategy": "VIX hedge sleeve", "Stress slack": null, "VIX vega finite": true}
- `optimizer` / `constraint audit rows written`: 7
- `empirical` / `post-cost returns no greater than gross for option strategies on average`: {"Beta/delta-neutral + VIX": -0.4223385590026113, "Cost-aware Sortino + VIX": -13.537098047870877, "Equal premium": -0.2591236114652083, "Equal risk": -0.11505201556888475, "Equity-option Greek Markowitz": -0.732869318060373, "Greek Markowitz + VIX": -0.6104277661483779, "VIX hedge sleeve": -0.4232794511500628}
- `empirical` / `post-cost return schema matches gross`: ["Beta/delta-neutral + VIX", "Cost-aware Sortino + VIX", "Delta-matched equities", "Equal premium", "Equal risk", "Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Underlying Markowitz", "VIX hedge sleeve"]
- `empirical` / `all executable cost scenarios present`: ["full_spread", "half_spread", "mid"]
- `empirical` / `strategy has all cost scenarios: Equity-option Greek Markowitz`: ["Equity-option Greek Markowitz::full_spread", "Equity-option Greek Markowitz::half_spread", "Equity-option Greek Markowitz::mid"]
- `empirical` / `strategy has all cost scenarios: Greek Markowitz + VIX`: ["Greek Markowitz + VIX::full_spread", "Greek Markowitz + VIX::half_spread", "Greek Markowitz + VIX::mid"]
- `empirical` / `strategy has all cost scenarios: Beta/delta-neutral + VIX`: ["Beta/delta-neutral + VIX::full_spread", "Beta/delta-neutral + VIX::half_spread", "Beta/delta-neutral + VIX::mid"]
- `empirical` / `strategy has all cost scenarios: Equal premium`: ["Equal premium::full_spread", "Equal premium::half_spread", "Equal premium::mid"]
- `empirical` / `strategy has all cost scenarios: Equal risk`: ["Equal risk::full_spread", "Equal risk::half_spread", "Equal risk::mid"]
- `empirical` / `strategy has all cost scenarios: VIX hedge sleeve`: ["VIX hedge sleeve::full_spread", "VIX hedge sleeve::half_spread", "VIX hedge sleeve::mid"]
- `empirical` / `required-capital returns artifact exists`: /Users/dhruvkohli/Desktop/Github Repos/Option_Portfolio_Greek_Risk_Premia/research/papers/option_only_markowitz/artifacts/required_capital_returns.csv
- `empirical` / `rejected/no-fill ledger is auditable`: ["return_date", "strategy", "scenario", "asset_id", "reject_reason", "foregone_gross_return_nav"]
- `empirical` / `hurdle_selection_ledger.csv schema`: ["hurdle", "asset_id", "expected_return", "expected_cost", "risk_estimate", "passed"]
- `empirical` / `liquidity_tier_performance.csv schema`: ["Liquidity tier", "Strategy", "Ann. return", "Ann. vol", "Sharpe", "Calmar", "Omega"]
- `empirical` / `forecast_ablation_performance.csv schema`: ["Ablation", "Ann. return", "Ann. vol", "Sharpe", "Calmar", "Omega", "Active assets"]
- `empirical` / `reality_check_inference.csv schema`: ["Variant", "Sharpe", "Monthly Sharpe", "Probabilistic Sharpe", "Deflated Sharpe", "DSR trials", "Reality check p", "Centered max stat", "Centered max p95", "N", "Block size", "Seed"]
- `empirical` / `capacity_market_impact_diagnostics.csv schema`: ["Strategy", "Scenario", "Avg contracts traded", "Avg quoted spread", "Avg monthly cost", "Max capacity used", "Rejected trades", "Mean margin/NAV", "Mean stress capital/NAV"]
- `empirical` / `simulation_summary.csv schema`: ["Return basis", "Strategy", "Requested method", "Status", "Simulation", "Reason", "N paths", "Defaulted path share", "Ann. return p05", "Ann. return p50", "Ann. return p95", "Sortino p05", "Sortino p50", "Sortino p95", "Max DD p05", "Max DD p50", "Max DD p95", "Terminal wealth p05", "Terminal wealth p50", "Terminal wealth p95"]
- `empirical` / `simulation_assumptions.csv schema`: ["Strategy", "Return basis", "Method", "Status", "N obs", "Source start", "Source end", "Periods/year", "Period mean", "Period volatility", "Skewness", "Excess kurtosis", "Lag1 autocorr", "Block length", "Path count", "Interpretation", "Reason"]
- `empirical` / `drawdown_breach_rates.csv schema`: ["Return basis", "Strategy", "Requested method", "Simulation", "Breach 10%", "Breach 25%", "Breach 50%", "Breach 75%", "Breach 90%"]
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Ann. return`: 0.894164572585072
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Ann. vol`: 1.0618025335815389
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Sharpe`: 0.8421194565895302
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Sortino`: 1.7700811621300996
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Calmar`: 1.0204897200736815
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Omega`: 1.887746709782267
- `empirical` / `performance metric matches summary: Equity-option Greek Markowitz / Info. ratio`: 0.5982743778565095
- `empirical` / `worst month matches summary: Equity-option Greek Markowitz`: -0.4548043633961028
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Ann. return`: 1.2463192201296207
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Ann. vol`: 0.9068172382561567
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Sharpe`: 1.374388539995489
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Sortino`: 3.3852590273086602
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Calmar`: 2.0339185818698295
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Omega`: 2.784363176427202
- `empirical` / `performance metric matches summary: Greek Markowitz + VIX / Info. ratio`: 1.1998006329123239
- `empirical` / `worst month matches summary: Greek Markowitz + VIX`: -0.3706500329201142
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Ann. return`: 1.1720642765798035
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Ann. vol`: 0.8291743699040247
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Sharpe`: 1.4135317239912608
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Sortino`: 3.410131121859385
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Calmar`: 2.0573541792299705
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Omega`: 2.947610573038662
- `empirical` / `performance metric matches summary: Beta/delta-neutral + VIX / Info. ratio`: 1.147073121558933
- `empirical` / `worst month matches summary: Beta/delta-neutral + VIX`: -0.3541829035716423
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Ann. return`: 5.01998964773432
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Ann. vol`: 4.948702018391607
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Sharpe`: 1.0144053186224946
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Sortino`: 3.349185377713025
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Calmar`: 4.992611798187288
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Omega`: 2.471099899395961
- `empirical` / `performance metric matches summary: Cost-aware Sortino + VIX / Info. ratio`: 0.9711886304435532
- `empirical` / `worst month matches summary: Cost-aware Sortino + VIX`: -1.2141176470588235
- `empirical` / `performance metric matches summary: Equal premium / Ann. return`: -0.0555215986452688
- `empirical` / `performance metric matches summary: Equal premium / Ann. vol`: 1.878473564887012
- `empirical` / `performance metric matches summary: Equal premium / Sharpe`: -0.029556763365263736
- `empirical` / `performance metric matches summary: Equal premium / Sortino`: -0.05531007631445422
- `empirical` / `performance metric matches summary: Equal premium / Calmar`: -0.05553271091792234
- `empirical` / `performance metric matches summary: Equal premium / Omega`: 0.977176547323733
- `empirical` / `performance metric matches summary: Equal premium / Info. ratio`: -0.11981879207294609
- `empirical` / `worst month matches summary: Equal premium`: -0.6623626401921293
- `empirical` / `performance metric matches summary: Equal risk / Ann. return`: 0.10434553413680833
- `empirical` / `performance metric matches summary: Equal risk / Ann. vol`: 1.275136625211684
- `empirical` / `performance metric matches summary: Equal risk / Sharpe`: 0.0818308658646567
- `empirical` / `performance metric matches summary: Equal risk / Sortino`: 0.14638589036583952
- `empirical` / `performance metric matches summary: Equal risk / Calmar`: 0.10693488072635131
- `empirical` / `performance metric matches summary: Equal risk / Omega`: 1.064138706582978
- `empirical` / `performance metric matches summary: Equal risk / Info. ratio`: -0.05774656874994709
- `empirical` / `worst month matches summary: Equal risk`: -0.5155029411966359
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Ann. return`: -9.28491238522107
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Ann. vol`: 1.4441818674927205
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Sharpe`: -6.429184989935397
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Sortino`: -3.0892753979009937
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Calmar`: nan
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Omega`: 0.040425690950258435
- `empirical` / `performance metric matches summary: VIX hedge sleeve / Info. ratio`: -6.13214961326551
- `empirical` / `worst month matches summary: VIX hedge sleeve`: -1.0
- `empirical` / `performance metric matches summary: Delta-matched equities / Ann. return`: 1.095404059234048
- `empirical` / `performance metric matches summary: Delta-matched equities / Ann. vol`: 0.9311728493163963
- `empirical` / `performance metric matches summary: Delta-matched equities / Sharpe`: 1.1763702732937489
- `empirical` / `performance metric matches summary: Delta-matched equities / Sortino`: 2.3173675259176236
- `empirical` / `performance metric matches summary: Delta-matched equities / Calmar`: 1.2251172051670711
- `empirical` / `performance metric matches summary: Delta-matched equities / Omega`: 2.366610102479196
- `empirical` / `performance metric matches summary: Delta-matched equities / Info. ratio`: 1.0407484819276789
- `empirical` / `worst month matches summary: Delta-matched equities`: -0.4540048150966957
- `empirical` / `performance metric matches summary: Underlying Markowitz / Ann. return`: 0.1785517211015295
- `empirical` / `performance metric matches summary: Underlying Markowitz / Ann. vol`: 0.19645175039322108
- `empirical` / `performance metric matches summary: Underlying Markowitz / Sharpe`: 0.9088833301008387
- `empirical` / `performance metric matches summary: Underlying Markowitz / Sortino`: 1.8560338426832323
- `empirical` / `performance metric matches summary: Underlying Markowitz / Calmar`: 0.5902286749234769
- `empirical` / `performance metric matches summary: Underlying Markowitz / Omega`: 1.9287455618155838
- `empirical` / `performance metric matches summary: Underlying Markowitz / Info. ratio`: 0.07169818618150241
- `empirical` / `worst month matches summary: Underlying Markowitz`: -0.0732331521875972
- `empirical` / `random feasible p95 matches summary`: 1.1449656633511185
- `empirical` / `random feasible seed output count`: 250
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Ann. alpha`: 0.4046822482056147
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / $R^2$`: 0.6182781587595915
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Residual ann. vol`: 0.6341524538729405
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Beta SPY`: 0.00475973050256312
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Beta VX front`: -1.3679497671888123
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Beta dVIX`: 0.045821494038335034
- `empirical` / `factor regression matches summary: Equity-option Greek Markowitz / Beta dVVIX`: 0.003051752035829535
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Ann. alpha`: 0.6747841146361057
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / $R^2$`: 0.5133361878818812
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Residual ann. vol`: 0.6156789424162552
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Beta SPY`: 1.907548364987137
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Beta VX front`: -0.7667338125037371
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Beta dVIX`: 0.05579114093524096
- `empirical` / `factor regression matches summary: Greek Markowitz + VIX / Beta dVVIX`: -0.001404049683069823
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Ann. alpha`: 0.8449573055730607
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / $R^2$`: 0.42889990365271735
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Residual ann. vol`: 0.601296224140347
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Beta SPY`: 0.7154500616975166
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Beta VX front`: -0.522020248937875
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Beta dVIX`: 0.037802167920126806
- `empirical` / `factor regression matches summary: Beta/delta-neutral + VIX / Beta dVVIX`: -0.0010577997552185447
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Ann. alpha`: 4.091151397943683
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / $R^2$`: 0.45078894296098715
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Residual ann. vol`: 3.6502041588229517
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Beta SPY`: 13.773864963136507
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Beta VX front`: -4.30221987337334
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Beta dVIX`: 0.19464231398624804
- `empirical` / `factor regression matches summary: Cost-aware Sortino + VIX / Beta dVVIX`: 0.008039009603954522
- `empirical` / `factor regression matches summary: Equal premium / Ann. alpha`: -0.41604126337318303
- `empirical` / `factor regression matches summary: Equal premium / $R^2$`: 0.25260243663169824
- `empirical` / `factor regression matches summary: Equal premium / Residual ann. vol`: 1.6469374713863754
- `empirical` / `factor regression matches summary: Equal premium / Beta SPY`: 2.522490477155059
- `empirical` / `factor regression matches summary: Equal premium / Beta VX front`: -0.7487384813045156
- `empirical` / `factor regression matches summary: Equal premium / Beta dVIX`: 0.03217385857447594
- `empirical` / `factor regression matches summary: Equal premium / Beta dVVIX`: -0.0005857303792825962
- `empirical` / `factor regression matches summary: Equal risk / Ann. alpha`: -0.1828755135900255
- `empirical` / `factor regression matches summary: Equal risk / $R^2$`: 0.253452337393129
- `empirical` / `factor regression matches summary: Equal risk / Residual ann. vol`: 1.1147547910664153
- `empirical` / `factor regression matches summary: Equal risk / Beta SPY`: 2.0414812843681496
- `empirical` / `factor regression matches summary: Equal risk / Beta VX front`: -0.3788834768155678
- `empirical` / `factor regression matches summary: Equal risk / Beta dVIX`: 0.020108125711849424
- `empirical` / `factor regression matches summary: Equal risk / Beta dVVIX`: -0.0027659763860344907
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Ann. alpha`: -8.806789511925738
- `empirical` / `factor regression matches summary: VIX hedge sleeve / $R^2$`: 0.25243620349716944
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Residual ann. vol`: 1.2638050706290416
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Beta SPY`: 3.5616785167856206
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Beta VX front`: -1.1800723747222541
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Beta dVIX`: 0.09271988778412793
- `empirical` / `factor regression matches summary: VIX hedge sleeve / Beta dVVIX`: -0.012626568451072198
- `empirical` / `factor regression matches summary: Delta-matched equities / Ann. alpha`: 8.451572774959004e-15
- `empirical` / `factor regression matches summary: Delta-matched equities / $R^2$`: 1.0
- `empirical` / `factor regression matches summary: Delta-matched equities / Residual ann. vol`: 6.650449409235076e-15
- `empirical` / `factor regression matches summary: Delta-matched equities / Beta SPY`: -1.6209256159527285e-14
- `empirical` / `factor regression matches summary: Delta-matched equities / Beta VX front`: -8.465450562766819e-16
- `empirical` / `factor regression matches summary: Delta-matched equities / Beta dVIX`: -5.204170427930421e-16
- `empirical` / `factor regression matches summary: Delta-matched equities / Beta dVVIX`: 2.5196858488563123e-16
- `empirical` / `factor regression matches summary: Underlying Markowitz / Ann. alpha`: 1.8839096949108125e-15
- `empirical` / `factor regression matches summary: Underlying Markowitz / $R^2$`: 1.0
- `empirical` / `factor regression matches summary: Underlying Markowitz / Residual ann. vol`: 9.443021598833129e-16
- `empirical` / `factor regression matches summary: Underlying Markowitz / Beta SPY`: -2.4147350785597155e-15
- `empirical` / `factor regression matches summary: Underlying Markowitz / Beta VX front`: -1.2836953722228372e-15
- `empirical` / `factor regression matches summary: Underlying Markowitz / Beta dVIX`: -7.112366251504909e-17
- `empirical` / `factor regression matches summary: Underlying Markowitz / Beta dVVIX`: 3.7513395167998453e-17
- `empirical` / `P&L attribution reconciles: Equity-option Greek Markowitz`: 0.8941645725850716
- `empirical` / `P&L attribution reconciles: Greek Markowitz + VIX`: 1.2463192201296205
- `empirical` / `P&L attribution reconciles: Beta/delta-neutral + VIX`: 1.1720642765798028
- `empirical` / `P&L attribution reconciles: Cost-aware Sortino + VIX`: 5.01998964773432
- `empirical` / `P&L attribution reconciles: Equal premium`: -0.05552159864526729
- `empirical` / `P&L attribution reconciles: Equal risk`: 0.10434553413680625
- `empirical` / `P&L attribution reconciles: VIX hedge sleeve`: -9.28491238522107
- `empirical` / `simulation covers central strategy/basis pairs`: [["Full-spread post-cost", "Beta/delta-neutral + VIX"], ["Full-spread post-cost", "Delta-matched equities"], ["Full-spread post-cost", "Equal premium"], ["Full-spread post-cost", "Equal risk"], ["Full-spread post-cost", "Equity-option Greek Markowitz"], ["Full-spread post-cost", "Greek Markowitz + VIX"], ["Full-spread post-cost", "Underlying Markowitz"], ["Full-spread post-cost", "VIX hedge sleeve"], ["Gross before costs", "Beta/delta-neutral + VIX"], ["Gross before costs", "Delta-matched equities"], ["Gross before costs", "Equal premium"], ["Gross before costs", "Equal risk"], ["Gross before costs", "Equity-option Greek Markowitz"], ["Gross before costs", "Greek Markowitz + VIX"], ["Gross before costs", "Underlying Markowitz"], ["Gross before costs", "VIX hedge sleeve"]]
- `empirical` / `simulation methods include block and volatility clustered`: ["circular_block_bootstrap", "egarch_or_ewma"]
- `empirical` / `simulation inputs are OOS only`: 2021-01-29T00:00:00
- `empirical` / `simulation source length matches OOS returns`: [{"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Equity-option Greek Markowitz"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Equity-option Greek Markowitz"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Greek Markowitz + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Greek Markowitz + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Beta/delta-neutral + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Beta/delta-neutral + VIX"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Delta-matched equities"}, {"N obs": 60, "Return basis": "Gross before costs", "Strategy": "Delta-matched equities"}]
- `empirical` / `simulation breach probabilities bounded`: {"max": 1.0, "min": 0.0}
- `empirical` / `VIX regime table complete`: 27
- `empirical` / `leave-one-out row present: No META`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No NVDA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No TSLA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `leave-one-out row present: No META/NVDA/TSLA`: ["All underlyings", "No AAPL", "No AMZN", "No GOOGL", "No JPM", "No META", "No MSFT", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]
- `empirical` / `rolling 36M OOS recorded`: 20.0
- `empirical` / `repaired trade ledger schema`: ["return_date", "strategy", "scenario", "asset_id", "repair_reason", "decision_mark", "effective_fill_price", "extra_cost_nav", "fill_fraction", "foregone_gross_return_nav"]
- `empirical` / `repaired trade ledger scenarios are suffixed`: []
- `empirical` / `repaired trade ledger excludes hard-gate reasons`: []
- `empirical` / `quote repairs respect 10 percent fill band`: 0.0
- `empirical` / `capacity partial repairs respect fill fraction bounds`: {"max": 0.9999504, "min": 0.1001644444444444}
- `empirical` / `repaired scenario return columns are suffixed`: ["full_spread_repaired", "half_spread_repaired", "mid_repaired"]
- `empirical` / `repaired trades are not hard-gate rejections`: []
- `empirical` / `Sortino entry cost schema`: ["asset_id", "n_train_rows", "mean_relative_spread", "mean_mark", "entry_cost", "source"]
- `empirical` / `Sortino entry costs finite nonnegative`: {"max": 0.0933653573385656, "min": 0.0075991223088257}
- `empirical` / `cost input spread source coverage schema`: ["relative_spread_source", "asset_class", "rows", "mean_relative_spread"]
- `empirical` / `cost input spread sources are recognized`: ["default", "panel_cbbo"]
- `empirical` / `data extension manifest schema`: ["dataset", "location", "size_approx", "status", "reason", "artifacts_produced"]
- `empirical` / `data extension manifest records expected dataset families`: ["opra_surface_full_day_cbbo", "opra_vix_chain_*", "opra_{UND}_slices_*", "sibling DATA_ANALYSIS loaders"]
- `robustness` / `distributional robustness outputs exist`: []
- `robustness` / `distributional robustness summary schema`: ["cv_config", "cv_context_consistency", "cv_cpcv_path_metrics", "cv_fold_ledger", "cv_fold_schedule", "cv_pbo", "cv_regime_performance", "mc_refit_summary", "mc_repriced_assumptions", "mc_repriced_summary", "mc_resampled_summary", "mc_universe_comparison", "runtime_seconds", "seeds"]
- `robustness` / `blocked k-fold count matches config`: 12
- `robustness` / `CPCV split count matches config`: 66
- `robustness` / `complete CPCV path count matches config`: {"complete": 11, "status_counts": {"complete": 132}}
- `robustness` / `CV purge/embargo invariant recomputed from schedule`: []
- `robustness` / `complete CPCV paths cover every month exactly once`: []
- `robustness` / `PBO values bounded`: [0.19230769230769232, 0.20512820512820512]
- `robustness` / `CV context consistency diffs are negligible`: {"max": 9.84455572616838e-17, "rows": 7}
- `robustness` / `MC fixed-weight path counts match config`: {"('resampled', 'full_spread_post_cost')": 1000, "('resampled', 'gross')": 1000, "('resampled_stratified', 'full_spread_post_cost')": 1000, "('resampled_stratified', 'gross')": 1000}
- `robustness` / `MC refit path count matches config`: 200
- `robustness` / `MC repriced path counts match config`: {"joint_garch_block": 1000}
- `robustness` / `MC gaussian-copula path counts match config`: {"gaussian_copula": 250}
- `robustness` / `repriced assumptions include one-step tenor row`: Pricing Tenor Rule
- `robustness` / `repriced assumptions include VX-front convention row`: VIX Forward Convention
- `robustness` / `robustness table headers escape underscores`: []
- `robustness` / `breadth CV purge-gap inputs exist`: []
- `robustness` / `breadth CV purge-gap artifact schemas`: {"breadth_cv_fold_ledger": [], "breadth_cv_fold_schedule": [], "breadth_cv_test_month_returns": [], "full_history_month_grid": []}
- `robustness` / `breadth CV purge gap exceeds monthly tenor span`: Minimum realized pre-test train/test calendar gap after purge/embargo is 59 days across 484 test blocks. Reconstruction issues: []
- `robustness` / `CPCV windows table inputs exist`: []
- `robustness` / `CPCV windows table schema`: {"header": ["Config", "Full net p05", "Full net p50", "Default share", "Claim net p05", "Claim net p50", "Rel full p05", "Rel claim p05"], "missing_cols": [], "rows": 4}
- `robustness` / `CPCV windows artifact schemas`: {"breadth_cv_claim_cpcv_path_metrics": [], "breadth_cv_claim_relative_paths": [], "breadth_cv_cpcv_path_metrics": [], "breadth_cv_relative_paths": [], "final_result_scoreboard": []}
- `robustness` / `CPCV windows table values match artifacts`: []
- `robustness` / `CPCV windows default shares bounded`: []
- `robustness` / `breadth claim CV purge-gap inputs exist`: []
- `robustness` / `breadth claim CV purge-gap artifact schemas`: {"breadth_cv_claim_fold_ledger": [], "breadth_cv_claim_fold_schedule": [], "breadth_cv_claim_test_month_returns": [], "full_history_month_grid": []}
- `robustness` / `breadth claim CV purge gap exceeds monthly tenor span`: Minimum realized pre-test train/test calendar gap after purge/embargo is 59 days across 532 test blocks. Reconstruction issues: []
- `inference` / `independent numpy stat matches table: Gross before costs / Equity-option Greek Markowitz / Ann. return`: 0.894164572585072
- `inference` / `independent numpy stat matches table: Gross before costs / Equity-option Greek Markowitz / Ann. vol`: 1.0618025335815389
- `inference` / `independent numpy stat matches table: Gross before costs / Equity-option Greek Markowitz / Sharpe`: 0.8421194565895302
- `inference` / `independent numpy stat matches table: Gross before costs / Greek Markowitz + VIX / Ann. return`: 1.2463192201296207
- `inference` / `independent numpy stat matches table: Gross before costs / Greek Markowitz + VIX / Ann. vol`: 0.9068172382561567
- `inference` / `independent numpy stat matches table: Gross before costs / Greek Markowitz + VIX / Sharpe`: 1.374388539995489
- `inference` / `independent numpy stat matches table: Gross before costs / Beta/delta-neutral + VIX / Ann. return`: 1.1720642765798035
- `inference` / `independent numpy stat matches table: Gross before costs / Beta/delta-neutral + VIX / Ann. vol`: 0.8291743699040247
- `inference` / `independent numpy stat matches table: Gross before costs / Beta/delta-neutral + VIX / Sharpe`: 1.4135317239912608
- `inference` / `independent numpy stat matches table: Gross before costs / Cost-aware Sortino + VIX / Ann. return`: 5.01998964773432
- `inference` / `independent numpy stat matches table: Gross before costs / Cost-aware Sortino + VIX / Ann. vol`: 4.948702018391607
- `inference` / `independent numpy stat matches table: Gross before costs / Cost-aware Sortino + VIX / Sharpe`: 1.0144053186224946
- `inference` / `independent numpy stat matches table: Gross before costs / Equal premium / Ann. return`: -0.0555215986452688
- `inference` / `independent numpy stat matches table: Gross before costs / Equal premium / Ann. vol`: 1.878473564887012
- `inference` / `independent numpy stat matches table: Gross before costs / Equal premium / Sharpe`: -0.029556763365263736
- `inference` / `independent numpy stat matches table: Gross before costs / Equal risk / Ann. return`: 0.10434553413680833
- `inference` / `independent numpy stat matches table: Gross before costs / Equal risk / Ann. vol`: 1.275136625211684
- `inference` / `independent numpy stat matches table: Gross before costs / Equal risk / Sharpe`: 0.0818308658646567
- `inference` / `independent numpy stat matches table: Gross before costs / VIX hedge sleeve / Ann. return`: -9.28491238522107
- `inference` / `independent numpy stat matches table: Gross before costs / VIX hedge sleeve / Ann. vol`: 1.4441818674927205
- `inference` / `independent numpy stat matches table: Gross before costs / VIX hedge sleeve / Sharpe`: -6.429184989935397
- `inference` / `independent numpy stat matches table: Gross before costs / Delta-matched equities / Ann. return`: 1.095404059234048
- `inference` / `independent numpy stat matches table: Gross before costs / Delta-matched equities / Ann. vol`: 0.9311728493163963
- `inference` / `independent numpy stat matches table: Gross before costs / Delta-matched equities / Sharpe`: 1.1763702732937489
- `inference` / `independent numpy stat matches table: Gross before costs / Underlying Markowitz / Ann. return`: 0.1785517211015295
- `inference` / `independent numpy stat matches table: Gross before costs / Underlying Markowitz / Ann. vol`: 0.19645175039322108
- `inference` / `independent numpy stat matches table: Gross before costs / Underlying Markowitz / Sharpe`: 0.9088833301008387
- `inference` / `independent numpy stat matches table: Post-cost research / Equity-option Greek Markowitz / Ann. return`: -7.900267244139405
- `inference` / `independent numpy stat matches table: Post-cost research / Equity-option Greek Markowitz / Ann. vol`: 5.476489096854859
- `inference` / `independent numpy stat matches table: Post-cost research / Equity-option Greek Markowitz / Sharpe`: -1.4425788318790809
- `inference` / `independent numpy stat matches table: Post-cost research / Greek Markowitz + VIX / Ann. return`: -6.078813973650915
- `inference` / `independent numpy stat matches table: Post-cost research / Greek Markowitz + VIX / Ann. vol`: 4.8272433122818565
- `inference` / `independent numpy stat matches table: Post-cost research / Greek Markowitz + VIX / Sharpe`: -1.2592723383519353
- `inference` / `independent numpy stat matches table: Post-cost research / Beta/delta-neutral + VIX / Ann. return`: -3.895998431451532
- `inference` / `independent numpy stat matches table: Post-cost research / Beta/delta-neutral + VIX / Ann. vol`: 2.8407272413436755
- `inference` / `independent numpy stat matches table: Post-cost research / Beta/delta-neutral + VIX / Sharpe`: -1.3714792376929188
- `inference` / `independent numpy stat matches table: Post-cost research / Cost-aware Sortino + VIX / Ann. return`: -157.4251869267162
- `inference` / `independent numpy stat matches table: Post-cost research / Cost-aware Sortino + VIX / Ann. vol`: 148.97625674304243
- `inference` / `independent numpy stat matches table: Post-cost research / Cost-aware Sortino + VIX / Sharpe`: -1.0567132667203922
- `inference` / `independent numpy stat matches table: Post-cost research / Equal premium / Ann. return`: -3.1650049362277684
- `inference` / `independent numpy stat matches table: Post-cost research / Equal premium / Ann. vol`: 2.3784680347046976
- `inference` / `independent numpy stat matches table: Post-cost research / Equal premium / Sharpe`: -1.330690549566593
- `inference` / `independent numpy stat matches table: Post-cost research / Equal risk / Ann. return`: -1.2762786526898087
- `inference` / `independent numpy stat matches table: Post-cost research / Equal risk / Ann. vol`: 1.3671688859639324
- `inference` / `independent numpy stat matches table: Post-cost research / Equal risk / Sharpe`: -0.9335193813966584
- `inference` / `independent numpy stat matches table: Post-cost research / VIX hedge sleeve / Ann. return`: -14.364265799021823
- `inference` / `independent numpy stat matches table: Post-cost research / VIX hedge sleeve / Ann. vol`: 4.209238542567987
- `inference` / `independent numpy stat matches table: Post-cost research / VIX hedge sleeve / Sharpe`: -3.4125568446064882
- `inference` / `independent numpy stat matches table: Post-cost research / Delta-matched equities / Ann. return`: 1.095404059234048
- `inference` / `independent numpy stat matches table: Post-cost research / Delta-matched equities / Ann. vol`: 0.9311728493163963
- `inference` / `independent numpy stat matches table: Post-cost research / Delta-matched equities / Sharpe`: 1.1763702732937489
- `inference` / `independent numpy stat matches table: Post-cost research / Underlying Markowitz / Ann. return`: 0.1785517211015295
- `inference` / `independent numpy stat matches table: Post-cost research / Underlying Markowitz / Ann. vol`: 0.19645175039322108
- `inference` / `independent numpy stat matches table: Post-cost research / Underlying Markowitz / Sharpe`: 0.9088833301008387
- `inference` / `independent stat recompute coverage`: 54
- `inference` / `final inference panel inputs exist`: []
- `inference` / `final inference panel table schema`: ["Config", "Net Sharpe", "CI lo", "CI hi", "PSR", "DSR", "dSR stock", "p stock", "dSR naive", "p naive"]
- `inference` / `final inference panel artifact schemas`: {"final_inference_panel": [], "p1_regularization_results": [], "scoreboard": []}
- `inference` / `final inference panel intervals and probabilities valid`: []
- `inference` / `final inference table Net Sharpe matches returns: orig`: {"recomputed": 0.7776672701424105, "table": 0.778}
- `inference` / `final inference scoreboard Net Sharpe matches returns: orig`: {"recomputed": 0.7776672701424105, "scoreboard": 0.7776672701424103}
- `inference` / `final inference table Net Sharpe matches returns: orig+VIX`: {"recomputed": 1.3831478314520487, "table": 1.383}
- `inference` / `final inference scoreboard Net Sharpe matches returns: orig+VIX`: {"recomputed": 1.3831478314520487, "scoreboard": 1.3831478314520482}
- `inference` / `final inference table Net Sharpe matches returns: larger`: {"recomputed": 0.5871547497312365, "table": 0.587}
- `inference` / `final inference scoreboard Net Sharpe matches returns: larger`: {"recomputed": 0.5871547497312365, "scoreboard": 0.5871547497312364}
- `inference` / `final inference table Net Sharpe matches returns: larger+VIX`: {"recomputed": 1.6282919496891226, "table": 1.628}
- `inference` / `final inference scoreboard Net Sharpe matches returns: larger+VIX`: {"recomputed": 1.6282919496891226, "scoreboard": 1.6282919496891228}
- `inference` / `final inference DSR trial counts match P1 grid`: []
- `inference` / `E1 channel ablation table schema`: ["Arm", "orig", "orig+VIX", "larger", "larger+VIX"]
- `inference` / `E1 channel ablation table cells finite`: []
- `inference` / `E1 channel ablation Full E1 row matches scoreboard`: []
- `inference` / `E1 concentration table schema`: ["Config", "Candidates", "Active", "Top 5 share", "Deployed gross", "At cap share", "Cap budget"]
- `inference` / `E1 realized candidate summary schema`: ["config", "strategy", "deployable", "mode", "solver_status", "capacity_infeasible", "sum_of_caps", "deployed_gross", "gross_sharpe", "net_sharpe", "gross_sortino", "net_sortino", "gross_max_drawdown", "net_max_drawdown"]
- `inference` / `E1 concentration table values match artifacts`: []
- `inference` / `CI lo <= CI hi in cv_regime_performance.tex`: []
- `inference` / `CI lo <= CI hi in inference_summary.tex`: []
- `inference` / `CI lo <= CI hi in leave_one_out.tex`: []
- `inference` / `CI lo <= CI hi in portfolio_performance_diagnostics.tex`: []
- `inference` / `CI lo <= CI hi in portfolio_performance_net_diagnostics.tex`: []
- `inference` / `CI lo <= CI hi in regime_performance.tex`: []
- `inference` / `CI lo <= CI hi in short_inference_panel.tex`: []
- `inference` / `CI lo <= CI hi in vix_regime_performance.tex`: []
- `inference` / `CI ordering audit covers tables`: 8
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
- `paper` / `LaTeX log clean`: []
- `paper` / `compiled PDF exists`: 534134
- `paper` / `PDF page count plausible`: 34
- `paper` / `PDF caveat text includes not claimed`: not claimed
- `paper` / `PDF caveat text includes premium weights`: premium weights
- `paper` / `PDF text includes exact VRO/SOQ`: VRO/SOQ
- `paper` / `PDF caveat text includes transaction costs`: transaction costs
- `paper` / `PDF caveat text includes slippage`: slippage
- `paper` / `PDF text includes tail-path simulation caveat`: tail-path simulation diagnostics
- `paper` / `references include option-risk-premium papers`: reference text
- `paper` / `PDF pages render to PNG`: exit=0, pages=34
- `paper` / `rendered PDF sample pages nonempty`: [165229, 166415]
- `paper` / `Appendix all-strategy growth figure has all strategy series`: ["Beta/delta-neutral + VIX", "Cost-aware Sortino + VIX", "Delta-matched equities", "Equal premium", "Equal risk", "Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Underlying Markowitz", "VIX hedge sleeve"]
- `paper` / `Appendix all-strategy growth figure series visible`: [{"Figure": "portfolio_growth_all_strategies.pdf", "Max": 18.35005283540228, "Min": 0.183011836336598, "Pass": "yes", "Series": "Equity-option Greek Markowitz", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 144.51861716831928, "Min": 0.6403949030925237, "Pass": "yes", "Series": "Greek Markowitz + VIX", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 98.96884463070484, "Min": 0.5735594003706624, "Pass": "yes", "Series": "Beta/delta-neutral + VIX", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 2.996100619612762, "Min": 0.0001, "Pass": "yes", "Series": "Cost-aware Sortino + VIX", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 0.8905477325031234, "Min": 0.0001782014429151, "Pass": "yes", "Series": "Equal premium", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 0.944996867354269, "Min": 0.0228823784998115, "Pass": "yes", "Series": "Equal risk", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 0.0001, "Min": 0.0001, "Pass": "yes", "Series": "VIX hedge sleeve", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 33.429827509128316, "Min": 0.4455022015082628, "Pass": "yes", "Series": "Delta-matched equities", "Visible points": 60}, {"Figure": "portfolio_growth_all_strategies.pdf", "Max": 2.459772612888149, "Min": 0.945644336357695, "Pass": "yes", "Series": "Underlying Markowitz", "Visible points": 60}]
- `artifacts` / `hash manifest covers outputs`: 201
