# P3 Combined Decision Table

POC note: spread inputs are source-audited; added-name and VIX rows use measured panel CBBO when present and otherwise use a point-in-time inferred CBBO proxy calibrated from the historical liquid equity/ETF CBBO surface; off-hours Cboe snapshots are rejected and the old blanket 10%/15% class defaults are not used in the breadth-solution reruns

Baseline note: `orig` uses measured historical panel CBBO for all eight equity underlyings. `orig+VIX` uses the same exact equity-option CBBO rows, while VIX option spreads use the inferred liquid-option CBBO proxy.

Net cells are marked with `*` when that config uses `inferred_cbbo_proxy` rows for VIX options or added-name equity options without historical panel CBBO. The regenerated checked-in run does not consume current Cboe spread fills; exact panel rows remain unmarked in the spread-source coverage table.

## Decision Table

| Config | Strategy | Mode | Gross Sharpe | Net@1M | Net@5M | Net@10M | Net@25M | Breakeven AUM | Capacity Infeasible AUMs | Deployed Gross@25M |
|---|---|---|---|---|---|---|---|---|---|---|
| orig+VIX | GM paper | uncapped | 1.374 | -1.196* | -1.425* | -1.435* | -1.439* | n/a |  | 1.000 |
| orig+VIX | Delta neutral | uncapped | 1.414 | -1.264* | -1.629* | -1.643* | -1.649* | n/a |  | 1.000 |
| orig+VIX | Equal premium | naive | -0.030 | -1.277* | -1.928* | -1.932* | -1.935* | n/a |  | 1.000 |
| orig+VIX | Equal risk | naive | 0.082 | -0.863* | -2.593* | -2.637* | -2.654* | n/a |  | 1.000 |
| orig+VIX | GM combined | hard | 1.628 | 1.149* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| orig+VIX | GM combined alt | hard | 1.613 | 1.287* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| orig+VIX | Equal premium capped | hard | -0.578 | -0.853* | n/a | n/a | n/a | n/a |  | n/a |
| orig+VIX | Equal risk capped | hard | -0.673 | -0.977* | n/a | n/a | n/a | n/a |  | n/a |
| orig+VIX | GM combined relaxed | relaxed | 2.014 | n/a | 1.591* | 1.592* | 1.591* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.001 |
| orig+VIX | GM combined alt relaxed | relaxed | 2.511 | n/a | 2.161* | 2.161* | 2.161* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.002 |
| orig+VIX | Equal premium capped | relaxed | -0.709 | n/a | -0.992* | -0.992* | -0.992* | n/a | $5,000,000, $10,000,000, $25,000,000 | 0.042 |
| orig+VIX | Equal risk capped | relaxed | -0.824 | n/a | -1.134* | -1.134* | -1.134* | n/a | $5,000,000, $10,000,000, $25,000,000 | 0.042 |
| larger+VIX | GM paper | uncapped | 0.765 | -1.837* | -1.901* | -1.906* | -1.909* | n/a |  | 1.000 |
| larger+VIX | Delta neutral | uncapped | 0.832 | -1.741* | -1.798* | -1.803* | -1.806* | n/a |  | 1.000 |
| larger+VIX | Equal premium | naive | 0.485 | -1.289* | -4.249* | -4.254* | -4.257* | n/a |  | 1.000 |
| larger+VIX | Equal risk | naive | 0.576 | -1.382* | -5.085* | -5.149* | -5.170* | n/a |  | 1.000 |
| larger+VIX | GM combined | hard | 1.783 | 1.182* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger+VIX | GM combined alt | hard | 1.915 | 1.499* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger+VIX | Equal premium capped | hard | 0.317 | 0.073* | n/a | n/a | n/a | $1,000,000 |  | n/a |
| larger+VIX | Equal risk capped | hard | 0.532 | 0.266* | n/a | n/a | n/a | $1,000,000 |  | n/a |
| larger+VIX | GM combined relaxed | relaxed | 1.300 | n/a | 0.438* | 0.438* | 0.438* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger+VIX | GM combined alt relaxed | relaxed | 1.285 | n/a | 0.855* | 0.855* | 0.855* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger+VIX | Equal premium capped | relaxed | -0.206 | n/a | -0.501* | -0.501* | -0.501* | n/a | $5,000,000, $10,000,000, $25,000,000 | 0.058 |
| larger+VIX | Equal risk capped | relaxed | -0.287 | n/a | -0.605* | -0.605* | -0.605* | n/a | $5,000,000, $10,000,000, $25,000,000 | 0.058 |
| orig | GM paper | uncapped | 0.842 | -1.443 | -1.595 | -1.603 | -1.607 | n/a |  | 1.000 |
| orig | Delta neutral | uncapped | 0.781 | -1.641 | -1.806 | -1.814 | -1.819 | n/a |  | 1.000 |
| orig | Equal premium | naive | 0.267 | -1.202 | -1.916 | -1.928 | -1.934 | n/a |  | 1.000 |
| orig | Equal risk | naive | 0.340 | -0.731 | -2.579 | -2.633 | -2.653 | n/a |  | 1.000 |
| orig | GM combined | hard | 0.994 | 0.695 | n/a | n/a | n/a | $1,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| orig | GM combined relaxed | relaxed | 1.061 | 0.832 | 0.397 | 0.397 | 0.396 | $25,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.001 |
| orig | GM combined alt | hard | 0.961 | 0.720 | n/a | n/a | n/a | $1,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| orig | GM combined alt relaxed | relaxed | 1.076 | 0.922 | 0.636 | 0.636 | 0.637 | $25,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.001 |
| orig | Equal premium capped | relaxed | 0.534 | 0.350 | 0.350 | 0.350 | 0.350 | $25,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.028 |
| orig | Equal risk capped | relaxed | 0.550 | 0.327 | 0.327 | 0.327 | 0.327 | $25,000,000 | $1,000,000, $5,000,000, $10,000,000, $25,000,000 | 0.028 |
| larger | GM paper | uncapped | 0.456 | -1.991* | -2.018* | -2.021* | -2.024* | n/a |  | 1.000 |
| larger | Delta neutral | uncapped | 0.432 | -1.798* | -1.819* | -1.822* | -1.824* | n/a |  | 1.000 |
| larger | Equal premium | naive | 0.557 | -1.278* | -4.243* | -4.253* | -4.257* | n/a |  | 1.000 |
| larger | Equal risk | naive | 0.638 | -1.373* | -5.082* | -5.148* | -5.170* | n/a |  | 1.000 |
| larger | GM combined | hard | 0.924 | 0.530* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger | GM combined alt | hard | 0.810 | 0.551* | n/a | n/a | n/a | $1,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger | Equal premium capped | hard | 0.690 | 0.475* | n/a | n/a | n/a | $1,000,000 |  | n/a |
| larger | Equal risk capped | hard | 0.815 | 0.550* | n/a | n/a | n/a | $1,000,000 |  | n/a |
| larger | GM combined relaxed | relaxed | 0.470 | n/a | -0.337* | -0.337* | -0.337* | n/a | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger | GM combined alt relaxed | relaxed | 0.454 | n/a | 0.093* | 0.093* | 0.093* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.000 |
| larger | Equal premium capped | relaxed | 0.696 | n/a | 0.474* | 0.474* | 0.474* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.044 |
| larger | Equal risk capped | relaxed | 0.725 | n/a | 0.472* | 0.472* | 0.472* | $25,000,000 | $5,000,000, $10,000,000, $25,000,000 | 0.044 |

## Spread Source Coverage

| config | relative_spread_source | asset_class | rows | asset_ids | underlyings | mean_relative_spread | median_relative_spread |
|---|---|---|---|---|---|---|---|
| orig+VIX | inferred_cbbo_proxy | vix_option | 536 | 5 | 1 | 0.0339252 | 0.0253175 |
| orig+VIX | panel_cbbo | equity_option | 5777 | 49 | 8 | 0.0262017 | 0.0189274 |
| larger+VIX | inferred_cbbo_proxy | equity_option | 23740 | 213 | 47 | 0.0251166 | 0.0197222 |
| larger+VIX | inferred_cbbo_proxy | vix_option | 536 | 5 | 1 | 0.0339252 | 0.0253175 |
| larger+VIX | panel_cbbo | equity_option | 5777 | 49 | 8 | 0.0262017 | 0.0189274 |
| orig | panel_cbbo | equity_option | 5777 | 49 | 8 | 0.0262017 | 0.0189274 |
| larger | inferred_cbbo_proxy | equity_option | 23740 | 213 | 47 | 0.0251166 | 0.0197222 |
| larger | panel_cbbo | equity_option | 5777 | 49 | 8 | 0.0262017 | 0.0189274 |

## Verdict

- Breadth pays gross? larger vs orig: FAIL on the primary regularized row (0.924 vs 0.994), and still marginally below the 8-name bar in the P1 best-gross no-VIX sweep (0.834 vs 0.842).
- Breadth pays gross? larger+VIX vs orig+VIX: PASS on the selected E1 regularized/capped row (1.915 vs 1.613); the primary combined row also passes (1.783 vs 1.628).
- Optimizer vs capped naive at breadth (net), larger: $1,000,000: 0.001. Crossover: none on grid.
- Optimizer vs capped naive at breadth (net), larger+VIX: $1,000,000: 1.232. Crossover: none on grid.
- Capacity, larger: best regularized GM breakeven $1,000,000 vs orig $1,000,000 and GM-paper n/a.
- Capacity, larger+VIX: best regularized GM breakeven $1,000,000 vs orig+VIX $1,000,000 and GM-paper n/a.
