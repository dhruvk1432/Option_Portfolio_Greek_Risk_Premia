# Breadth Robustness Validation

Cost policy: full cost stack, NAV $1,000,000, X=0.05, current Cboe fills disabled, inferred CBBO proxy enabled.
CV policy: 12 chronological groups, 2 test groups, purge/embargo=1/1 month(s), 78 splits per config.
Spread-source audit: pass (current Cboe rows=0, default rows=0).

| config | strategy | deployable | verdict | net_sharpe | net_sortino | deployed_gross | sum_of_caps | cpcv_net_p05 | cpcv_net_p50 | cpcv_net_p95 | cpcv_gross_p50 | mc_resampled_net_p05 | mc_resampled_net_p50 | mc_refit_net_p05 | mc_refit_net_p50 | repriced_net_overlay_p05 | repriced_net_overlay_p50 | rolling_net_sharpe | reality_check_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | E1 capped | False | diagnostic_capacity_infeasible | 0.720 | 1.629 | 0.537 | 0.739 | -0.171 | -0.134 | -0.025 | 0.967 | 0.140 | 0.721 | 0.540 | 0.779 | -3.596 | 0.556 | 0.584 | 0.001 |
| orig+VIX | E1 capped | True | pass | 1.287 | 3.256 | 0.773 | 1.093 | -0.147 | -0.128 | -0.055 | 0.840 | 0.768 | 1.291 | 1.118 | 1.300 | -3.142 | 0.291 | 1.007 | 0.001 |
| larger | E1 capped | True | mixed | 0.551 | 1.196 | 0.638 | 1.160 | -0.480 | -0.349 | -0.261 | 1.147 | -0.038 | 0.553 | 0.491 | 0.680 | -288.008 | -94.975 | 0.466 | 0.001 |
| larger+VIX | E1 capped | True | pass | 1.499 | 4.004 | 0.906 | 1.515 | -0.541 | -0.420 | -0.306 | 0.814 | 0.989 | 1.495 | 1.262 | 1.542 | -13.614 | -13.016 | 1.217 | 0.001 |

Repriced synthetic net paths use a circular-block sample of realized full-cost drag; they are not synthetic NBBO/CBBO quotes.
