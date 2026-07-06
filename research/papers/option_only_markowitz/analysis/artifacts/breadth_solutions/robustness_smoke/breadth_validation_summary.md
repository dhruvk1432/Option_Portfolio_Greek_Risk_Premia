# Breadth Robustness Validation

Cost policy: full cost stack, NAV $1,000,000, X=0.05, current Cboe fills disabled, inferred CBBO proxy enabled.
CV policy: 3 chronological groups, 1 test groups, purge/embargo=1/1 month(s), 6 splits per config.
Spread-source audit: pass (current Cboe rows=0, default rows=0).

| config | strategy | deployable | verdict | net_sharpe | net_sortino | deployed_gross | sum_of_caps | cpcv_net_p05 | cpcv_net_p50 | cpcv_net_p95 | cpcv_gross_p50 | mc_resampled_net_p05 | mc_resampled_net_p50 | mc_refit_net_p05 | mc_refit_net_p50 | repriced_net_overlay_p05 | repriced_net_overlay_p50 | rolling_net_sharpe | reality_check_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | E1 capped | False | diagnostic_capacity_infeasible | 0.724 | 1.644 | 0.535 | 0.731 | -0.152 | -0.152 | -0.152 | 0.999 |  |  | 0.708 | 0.802 | -3.422 | 0.383 | 1.189 | 0.100 |

Repriced synthetic net paths use a circular-block sample of realized full-cost drag; they are not synthetic NBBO/CBBO quotes.
