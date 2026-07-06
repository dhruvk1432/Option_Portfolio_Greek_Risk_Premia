# Breadth Robustness Validation

Cost policy: full cost stack, NAV $1,000,000, X=0.05, current Cboe fills disabled, inferred CBBO proxy enabled.
CV policy: 4 chronological groups, 1 test groups, purge/embargo=1/1 month(s), 8 splits per config.
Spread-source audit: pass (current Cboe rows=0, default rows=0).

| config | strategy | deployable | verdict | net_sharpe | net_sortino | deployed_gross | sum_of_caps | cpcv_net_p05 | cpcv_net_p50 | cpcv_net_p95 | cpcv_gross_p50 | mc_resampled_net_p05 | mc_resampled_net_p50 | mc_refit_net_p05 | mc_refit_net_p50 | repriced_net_overlay_p05 | repriced_net_overlay_p50 | rolling_net_sharpe | reality_check_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | E1 capped | False | diagnostic_capacity_infeasible | 0.724 | 1.644 | 0.535 | 0.731 | -0.170 | -0.170 | -0.170 | 0.890 | -0.005 | 0.666 | 0.708 | 0.802 | -3.422 | 0.383 | 1.189 | 0.000 |
| orig+VIX | E1 capped | True | pass | 1.288 | 3.264 | 0.770 | 1.079 | -0.206 | -0.206 | -0.206 | 0.629 | 0.759 | 1.278 | 1.089 | 1.309 | -2.683 | 0.176 | 2.207 | 0.000 |
| larger | E1 capped | True | mixed | 0.563 | 1.219 | 0.641 | 1.155 | -0.342 | -0.342 | -0.342 | 1.013 | -0.295 | 0.475 | 0.520 | 0.729 | -269.315 | -101.024 | 1.440 | 0.000 |
| larger+VIX | E1 capped | True | pass | 1.514 | 4.041 | 0.905 | 1.503 | -0.329 | -0.329 | -0.329 | 0.893 | 0.987 | 1.389 | 1.190 | 1.450 | -13.562 | -13.062 | 2.763 | 0.000 |

Repriced synthetic net paths use a circular-block sample of realized full-cost drag; they are not synthetic NBBO/CBBO quotes.
