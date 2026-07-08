# Breadth Robustness Validation

Cost policy: full cost stack, NAV $1,000,000, X=0.05, current Cboe fills disabled, inferred CBBO proxy enabled.
CV policy: 12 chronological groups, 2 test groups, purge/embargo=1/1 month(s), 78 splits per config.
Spread-source audit: pass (current Cboe rows=0, default rows=0).

| config | strategy | deployable | verdict | net_sharpe | net_sortino | deployed_gross | sum_of_caps | cpcv_net_p05 | cpcv_net_p50 | cpcv_net_p95 | cpcv_gross_p50 | mc_resampled_net_p05 | mc_resampled_net_p50 | mc_refit_net_p05 | mc_refit_net_p50 | repriced_net_overlay_p05 | repriced_net_overlay_p50 | rolling_net_sharpe | reality_check_p | cpcv_claim_net_p05 | cpcv_claim_net_p50 | cpcv_claim_net_p95 | rel_cpcv_net_p05 | rel_cpcv_net_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | E1 capped | False | diagnostic_capacity_infeasible | 0.778 | 1.783 | 0.497 | 0.739 | 0.860 | 0.908 | 0.973 | 1.057 | 0.199 | 0.778 | 0.570 | 0.818 | -3.654 | 0.547 | 0.585 | 0.001 | 0.437 | 0.490 | 0.722 | -0.561 | -0.340 |
| orig+VIX | E1 capped | True | pass | 1.383 | 3.551 | 0.723 | 1.093 | 0.727 | 0.749 | 0.794 | 0.981 | 0.873 | 1.377 | 1.178 | 1.375 | -3.096 | 0.258 | 1.033 | 0.001 | 1.003 | 1.062 | 1.249 | 0.214 | 0.246 |
| larger | E1 capped | True | pass | 0.587 | 1.298 | 0.594 | 1.160 | 0.766 | 0.802 | 0.913 | 1.329 | -0.001 | 0.589 | 0.537 | 0.719 | -269.863 | -68.685 | 0.480 | 0.001 | 0.660 | 0.706 | 0.827 | -0.265 | 0.331 |
| larger+VIX | E1 capped | True | pass | 1.628 | 4.377 | 0.836 | 1.515 | 0.500 | 0.532 | 0.692 | 0.990 | 1.096 | 1.636 | 1.360 | 1.644 | -12.315 | -11.977 | 1.273 | 0.001 | 1.445 | 1.474 | 1.722 | 0.103 | 0.250 |

Repriced synthetic net paths use a circular-block sample of realized full-cost drag; they are not synthetic NBBO/CBBO quotes.
