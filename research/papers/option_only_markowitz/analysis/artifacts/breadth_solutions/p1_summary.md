# P1 Regularization Sweep

POC note: spread inputs are source-audited; added-name and VIX rows use measured panel CBBO when present and otherwise use a point-in-time inferred CBBO proxy calibrated from the historical liquid equity/ETF CBBO surface; off-hours Cboe snapshots are rejected and the old blanket 10%/15% class defaults are not used in the breadth-solution reruns

Baseline note: `orig` uses measured historical panel CBBO for all eight equity underlyings. `orig+VIX` uses the same exact equity-option CBBO rows, while VIX option spreads use the inferred liquid-option CBBO proxy.

## Results

| Config | Strategy | Arm | Point | Status | Underlyings | Contracts | Gross Sharpe | Net Sharpe No-Impact |
|---|---|---:|---|---|---:|---:|---:|---:|
| larger | Greek Markowitz | D | D_stz_0p75_hw_0p00 | optimal | 56 | 262 | 0.834 | 0.288 |
| larger | Greek Markowitz | D | D_stz_0p90_hw_0p00 | optimal | 56 | 262 | 0.834 | 0.288 |
| larger | Greek Markowitz | D | D_stz_0p60_hw_0p00 | optimal | 56 | 262 | 0.834 | 0.288 |
| larger | Greek Markowitz | B | B_residual_lw_cov_0p20 | optimal | 56 | 262 | 0.742 | 0.470 |
| larger | Greek Markowitz | B | B_residual_diag_cov_0p20 | optimal | 56 | 262 | 0.714 | 0.287 |
| larger | Greek Markowitz | B | B_residual_lw_cov_0p50 | optimal | 56 | 262 | 0.695 | 0.433 |
| larger | Greek Markowitz | B | B_residual_diag_cov_0p50 | optimal | 56 | 262 | 0.690 | 0.318 |
| larger | Greek Markowitz | E | E4_residual_diag_cov065_stz075 | optimal | 56 | 262 | 0.687 | 0.338 |
| larger | Greek Markowitz | E | E1_residual_diag_n_scaled_hw0_stz075 | optimal | 56 | 262 | 0.654 | 0.368 |
| larger | Greek Markowitz | E | E3_under_lw_residual_diag_n_scaled | optimal | 56 | 262 | 0.653 | 0.334 |
| larger | Equal risk | naive | Equal risk | reference | 56 | 262 | 0.638 | 0.348 |
| larger | Greek Markowitz | A | A_cov_n_scaled | optimal | 56 | 262 | 0.623 | 0.266 |
| larger | Greek Markowitz | A | A_cov_0p80 | optimal | 56 | 262 | 0.622 | 0.268 |
| larger | Greek Markowitz | A | A_cov_0p65 | optimal | 56 | 262 | 0.618 | 0.229 |
| larger | Greek Markowitz | A | A_cov_0p90 | optimal | 56 | 262 | 0.595 | 0.270 |
| larger | Greek Markowitz | A | A_cov_0p50 | optimal | 56 | 262 | 0.563 | 0.127 |
| larger | Greek Markowitz | C | C_under_lw | optimal | 56 | 262 | 0.560 | 0.020 |
| larger | Equal premium | naive | Equal premium | reference | 56 | 262 | 0.557 | 0.336 |
| larger | Greek Markowitz | E | E2_residual_lw_cov050_hw0_stz060 | optimal | 56 | 262 | 0.510 | 0.275 |
| larger | Greek Markowitz | A | A_cov_0p35 | optimal | 56 | 262 | 0.509 | 0.030 |
| larger | Greek Markowitz | D | D_stz_0p75_hw_0p25 | optimal | 56 | 262 | 0.456 | -0.075 |
| larger | Greek Markowitz | D | D_stz_0p90_hw_0p25 | optimal | 56 | 262 | 0.456 | -0.075 |
| larger | Greek Markowitz | default | default | optimal | 56 | 262 | 0.456 | -0.075 |
| larger | Greek Markowitz | C | C_under_single_factor | optimal | 56 | 262 | 0.443 | -0.014 |
| larger+VIX | Greek Markowitz | E | E1_residual_diag_n_scaled_hw0_stz075 | optimal | 57 | 267 | 1.501 | 1.157 |
| larger+VIX | Greek Markowitz | E | E2_residual_lw_cov050_hw0_stz060 | optimal | 57 | 267 | 1.458 | 1.159 |
| larger+VIX | Greek Markowitz | D | D_stz_0p75_hw_0p00 | optimal | 57 | 267 | 1.427 | 0.757 |
| larger+VIX | Greek Markowitz | D | D_stz_0p90_hw_0p00 | optimal | 57 | 267 | 1.412 | 0.694 |
| larger+VIX | Greek Markowitz | D | D_stz_0p60_hw_0p00 | optimal | 57 | 267 | 1.368 | 0.728 |
| larger+VIX | Greek Markowitz | E | E4_residual_diag_cov065_stz075 | optimal | 57 | 267 | 1.264 | 0.869 |
| larger+VIX | Greek Markowitz | B | B_residual_lw_cov_0p20 | optimal | 57 | 267 | 1.241 | 0.952 |
| larger+VIX | Greek Markowitz | B | B_residual_diag_cov_0p20 | optimal | 57 | 267 | 1.221 | 0.759 |
| larger+VIX | Greek Markowitz | B | B_residual_lw_cov_0p50 | optimal | 57 | 267 | 1.194 | 0.912 |
| larger+VIX | Greek Markowitz | B | B_residual_diag_cov_0p50 | optimal | 57 | 267 | 1.187 | 0.775 |
| larger+VIX | Greek Markowitz | E | E3_under_lw_residual_diag_n_scaled | optimal | 57 | 267 | 1.091 | 0.742 |
| larger+VIX | Greek Markowitz | A | A_cov_n_scaled | optimal | 57 | 267 | 0.995 | 0.596 |
| larger+VIX | Greek Markowitz | A | A_cov_0p80 | optimal | 57 | 267 | 0.993 | 0.596 |
| larger+VIX | Greek Markowitz | A | A_cov_0p65 | optimal | 57 | 267 | 0.978 | 0.530 |
| larger+VIX | Greek Markowitz | C | C_under_lw | optimal | 57 | 267 | 0.974 | 0.397 |
| larger+VIX | Greek Markowitz | A | A_cov_0p90 | optimal | 57 | 267 | 0.940 | 0.579 |
| larger+VIX | Greek Markowitz | D | D_stz_0p90_hw_0p25 | optimal | 57 | 267 | 0.928 | 0.334 |
| larger+VIX | Greek Markowitz | A | A_cov_0p50 | optimal | 57 | 267 | 0.912 | 0.415 |
| larger+VIX | Greek Markowitz | D | D_stz_0p75_hw_0p25 | optimal | 57 | 267 | 0.848 | 0.272 |
| larger+VIX | Greek Markowitz | A | A_cov_0p35 | optimal | 57 | 267 | 0.840 | 0.306 |
| larger+VIX | Greek Markowitz | C | C_under_single_factor | optimal | 57 | 267 | 0.819 | 0.323 |
| larger+VIX | Greek Markowitz | default | default | optimal | 57 | 267 | 0.765 | 0.202 |
| larger+VIX | Equal risk | naive | Equal risk | reference | 57 | 267 | 0.576 | 0.282 |
| larger+VIX | Equal premium | naive | Equal premium | reference | 57 | 267 | 0.485 | 0.258 |
| orig | Greek Markowitz | B | B_residual_diag_cov_0p20 | optimal | 8 | 49 | 0.955 | 0.742 |
| orig | Greek Markowitz | B | B_residual_lw_cov_0p20 | optimal | 8 | 49 | 0.951 | 0.723 |
| orig | Greek Markowitz | E | E3_under_lw_residual_diag_n_scaled | optimal | 8 | 49 | 0.948 | 0.737 |
| orig | Greek Markowitz | C | C_under_single_factor | optimal | 8 | 49 | 0.911 | 0.638 |
| orig | Greek Markowitz | B | B_residual_diag_cov_0p50 | optimal | 8 | 49 | 0.866 | 0.677 |
| orig | Greek Markowitz | A | A_cov_0p35 | optimal | 8 | 49 | 0.866 | 0.629 |
| orig | Greek Markowitz | A | A_cov_0p50 | optimal | 8 | 49 | 0.859 | 0.647 |
| orig | Greek Markowitz | E | E1_residual_diag_n_scaled_hw0_stz075 | optimal | 8 | 49 | 0.845 | 0.661 |
| orig | Greek Markowitz | default | default | optimal | 8 | 49 | 0.842 | 0.578 |
| orig | Greek Markowitz | A | A_cov_n_scaled | optimal | 8 | 49 | 0.842 | 0.578 |
| orig | Greek Markowitz | D | D_stz_0p90_hw_0p25 | optimal | 8 | 49 | 0.842 | 0.578 |
| orig | Greek Markowitz | D | D_stz_0p75_hw_0p25 | optimal | 8 | 49 | 0.842 | 0.578 |
| orig | Greek Markowitz | B | B_residual_lw_cov_0p50 | optimal | 8 | 49 | 0.838 | 0.644 |
| orig | Greek Markowitz | A | A_cov_0p65 | optimal | 8 | 49 | 0.828 | 0.638 |
| orig | Greek Markowitz | C | C_under_lw | optimal | 8 | 49 | 0.828 | 0.570 |
| orig | Greek Markowitz | E | E4_residual_diag_cov065_stz075 | optimal | 8 | 49 | 0.812 | 0.635 |
| orig | Greek Markowitz | E | E2_residual_lw_cov050_hw0_stz060 | optimal | 8 | 49 | 0.780 | 0.586 |
| orig | Greek Markowitz | A | A_cov_0p80 | optimal | 8 | 49 | 0.780 | 0.608 |
| orig | Greek Markowitz | A | A_cov_0p90 | optimal | 8 | 49 | 0.734 | 0.570 |
| orig | Greek Markowitz | D | D_stz_0p60_hw_0p00 | optimal | 8 | 49 | 0.725 | 0.498 |
| orig | Greek Markowitz | D | D_stz_0p75_hw_0p00 | optimal | 8 | 49 | 0.725 | 0.498 |
| orig | Greek Markowitz | D | D_stz_0p90_hw_0p00 | optimal | 8 | 49 | 0.725 | 0.498 |
| orig | Equal risk | naive | Equal risk | reference | 8 | 49 | 0.340 | 0.149 |
| orig | Equal premium | naive | Equal premium | reference | 8 | 49 | 0.267 | 0.115 |
| orig+VIX | Greek Markowitz | E | E1_residual_diag_n_scaled_hw0_stz075 | optimal | 9 | 54 | 1.747 | 1.409 |
| orig+VIX | Greek Markowitz | D | D_stz_0p90_hw_0p25 | optimal | 9 | 54 | 1.629 | 1.106 |
| orig+VIX | Greek Markowitz | B | B_residual_diag_cov_0p20 | optimal | 9 | 54 | 1.625 | 1.341 |
| orig+VIX | Greek Markowitz | E | E3_under_lw_residual_diag_n_scaled | optimal | 9 | 54 | 1.619 | 1.344 |
| orig+VIX | Greek Markowitz | E | E4_residual_diag_cov065_stz075 | optimal | 9 | 54 | 1.609 | 1.346 |
| orig+VIX | Greek Markowitz | D | D_stz_0p90_hw_0p00 | optimal | 9 | 54 | 1.605 | 0.966 |
| orig+VIX | Greek Markowitz | C | C_under_single_factor | optimal | 9 | 54 | 1.570 | 1.200 |
| orig+VIX | Greek Markowitz | E | E2_residual_lw_cov050_hw0_stz060 | optimal | 9 | 54 | 1.567 | 1.292 |
| orig+VIX | Greek Markowitz | D | D_stz_0p75_hw_0p00 | optimal | 9 | 54 | 1.557 | 1.136 |
| orig+VIX | Greek Markowitz | B | B_residual_diag_cov_0p50 | optimal | 9 | 54 | 1.541 | 1.282 |
| orig+VIX | Greek Markowitz | B | B_residual_lw_cov_0p20 | optimal | 9 | 54 | 1.476 | 1.237 |
| orig+VIX | Greek Markowitz | D | D_stz_0p75_hw_0p25 | optimal | 9 | 54 | 1.445 | 1.104 |
| orig+VIX | Greek Markowitz | B | B_residual_lw_cov_0p50 | optimal | 9 | 54 | 1.397 | 1.168 |
| orig+VIX | Greek Markowitz | C | C_under_lw | optimal | 9 | 54 | 1.386 | 1.062 |
| orig+VIX | Greek Markowitz | A | A_cov_0p35 | optimal | 9 | 54 | 1.381 | 1.099 |
| orig+VIX | Greek Markowitz | default | default | optimal | 9 | 54 | 1.374 | 1.036 |
| orig+VIX | Greek Markowitz | A | A_cov_n_scaled | optimal | 9 | 54 | 1.374 | 1.036 |
| orig+VIX | Greek Markowitz | D | D_stz_0p60_hw_0p00 | optimal | 9 | 54 | 1.373 | 0.948 |
| orig+VIX | Greek Markowitz | A | A_cov_0p50 | optimal | 9 | 54 | 1.367 | 1.107 |
| orig+VIX | Greek Markowitz | A | A_cov_0p65 | optimal | 9 | 54 | 1.342 | 1.096 |
| orig+VIX | Greek Markowitz | A | A_cov_0p80 | optimal | 9 | 54 | 1.325 | 1.091 |
| orig+VIX | Greek Markowitz | A | A_cov_0p90 | optimal | 9 | 54 | 1.310 | 1.084 |
| orig+VIX | Equal risk | naive | Equal risk | reference | 9 | 54 | 0.082 | -0.129 |
| orig+VIX | Equal premium | naive | Equal premium | reference | 9 | 54 | -0.030 | -0.207 |

## Verdict

- larger: best GM 0.834 (D/D_stz_0p75_hw_0p00) vs 8-name default gross bar 0.842 and Equal-premium gross 0.557. FAIL regularized 56-name gross >= 8-name gross.
- larger+VIX: best GM 1.501 (E/E1_residual_diag_n_scaled_hw0_stz075) vs 8-name default gross bar 1.374 and Equal-premium gross 0.485. PASS regularized 56-name gross >= 8-name gross.
