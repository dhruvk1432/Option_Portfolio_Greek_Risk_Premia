# P2 Liquidity-Cap Sweep

POC note: spread inputs are source-audited; added-name and VIX rows use measured panel CBBO when present and otherwise use a point-in-time inferred CBBO proxy calibrated from the historical liquid equity/ETF CBBO surface; off-hours Cboe snapshots are rejected and the old blanket 10%/15% class defaults are not used in the breadth-solution reruns

Baseline note: `orig` uses measured historical panel CBBO for all eight equity underlyings. `orig+VIX` uses the same exact equity-option CBBO rows, while VIX option spreads use the inferred liquid-option CBBO proxy.

## orig+VIX

| Strategy | X | AUM | Mode | Net Sharpe | Max Capacity Ratio | Capacity Infeasible |
|---|---:|---:|---|---:|---:|---|
| Equal premium |  | 1000000 | naive | -1.277 | 206.449 | False |
| Equal premium capped X=0.02 | 0.02 | 1000000 | relaxed | -0.898 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 1000000 | hard | -0.853 | 32.638 | False |
| Equal premium capped X=0.10 | 0.10 | 1000000 | hard | -0.333 | 65.048 | False |
| Equal risk |  | 1000000 | naive | -0.863 | 115.882 | False |
| Equal risk capped X=0.02 | 0.02 | 1000000 | relaxed | -1.027 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 1000000 | hard | -0.977 | 32.638 | False |
| Equal risk capped X=0.10 | 0.10 | 1000000 | hard | -0.114 | 65.048 | False |
| GM X=0.02 | 0.02 | 1000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 1000000 | relaxed | 1.185 | 1.785 | True |
| GM X=0.05 | 0.05 | 1000000 | hard | 1.080 | 32.638 | False |
| GM X=0.10 | 0.10 | 1000000 | hard | 0.484 | 65.048 | False |
| GM X=inf | inf | 1000000 | uncapped | -1.196 | 187.756 | False |
| Equal premium |  | 5000000 | naive | -1.928 | 1032.247 | False |
| Equal premium capped X=0.02 | 0.02 | 5000000 | relaxed | -0.898 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 5000000 | relaxed | -0.992 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 5000000 | relaxed | -1.239 | 65.275 | True |
| Equal risk |  | 5000000 | naive | -2.593 | 579.411 | False |
| Equal risk capped X=0.02 | 0.02 | 5000000 | relaxed | -1.027 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 5000000 | relaxed | -1.134 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 5000000 | relaxed | -1.406 | 65.275 | True |
| GM X=0.02 | 0.02 | 5000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 5000000 | relaxed | 1.185 | 1.809 | True |
| GM X=0.05 | 0.05 | 5000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 5000000 | relaxed | 1.185 | 4.755 | True |
| GM X=0.10 | 0.10 | 5000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 5000000 | relaxed | 1.183 | 8.773 | True |
| GM X=inf | inf | 5000000 | uncapped | -1.425 | 938.778 | False |
| Equal premium |  | 10000000 | naive | -1.932 | 2064.495 | False |
| Equal premium capped X=0.02 | 0.02 | 10000000 | relaxed | -0.898 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 10000000 | relaxed | -0.992 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 10000000 | relaxed | -1.239 | 65.275 | True |
| Equal risk |  | 10000000 | naive | -2.637 | 1158.822 | False |
| Equal risk capped X=0.02 | 0.02 | 10000000 | relaxed | -1.027 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 10000000 | relaxed | -1.134 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 10000000 | relaxed | -1.406 | 65.275 | True |
| GM X=0.02 | 0.02 | 10000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 10000000 | relaxed | 1.185 | 1.837 | True |
| GM X=0.05 | 0.05 | 10000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 10000000 | relaxed | 1.185 | 4.326 | True |
| GM X=0.10 | 0.10 | 10000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 10000000 | relaxed | 1.183 | 9.511 | True |
| GM X=inf | inf | 10000000 | uncapped | -1.435 | 1877.557 | False |
| Equal premium |  | 25000000 | naive | -1.935 | 5161.237 | False |
| Equal premium capped X=0.02 | 0.02 | 25000000 | relaxed | -0.898 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 25000000 | relaxed | -0.992 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 25000000 | relaxed | -1.239 | 65.275 | True |
| Equal risk |  | 25000000 | naive | -2.654 | 2897.056 | False |
| Equal risk capped X=0.02 | 0.02 | 25000000 | relaxed | -1.027 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 25000000 | relaxed | -1.134 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 25000000 | relaxed | -1.406 | 65.275 | True |
| GM X=0.02 | 0.02 | 25000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 25000000 | relaxed | 1.185 | 1.930 | True |
| GM X=0.05 | 0.05 | 25000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 25000000 | relaxed | 1.185 | 4.502 | True |
| GM X=0.10 | 0.10 | 25000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 25000000 | relaxed | 1.183 | 9.426 | True |
| GM X=inf | inf | 25000000 | uncapped | -1.439 | 4693.892 | False |

Verdict: orig+VIX largest positive-net capped AUM=$25,000,000 vs uncapped positive-net AUM=n/a. Caps do collapse max capacity ratio vs X=inf (best capped 1.785, uncapped max 4693.892).

## larger+VIX

| Strategy | X | AUM | Mode | Net Sharpe | Max Capacity Ratio | Capacity Infeasible |
|---|---:|---:|---|---:|---:|---|
| Equal premium |  | 1000000 | naive | -1.289 | 149.813 | False |
| Equal premium capped X=0.02 | 0.02 | 1000000 | relaxed | -0.408 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 1000000 | hard | 0.073 | 47.260 | False |
| Equal premium capped X=0.10 | 0.10 | 1000000 | hard | 0.202 | 94.520 | False |
| Equal risk |  | 1000000 | naive | -1.382 | 159.889 | False |
| Equal risk capped X=0.02 | 0.02 | 1000000 | relaxed | -0.500 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 1000000 | hard | 0.266 | 47.260 | False |
| Equal risk capped X=0.10 | 0.10 | 1000000 | hard | 0.214 | 80.481 | False |
| GM X=0.02 | 0.02 | 1000000 | hard | 0.500 | 18.904 | True |
| GM X=0.02 | 0.02 | 1000000 | relaxed | 1.028 | 11.141 | True |
| GM X=0.05 | 0.05 | 1000000 | hard | 1.109 | 30.808 | False |
| GM X=0.10 | 0.10 | 1000000 | hard | 0.705 | 55.704 | False |
| GM X=inf | inf | 1000000 | uncapped | -1.837 | 539.826 | False |
| Equal premium |  | 5000000 | naive | -4.249 | 749.064 | False |
| Equal premium capped X=0.02 | 0.02 | 5000000 | relaxed | -0.408 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 5000000 | relaxed | -0.501 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 5000000 | relaxed | -0.779 | 94.520 | True |
| Equal risk |  | 5000000 | naive | -5.085 | 799.445 | False |
| Equal risk capped X=0.02 | 0.02 | 5000000 | relaxed | -0.500 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 5000000 | relaxed | -0.605 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 5000000 | relaxed | -0.911 | 94.520 | True |
| GM X=0.02 | 0.02 | 5000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 5000000 | relaxed | -0.224 | 1.025 | True |
| GM X=0.05 | 0.05 | 5000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 5000000 | relaxed | -0.224 | 1.957 | True |
| GM X=0.10 | 0.10 | 5000000 | hard | 0.040 | 94.520 | True |
| GM X=0.10 | 0.10 | 5000000 | relaxed | 0.639 | 55.704 | True |
| GM X=inf | inf | 5000000 | uncapped | -1.901 | 2699.129 | False |
| Equal premium |  | 10000000 | naive | -4.254 | 1498.127 | False |
| Equal premium capped X=0.02 | 0.02 | 10000000 | relaxed | -0.408 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 10000000 | relaxed | -0.501 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 10000000 | relaxed | -0.779 | 94.520 | True |
| Equal risk |  | 10000000 | naive | -5.149 | 1598.891 | False |
| Equal risk capped X=0.02 | 0.02 | 10000000 | relaxed | -0.500 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 10000000 | relaxed | -0.605 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 10000000 | relaxed | -0.911 | 94.520 | True |
| GM X=0.02 | 0.02 | 10000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 10000000 | relaxed | -0.224 | 1.003 | True |
| GM X=0.05 | 0.05 | 10000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 10000000 | relaxed | -0.224 | 1.909 | True |
| GM X=0.10 | 0.10 | 10000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 10000000 | relaxed | -0.224 | 3.913 | True |
| GM X=inf | inf | 10000000 | uncapped | -1.906 | 5398.258 | False |
| Equal premium |  | 25000000 | naive | -4.257 | 3745.318 | False |
| Equal premium capped X=0.02 | 0.02 | 25000000 | relaxed | -0.408 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 25000000 | relaxed | -0.501 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 25000000 | relaxed | -0.779 | 94.520 | True |
| Equal risk |  | 25000000 | naive | -5.170 | 3997.227 | False |
| Equal risk capped X=0.02 | 0.02 | 25000000 | relaxed | -0.500 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 25000000 | relaxed | -0.605 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 25000000 | relaxed | -0.911 | 94.520 | True |
| GM X=0.02 | 0.02 | 25000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 25000000 | relaxed | -0.224 | 1.012 | True |
| GM X=0.05 | 0.05 | 25000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 25000000 | relaxed | -0.224 | 1.913 | True |
| GM X=0.10 | 0.10 | 25000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 25000000 | relaxed | -0.224 | 3.493 | True |
| GM X=inf | inf | 25000000 | uncapped | -1.909 | 13495.646 | False |

Verdict: larger+VIX largest positive-net capped AUM=$5,000,000 vs uncapped positive-net AUM=n/a. Caps do collapse max capacity ratio vs X=inf (best capped 1.003, uncapped max 13495.646).

## orig

| Strategy | X | AUM | Mode | Net Sharpe | Max Capacity Ratio | Capacity Infeasible |
|---|---:|---:|---|---:|---:|---|
| Equal premium |  | 1000000 | naive | -1.202 | 227.516 | False |
| Equal premium capped X=0.02 | 0.02 | 1000000 | relaxed | 0.405 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 1000000 | relaxed | 0.350 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 1000000 | hard | 0.220 | 65.275 | False |
| Equal risk |  | 1000000 | naive | -0.731 | 124.002 | False |
| Equal risk capped X=0.02 | 0.02 | 1000000 | relaxed | 0.414 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 1000000 | relaxed | 0.327 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 1000000 | hard | 0.252 | 65.048 | False |
| GM X=0.02 | 0.02 | 1000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 1000000 | relaxed | 0.492 | 1.169 | True |
| GM X=0.05 | 0.05 | 1000000 | hard | 0.580 | 32.524 | True |
| GM X=0.05 | 0.05 | 1000000 | relaxed | 0.909 | 32.524 | True |
| GM X=0.10 | 0.10 | 1000000 | hard | 0.779 | 65.048 | False |
| GM X=inf | inf | 1000000 | uncapped | -1.443 | 184.866 | False |
| Equal premium |  | 5000000 | naive | -1.916 | 1137.579 | False |
| Equal premium capped X=0.02 | 0.02 | 5000000 | relaxed | 0.405 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 5000000 | relaxed | 0.350 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 5000000 | relaxed | 0.150 | 65.275 | True |
| Equal risk |  | 5000000 | naive | -2.579 | 620.008 | False |
| Equal risk capped X=0.02 | 0.02 | 5000000 | relaxed | 0.414 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 5000000 | relaxed | 0.327 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 5000000 | relaxed | 0.013 | 65.275 | True |
| GM X=0.02 | 0.02 | 5000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 5000000 | relaxed | 0.492 | 1.160 | True |
| GM X=0.05 | 0.05 | 5000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 5000000 | relaxed | 0.491 | 2.586 | True |
| GM X=0.10 | 0.10 | 5000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 5000000 | relaxed | 0.484 | 5.659 | True |
| GM X=inf | inf | 5000000 | uncapped | -1.595 | 924.332 | False |
| Equal premium |  | 10000000 | naive | -1.928 | 2275.158 | False |
| Equal premium capped X=0.02 | 0.02 | 10000000 | relaxed | 0.405 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 10000000 | relaxed | 0.350 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 10000000 | relaxed | 0.150 | 65.275 | True |
| Equal risk |  | 10000000 | naive | -2.633 | 1240.015 | False |
| Equal risk capped X=0.02 | 0.02 | 10000000 | relaxed | 0.414 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 10000000 | relaxed | 0.327 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 10000000 | relaxed | 0.013 | 65.275 | True |
| GM X=0.02 | 0.02 | 10000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 10000000 | relaxed | 0.492 | 0.993 | True |
| GM X=0.05 | 0.05 | 10000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 10000000 | relaxed | 0.491 | 2.667 | True |
| GM X=0.10 | 0.10 | 10000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 10000000 | relaxed | 0.486 | 5.173 | True |
| GM X=inf | inf | 10000000 | uncapped | -1.603 | 1848.665 | False |
| Equal premium |  | 25000000 | naive | -1.934 | 5687.894 | False |
| Equal premium capped X=0.02 | 0.02 | 25000000 | relaxed | 0.405 | 13.055 | True |
| Equal premium capped X=0.05 | 0.05 | 25000000 | relaxed | 0.350 | 32.638 | True |
| Equal premium capped X=0.10 | 0.10 | 25000000 | relaxed | 0.150 | 65.275 | True |
| Equal risk |  | 25000000 | naive | -2.653 | 3100.038 | False |
| Equal risk capped X=0.02 | 0.02 | 25000000 | relaxed | 0.414 | 13.055 | True |
| Equal risk capped X=0.05 | 0.05 | 25000000 | relaxed | 0.327 | 32.638 | True |
| Equal risk capped X=0.10 | 0.10 | 25000000 | relaxed | 0.013 | 65.275 | True |
| GM X=0.02 | 0.02 | 25000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 25000000 | relaxed | 0.492 | 1.081 | True |
| GM X=0.05 | 0.05 | 25000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 25000000 | relaxed | 0.492 | 2.374 | True |
| GM X=0.10 | 0.10 | 25000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 25000000 | relaxed | 0.484 | 5.622 | True |
| GM X=inf | inf | 25000000 | uncapped | -1.607 | 4621.662 | False |

Verdict: orig largest positive-net capped AUM=$25,000,000 vs uncapped positive-net AUM=n/a. Caps do collapse max capacity ratio vs X=inf (best capped 0.993, uncapped max 4621.662).

## larger

| Strategy | X | AUM | Mode | Net Sharpe | Max Capacity Ratio | Capacity Infeasible |
|---|---:|---:|---|---:|---:|---|
| Equal premium |  | 1000000 | naive | -1.278 | 152.672 | False |
| Equal premium capped X=0.02 | 0.02 | 1000000 | relaxed | 0.516 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 1000000 | hard | 0.475 | 47.260 | False |
| Equal premium capped X=0.10 | 0.10 | 1000000 | hard | 0.369 | 94.520 | False |
| Equal risk |  | 1000000 | naive | -1.373 | 162.011 | False |
| Equal risk capped X=0.02 | 0.02 | 1000000 | relaxed | 0.544 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 1000000 | hard | 0.550 | 47.260 | False |
| Equal risk capped X=0.10 | 0.10 | 1000000 | hard | 0.348 | 80.481 | False |
| GM X=0.02 | 0.02 | 1000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 1000000 | relaxed | -0.557 | 1.156 | True |
| GM X=0.05 | 0.05 | 1000000 | hard | 0.741 | 40.241 | False |
| GM X=0.10 | 0.10 | 1000000 | hard | 0.259 | 55.704 | False |
| GM X=inf | inf | 1000000 | uncapped | -1.991 | 587.952 | False |
| Equal premium |  | 5000000 | naive | -4.243 | 763.359 | False |
| Equal premium capped X=0.02 | 0.02 | 5000000 | relaxed | 0.516 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 5000000 | relaxed | 0.474 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 5000000 | relaxed | 0.271 | 94.520 | True |
| Equal risk |  | 5000000 | naive | -5.082 | 810.055 | False |
| Equal risk capped X=0.02 | 0.02 | 5000000 | relaxed | 0.544 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 5000000 | relaxed | 0.472 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 5000000 | relaxed | 0.161 | 94.520 | True |
| GM X=0.02 | 0.02 | 5000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 5000000 | relaxed | -0.557 | 1.033 | True |
| GM X=0.05 | 0.05 | 5000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 5000000 | relaxed | -0.558 | 1.996 | True |
| GM X=0.10 | 0.10 | 5000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 5000000 | relaxed | -0.558 | 4.042 | True |
| GM X=inf | inf | 5000000 | uncapped | -2.018 | 2939.758 | False |
| Equal premium |  | 10000000 | naive | -4.253 | 1526.718 | False |
| Equal premium capped X=0.02 | 0.02 | 10000000 | relaxed | 0.516 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 10000000 | relaxed | 0.474 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 10000000 | relaxed | 0.271 | 94.520 | True |
| Equal risk |  | 10000000 | naive | -5.148 | 1620.110 | False |
| Equal risk capped X=0.02 | 0.02 | 10000000 | relaxed | 0.544 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 10000000 | relaxed | 0.472 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 10000000 | relaxed | 0.161 | 94.520 | True |
| GM X=0.02 | 0.02 | 10000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 10000000 | relaxed | -0.557 | 1.185 | True |
| GM X=0.05 | 0.05 | 10000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 10000000 | relaxed | -0.557 | 1.856 | True |
| GM X=0.10 | 0.10 | 10000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 10000000 | relaxed | -0.558 | 3.993 | True |
| GM X=inf | inf | 10000000 | uncapped | -2.021 | 5879.516 | False |
| Equal premium |  | 25000000 | naive | -4.257 | 3816.794 | False |
| Equal premium capped X=0.02 | 0.02 | 25000000 | relaxed | 0.516 | 18.904 | True |
| Equal premium capped X=0.05 | 0.05 | 25000000 | relaxed | 0.474 | 47.260 | True |
| Equal premium capped X=0.10 | 0.10 | 25000000 | relaxed | 0.271 | 94.520 | True |
| Equal risk |  | 25000000 | naive | -5.170 | 4050.276 | False |
| Equal risk capped X=0.02 | 0.02 | 25000000 | relaxed | 0.544 | 18.904 | True |
| Equal risk capped X=0.05 | 0.05 | 25000000 | relaxed | 0.472 | 47.260 | True |
| Equal risk capped X=0.10 | 0.10 | 25000000 | relaxed | 0.161 | 94.520 | True |
| GM X=0.02 | 0.02 | 25000000 | hard | n/a | n/a | True |
| GM X=0.02 | 0.02 | 25000000 | relaxed | -0.557 | 1.005 | True |
| GM X=0.05 | 0.05 | 25000000 | hard | n/a | n/a | True |
| GM X=0.05 | 0.05 | 25000000 | relaxed | -0.557 | 2.065 | True |
| GM X=0.10 | 0.10 | 25000000 | hard | n/a | n/a | True |
| GM X=0.10 | 0.10 | 25000000 | relaxed | -0.558 | 3.653 | True |
| GM X=inf | inf | 25000000 | uncapped | -2.024 | 14698.789 | False |

Verdict: larger largest positive-net capped AUM=$1,000,000 vs uncapped positive-net AUM=n/a. Caps do collapse max capacity ratio vs X=inf (best capped 1.005, uncapped max 14698.789).

