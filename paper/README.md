# Working Paper Guide

The complete paper is [paper.pdf](paper.pdf). The manuscript develops a funded option model,
derives the full covariance of Greek-factor and residual returns, and studies two constrained
portfolio specifications. R1 imposes a 15% volatility ceiling. R1.1 uses the same policy with a
25% ceiling.

The main R1.1 development result uses eight equity names plus VIX over 93 months. Under modeled
costs, its annualized mean return is 15.8%, its CAGR is 16.0%, its Sortino ratio is 2.275, and
its maximum drawdown is 17.1%. The touch-price sensitivity has a Sortino ratio of 2.052. Quote
coverage is 94.1% at entry and 25.5% for complete round trips. These are retrospective
sensitivity results, not routed orders or realized fills.

The counts in the paper refer to different samples. The R1 result uses the 60-month post-2020
claim window; the R1 and R1.1 comparison uses the 93-month aligned development window. The PBO
analysis is likewise reported for two scopes: a liquid-era window and the narrower claim
window. The search ledgers contain 598 recorded R1 trials and 607 recorded R1.1 trials. Those
counts are lower bounds because the historical registries are incomplete.

The return expansion holds Greeks fixed over each evaluation interval. It therefore omits
intraperiod re-hedging, higher-order Greeks, smile dynamics beyond the retained factors, and
path-dependent exercise effects. The structural-mean bound checks the magnitude of the stated
premia under this model; it is not evidence that those premia are identified or will persist.

## Files

```text
paper.tex          manuscript root
sections/          human-edited section source
figures/           12 referenced figures
tables/            12 referenced compact tables
evidence/          portfolio-level results, claim checks, counts, and hashes
references.bib     cited references only
paper.pdf          sole reader-facing output
```

The exact R1 and R1.1 monthly return files retain their historical machine labels so their
source hashes remain unchanged. Reader-facing prose, tables, figures, and regenerated summaries
use neutral specification names.

## Building and checking

Run these commands from the repository root:

```bash
make paper
make verify-artifacts
```

`make paper` compiles a clean candidate under `build/` and does not alter this directory.
Maintainers use `make release` to promote a verified candidate.

## Data availability

The public evidence is aggregated at the portfolio-month level. Raw quote, trade, contract, and
licensed market data are not distributed, and no controlled reviewer-access program is
promised. The retained hashes document the historical source artifacts without implying that
the refactored code recreated the unavailable feature stores. The required private input paths
and the two verification modes are described in [../data/README.md](../data/README.md).
