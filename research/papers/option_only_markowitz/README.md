# Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional Risk Premia

This folder is the self-contained publication bundle for the paper:

```text
option_only_portfolio_optimization_dhruv_kohli.pdf
```

The paper develops a premium-weighted option-only Markowitz framework for listed calls and puts. Options are treated as funded, expiring, state-contingent cashflows rather than ordinary asset-return columns. The empirical layer is a point-in-time research simulation with exact VRO/SOQ settlement gating, post-cost diagnostics, inference, and claim-audit checks. It is not broker-executed or live-trading evidence.

## What Is Included

- `option_only_portfolio_optimization_dhruv_kohli.tex`: LaTeX root.
- `option_only_portfolio_optimization_dhruv_kohli.pdf`: final compiled paper.
- `sections/`: article sections and Appendix A.
- `analysis/`: empirical pipeline, VIX panel construction, cost scenarios, inference, and publication utilities.
- `tables/`, `figures/`, `artifacts/`: generated paper outputs and machine-readable ledgers.
- `verification/`: independent paper verifier and pass/fail reports.
- `docs/`: source ledger, replication package, and release notes.
- `REPRODUCIBILITY.md`: compact reproducibility note.

## Quickstart

Run from the repository root after creating `.env` from `.env.example` and installing `requirements.txt` into `.venv`.

```bash
# Dry-run the paper data plan; no downloads and no paid data pulls.
.venv/bin/python -m data_pull.pull --preset option-paper

# Optional: pull credential-free public inputs, including Cboe VRO/SOQ settlement.
make data-public

# Regenerate paper artifacts and compile the PDF.
make paper

# Run the independent verifier.
make verify

# Run focused publication tests.
make test
```

The canonical PDF path is:

```text
research/papers/option_only_markowitz/option_only_portfolio_optimization_dhruv_kohli.pdf
```

## Data Requirements

The paper is reproducible for a licensed local user, but raw OPRA/Databento data are not redistributed. A reproducer must supply or regenerate the expected local files:

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/option_greek_quality.csv`
- `data/universe/multi_raw_close.csv`
- `data/databento_cache/opra_vix_chain_*.parquet`
- `data/universe/vx_futures_daily.parquet`
- `data/universe/vix_complex.parquet`

Exact VRO/SOQ settlement can be downloaded from public Cboe inputs with `make data-public` or supplied through `OPTION_MARKOWITZ_VRO_FILE` / `OPTION_MARKOWITZ_VRO_DIR`. VIX option results are headline-grade only when every required VIX expiry row uses `vro_soq_exact` settlement.

## Verification Standard

The paper is considered release-ready only when these commands pass:

```bash
make paper
make verify
make test
```

The verifier records the current check count in `verification/verification_summary.json` and covers data lineage, point-in-time timing, optimizer constraints, settlement coverage, cost ledgers, figures, bibliography scope, PDF rendering, and claim boundaries, including independent recomputation of the reported performance metrics and ordering checks on every bootstrap confidence interval. The latest verification report is in `verification/verification_report.md`.

## New Research Diagnostics

The pipeline also emits repaired execution-sensitivity scenarios, a cost-aware Sortino variant, a CBBO spread cost surface, and VIX chain state diagnostics. The repaired scenarios use `_repaired` labels and are execution-sensitivity analysis only; they are deliberately excluded from headline growth tables and the reality-check family. The `Cost-aware Sortino + VIX` strategy uses train-window-only entry-cost estimates and is diagnostic rather than a headline or simulation strategy, although its gross and scenario columns expand the reality-check family. `make cbbo-surface` builds `data/feature_store/cbbo_spread_surface.parquet` from the local OPRA full-day CBBO cache when available, refining rows that otherwise fall back to class-default spreads. VIX chain features and vol-of-vol regime tables are diagnostics only and do not feed expected returns. All of these outputs remain research simulation evidence, not live tradability evidence.

### Distributional Robustness Stage

Run `make robustness` to build the distributional-robustness layer. The stage takes roughly
35-40 minutes on the publication machine, uses fixed seeds recorded in
`tables/distributional_robustness_summary.json`, and writes `artifacts/cv_*.csv`,
`artifacts/mc_*.csv`, `tables/cv_*.tex`, `tables/mc_*.tex`, and the summary JSON. CPCV in
this layer is intentionally non-PIT: it is an overfitting and path-distribution diagnostic,
not a tradable out-of-sample claim.

## Claim Boundary

The durable claim is a disciplined option allocation framework and validation layer. The paper does not claim live alpha, production tradability, broker-executed fills, live margin parity, or deployable option trading performance. Production deployment would require broker-grade quotes, fills, margin previews, assignment/exercise handling, borrow, capacity, order routing, and reconciliation.
