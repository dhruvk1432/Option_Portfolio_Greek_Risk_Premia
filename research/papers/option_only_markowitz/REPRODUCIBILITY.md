# Reproducibility

Paper: **Option-Only Portfolio Optimization with Greek-Induced Covariance and Conditional
Risk Premia**. All commands run from the repository root with `.venv` active and `.env`
copied from `.env.example`.

## 1. Environment and data

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m data_pull.pull --preset option-paper   # dry run: prints expected paths, no downloads
```

Raw licensed OPRA/Databento data are **not** redistributed. A licensed local user
reproduces by placing (or symlinking) these files under `data/`:

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/cbbo_spread_surface.parquet` (derived; see step 2)
- `data/feature_store/option_greek_quality.csv`
- `data/universe/multi_raw_close.csv`
- `data/databento_cache/opra_vix_chain_*.parquet`
- `data/universe/vx_futures_daily.parquet`, `data/universe/vix_complex.parquet`

Exact VIX settlement (VRO/SOQ) is credential-free from public Cboe endpoints:

```bash
make data-public
# or supply via OPTION_MARKOWITZ_VRO_DIR / OPTION_MARKOWITZ_VRO_FILE
```

VIX-enabled results are headline-grade only when every required VIX expiry row uses
`vro_soq_exact` settlement; otherwise those rows are labeled `vix_close_settlement_proxy`
and are diagnostic.

## 2. Fast path — verify the shipped results

If the checked-in artifacts are present, this reproduces the tables and independently
audits every reported number in a few minutes:

```bash
make verify   # regenerates the fast core (run_empirics), recompiles the PDF, runs 386 checks
make test     # 155 focused unit tests
```

`make verify` passes only with zero critical failures. It does **not** rerun the
multi-hour breadth or distributional-robustness stages (those are opt-in and their
artifacts are checked as shipped). The report is written to
`verification/verification_report.md` and `verification/verification_summary.json`.

## 3. Full regeneration of the headline results

The four E1 books are produced by the breadth robustness pipeline, which `make paper` does
**not** run. Regenerate them explicitly, in order (the scoreboard must exist before the
CPCV-windows table selects each book's matched naive baseline):

```bash
# 0. Optional but recommended: build the derived CBBO spread surface used for the
#    inferred-proxy rows. Requires data/databento_cache/opra_surface_full_day_cbbo
#    (usually a symlink to the sibling OPRA cache). Absent -> class-default spread fallback.
make cbbo-surface

PD=research.papers.option_only_markowitz.analysis
# 1. Headline books, CV/CPCV, MC (resampled/refit/repriced), rolling OOS, path sims.  Hours.
.venv/bin/python -m $PD.breadth_robustness_experiment --stage all
# 2. Claim-window CPCV.  ~75 min.
.venv/bin/python -m $PD.breadth_robustness_experiment --stage claim-cv
# 3. Scoreboard + inference panel (reads the fresh return panels).
make final-results
# 4. Channel ablation (asserts Full-E1 == scoreboard to 1e-6) + concentration.
make e1-ablation
# 5. Refresh the validation summary and CPCV-windows table against the new scoreboard.
.venv/bin/python -m $PD.breadth_robustness_experiment --stage summary
make final-results
# 6. Core artifacts, base tables, and compile.
make option-paper && make paper
# 7. Audit.
make verify && make test
```

Whole-contract integer execution is applied inside `fit_books`
(`analysis/breadth_robustness_experiment.py`) and the ablation, via
`integerize_book_weights` in `analysis/breadth_solutions_lib.py`: continuous premium
weights are rounded to the nearest signed integer contract count at the 100 multiplier and
clipped so the realized weight never exceeds its per-contract liquidity cap. Because this
is the single chokepoint, it propagates to every CV/MC/rolling refit automatically.

The checked run uses `nav=1_000_000`, `participation=0.05`, 12 chronological groups, 66
CPCV splits in each of two designs (a liquid-era post-2018 primary and a claim-window
post-2020 design), 78 total CV/PBO splits per config, one-month purge/embargo, 1,000 resampled
paths, 200 refit paths, 1,000 repriced paths, 1,000 path simulations, and rolling 36-month
OOS refits. The spread-source audit passes with zero `current_cboe_liquid_quote` rows and
zero `default` rows.

### Exploratory breadth diagnostics (optional)

The P1/P2/P3 experiments document the estimator/liquidity/combined design search. They stay
on **continuous** weights on purpose (they characterize the optimizer, not the traded
book). P3 reads the P1 ledger, so run them in order:

```bash
.venv/bin/python -m $PD.breadth_p1_regularization_experiment
.venv/bin/python -m $PD.breadth_p2_liquidity_experiment --include-no-vix
.venv/bin/python -m $PD.breadth_p3_combined_experiment
```

`p1_regularization_results.csv` is consumed by the verifier; the others feed audit tables.

## 4. Cost, spread, and volume data in practice

Gross option returns come from listed-expiry payoff returns times signed premium/NAV
weights, so option premium cost is already inside the gross number. The full-cost stack then
charges spread crossing, explicit fees, slippage, borrow, margin drag, and short-option
assignment penalties. Spread sourcing is the most consequential choice: the eight equity
names use **exact historical market-hours panel CBBO**; VIX rows and unmatched broad-name
rows use a **point-in-time inferred CBBO proxy** calibrated from the liquid-option surface
(median relative spread ≈2.0% equity, ≈2.5% VIX). Proxy rows are calibrated
execution-sensitivity inputs, not final execution truth, and the paper weights the exact
`orig`/`orig+VIX` equity rows accordingly.

Liquidity caps are volume-aware and point-in-time: contract `i` is bounded by
`min(0.18, X · V_train · mark · 100 / NAV)` with `X = 0.05` and `V_train` the
median training-window daily volume. Above ≈\$2M NAV the summed caps stop supporting full
deployment; at \$5M+ they fall below one NAV for every configuration, which is the paper's
capacity result.

## 5. Point-in-time integrity

Fold-specific and rolling refits rebuild the representative contracts, mean/covariance
estimates, and liquidity caps from the fold's training window; caps use train-window volume
only. Purge/embargo drops one month around fold boundaries (verifier-measured minimum gap
59 days > the 44-day maximum decision-to-expiry span). CPCV is a distributional/overfitting
diagnostic rather than a tradable out-of-sample claim, because some recombined training sets
include calendar months after a tested block; rolling 36-month OOS is the trading-like
protocol. Known, by-design caveats: the inferred CBBO proxy for broad/VIX rows, and E1
specification selection that was in-sample across the candidate grid (mechanism-consistent,
and charged for via the deflated Sharpe ratio). The full historical audit, including
pipeline-internal findings, is in [`leakage.md`](leakage.md).

## 6. Provenance and determinism

- `environment_lock.json` — pinned interpreter and package versions.
- `artifact_hash_manifest.csv` and `verification/hash_manifest.csv` — SHA hashes of the
  shipped artifacts and paper inputs, re-checked by the verifier.
- `docs/source_ledger.md` — data source provenance; `docs/release_notes.md` — changelog.
- The distributional-robustness stage (`make robustness`, ≈35–40 min) is deterministic
  under the fixed seeds in `tables/distributional_robustness_summary.json`.
- The paper bibliography cites only papers or books; operational data-source URLs live here
  and in `docs/source_ledger.md`, not as scholarly references.

## 7. Known limitations

- Rolling buckets are representative month-end contracts, not permanent listed contracts.
- Expiry payoff uses raw daily underlying closes, not same-contract option liquidation
  quotes.
- Expected returns are estimated from training-window structural channels and remain noisy.
- The Greek factor approximation leaves material residual option-return variation.
- This validates the option-only Markowitz machinery as a backtest; it is not an
  execution-ready trading strategy. The `src/option_portfolio_production` verifier stays
  red until real settlement, order, fill, margin, assignment, and reconciliation ledgers are
  supplied.
