# Point-in-Time Leakage Audit Report

**Last Updated**: 2026-07-03
**Audit Scope**: research/papers/option_only_markowitz/analysis/{run_empirics.py, conditional_premia.py, vix_option_panel.py, vix_chain_features.py, simulation.py, publication_costs.py, execution_cost_scenarios.py, inference.py}; src/portfolio/option_only_markowitz_model.py

## Summary

- Critical Issues: 0
- High Severity: 0
- Medium Severity: 1
- Low Severity: 2
- Info: 7

No confirmed future-to-past contamination affects the headline strategies. The headline
train/test split is honest. Findings below concern (1) one supplementary robustness table that
selects a universe on future-observed liquidity, and (2) a "rolling/walk-forward" claim that the
code does not actually implement.

## Findings

### [MEDIUM] Liquidity-tier universe selected on future (end-of-sample) liquidity

**File**: `research/papers/option_only_markowitz/analysis/execution_cost_scenarios.py`
**Lines**: 346-373 (label construction); consumed in `run_empirics.py` 1665-1666, `liquidity_tier_rerun_tables` 1339-1398
**Status**: Open
**Date Identified**: 2026-07-02

**Code Evidence**:
```python
def liquidity_tier_labels(cost_inputs: pd.DataFrame) -> pd.DataFrame:
    ...
    last = cost_inputs.sort_values("return_date").groupby("asset_id", as_index=False).tail(1).copy()
    volume = pd.to_numeric(last.get("available_volume_contracts", np.nan), errors="coerce")
    ...
    vol_cut = volume.quantile(0.75) if volume.notna().any() else np.nan
    oi_cut = oi.quantile(0.75) if oi.notna().any() else np.nan
    spread_cut = spread.quantile(0.25) if spread.notna().any() else np.nan
```
`cost_inputs` spans the full sample (built from the full `return_detail`, run_empirics.py:1564).
`tail(1)` per asset therefore takes each option bucket's liquidity as observed on its LAST date,
which for surviving assets is deep in the out-of-sample window.

**Description of Leak**:
Each option bucket is assigned to `top_volume_quartile` / `tight_spread_quartile` /
`high_open_interest_quartile` / `combined_liquid` using its liquidity measured at the end of the
sample, then `liquidity_tier_rerun_tables` re-runs the optimizer inside that tier and reports its
out-of-sample Sharpe across the whole OOS window — including months before that end-of-sample
liquidity was knowable.

**Mechanism**:
A cross-sectional quantile cut over end-of-sample liquidity is a future-observed grouping key. An
asset can only be classified "top volume quartile" because we looked ahead to its final liquidity.
The tier-conditional OOS performance is then computed over dates preceding that observation.

**Impact**:
Biases the liquidity-tier robustness tables (`liquidity_tier_performance.tex`,
`liquidity_tier_diagnostics.tex`) — a mild survivorship/selection tilt toward buckets that turned
out liquid. Does NOT touch headline strategies. Magnitude is limited because tier labels are coarse
quartiles and the main result is unaffected, hence MEDIUM not HIGH.

**Severity Justification**:
Genuine point-in-time violation (grouping key uses future data) but confined to a supplementary
robustness section, not the headline claim.

**Recommended Fix**:
Compute tier labels from liquidity observed at or before `TRAIN_END` only, e.g.
`last = cost_inputs[cost_inputs["return_date"] <= TRAIN_END].sort_values("return_date").groupby("asset_id").tail(1)`,
or compute quartile cuts within the training window. Alternatively, assign each period its own
point-in-time tier and evaluate tier-conditional returns period-by-period. Document that tiers are
train-window liquidity classifications.

---

### [LOW] "Rolling / walk-forward" OOS table does not actually roll — make_model hard-codes global TRAIN_END

**File**: `research/papers/option_only_markowitz/analysis/run_empirics.py`
**Lines**: 517 (in `make_model`) interacting with 1296-1336 (`rolling_oos_table`)
**Status**: Open
**Date Identified**: 2026-07-02

**Code Evidence**:
```python
# make_model (line 517), used by every fit path:
train_returns = returns.loc[:TRAIN_END, spec.index].dropna(how="all")

# rolling_oos_table passes a rolling window that make_model then ignores:
train_end = dates[pos - 1]
...
model, _ = make_model(sub_spec, sub_returns.loc[:train_end], sub_reps[...le(train_end)], universe)
```
`rolling_oos_table` intends to fit each fold on a 36-month rolling window ending at `train_end`
(a date in 2021+). But `make_model` re-slices whatever it receives to the GLOBAL
`TRAIN_END = 2020-12-31`, so every fold is fit on the identical fixed 2020 training set.

**Description of Leak**:
Not a look-ahead leak — the fold never sees future data (it sees LESS recent data than intended).
It is a correctness/claim discrepancy: the paper's "walk-forward forecasts" and "Rolling 36M OOS"
diagnostic (`rolling_oos.tex`) are computed from a single static 2020 fit, not a rolling refit.

**Mechanism**:
Because all rolling OOS return dates satisfy `dt > TRAIN_END`, the internal `<=TRAIN_END` slice can
never include information after the test date, so there is no forward contamination. The label
"rolling / walk-forward," however, misrepresents the estimator.

**Impact**:
No inflation of returns from leakage. Risk is a false methodological claim: readers will believe the
strategy was re-estimated through time when it was not. Also, the `conditional_premia.py`
`rolling_walk_forward_weights` helper (lines 135-156) is defined but unused by the pipeline.

**Severity Justification**:
No look-ahead; purely a claim/correctness issue. LOW because it can mislead about methodology even
though it does not bias the numbers upward.

**Recommended Fix**:
Parameterize `make_model` with an explicit `train_end` argument (default `TRAIN_END`) and use it in
place of the hard-coded constant at line 517 and in `representative_specs`, `_augment_spec...`, and
`conditional_expected_returns` calls. Then `rolling_oos_table` produces a genuine walk-forward path.
Alternatively, relabel the table as a fixed-training holdout to match what the code does.

---

### [LOW] Static representative spec (Greeks/mark frozen at end-of-train) applied across full OOS window

**File**: `research/papers/option_only_markowitz/analysis/run_empirics.py`
**Lines**: 430-452 (`representative_specs`)
**Status**: Acknowledged
**Date Identified**: 2026-07-02

**Code Evidence**:
```python
train_reps = reps[reps["snap_date"].le(TRAIN_END)]
for asset_id, grp in train_reps.groupby("asset_id"):
    last = grp.sort_values("snap_date").iloc[-1]   # frozen end-of-train contract snapshot
```

**Description of Leak**:
No leak — the spec uses only `<=TRAIN_END` data (the last train observation). Flagged as LOW/risk
because the model's Greeks, mark, and spot are a single train-era snapshot held fixed while
evaluated against realized OOS option returns. This is a modeling simplification, and it is PIT-safe
in the correct direction (uses past, never future). It is called out so a future refactor that tries
to "update the spec each period" does not accidentally pull the update from the realization date.

**Mechanism**:
Weights are computed once from train-fit Sigma/mu on the frozen spec; OOS returns come from the
independent realized `option_return` panel. The frozen spec never reads test-period Greeks.

**Impact**:
None on PIT integrity. Possible model-staleness (Greeks drift over the OOS window) but that hurts,
not helps, reported performance.

**Severity Justification**:
Not a violation; recorded as a guardrail for future edits.

**Recommended Fix**:
None required. If per-period spec updates are later added, source each period's Greeks/mark strictly
from that period's decision-date snapshot (`snap_date == decision_date`), never from `return_date`.

---

### [INFO] Regime / volatility-regime performance tables use full-OOS-window terciles (reporting only)

**File**: `research/papers/option_only_markowitz/analysis/run_empirics.py`
**Lines**: 846-873 (`regime_performance_table`), 1262-1293 (`volatility_regime_performance_table`)
**Status**: Acknowledged
**Date Identified**: 2026-07-02

**Code Evidence**:
```python
q1, q2 = spy.quantile([1.0 / 3.0, 2.0 / 3.0])   # SPY terciles over ret_frame (OOS) index
...
q1, q2 = vix.quantile([1.0 / 3.0, 2.0 / 3.0])   # VIX terciles over OOS index
```

**Description**:
Terciles are computed over the out-of-sample window and used only to bucket already-realized OOS
strategy returns for conditional reporting. They do NOT influence contract selection, weights, or
the universe. This is standard conditional-performance reporting, not a backtest leak.

**Impact**:
None on strategy returns. The only caveat is presentational: the "Down/Flat/Up" and
"Low/Mid/High VIX" cut points are known only ex-post, so these are in-sample conditional summaries,
not tradable regime timing signals — which is how they are labeled.

**Severity Justification**:
Full-sample statistic used for reporting, not selection. Explicitly verified clean.

**Recommended Fix**:
None. Optionally state in the paper that regime cut points are ex-post OOS terciles.

---

### [INFO] P&L attribution uses next-period IV to decompose realized returns (ex-post attribution)

**File**: `research/papers/option_only_markowitz/analysis/run_empirics.py`
**Lines**: 762-843 (`pnl_attribution_table`), esp. 772-786
**Status**: Acknowledged
**Date Identified**: 2026-07-02

**Description**:
`next_iv - current_iv` uses the following snapshot's implied vol to attribute the vega component of
realized P&L. This is legitimate ex-post attribution of returns that already happened; it is applied
only to the attribution table and never feeds weights, selection, or the reported returns.

**Impact**:
None. Attribution correctly decomposes realized OOS P&L.

**Severity Justification**:
Reporting/attribution only, verified not to touch portfolio construction.

**Recommended Fix**:
None.

---

### [INFO] Verified-clean core components

**Status**: Acknowledged
**Date Identified**: 2026-07-02

The following were audited and found PIT-clean:

- **`src/portfolio/option_only_markowitz_model.py`** — state-free optimizer. Consumes pre-built
  Sigma/mu/Greeks; no rolling windows, shifts, or time semantics. Covariance shrinkage/PSD repair
  operate on whatever matrix is passed.
- **Covariance and mu at decision date** (`make_model`, lines 517-573) — `under_cov`, `vol_cov`,
  residual cov, conditional mu, and betas are all computed strictly from `returns.loc[:TRAIN_END]`
  and `train_returns.index`. No test-window data enters Sigma or mu. (Answers audit Q1: clean for
  the headline model.)
- **`conditional_premia.py`** — every estimator takes already-train-restricted inputs; no forward
  fill; `_safe_zscore` and the `iv.groupby(underlying).transform(median)` are cross-sectional on the
  contract snapshot, not time-series. Clean.
- **`vix_option_panel.py`** — decision-date selection uses `panel["trade_date"].le(d)`
  (build_vix_option_bucket_panel:379). VRO/SOQ used only at exact expiry (build_vix_expiry_proxy_
  returns:424-427); exact mode drops rows without exact settlement rather than substituting. VIX-
  close proxy bounded by `_last_value_on_or_before(vix, expiry, decision_date)`. `vix_state_panel`
  ffill is past-only. Settlement forward looked up as-of settlement date, used only post-expiry for
  attribution. (Answers Q5: settlement values used only at/after expiry, never for pre-expiry
  selection.)
- **Split/corporate actions** (`split_adjusted_spot_panel`, lines 149-194;
  `build_expiry_proxy_return_panel`, 221-309) — factors accumulate forward-in-time with ffill; the
  terminal spot is converted into decision-date contract units before payoff. No future split ratio
  is applied before its detection date. (Answers Q6: split handling is PIT-safe.)
- **`simulation.py`, `inference.py`, `publication_costs.py`, `execution_cost_scenarios.py`
  (cost mechanics)** — evaluation-only. Costs are subtracted from realized returns using
  decision-date spread/borrow attributes (`build_cost_input_ledger` merges on `decision_date`).
  Tail-path simulations and bootstrap inference consume realized OOS paths and never feed back into
  weights.
- **Leave-one-out table** (`leave_one_out_table`, 876-916) and headline OOS split (1490-1491) — each
  sub-universe is refit via `make_model` on train only, then evaluated on `> TRAIN_END`. Honest OOS.
  (Answers Q4: the leave-one-out split is honest. The separate "rolling" table is not truly rolling —
  see the LOW finding above.)
- **`random_feasible`** (631-642) evaluates random weights on the same OOS window as the optimized
  book; the p95 comparison is a fair OOS-vs-OOS benchmark. (Answers Q3: no full-sample normalization,
  scaling, or shrinkage target is fit on the full sample for the headline path.)

## New PIT extension entries (2026-07-03)

### [INFO] Repaired execution-sensitivity scenarios use only the row that fired the gate

**File**: `research/papers/option_only_markowitz/analysis/execution_cost_scenarios.py`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The repaired execution scenarios are diagnostics, not headline results. Quote repairs and
capacity partial fills use only the same decision-date cost-input row that triggered the
original rejection. Missing cost-input rows and assignment/dividend hard gates are never
repaired, so the repair pass cannot import later quote or risk information.

---

### [INFO] Cost-aware Sortino entry costs are train-window-only

**File**: `research/papers/option_only_markowitz/analysis/publication_costs.py`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

`derive_entry_cost_series` restricts observed entry-cost means to rows with
`return_date <= TRAIN_END`. The estimate is `0.5 * relative_spread +
fees/(mark * multiplier)`, and assets without train-window observations receive
class-default imputation. The resulting `Cost-aware Sortino + VIX` strategy is diagnostic
and does not enter headline or simulation strategy sets.

---

### [INFO] CBBO spread surface joins exact decision dates only

**File**: `research/papers/option_only_markowitz/analysis/publication_costs.py`; `data_ingestion/build_cbbo_cost_surface.py`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The CBBO spread surface is built from the same-day end-of-day 15:30--16:00 ET window and
is joined on the exact decision date, underlying, moneyness bucket, and tenor bucket.
There is no as-of merge or forward fill. If no panel CBBO or surface CBBO match exists,
the cost-input ledger records `relative_spread_source = default`.

---

### [INFO] VIX chain features are diagnostic and prior-date conditioned

**File**: `research/papers/option_only_markowitz/analysis/vix_chain_features.py`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

VIX chain state features are computed from chain rows observable at the decision date,
using the same staleness convention as `vix_option_panel`. They are not fed into expected
returns. The vol-of-vol regime table conditions realized strategy returns on the prior
decision date's feature, so the reported regime label is observable before the return it
summarizes.

---

## Distributional-robustness layer (2026-07-03)

**Audit result**: Clean. No confirmed leak affects the headline point-in-time strategies.
The entries below are INFO/by-design disclosures for the new robustness diagnostics.

### [INFO/by-design] CPCV is deliberately non-PIT

**Files**: `research/papers/option_only_markowitz/analysis/cross_validation.py:4-7`,
`research/papers/option_only_markowitz/analysis/run_empirics.py:2199-2225`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

CPCV trains on data after some test folds by construction. It is a distributional-
robustness and overfitting diagnostic (backtest-path distribution and PBO), not a
tradable out-of-sample claim. This does not alter the headline claim boundary, which is
still the fixed point-in-time train/test simulation.

---

### [INFO/by-design] Purge and embargo are adequate for payoff labels

**File**: `research/papers/option_only_markowitz/analysis/cross_validation.py:173-224`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The fold builder removes one purge snapshot around test groups and one embargo snapshot
after the purge. Labels' payoffs realize at most 44 days after the decision date; purge=1
plus embargo=1 enforce an at least two-snapshot train/test gap, at least 57 calendar days
in the monthly grid. Since 57 > 44, the retained training labels cannot overlap the test
label payoff window.

---

### [INFO/by-design] Fold eligibility is fixed at the headline anchor

**File**: `research/papers/option_only_markowitz/analysis/run_empirics.py:441`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The contract-universe eligibility filter requires at least 36 training observations
through the global `TRAIN_END` and is reused unchanged in every CV fold. That would be
too strong for a tradable per-fold OOS claim, but it is acceptable for this diagnostic
because CPCV itself is explicitly non-PIT and used only to inspect distributional
robustness and overfitting sensitivity.

---

### [INFO/by-design] Resampled-refit slot relabeling affects constraints only

**Files**: `research/papers/option_only_markowitz/analysis/resampled_universes.py:124-176`,
`research/papers/option_only_markowitz/analysis/resampled_universes.py:303-306`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The resampled-refit tier relabels pseudo rows to the original training-date slots before
calling the model. SPY beta and stress augmentation therefore use slot-calendar dates
while moment inputs are slot-relabeled resampled months. This approximation affects
constraint bounds only. It does not enter expected returns or realized P&L.

---

### [INFO/by-design] MC generator initial state is filled only within train

**Files**: `research/papers/option_only_markowitz/analysis/run_empirics.py:2384-2385`,
`research/papers/option_only_markowitz/analysis/monte_carlo_repricing.py:64-151`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

The MC state model receives IV and VIX levels restricted to `loc[:TRAIN_END]` before
`ffill().bfill()` is applied. The fill can move information within the training window
but does not cross into the test window. The resulting initial IV/VIX state is therefore
train-contained.

---

### [INFO/by-design] MC tiers answer different robustness questions

**Files**: `research/papers/option_only_markowitz/analysis/resampled_universes.py:90-123`,
`research/papers/option_only_markowitz/analysis/resampled_universes.py:124-176`,
`research/papers/option_only_markowitz/analysis/run_empirics.py:2306-2354`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

Fixed-weight MC paths measure robustness of the fixed train-estimated portfolio. The
separate refit tier measures procedure and estimation-risk robustness. The per-fold CV
set excludes `Cost-aware Sortino + VIX` because its solver is expensive fold-by-fold, but
the fixed-weight MC tier includes it. Equity benchmarks remain headline-only in CV
because their weights are underlying weights, not option-contract weights.

---

### [INFO/by-design] Repriced MC is a variance-risk-premium stress world

**Files**: `research/papers/option_only_markowitz/analysis/monte_carlo_repricing.py:162-211`,
`research/papers/option_only_markowitz/analysis/monte_carlo_repricing.py:345-395`,
`research/papers/option_only_markowitz/analysis/monte_carlo_repricing.py:502-518`
**Status**: Acknowledged
**Date Identified**: 2026-07-03

Repriced synthetic contracts are one-step monthly options. Equity contracts are priced
with Black-Scholes; VIX contracts are priced with Black-76 off the simulated VX-front
state and settled against the simulated VIX level. Entry premiums are model-priced at
the simulated IV state, so the empirical variance-risk-premium wedge embedded in observed
market premia is largely absent by construction. Rank changes versus the resampled
universes are informative about premium-dependence, not evidence of headline backtest
failure.

---
