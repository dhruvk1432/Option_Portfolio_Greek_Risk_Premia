---
name: pit-verified-patterns
description: Verified-clean point-in-time idioms and the return-panel timing facts for the option-only Markowitz distributional-robustness layer (cross_validation.py, resampled_universes.py, monte_carlo_repricing.py)
metadata:
  type: project
---

Audit of the distributional-robustness layer (2026-07-03). Verified-clean patterns and load-bearing timing facts, so future audits do not re-derive them.

## Return-panel timing (build_expiry_proxy_return_panel, run_empirics.py:288)
- The panel indexes each option return at `return_date = the NEXT month-end snapshot`, but the DECISION is at the current snap (`decision_date`) and the PAYOFF is realized at the listed `expiry`.
- Empirical spans (from artifacts/holding_return_detail.csv, 6313 rows; vix_holding_return_detail.csv, 536 rows):
  - decision->expiry: median 18d, p99 21d, **max 44 days** (both equity and VIX).
  - For ~99.95% of rows the payoff realizes in the PRIOR label-month bucket (`bleed=1`); every label's decision is exactly 1 label-month before its return_date (`dec_to_lab=1` for all rows).
  - Combined equity+VIX snap grid = 129 monthly labels; **tightest 2-snap calendar span = 57 days**.

## CPCV purge/embargo adequacy — VERIFIED SUFFICIENT at purge_months=1
- build_folds (cross_validation.py:182) with n_groups=12, n_test_groups=2, purge=1, embargo=1 yields, on the real grid: nearest train month ABOVE a test block sits at test_end+3; nearest train month BELOW sits at test_start-2. The test-decision month (test_start-1) is always purged out of train.
- Max decision->payoff span 44d < tightest 2-snap span 57d => no train contract's payoff window can overlap any test contract's payoff window. **purge_months=2 is NOT required.** This is the key referee defense.

## Verified-clean idioms (do not re-flag)
- **Restriction-first specs**: `_restricted_reps`/`_restrict_reps_to_train_dates` filter reps to fold train_dates BEFORE `representative_specs`, which itself masks `snap_date.le(train_end) & .ge(train_start)`. make_model re-slices `returns.loc[start:end]` and asserts `train_returns.index.max() <= end` (run_empirics.py:609).
- **Slot relabeling** (resampled_universes.py `_slot_relabel` + `month_index_paths(len(train_index),...)`): index paths are drawn over TRAIN positions only; cannot reach test months. Spec geometry uses non-resampled train_returns; pseudo returns only feed the moment estimates.
- **Slot-calendar SPY approximation** (`_augment_spec_with_beta_and_stress`, run_empirics.py:563): reloads SPY at real train dates while underlying rows are slot-relabeled -> beta/stress are CONSTRAINTS-ONLY (bounds), never expected returns or P&L. Benign for PIT; disclosure item only.
- **Slice-before-fill**: `iv_levels.loc[:TRAIN_END].ffill().bfill()` and `vix_level.loc[:TRAIN_END].ffill().bfill()` (run_mc_stage) slice to train FIRST, so bfill only pulls within-train-window; never crosses the train/test boundary. State model is a train-only synthetic-universe generator, so within-train bfill is at most INFO.
- **contract_static_params** (monte_carlo_repricing.py:234): `frame[frame.snap_date.le(train_end)]` restriction-first. run_mc_stage passes TRAIN_END.
- **Fold cost eval**: build_execution_cost_scenarios (execution_cost_scenarios.py:277) sets a (return_date, asset_id) index and iterates `for return_date in gross_returns.index`. Passing the full cost_inputs ledger is harmless because iteration is driven by the test-month gross_frame; only test-month cost rows are read.
- **No global mutation**: `_robustness_context()` builds a fresh context; `make_model(spec, returns, reps, universe)` reproduces the headline fit with default TRAIN_END. run_robustness passes ctx to cv/mc stages without mutating module state.

## CPCV is deliberately non-PIT BY DESIGN
CPCV trains on data after test folds (backtest-path distribution / PBO overfitting diagnostic). NOT a leak — must be disclosed as non-tradable, not corrected.
