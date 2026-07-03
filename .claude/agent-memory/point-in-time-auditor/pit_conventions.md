---
name: pit-conventions
description: Verified-clean PIT patterns and the recurring leak-prone idioms in the option Markowitz repo
metadata:
  type: project
---

Verified CLEAN patterns in this repo (as of 2026-07-02 audit):
- Universe survivorship filter `returns.loc[:TRAIN_END].count() >= 36` (load_bucket_panel) uses only
  train data — PIT-safe.
- conditional_premia.py: all estimators take pre-restricted train inputs; no forward fill; cross-
  sectional zscore only.
- vix_option_panel.py: selection `panel["trade_date"].le(d)`; VRO used only at exact expiry; VIX-close
  proxy bounded by `_last_value_on_or_before(vix, expiry, decision_date)`; state ffill is past-only.
- simulation.py / inference.py / publication_costs.py: evaluation-only, consume realized OOS paths,
  never feed back into weights. `bfill` in simulation is on a diagnostic vol series only.
- split_adjusted_spot_panel: forward-scanning cumulative split factors, ffill only — PIT-safe.

Leak-prone idioms found (see leakage.md for full writeups):
- liquidity_tier_labels (execution_cost_scenarios.py:349) uses `groupby(asset_id).tail(1)` over the
  FULL sample then cross-asset quantile cuts -> future-observed liquidity used to define tier universe
  re-run in liquidity_tier_rerun_tables. MEDIUM look-ahead in a supplementary robustness table.
- pnl_attribution_table uses next-period IV to attribute realized P&L — legitimate ex-post
  attribution, reporting only.
- regime_performance_table / volatility_regime_performance_table use full-OOS-window terciles to bucket
  reported performance — reporting only, not selection. INFO.

**Why:** These distinctions (selection vs reporting; less-data vs future-data) determine severity.
**How to apply:** Flag full-sample stats only when they drive SELECTION/universe/weights. Full-sample
terciles used purely to slice an already-realized OOS return series for a conditional table are not
a backtest leak — call them INFO/LOW and say so explicitly.

Additional verified CLEAN patterns (2026-07-03 audit of cost-aware Sortino / CBBO / repair paths):
- return_date vs decision_date semantics: a trade is DECIDED at decision_date (= snap_date, month-end)
  and REALIZED at return_date (= expiry/payoff, strictly later; asserted decision_date < payoff_date
  at vix_option_panel.py:546). Entry-cost fields (mark, relative_spread) are decision-date
  contemporaneous but live on rows keyed by the later return_date.
- derive_entry_cost_series (publication_costs.py ~280) filters pool["return_date"].le(train_end).
  This is STRICTER than needed but conservative/leak-free: no post-TRAIN_END decision can survive
  (return_date > decision_date > TRAIN_END). It over-drops trades decided pre-TRAIN_END but expiring
  post-TRAIN_END (discards valid same-period info, never admits future). Correct direction for
  entry-cost estimation. train_scenarios = returns.loc[:TRAIN_END,...] is train-window only; OOS path
  (test_returns = returns.loc[>TRAIN_END]) unchanged; Sortino weights solved on train only via
  _sortino_weights_with_guard/model.solve_max_sortino.
- Execution repair (execution_cost_scenarios.py ~347-470, RepairConfig): quote repair and capacity
  pro-rata partial fill read ONLY row = indexed.loc[(return_date, asset_id)] — the single same-date
  cost-input row that fired the gate — plus static config/repair params. No cross-date access. Repaired
  fills pay 0.5*rel_spread (actual half-spread), remainder fails closed. PIT-clean by construction.
- CBBO surface builder (data_ingestion/build_cbbo_cost_surface.py): every surface row is a pure
  same-snap_date EOD (15:30-16:00 ET) aggregation. Quotes filtered quote_date.eq(snap_date);
  days_to_expiry = expiry - snap_date; groupby includes snap_date; NO cross-date ffill/smoothing.
  _surface_relative_spread joins on exact (underlying, decision_date->snap_date, moneyness, tenor) —
  cannot pull a future date's row. Same-decision-date EOD quotes for entry cost = contemporaneous,
  acceptable. _tenor_days_for_cost_rows uses expiry - decision_date (no future date).
- vix_chain_features.py build_vix_chain_state_features mirrors vix_option_panel trade_date<=snap_date
  via searchsorted side="right"-1 with 5-calendar-day staleness guard, no output ffill (NaN row if
  stale/empty). vol_of_vol_regime_table conditions realized ret_frame on features[feature].shift(1)
  (prior decision date) — correct shift direction, reporting-only (feeds vol_regimes table, not
  weights). Its pd.qcut over full feature history is a REPORTING full-sample bucketer -> INFO, not a
  backtest leak (same class as regime_performance_table).
- production repair (src/option_portfolio_production/repair.py) attempt_order_repair uses only the
  passed quote + decision_mark; adverse_drift_bps direction correct: BUY drift = (touch-mark)/mark,
  SELL drift = (mark-touch)/mark, clipped at 0 for favorable moves. Certification machinery, not
  backtest; no future data.
