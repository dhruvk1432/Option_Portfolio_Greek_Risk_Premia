---
name: execution-audit-pit
description: PIT design of the retrospective quote-grounded execution audit (r1_r11_execution_audit.py) and why it is leak-free
metadata:
  type: project
---

`research/papers/option_only_markowitz/analysis/r1_r11_execution_audit.py` is a RETROSPECTIVE
execution audit of the already-frozen R1/R1.1 replay. It recomputes ONLY the cost leg of monthly
returns under observed licensed NBBO quotes. Audited 2026-07-15: no look-ahead into decision
variables. Key invariants (verify these lines still hold before trusting future edits):

- gross_return is frozen: `_load_monthly` snapshots `_gross_return_frozen_text`; `recompute_costs`
  restores it verbatim (lines ~636 and ~725-726) AFTER computing net_return_* from the numeric copy.
  Order matters — net computed at 718-720, text restored at 725. Test asserts exact string + frozen
  file hash unchanged.
- weights / integer_contracts / mark all come from frozen weight CSVs (build_trade_table). Observed
  quotes only populate NEW columns (obs_*, exit_obs_*, position_cost_*, coverage). No write-back.
- Entry cost = decision-date CLOSE window only. `close_window(decision)` = [close-10min, close] on the
  decision session (previous-session snap if non-trading). `_quote_snapshot` double-clips with
  deterministic window_end via `.gt(window_start) & .le(window_end)` — so even an over-pulled request
  (request_end padded into the future) cannot leak; the recomputed close_window bound clips it. This
  is exactly what test_match_quotes_last_valid_and_early_close_boundary guards.
- `_attach_modeled_costs` merge_asof is direction="backward" on decision_date by (config, asset_id):
  never attaches a cost input dated after the decision. Worst case is a STALE prior-date input if the
  exact decision-date row is missing — conservative (past-only), not future.
- next-open (`held_next_open`, open_obs) is DIAGNOSTIC ONLY — never used in position_cost. Entry fill
  is modeled at decision close, contemporaneous with the decision.
- VIX exit fallback `_last_path_exit_snapshot`: quotes filtered `ts_recv.normalize() < expiry` (last
  tradable session strictly before settlement Wednesday), used only when is_vix & direct exit not
  covered. Exit cost naturally uses exit-time data; hold-to-expiry is deterministic — not look-ahead.
- Liquidity validation (`_liquidity_validation`) uses decision-day ohlcv-1d volume + OI: entry-day
  diagnostic, labeled `validation_scope="entry-day only; not the holding path"`. Does NOT feed weights
  (static optimizer caps use TRAIN-window volume, per breadth_solutions_lib memory). "breach_optimizer_
  volume_0_05" is an EX-POST capacity check, not a claim of decision-time admissibility.

Timezone: exchange_calendars XNYS session_close/open are tz-aware UTC evening/morning stamps (e.g.
early close 2018-07-03 17:00 UTC). All market hours 13:30-21:00 UTC stay on one calendar date, so
`_request_dates`/`_window_keys` normalize() never crosses midnight — key_date == decision session date.
