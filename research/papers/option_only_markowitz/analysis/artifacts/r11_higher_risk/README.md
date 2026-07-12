# R1.1 development artifacts

These files are retrospective development evidence for the separately versioned R1.1
policy. They do not modify the frozen R1 artifacts.

- `r11_monthly_development_returns.csv`: 25% base, EGARCH diagnostic, and explicitly
  unscored VIX-40 rows for the January 2018 decision--April 2026 return replay.
- `r11_monthly_weights.csv`: whole-contract shadow and executed base/EGARCH positions.
- `r11_integer_repair_candidates.csv` and `r11_integer_repair_method_summary.csv`:
  the attempted direct whole-contract conversion and the cash-abstention outcome. If the
  direct book is infeasible, no alternative risky portfolio is substituted. Its original
  constraint values remain recorded even though the selected book is cash.
- `r11_survival_summary.csv`: survival-first results, including the number of valid cash
  abstentions. Abstentions are not classified as integer-execution failures.
- `r11_vix_risk_off_events.csv` and `r11_vix_exposure_calendar.csv`: official-close
  state transitions and start-of-session exposure state.
- `r11_event_quote_request.csv`: exact held OSI symbols and event dates requiring licensed
  OPRA CBBO. Missing quotes are not replaced by model prices.
- `r11_intervention_execution_summary.csv` and `r11_intervention_fill_ledger.csv`:
  execution status and fills. The checked ledger is unscored because the licensed files
  are absent.
- `r11_egarch_forecasts.csv` and `r11_egarch_gate.json`: cutoff-safe forecasts and the
  prespecified promotion decision; EGARCH remains diagnostic.
- `r11_research_trial_registry.*`, `r11_specification_status.csv`, and
  `r11_prospective_freeze_manifest.json`: research chronology, evidence labels, and the
  separate 36-month prospective protocol.
