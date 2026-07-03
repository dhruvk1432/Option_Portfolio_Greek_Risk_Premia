# Release Notes: Option-Only Portfolio Optimization

## Distributional-robustness layer (2026-07-03)

- Added the paper's distributional-robustness section, covering blocked k-fold, CPCV/PBO,
  regime-tagged CPCV paths, resampled universes, refit stability, and repriced synthetic
  universes.
- Added `make robustness` outputs: CV ledgers/tables, MC ledgers/tables, repricing
  assumptions, and `tables/distributional_robustness_summary.json`.
- Documented the claim boundary: CPCV is deliberately non-PIT and diagnostic only; the
  repriced universe is a variance-risk-premium stress world rather than a replay of
  realized market premia.
- Extended the verifier with robustness artifact/schema checks while keeping robustness
  regeneration opt-in.

## Pipeline-output verification update (2026-07-03)

- Added repaired execution-sensitivity scenarios from a second `build_execution_cost_scenarios(repair=RepairConfig())` pass. Repaired outputs use `_repaired` scenario labels, write repaired cost/reject/capital/repair ledgers plus execution-repair diagnostics and comparison tables, and remain excluded from headline growth tables and the reality-check family.
- Added the diagnostic `Cost-aware Sortino + VIX` strategy using train-window-only entry-cost estimates. It is not a headline or simulation strategy, but its gross and cost-scenario columns expand the reality-check family; PSR/DSR and family-wise reality-check values can therefore move for all variants.
- Added `make cbbo-surface` and the CBBO spread cost surface path. `build_cost_input_ledger` now records spread-source precedence (`panel_cbbo`, `surface_cbbo`, `default`), and rows that formerly used class-default spreads may now use measured surface spreads, moving post-cost and cost-scenario numbers while leaving gross returns unchanged.
- Added VIX chain state features and vol-of-vol regime diagnostics. These are diagnostic only, use the prior decision date for regime conditioning, and do not feed expected returns.
- Added the data-extension manifest recording integrated, extended, deferred, and referenced data families, including the deferred IWM/QQQ universe expansion.
- Added the production paper-broker repair module and repair-ledger validation path for research certification. The claim boundary is unchanged: the paper reports research simulations and repaired research scenarios, not broker-executed live fills, live margin parity, order routing evidence, or live tradability.

## Referee-audit revision (2026-07-02)

- Formal mathematics: the cashflow and portfolio sections now state the NAV accounting identity, the projection assumption, the covariance identity/PSD result, the ray-frontier result, and the conic solver reduction as numbered propositions with proofs (Propositions 3.1, 4.2, A.1-A.3; Assumption 1), plus a formal Sharpe estimator definition (Definition 5.1).
- Corrected inference statistics: `regenerate_from_artifacts.py` rebuilt the reality-check, PSR/DSR, inference-summary, simulation, and performance-diagnostics tables. PSR/DSR are now in per-month units, the reality check is a centered max-statistic block bootstrap (family p = 0.078), DSR = 0.000 for all variants, bootstrap CI columns are correctly ordered, and wealth-path simulations absorb defaulted paths at zero and report a defaulted-path share column.
- Reconciled cost-layer presentation: the manuscript now distinguishes the executable-cost scenarios (entry frictions; leading sleeves survive) from the conservative stress-cost layer (round-trip spreads, margin funding, capacity, assignment penalties; no sleeve survives), and states that the deployable claim is scenario-dependent and sits between them.
- Point-in-time upgrades executed: the empirical pipeline was re-run in full on the raw local data, so the shipped liquidity-tier table now classifies tiers on training-window liquidity and the "rolling 36M OOS" exhibit is a genuine walk-forward re-estimation (trailing 36-month refits with the constrained headline solver). Post-cost scenario tables are fail-closed: rejected or unfillable positions forfeit their gross P&L instead of keeping it.
- Sample-period statement: OOS window 2021-01-29 to 2026-04-30, 60 monthly snapshots over 64 calendar months, with the four absent months and the zero-return cash convention documented; the VIX exact-settlement selection effect (536 of 630 representative rows, exclusions concentrated 2015-2018) is disclosed.
- Quantitative results discussion, appendix regrouping with in-text references for every exhibit, new citations (Faias & Santa-Clara 2017; Driessen & Maenhout 2007; Frazzini & Pedersen 2022; Newey & West 1987; Politis & Romano 1994; Nelson 1991), and the previously orphaned `regime_sharpes.pdf` and `vix_regime_sharpes.pdf` figures included in Appendix A. `plot_regime_sharpes` now derives regime labels from the data (fixing the previously empty VIX-regime panel) and clips Sharpe values below -8 with an annotated marker so outlier sleeves no longer compress the axis.

## Publication polish pass

- Final PDF: `research/papers/option_only_markowitz/option_only_portfolio_optimization_dhruv_kohli.pdf`.
- The manuscript frames the contribution as an option-only portfolio optimization and risk-accounting framework, not a live-trading claim.
- The introduction includes formal research-question, contribution, and related-work positioning paragraphs.
- The empirical design clarifies the option universe, monthly train/test convention, equity-option buckets, VIX/VX-forward treatment, exact VRO/SOQ settlement gate, and post-cost research assumptions.
- Main exhibits are curated around OOS performance, post-cost survival, growth evidence, random feasible comparison, factor controls, claim strength, and claim audit.
- Appendix A keeps the all-strategy growth diagnostic, option/VIX/cost/settlement checks, inference, robustness, conventions, proof sketches, and reproducibility commands.
- No v1.0 tag has been created.
