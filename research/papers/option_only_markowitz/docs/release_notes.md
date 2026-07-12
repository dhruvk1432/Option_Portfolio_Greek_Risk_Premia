# Release Notes: Option-Only Portfolio Optimization

## R1.1 direct-or-abstain integer execution (2026-07-12)

- Restored the R1-style direct whole-contract conversion as the only executable risky
  portfolio. If it violates a hard constraint, that month is explicitly held in cash.
- A cash abstention is a valid decision, not an integer-execution or survival failure.
- Preserved the rejected conversion's CVaR, stress, volatility, margin, collateral,
  assignment, liquidity-cap, and base-constraint diagnostics in the artifact ledger.
- No risk-leg reduction, extra VIX contract, iterative repair, or mixed-integer substitute
  portfolio contributes to the restored strategy's returns.

## Rejected R1.1 integer-repair development comparison (2026-07-12)

- Tested five registered candidates in a superseded development experiment:
  direct truncation, whole risk-leg removal, retention of an additional protective VIX
  contract, binding-constraint-guided one-contract reduction, and a sign-restricted
  mixed-integer conic net-utility solve using ECOS_BB.
- Candidate selection required every hard constraint to pass, then maximized net
  utility with distance from the continuous target as a deterministic tie-break.
- Preserved failed pre-repair CVaR, stress, volatility, margin, collateral, assignment,
  liquidity-cap, and base-constraint values instead of overwriting them with cash values.
- That superseded replay repaired all 417 direct-truncation failures. ECOS_BB certified 439 of
  744 books under a 1,000-node, 32-active-direction operational gate; every remaining
  book is resolved by a checked deterministic repair.

## R1.1 higher-risk and VIX risk-off development arm (2026-07-12)

- Added a separately versioned 25% volatility policy with a positive-edge, sign-restricted
  50% deployment feasibility test; no gross floor or split-leg burn is introduced.
- Added official-close VIX-40 state transitions, deduplicated March 1 user-attested exit,
  sequential displayed-size CBBO execution, and a cost-guarded licensed quote pull.
- Added the 2018--2026 crisis replay, intervention request/ledger, EGARCH(1,1)-Student-t
  joint-covariance overlay and promotion gate, trial registry, and R1.1 freeze manifest.
- The checked risk-off arm remains unscored without event-date CBBO and EGARCH is
  diagnostic. Legacy E1 absorbed-zero CPCV evidence and the original R1 freeze remain
  unchanged.

## R1 mathematical and research-design repair (2026-07-11)

- Added the complete joint Greek factor/residual covariance, including both cross terms.
- Added a cost-aware, cash-permitting net-utility solver with fixed volatility, CVaR,
  stress, margin, collateral, assignment, liquidity, and integer-execution controls.
- Added a monthly point-in-time R1 development pipeline, hard survival gate, known-trial
  registry, and prospective 36-month freeze manifest.
- Reclassified every existing result as retrospective development evidence. E1 remains in
  the paper only as legacy research history and its absorbed-zero VIX paths fail R1's
  survival standard.
- Reframed the mathematical contribution as a self-contained implementation of standard
  results rather than new theory.

## Whole-contract integer execution as the headline (2026-07-07)

- Every headline book is now scored in **whole option contracts** at the standard
  100-share multiplier, not fractional contracts. The maximum-Sharpe solver still returns
  continuous premium weights, but each weight is rounded to the nearest signed integer
  contract count and clipped so the realized weight never exceeds its per-contract
  liquidity cap (`integerize_book_weights` in `analysis/breadth_solutions_lib.py`, wired
  into `fit_books` and the E1 ablation).
- Effect: full-cost net Sharpe rose on all four books to 0.778 / 1.383 / 0.587 / 1.628
  (gross 0.975 / 1.675 / 0.847 / 2.010). Because the continuous solution binds its
  liquidity caps on the thinnest, most expensive legs, cap-respecting integer rounding can
  only round those down, slightly de-levering the book and reducing net cost drag.
- Paper updates: §2.2 now states results are whole-contract and adds the "Whole-contract
  rounding" remark; §3 opens with an explicit three-layer bridge from the clean conic
  theory to an exchange-realistic book (CBBO-sourced cost stack, volume-aware pre-trade
  caps, whole-contract execution); every prose number was refreshed to the regenerated
  artifacts. Verification: 386/386 checks pass, 155 unit tests pass, 34 pages.

## Mid-length theory-first canonical paper (2026-07-06)

- Replaced the 60-page canonical manuscript with a mid-length theory-first paper targeted
  at the 25-30 page range. The old long-form section files and dense diagnostic artifacts
  remain in the repository for audit, but they are no longer printed as a table-heavy
  appendix in the canonical PDF.
- Added the paper exhibit set:
  `figures/short_theory_flow.pdf`, `figures/short_four_variant_scoreboard.pdf`,
  `figures/short_walk_forward_return_paths.pdf`,
  `figures/short_validation_distributions.pdf`, `figures/short_robustness_heatmap.pdf`,
  and `figures/short_capacity_spread_panel.pdf`.
- Added compact manuscript tables:
  `tables/short_four_scenario_assumptions.tex`, `tables/short_spread_source_ladder.tex`,
  `tables/short_final_scoreboard.tex`, and `tables/short_robustness_summary.tex`.
- Reorganized the paper around theory, the four locked E1 scenarios, corrected
  spread-source assumptions, baselines, distributional validation, real-world capacity,
  and the forward shadow/production-readiness bridge.
- Added an explicit front-matter subsection documenting what was retained from the
  previously pushed long version: cashflow theory, Greek covariance, VIX settlement,
  cost-stack discipline, breadth/capacity diagnostics, robustness gates, and verification
  culture.
- Added a compact technical appendix that restores the useful formal material from the
  longer draft without reintroducing the dense appendix tables: no-free-exposure scale
  control, cost/constraint monotonicity, conic maximum-Sharpe reduction, net-cap solver
  handling, and PSD residual-estimator repair.
- Updated the verifier's PDF page-count expectation to the 25-30 page range while
  keeping artifact, claim-boundary, and rendered-PDF checks active.

## Visual final-result scoreboard (2026-07-06)

- Added `analysis/build_final_results_summary.py` and `make final-results`.
- Added conclusion figures `figures/final_breadth_validation_distributions.pdf`,
  `figures/final_baseline_comparison.pdf`, and
  `figures/final_walk_forward_return_paths.pdf`, backed by
  `analysis/artifacts/breadth_solutions/robustness/final_result_scoreboard.csv` and
  `final_validation_distribution_summary.csv`, plus
  `final_walk_forward_return_paths.csv`.
- Reframed the paper's ending around charts rather than dense tables. The final visual
  answer compares the four locked E1 configurations against CPCV/resampled/refit
  distributions, ordinary underlying Markowitz, matched capped naive option books, and
  rolling OOS cumulative return paths.
- Clarified the final claim: the VIX-enabled E1 books beat both simple baselines;
  `larger` no-VIX is mixed because it essentially ties naive options and remains below
  stock Markowitz; `orig` no-VIX is a capacity diagnostic.

## Production-grade claim-boundary pass (2026-07-05)

- Narrowed the headline claim: locked E1 with structural-only mean, diagonal residual covariance, N-scaled shrinkage, and net liquidity caps is a robust small-account research candidate, strongest with VIX. It is not live-trading proof.
- Added an artifact-backed spread methodology table. Baseline eight-name equity rows use historical panel CBBO; missing broad-name and VIX rows use a point-in-time inferred CBBO proxy and are labeled calibrated execution sensitivity rather than final execution truth.
- Added operational footnote references for OPRA, Cboe DataShop option EOD/NBBO snapshots, and IBKR free-trial delayed-data constraints while keeping the scholarly bibliography limited to article/book entries.
- Added a robust conic production objective, plus propositions for no-free-option exposure, cost/constraint monotonicity, net-cap representation invariance, and historical-mean fragility when \(n/T\) rises.
- Added `src/option_portfolio_production/shadow.py` and `analysis/export_shadow_targets.py`. The workflow exports locked E1 targets, ingests market-hours NBBO/CBBO CSV inputs, and writes `shadow_*` ledgers. Shadow fills use `shadow_nbbo_displayed_size_cross` and intentionally do not satisfy production-live verification.
- Added a production-readiness appendix checklist for quote source, fill model, margin, assignment/exercise, ex-dividend closeout, VIX settlement, order caps, kill switch, reconciliation, and broker statement audit.

## Breadth and capacity diagnostic (2026-07-05)

- Added the breadth-solution experiment bundle under `analysis/artifacts/breadth_solutions/`, with P1 estimator regularization, P2 pre-trade liquidity caps, capped-naive benchmarks, and P3 combined decision tables for 8/9-name and 56/57-name universes.
- Replaced the old POC 10%/15% spread fill path with a point-in-time inferred CBBO proxy for missing added-name and VIX spread rows. The proxy is calibrated from the historical liquid equity/ETF CBBO surface and is used only after exact panel CBBO and exact decision-date surface rows are unavailable. The optional Cboe delayed-chain builder and cost loader still reject weekend, holiday, and after-hours snapshots, but the regenerated breadth tables no longer consume the stale off-hours Cboe file.
- Clarified that the no-VIX 8-name baseline is already exact for equity-option spread inputs: `p3_spread_source_coverage.csv` reports 5,777 `panel_cbbo` rows across 49 asset IDs and all eight baseline underlyings. The VIX-enabled baseline uses the same exact equity-option spread rows, while VIX option spread costs use the inferred liquid-option CBBO proxy.
- Documented the main diagnostic result: at `$1M` NAV, the inferred-spread ledger moves the 57-name-with-VIX Greek-Markowitz book from net Sharpe `-1.837` in the uncapped paper configuration to gross Sharpe `1.915` and net Sharpe `+1.499` when the historical mean is removed, diagonal residual covariance plus N-scaled covariance shrinkage are used, and liquidity caps bind net contract positions; it beats the best capped-naive book by `1.232` net Sharpe. The no-VIX 56-name book is positive but essentially ties capped equal-risk naive (`0.551` versus `0.550` net Sharpe).
- Documented the capacity boundary: at `$5M+`, the sum of liquidity caps falls below full deployment for the tested configurations and participation rates, so positive relaxed rows above that scale are economically tiny.
- Added the net-cap formulation note. The liquidity cap must bind `abs(q_i)` after long/short split variables are netted; split-leg caps can burn offsetting exposure as pseudo-cash and consume scarce contract capacity.
- Carried forward the P1/P2/P3 decision-table caveats: inferred spread fills are a source-audited proxy rather than matched historical executable CBBO for every added name and VIX option, and best-knob selection is in-sample across the diagnostic grid. The separate breadth robustness runner locks E1 and evaluates fold-local CV, resampled, synthetic, and rolling diagnostics.

## Breadth robustness validation (2026-07-05)

- Added `analysis/breadth_robustness_experiment.py` and `tests/test_breadth_robustness_experiment.py`.
- Validated the locked E1 capped candidates for `orig`, `orig+VIX`, `larger`, and `larger+VIX` using the corrected cost stack (`use_current_spread_assumptions=False`, `use_inferred_spread_proxy=True`) with full spread, fees, slippage, borrow proxy, margin drag, assignment penalty, and capacity/impact costs.
- The full run writes `analysis/artifacts/breadth_solutions/robustness/` with 12 chronological groups, 66 CPCV splits, 78 total CV/PBO splits per config, one-month purge/embargo, 1,000 resampled paths, 200 refit paths, 1,000 repriced paths, circular-block and GARCH-style path simulations, drawdown breach rates, reality-check inference, and true rolling 36-month monthly OOS refits.
- The spread-source audit passes: zero `current_cboe_liquid_quote` rows and zero `default` spread rows. The fold schedule has 312 data rows (`4 x 78`) and all fold schedule statuses are `ok`. E1 CV ledger rows are `optimal`; broad rolling refits emit solver accuracy warnings but record E1 solver status as `optimal`.
- Main E1 static results: `orig` is diagnostic capacity-infeasible (`0.720` net Sharpe, `1.629` net Sortino); `orig+VIX` passes (`1.287` net Sharpe, `3.256` net Sortino); `larger` is mixed (`0.551` net Sharpe, `1.196` net Sortino); `larger+VIX` passes (`1.499` net Sharpe, `4.004` net Sortino).
- Robustness details: `larger+VIX` E1 has MC resampled p05/p50 net Sharpe `0.989`/`1.495`, MC refit p05/p50 net Sharpe `1.262`/`1.542`, and rolling net Sharpe `1.217`. CPCV complete-path net Sharpe is negative for all E1 configs, while CPCV gross medians remain positive; this is reported as a short-window cost-timing fragility, not as a tradable OOS failure.
- Repriced synthetic net paths use a circular-block sample of realized full-cost drag; they are a documented historical-cost overlay, not synthetic NBBO/CBBO.

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
