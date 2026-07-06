# Release Notes: Option-Only Portfolio Optimization

## Visual final-result scoreboard (2026-07-06)

- Added `analysis/build_final_results_summary.py` and `make final-results`.
- Added conclusion figures `figures/final_breadth_validation_distributions.pdf` and
  `figures/final_baseline_comparison.pdf`, backed by
  `analysis/artifacts/breadth_solutions/robustness/final_result_scoreboard.csv` and
  `final_validation_distribution_summary.csv`.
- Reframed the paper's ending around charts rather than dense tables. The final visual
  answer compares the four locked E1 configurations against CPCV/resampled/refit
  distributions, ordinary underlying Markowitz, and the best capped naive option book.
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
