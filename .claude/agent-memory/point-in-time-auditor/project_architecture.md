---
name: project-architecture
description: Data flow and train/test architecture of the option-only Markowitz empirical pipeline
metadata:
  type: project
---

Option-only Markowitz paper pipeline (research/papers/option_only_markowitz/analysis).

Fact: The headline backtest is a STATIC single train/test split, NOT walk-forward.
`TRAIN_END = pd.Timestamp("2020-12-31")` (run_empirics.py:85). `make_model()` fits Sigma,
mu, betas, residual cov, and weights once on `returns.loc[:TRAIN_END]` (line 517) and applies
fixed weights to `returns.loc[>TRAIN_END]`. The optimizer in
src/portfolio/option_only_markowitz_model.py is pure — it consumes pre-built Sigma/mu/Greeks and
introduces no time semantics of its own.

Fact: `make_model` HARD-CODES the global `TRAIN_END` slice internally (line 517), ignoring any
narrower window a caller passes. So `rolling_oos_table` (which passes rolling windows) still trains
every fold on `<=2020-12-31`. This makes the "rolling 36M OOS" table not actually rolling. It is a
correctness/claim bug, NOT a look-ahead leak (it uses less-recent, never future, data).

Fact: Returns are realized option payoffs from build_expiry_proxy_return_panel (equities) and
build_vix_expiry_proxy_returns (VIX). Decision at snap_date t selects the contract; payoff computed
at listed expiry from raw daily closes (equities) or VRO/SOQ exact / VIX-close proxy (VIX).

**Why:** Understanding this separation is essential — the leak surface is entirely in data
construction and universe/tier selection, not in the optimizer.
**How to apply:** When auditing this repo, focus on (1) what window feeds make_model, (2) universe
and liquidity-tier selection functions, (3) settlement date bounds. Do not re-audit the optimizer
math for PIT — it is state-free.
