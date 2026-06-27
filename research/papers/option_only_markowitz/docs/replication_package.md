# Publication Replication Package


## Data pull preflight

From the repository root, copy `.env.example` to `.env`, fill local keys/paths, and dry-run the acquisition plan:

```bash
cp .env.example .env
.venv/bin/python -m data_pull.pull --preset option-paper
```

Execute credential-free public inputs with:

```bash
.venv/bin/python -m data_pull.pull --preset public --execute
```

Execute paid Databento OPRA jobs only after reviewing the dry-run manifest:

```bash
.venv/bin/python -m data_pull.pull --preset option-paper --execute --allow-paid
```

Raw OPRA data are not redistributed. The command records file-status and credential-presence booleans in `research/reports/pipeline_reports/data_pull_manifest.json`, but never writes credential values.

## Exact commands

Run from the repository root.

```bash
# 1. Pull public Cboe VRO/SOQ settlement for the paper's VIX expiries.
.venv/bin/python -m data_pull.pull --preset validate --jobs public-vro-soq --execute

# Optional override when using a separately versioned exact-settlement file.
export OPTION_MARKOWITZ_VRO_FILE=data/public/cboe/vro_soq/vro_soq_settlements.csv

# 2. Regenerate all empirical artifacts, gross and post-cost tables, inference, and hashes.
.venv/bin/python -m research.papers.option_only_markowitz.analysis.run_empirics --stage all

# 3. Compile the paper.
cd research/papers/option_only_markowitz
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
bibtex option_only_portfolio_optimization_dhruv_kohli
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
lualatex option_only_portfolio_optimization_dhruv_kohli.tex
cd ../../../..

# 4. Run the independent verifier.
.venv/bin/python -m research.papers.option_only_markowitz.verification.verify
```

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_option_only_markowitz_model.py tests/test_option_only_markowitz_verification.py tests/test_option_only_publication_upgrade.py -q -p no:cacheprovider
```

## Generated publication artifacts

- `tables/empirical_summary.json` contains every headline number used by the paper.
- `artifacts/strategy_returns.csv` contains gross research returns.
- `artifacts/strategy_returns_post_cost.csv` contains conservative post-cost research returns.
- `artifacts/vix_settlement_coverage.csv` determines whether VIX-enabled results are headline-grade or diagnostic.
- `artifacts/vix_settlement_audit.csv` records exact/proxy settlement coverage by VIX expiry.
- `artifacts/inference_summary.csv` contains bootstrap confidence intervals.
- `artifacts/cost_ledger.csv`, `capacity_ledger.csv`, `research_margin_ledger.csv`, and `assignment_risk_ledger.csv` contain implementation diagnostics.
- `environment_lock.json` records Python and package versions.
- `artifact_hash_manifest.csv` records SHA-256 hashes for generated tables, figures, artifacts, and paper metadata.

## Data availability and licensing

The code is reproducible for a licensed local user, but raw Databento/OPRA options data must not be redistributed through the replication package. The package provides schemas, code, generated output hashes, and local path conventions. A replicating user must obtain OPRA/Databento data under their own license and place it in the expected local data directories.

Exact VRO/SOQ files are downloaded from public Cboe settlement endpoints by `public-vro-soq`, or may be supplied locally through `OPTION_MARKOWITZ_VRO_DIR` or `OPTION_MARKOWITZ_VRO_FILE`. If exact settlement is absent or incomplete, VIX option rows remain proxy-labeled and VIX-enabled results are not headline-grade. Raw Databento/OPRA option data remains licensed and is not redistributed.
