# Data Directory

This standalone repository intentionally does **not** redistribute raw OPRA or Databento data. The paper can be inspected from the included generated artifacts, but a full empirical rebuild requires licensed local inputs under the paths below.

## Required Local Inputs For Full Rebuild

- `data/feature_store/option_greek_proxy_panel.parquet`
- `data/feature_store/opra_surface_panel.parquet`
- `data/feature_store/option_greek_quality.csv`
- `data/universe/multi_raw_close.csv`
- `data/universe/vx_futures_daily.parquet`
- `data/universe/vix_complex.parquet`
- `data/databento_cache/opra_vix_chain_*.parquet`

The breadth/capacity diagnostic under `research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/` uses the same local inputs. It does not introduce a new raw data source; the 48 added equity names are read from the existing OPRA-derived feature store. To reproduce the checked-in breadth net cells, also build the derived CBBO spread surface from the licensed local OPRA full-day CBBO cache:

- `data/feature_store/cbbo_spread_surface.parquet`
- `data/databento_cache/opra_surface_full_day_cbbo`

The eight-name no-VIX baseline is exact on equity-option spreads (`panel_cbbo` for 5,777 cost rows across all eight baseline underlyings). Missing added-name and VIX spread rows in the breadth reruns use a point-in-time inferred CBBO proxy calibrated from that derived surface, not blanket 10%/15% class defaults or stale off-hours current-chain quotes.

The breadth robustness artifacts under `research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/robustness/` use the same data boundary. The checked run reports zero current-Cboe spread rows and zero default-spread rows; all non-panel breadth spread inputs come from the inferred historical CBBO proxy. Repriced synthetic net paths do not create synthetic bid/ask quotes: they subtract a resampled historical full-cost drag from gross repriced paths.

The broad inferred-spread rows are not a substitute for matched historical market-hours
NBBO/CBBO. A production-grade historical execution proof would require OPRA/NBBO or
broker CBBO with displayed size matched to every backtest decision row. The forward shadow
runner accepts user-supplied market-hours quote exports, margin previews, and rejection
notes, but those files are local operational inputs and should not be committed if they
contain licensed or account-specific data.

R1.1's VIX-40 intervention is subject to the same boundary. Its generated
`r11_event_quote_request.csv` identifies the held OSI symbols and event dates. Licensed
`cbbo-1m` responses belong under `data/databento_cache/r11_event_cbbo/` and are never
committed. Until every requested order has a complete displayed-size execution and re-entry
constraint check, the risk-off arm remains unscored.

The staged R1/R1.1 Databento audit uses
`data_ingestion.market_data.fetch_r1_r11_databento_audit`. It reads only
`DATABENTO_API_KEY2` from the repository `.env`, enforces a cumulative $40 cost ceiling,
and now limits newly requested execution audit records to option entry, exit, and
intervention quote windows under `data/databento_cache/r1_r11_audit/`. Run `make
databento-audit-plan` for cost estimation, `make databento-audit-execute` for the initial
pull, `make databento-audit-resume` after an interrupted pull, and `make
databento-audit-verify` to verify hashes. Corporate actions and assignment records are not
part of this cache; assignment remains unverified.

## Included Public Inputs

The repository includes the normalized public Cboe VRO/SOQ settlement outputs used by the paper:

- `data/public/cboe/vro_soq/vro_soq_settlements.csv`
- `data/public/cboe/vro_soq/vro_soq_download_audit.csv`
- `data/public/cboe/vro_soq/vro_soq_manifest.json`

To refresh public settlement files, run `make data-public` or `python -m data_pull.pull --preset validate --jobs public-vro-soq --execute`. This is not run during project extraction.

## Licensing Boundary

Licensed users can reproduce the paper by supplying equivalent OPRA/Databento inputs locally or by running the paid data-pull jobs with their own credentials. Do not commit raw licensed market data or `.env` files.
