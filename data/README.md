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

## Included Public Inputs

The repository includes the normalized public Cboe VRO/SOQ settlement outputs used by the paper:

- `data/public/cboe/vro_soq/vro_soq_settlements.csv`
- `data/public/cboe/vro_soq/vro_soq_download_audit.csv`
- `data/public/cboe/vro_soq/vro_soq_manifest.json`

To refresh public settlement files, run `make data-public` or `python -m data_pull.pull --preset validate --jobs public-vro-soq --execute`. This is not run during project extraction.

## Licensing Boundary

Licensed users can reproduce the paper by supplying equivalent OPRA/Databento inputs locally or by running the paid data-pull jobs with their own credentials. Do not commit raw licensed market data or `.env` files.
