# Private Data Boundary

No raw or licensed market data belong in this repository. Public evidence is limited to
portfolio-level returns, summarized weights and exposures, claim tables, trial counts, source
hashes, and verification metadata under `paper/evidence/`.

Historical regeneration depended on six externally prepared inputs:

```text
data/feature_store/option_greek_proxy_panel.parquet
data/feature_store/opra_surface_panel.parquet
data/feature_store/option_greek_quality.csv
data/universe/multi_raw_close.csv
data/universe/vx_futures_daily.parquet
data/universe/vix_complex.parquet
```

These files are intentionally ignored. They may contain licensed or security-level information
and must not be committed.

`make verify-artifacts` is the public, read-only check. It verifies the committed derived
evidence, release manifest, manuscript assets, and PDF. `make verify-full` first reports every
missing private input. When all six inputs exist, it requires the ignored maintainer hook
`data/private_rebuild.py`, calls it with
`--destination build/private-release`, and verifies the resulting complete candidate against
the public release contract. The hook and licensed inputs are external to this repository.
This interface distinguishes public code and artifact verification from historical data
reconstruction without claiming a standalone raw-to-paper rebuild.

The project does not promise controlled reviewer access to the private inputs.
