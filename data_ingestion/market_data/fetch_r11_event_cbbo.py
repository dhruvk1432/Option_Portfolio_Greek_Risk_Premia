"""Cost-guarded targeted OPRA CBBO pulls for R1.1 state transitions.

The command estimates cost by default and downloads only with ``--execute``.
It reads the generated event request manifest, never logs credentials, and
stores licensed data outside the paper artifact directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "research/papers/option_only_markowitz/analysis/artifacts/r11_higher_risk/r11_event_quote_request.csv"
)
DEFAULT_OUT = ROOT / "data/databento_cache/r11_event_cbbo"


def _client():
    load_dotenv(ROOT / ".env")
    if os.environ.get("DATABENTO_API_KEY2"):
        os.environ["DATABENTO_API_KEY"] = os.environ["DATABENTO_API_KEY2"]
    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("Databento credentials unavailable; no request was made")
    import databento as db

    return db.Historical()


def build_requests(manifest: pd.DataFrame) -> list[dict[str, object]]:
    required = {"execution_date", "symbol"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest must contain {sorted(required)}")
    frame = manifest.copy()
    frame["execution_date"] = pd.to_datetime(frame["execution_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["execution_date", "symbol"])
    requests: list[dict[str, object]] = []
    for date, group in frame.groupby("execution_date", observed=True):
        symbols = sorted(set(group["symbol"].astype(str)))
        requests.append(
            {
                "dataset": "OPRA.PILLAR",
                "schema": "cbbo-1m",
                "symbols": symbols,
                "start": pd.Timestamp(date).date().isoformat(),
                "end": (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat(),
            }
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_cost <= 0:
        raise SystemExit("--max-cost must be positive")
    if not args.manifest.exists():
        raise SystemExit(f"request manifest missing: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    requests = build_requests(manifest)
    client = _client()
    estimates: list[tuple[dict[str, object], float]] = []
    for request in requests:
        estimate = float(client.metadata.get_cost(**request))
        estimates.append((request, estimate))
        print(
            f"{request['start']}: {len(request['symbols'])} held symbols, "
            f"estimated ${estimate:.4f}"
        )
    total = sum(value for _, value in estimates)
    print(f"total estimated cost: ${total:.4f}; configured ceiling: ${args.max_cost:.2f}")
    if total > args.max_cost:
        raise SystemExit("estimated cost exceeds --max-cost; nothing downloaded")
    if not args.execute:
        print("dry run only; pass --execute to download licensed CBBO")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for request, _ in estimates:
        date = str(request["start"])
        destination = args.out_dir / f"opra_cbbo1m_{date}.parquet"
        if destination.exists():
            print(f"{date}: cached")
            continue
        data = client.timeseries.get_range(**request).to_df().reset_index()
        if data.empty:
            print(f"{date}: no CBBO returned")
            continue
        data.to_parquet(destination)
        print(f"{date}: wrote {len(data):,} licensed rows")


if __name__ == "__main__":
    main()
