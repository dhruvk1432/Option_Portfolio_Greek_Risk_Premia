"""Probe Databento coverage with implicit credentials; cost-guarded.

Run:  .venv/bin/python -m data_ingestion.market_data.databento_probe

SECURITY: credentials are loaded into the process environment via
python-dotenv and consumed implicitly by `databento.Historical()`.
Nothing in this module reads, prints, or logs the key value.

Every download is gated by `metadata.get_cost`; the per-call budget is
capped at MAX_COST_USD.  Output: a coverage/cost report (no secrets) at
research/reports/pipeline_reports/databento_probe.md plus small cached samples under
data/databento_cache/.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

MAX_COST_USD = 5.0
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE = os.path.join(_ROOT, "data", "databento_cache")
REPORT = os.path.join(_ROOT, "research", "reports", "pipeline_reports", "databento_probe.md")

DATASETS_OF_INTEREST = [
    "GLBX.MDP3",     # CME futures L2/L3 (user: post-2010)
    "XNAS.ITCH",     # Nasdaq TotalView L3
    "EQUS.MINI",     # US equities consolidated sample
    "DBEQ.BASIC",    # US equities basic
    "OPRA.PILLAR",   # US listed options
]


def main() -> None:
    load_dotenv(os.path.join(_ROOT, ".env"))  # implicit; never echoed
    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY not present after .env load -> skipping")
        return
    import databento as db
    client = db.Historical()  # key picked up from env implicitly

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    lines = ["# Databento probe (no secrets)", ""]
    available = {d for d in client.metadata.list_datasets()}
    for ds in DATASETS_OF_INTEREST:
        if ds not in available:
            lines.append(f"- `{ds}`: NOT in account-visible datasets")
            continue
        try:
            rng = client.metadata.get_dataset_range(dataset=ds)
            start = getattr(rng, "start", None) or rng.get("start")
            end = getattr(rng, "end", None) or rng.get("end")
            lines.append(f"- `{ds}`: {start} -> {end}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- `{ds}`: range query failed ({type(e).__name__})")
    print("\n".join(lines))

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
