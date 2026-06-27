"""Exact VIX VRO/SOQ settlement ingestion and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import hashlib

import numpy as np
import pandas as pd


SETTLEMENT_COLUMNS = [
    "settlement_date",
    "expiration",
    "product",
    "root",
    "settlement_symbol",
    "settlement_value",
    "source",
    "source_url",
    "source_file_hash",
    "published_timestamp",
    "ingested_timestamp",
]


@dataclass(frozen=True)
class SettlementCoverage:
    required_expirations: int
    exact_expirations: int
    missing_expirations: tuple[pd.Timestamp, ...]
    status: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_vro_soq_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    source_url: str = "",
    source_file_hash: str = "",
    ingested_timestamp: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Normalize a vendor/Cboe VRO table into the production settlement schema.

    Accepts common column names such as ``date``, ``expiration``, ``settlement_date``,
    ``VRO``, ``settle`` and ``value``.  It rejects non-positive settlement values because
    they would create artificial option payoffs.
    """

    if frame.empty:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    lower = {str(c).lower().strip(): c for c in frame.columns}
    date_col = next((lower[c] for c in ("settlement_date", "expiration", "date", "trade_date") if c in lower), None)
    value_col = next((lower[c] for c in ("settlement_value", "vro", "settle", "settlement", "value", "close") if c in lower), None)
    if date_col is None or value_col is None:
        raise ValueError("VRO/SOQ table must contain a settlement date and settlement value column")
    settlement_date = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    value = pd.to_numeric(frame[value_col], errors="coerce")
    out = pd.DataFrame({
        "settlement_date": settlement_date,
        "expiration": settlement_date,
        "product": frame[lower["product"]] if "product" in lower else "VIX option",
        "root": frame[lower["root"]] if "root" in lower else "VIX",
        "settlement_symbol": frame[lower["settlement_symbol"]] if "settlement_symbol" in lower else "VRO",
        "settlement_value": value,
        "source": source,
        "source_url": source_url,
        "source_file_hash": source_file_hash,
        "published_timestamp": pd.to_datetime(frame[lower["published_timestamp"]], errors="coerce", utc=True) if "published_timestamp" in lower else pd.NaT,
        "ingested_timestamp": pd.Timestamp.utcnow() if ingested_timestamp is None else pd.Timestamp(ingested_timestamp),
    })
    out["published_timestamp"] = pd.to_datetime(out["published_timestamp"], errors="coerce", utc=True)
    out["ingested_timestamp"] = pd.to_datetime(out["ingested_timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["settlement_date", "settlement_value"])
    out = out[out["settlement_value"].gt(0)].copy()
    out["root"] = out["root"].astype(str).str.upper().str.strip()
    out["settlement_symbol"] = out["settlement_symbol"].astype(str).str.upper().str.strip()
    out = out.sort_values(["settlement_date", "ingested_timestamp"]).drop_duplicates(["settlement_date", "root"], keep="last")
    return out[SETTLEMENT_COLUMNS].reset_index(drop=True)


def load_vro_soq_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported settlement file type: {path.suffix}")
    return normalize_vro_soq_frame(frame, source=path.name, source_file_hash=file_sha256(path))


def find_vro_soq_tables(root: Path | str) -> list[Path]:
    root = Path(root)
    patterns = ["**/*vro*.*", "**/*VRO*.*", "**/*soq*.*", "**/*SOQ*.*"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(root.glob(pat))
    return sorted({p for p in paths if p.suffix.lower() in {".csv", ".parquet", ".pq"}})


def load_all_vro_soq(root: Path | str) -> pd.DataFrame:
    frames = []
    for path in find_vro_soq_tables(root):
        try:
            frames.append(load_vro_soq_table(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=SETTLEMENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["settlement_date", "ingested_timestamp"]).drop_duplicates(["settlement_date", "root"], keep="last")
    return out[SETTLEMENT_COLUMNS].reset_index(drop=True)


def settlement_coverage(required_expirations: Iterable[Any], settlements: pd.DataFrame) -> SettlementCoverage:
    required = pd.DatetimeIndex(pd.to_datetime(list(required_expirations), errors="coerce")).dropna().normalize().unique()
    exact = pd.DatetimeIndex(pd.to_datetime(settlements.get("expiration", []), errors="coerce")).dropna().normalize().unique()
    missing = tuple(pd.Timestamp(x).normalize() for x in required if x not in set(exact))
    status = "pass" if not missing else "fail"
    return SettlementCoverage(len(required), len(required) - len(missing), missing, status)


def attach_exact_vix_settlement(holding_detail: pd.DataFrame, settlements: pd.DataFrame) -> pd.DataFrame:
    """Attach exact VRO/SOQ values to a VIX option holding ledger.

    Rows without exact settlement remain visible and are labeled ``missing_vro_soq``.
    This function never silently falls back to spot VIX for production results.
    """

    if holding_detail.empty:
        return holding_detail.copy()
    out = holding_detail.copy()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    exact = settlements.copy()
    if exact.empty:
        out["production_settlement_source"] = "missing_vro_soq"
        out["production_settlement_value"] = np.nan
        return out
    exact["expiration"] = pd.to_datetime(exact["expiration"], errors="coerce").dt.normalize()
    exact = exact[exact["root"].astype(str).str.upper().eq("VIX")].copy()
    merged = out.merge(
        exact[["expiration", "settlement_value", "settlement_symbol", "source", "source_file_hash"]],
        how="left",
        left_on="expiry",
        right_on="expiration",
    )
    merged["production_settlement_source"] = np.where(merged["settlement_value"].notna(), "vro_soq_exact", "missing_vro_soq")
    merged["production_settlement_value"] = merged["settlement_value"]
    return merged.drop(columns=[c for c in ["expiration", "settlement_value"] if c in merged.columns])


def require_exact_vix_settlement(ledger: pd.DataFrame) -> tuple[bool, list[str]]:
    if ledger.empty:
        return False, ["empty_settlement_ledger"]
    if "production_settlement_source" not in ledger:
        return False, ["missing_production_settlement_source"]
    vix_rows = ledger.get("asset_class", pd.Series("vix_option", index=ledger.index)).astype(str).str.lower().eq("vix_option")
    bad = ledger.loc[vix_rows & ~ledger["production_settlement_source"].astype(str).eq("vro_soq_exact")]
    if bad.empty:
        return True, []
    expiries = sorted(pd.to_datetime(bad.get("expiry"), errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    return False, ["missing_exact_vro_soq:" + ",".join(expiries[:20])]
