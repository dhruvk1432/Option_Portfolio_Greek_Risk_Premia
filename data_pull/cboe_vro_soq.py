from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from data_pull.config import REPO_ROOT, load_repo_env


SCALAR_SETTLEMENT_URL = "https://www-api.cboe.com/us/futures/market_statistics/settlement/csv?dt={date}"
INDEX_SETTLEMENT_URL = "https://www.cboe.com/index_settlement_values/get_sv_data/{sv_type}/{year}/"
SOQ_COMPONENT_URL = "https://www.cboe.com/us/futures/market_statistics/vix_settlement_series/{year}/{month}/soq_vxs_{yyyymmdd}.csv-dl"
SOQ_ARCHIVE_URL = "https://cdn.cboe.com/resources/futures/archive/settlements/Vix_Series_{mmddyyyy}.xls"
DEFAULT_OUTPUT_DIR = Path("data/public/cboe/vro_soq")
USER_AGENT = "OptionsPortfolioModel/1.0 (+https://www.cboe.com/)"


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: str
    sha256: str
    status: str
    error: str = ""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - public Cboe HTTPS endpoint.
        return response.read()


def _write_download(path: Path, payload: bytes) -> DownloadResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return DownloadResult(url="", path=str(path), sha256=sha256_bytes(payload), status="downloaded")


def fetch_url_to_file(url: str, path: Path, *, allow_missing: bool = False) -> DownloadResult:
    try:
        payload = _download_bytes(url)
    except HTTPError as exc:
        if allow_missing and exc.code == 404:
            return DownloadResult(url=url, path=str(path), sha256="", status="unavailable", error="404")
        return DownloadResult(url=url, path=str(path), sha256="", status="failed", error=f"HTTP {exc.code}")
    except URLError as exc:
        return DownloadResult(url=url, path=str(path), sha256="", status="failed", error=str(exc.reason))
    except TimeoutError as exc:
        return DownloadResult(url=url, path=str(path), sha256="", status="failed", error=str(exc))
    result = _write_download(path, payload)
    return DownloadResult(url=url, path=result.path, sha256=result.sha256, status=result.status)


def parse_cboe_scalar_settlement(payload: bytes, settlement_date: pd.Timestamp) -> dict[str, object] | None:
    """Parse Cboe futures settlement CSV and return the expiring VX row."""

    text = payload.decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    target = pd.Timestamp(settlement_date).normalize()
    candidates: list[dict[str, object]] = []
    for row in rows:
        product = str(row.get("Product", "")).strip().upper()
        expiry = pd.to_datetime(row.get("Expiration Date"), errors="coerce")
        price = pd.to_numeric(row.get("Price"), errors="coerce")
        if product != "VX" or pd.isna(expiry) or pd.isna(price) or float(price) <= 0:
            continue
        if pd.Timestamp(expiry).normalize() == target:
            candidates.append(
                {
                    "settlement_date": target,
                    "expiration": target,
                    "product": "VIX option",
                    "root": "VIX",
                    "settlement_symbol": str(row.get("Symbol", "VRO")).strip() or "VRO",
                    "settlement_value": float(price),
                }
            )
    if not candidates:
        return None
    return candidates[-1]


def soq_component_url(settlement_date: pd.Timestamp) -> str:
    dt = pd.Timestamp(settlement_date).normalize()
    return SOQ_COMPONENT_URL.format(year=dt.strftime("%Y"), month=dt.strftime("%m"), yyyymmdd=dt.strftime("%Y%m%d"))


def soq_archive_url(settlement_date: pd.Timestamp) -> str:
    dt = pd.Timestamp(settlement_date).normalize()
    return SOQ_ARCHIVE_URL.format(mmddyyyy=dt.strftime("%m%d%Y"))


def scalar_url(settlement_date: pd.Timestamp) -> str:
    return SCALAR_SETTLEMENT_URL.format(date=pd.Timestamp(settlement_date).strftime("%Y-%m-%d"))


def index_settlement_url(year: int, sv_type: str = "O") -> str:
    return INDEX_SETTLEMENT_URL.format(sv_type=sv_type, year=int(year))


def parse_cboe_index_settlement_payload(payload: bytes) -> dict[object, float]:
    """Parse Cboe Index Settlement Values API payload into monthly VRO values."""

    obj = json.loads(payload.decode("utf-8-sig", errors="replace"))
    data = obj.get("data", [])
    values: dict[object, float] = {}
    if isinstance(data, list):
        for row in data:
            symbol = str(row.get("settlement_symbol", "")).upper()
            desc = str(row.get("description", "")).upper()
            if symbol != "VRO" and "VIX" not in desc:
                continue
            expiry = pd.to_datetime(row.get("expiration_date"), errors="coerce")
            value = pd.to_numeric(row.get("settlement_value"), errors="coerce")
            if pd.isna(expiry) or pd.isna(value) or float(value) <= 0:
                continue
            values[pd.Timestamp(expiry).normalize()] = float(value)
        return values
    if not isinstance(data, str) or not data.strip():
        return values
    labels = re.findall(r"<h4>\s*([A-Za-z]+)\s+(\d{4})\s+Settlement Values\s*</h4>", data)
    try:
        tables = pd.read_html(io.StringIO(data))
    except ValueError:
        return values
    for (month_name, year), table in zip(labels, tables):
        if table.empty or table.shape[1] < 2:
            continue
        left = table.iloc[:, 0].astype(str)
        mask = left.str.contains("VIX", case=False, na=False) & left.str.contains("VRO", case=False, na=False)
        if not mask.any():
            continue
        value = pd.to_numeric(table.loc[mask, table.columns[1]], errors="coerce").dropna()
        if value.empty or float(value.iloc[0]) <= 0:
            continue
        period = pd.Period(pd.Timestamp(f"{month_name} 1 {year}"), freq="M")
        values[period] = float(value.iloc[0])
    return values


def required_vix_expiries(repo_root: Path = REPO_ROOT) -> list[pd.Timestamp]:
    """Return VIX expiries needed by the option-only paper's current ledger.

    The empirical ledger is the authoritative default because it contains the exact
    contract universe used in the paper. If it is absent, fall back to local raw VIX
    OPRA shards and parse the OSI expiration from the symbol.
    """

    detail = repo_root / "research/papers/option_only_markowitz/artifacts/vix_holding_return_detail.csv"
    if detail.exists():
        frame = pd.read_csv(detail, usecols=["expiry"])
        dates = pd.to_datetime(frame["expiry"], errors="coerce").dropna().dt.normalize().unique()
        return sorted(pd.Timestamp(x).normalize() for x in dates)

    shards = sorted((repo_root / "data/databento_cache").glob("opra_vix_chain_*.parquet"))
    expiries: set[pd.Timestamp] = set()
    for path in shards:
        try:
            frame = pd.read_parquet(path, columns=["symbol"])
        except Exception:
            continue
        symbols = frame["symbol"].astype(str).dropna().unique()
        for symbol in symbols:
            compact = symbol.strip()
            if len(compact) < 15 or not compact.upper().startswith("VIX"):
                continue
            date_token = compact.replace(" ", "")[3:9]
            try:
                expiry = pd.to_datetime(date_token, format="%y%m%d", errors="raise")
            except Exception:
                continue
            expiries.add(pd.Timestamp(expiry).normalize())
    return sorted(expiries)


def normalize_cboe_settlements(
    rows: Iterable[dict[str, object]],
    *,
    ingested_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    timestamp = pd.Timestamp.utcnow() if ingested_timestamp is None else pd.Timestamp(ingested_timestamp)
    out = pd.DataFrame(list(rows))
    columns = [
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
        "component_source_url",
        "component_file_hash",
        "component_status",
    ]
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["settlement_date"] = pd.to_datetime(out["settlement_date"], errors="coerce").dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["settlement_value"] = pd.to_numeric(out["settlement_value"], errors="coerce")
    out["published_timestamp"] = pd.NaT
    out["ingested_timestamp"] = pd.to_datetime(timestamp, utc=True)
    out = out.dropna(subset=["settlement_date", "settlement_value"])
    out = out[out["settlement_value"].gt(0)].copy()
    out["root"] = out["root"].astype(str).str.upper().str.strip()
    out["settlement_symbol"] = out["settlement_symbol"].astype(str).str.upper().str.strip()
    out = out.sort_values(["settlement_date", "source"]).drop_duplicates(["settlement_date", "root"], keep="last")
    return out[columns].reset_index(drop=True)


def pull_cboe_vro_soq(
    repo_root: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expiries: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_root = Path(repo_root)
    out_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    raw_dir = out_dir / "raw"
    required = sorted(pd.Timestamp(x).normalize() for x in (expiries if expiries is not None else required_vix_expiries(repo_root)))
    required_by_month = pd.Series(required).map(lambda x: pd.Period(pd.Timestamp(x), freq="M")).value_counts().to_dict() if required else {}
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    index_cache: dict[tuple[int, str], tuple[DownloadResult, dict[object, float]]] = {}
    for expiry in required:
        ymd = expiry.strftime("%Y%m%d")
        scalar = scalar_url(expiry)
        scalar_path = raw_dir / "scalar_settlement" / expiry.strftime("%Y") / f"cboe_vx_settlement_{ymd}.csv"
        scalar_result = fetch_url_to_file(scalar, scalar_path)
        component = soq_component_url(expiry)
        component_path = raw_dir / "soq_components" / expiry.strftime("%Y") / f"soq_vxs_{ymd}.csv"
        component_result = fetch_url_to_file(component, component_path, allow_missing=True)
        if component_result.status == "unavailable":
            archive_component = soq_archive_url(expiry)
            archive_component_path = raw_dir / "soq_components" / expiry.strftime("%Y") / f"Vix_Series_{expiry.strftime('%m%d%Y')}.xls"
            archive_result = fetch_url_to_file(archive_component, archive_component_path, allow_missing=True)
            if archive_result.status == "downloaded":
                component = archive_component
                component_result = archive_result

        parsed = None
        scalar_source = "cboe_vx_final_settlement"
        if scalar_result.status == "downloaded":
            parsed = parse_cboe_scalar_settlement(Path(scalar_result.path).read_bytes(), expiry)
        if parsed is None:
            year = int(expiry.year)
            month = pd.Period(expiry, freq="M")
            for sv_type in ("O", "S", "V"):
                cache_key = (year, sv_type)
                if cache_key not in index_cache:
                    index_url = index_settlement_url(year, sv_type)
                    index_path = raw_dir / "index_settlement_values" / f"cboe_index_settlement_values_{sv_type}_{year}.json"
                    index_result = fetch_url_to_file(index_url, index_path, allow_missing=True)
                    index_values = parse_cboe_index_settlement_payload(Path(index_result.path).read_bytes()) if index_result.status == "downloaded" else {}
                    index_cache[cache_key] = (index_result, index_values)
                index_result, index_values = index_cache[cache_key]
                settlement_value = None
                if expiry in index_values:
                    settlement_value = float(index_values[expiry])
                elif required_by_month.get(month, 0) == 1 and month in index_values:
                    settlement_value = float(index_values[month])
                if settlement_value is None:
                    continue
                parsed = {
                    "settlement_date": expiry,
                    "expiration": expiry,
                    "product": "VIX option",
                    "root": "VIX",
                    "settlement_symbol": "VRO",
                    "settlement_value": settlement_value,
                }
                scalar = index_result.url
                scalar_result = index_result
                scalar_source = "cboe_index_settlement_values"
                break
        if parsed is not None:
            parsed.update(
                {
                    "source": scalar_source,
                    "source_url": scalar,
                    "source_file_hash": scalar_result.sha256,
                    "component_source_url": component,
                    "component_file_hash": component_result.sha256,
                    "component_status": component_result.status,
                }
            )
            rows.append(parsed)

        audit_rows.append(
            {
                "expiration": expiry.strftime("%Y-%m-%d"),
                "scalar_status": scalar_result.status,
                "scalar_url": scalar,
                "scalar_file": scalar_result.path,
                "scalar_sha256": scalar_result.sha256,
                "scalar_error": scalar_result.error,
                "component_status": component_result.status,
                "component_url": component,
                "component_file": component_result.path,
                "component_sha256": component_result.sha256,
                "component_error": component_result.error,
                "parsed_settlement": parsed is not None,
                "settlement_value": parsed["settlement_value"] if parsed is not None else "",
            }
        )

    settlements = normalize_cboe_settlements(rows)
    audit = pd.DataFrame(audit_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    settlements.to_csv(out_dir / "vro_soq_settlements.csv", index=False)
    audit.to_csv(out_dir / "vro_soq_download_audit.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_expiries": len(required),
        "exact_rows": int(len(settlements)),
        "missing_expiries": sorted(set(str(x)[:10] for x in required) - set(settlements["settlement_date"].dt.strftime("%Y-%m-%d")) if not settlements.empty else [str(x)[:10] for x in required]),
        "output_file": str(out_dir / "vro_soq_settlements.csv"),
        "audit_file": str(out_dir / "vro_soq_download_audit.csv"),
    }
    (out_dir / "vro_soq_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return settlements, audit


def _parse_date_list(value: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(part.strip()).normalize() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Cboe VIX VRO/SOQ settlement inputs.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--expiries", default="", help="Optional comma-separated expiry dates. Defaults to paper VIX ledger.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    load_repo_env(repo_root, args.env_file)
    env_output = Path(args.output_dir or os.environ.get("OPTION_MARKOWITZ_VRO_OUTPUT_DIR", "") or DEFAULT_OUTPUT_DIR)
    expiries = _parse_date_list(args.expiries) if args.expiries else None
    settlements, audit = pull_cboe_vro_soq(repo_root=repo_root, output_dir=env_output, expiries=expiries)
    print(json.dumps({"settlement_rows": len(settlements), "audit_rows": len(audit), "output_dir": str((repo_root / env_output) if not env_output.is_absolute() else env_output)}, indent=2))


if __name__ == "__main__":
    main()
