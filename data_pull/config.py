from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

LICENSED_KEYS = (
    "DATABENTO_API_KEY",
    "DATABENTO_API_KEY2",
    "POLYGON_API_KEY",
    "LSEG_APP_KEY",
    "LSEG_APP_KEY_SIDE_BY_SIDE",
    "LSEG_APP_KEY_EIKON",
    "WRDS_USERNAME",
)

PUBLIC_KEYS = (
    "FRED_API_KEY",
    "EIA_API_KEY",
    "BEA_API_KEY",
)

OPTION_PAPER_ENV = (
    "OPTION_MARKOWITZ_VRO_FILE",
    "OPTION_MARKOWITZ_VRO_DIR",
    "OPTION_MARKOWITZ_VRO_OUTPUT_DIR",
    "OPTION_MARKOWITZ_DOWNLOAD_START",
    "OPTION_MARKOWITZ_DOWNLOAD_END",
    "OPTION_MARKOWITZ_UNDERLYINGS",
)


@dataclass(frozen=True)
class InputSpec:
    name: str
    pattern: str
    required_for: str
    licensed: bool = False


OPTION_MARKOWITZ_INPUTS: tuple[InputSpec, ...] = (
    InputSpec("Greek proxy panel", "data/feature_store/option_greek_proxy_panel.parquet", "headline equity-option run"),
    InputSpec("OPRA surface panel", "data/feature_store/opra_surface_panel.parquet", "headline equity-option run", licensed=True),
    InputSpec("Greek quality report", "data/feature_store/option_greek_quality.csv", "data audit"),
    InputSpec("Raw underlying closes", "data/universe/multi_raw_close.csv", "expiry payoff and equity benchmarks"),
    InputSpec("VIX option shards", "data/databento_cache/opra_vix_chain_*.parquet", "VIX option diagnostics/headline if VRO is exact", licensed=True),
    InputSpec("VX futures curve", "data/universe/vx_futures_daily.parquet", "VIX option Black-76 forward alignment"),
    InputSpec("VIX complex", "data/universe/vix_complex.parquet", "volatility regimes and controls"),
)


def load_repo_env(repo_root: Path = REPO_ROOT, env_file: str | Path = ".env") -> Path:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = repo_root / env_path
    load_dotenv(env_path, override=False)
    return env_path


def configured_keys(keys: Iterable[str]) -> dict[str, bool]:
    return {key: bool(os.environ.get(key)) for key in keys}


def credential_status() -> dict[str, dict[str, bool]]:
    return {
        "licensed": configured_keys(LICENSED_KEYS),
        "public": configured_keys(PUBLIC_KEYS),
        "option_paper": configured_keys(OPTION_PAPER_ENV),
    }


def validate_expected_inputs(repo_root: Path = REPO_ROOT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in OPTION_MARKOWITZ_INPUTS:
        matches = sorted(repo_root.glob(spec.pattern))
        rows.append(
            {
                "name": spec.name,
                "pattern": spec.pattern,
                "required_for": spec.required_for,
                "licensed": spec.licensed,
                "exists": bool(matches),
                "n_files": len(matches),
                "total_bytes": int(sum(path.stat().st_size for path in matches if path.exists())),
                "example": str(matches[0].relative_to(repo_root)) if matches else "",
            }
        )
    return rows
