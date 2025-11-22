"""Application configuration helpers for Streamlit surfaces."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import streamlit as st


DEFAULT_METRIC_FLOORS = {
    "tokens_deduped": 12500.0,
    "ledger_integrity": 0.97,
    "durability_h": 36.0,
}


@dataclass(frozen=True)
class AppSettings:
    """Immutable configuration bundle for the chat demo."""

    api_base: str
    api_key: str | None
    default_entity: str
    default_ledger_id: str
    metric_floors: dict[str, float]
    genai_api_key: str | None
    openai_api_key: str | None
    rocksdb_data_path: str
    enable_advanced_probes: bool
    enable_ledger_management: bool


def _safe_secret(key: str) -> Any:
    """Return a Streamlit secret when available."""

    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _load_metric_floors(raw: str | Mapping[str, Any] | None) -> dict[str, float]:
    """Parse the metric floors configuration into numeric values."""

    if not raw:
        return {}
    if isinstance(raw, Mapping):
        source = raw
    else:
        try:
            source = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return {}
    floors: dict[str, float] = {}
    for key, value in source.items():
        try:
            floors[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return floors


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse truthy/falsey strings and primitives into booleans."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return False
    return default


def load_settings() -> AppSettings:
    """Collect runtime configuration from environment and secrets."""

    api_base = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
    api_key = _safe_secret("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY")
    default_entity = os.getenv("DEFAULT_ENTITY", "Demo_dev")
    default_ledger_id = os.getenv("DEFAULT_LEDGER_ID", "default")
    metric_source = (
        _safe_secret("METRIC_FLOORS")
        or os.getenv("METRIC_FLOORS")
        or {}
    )
    metric_floors = {**DEFAULT_METRIC_FLOORS, **_load_metric_floors(metric_source)}
    genai_api_key = _safe_secret("API_KEY") or os.getenv("API_KEY")
    openai_api_key = _safe_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    rocksdb_data_path = os.getenv("ROCKSDB_DATA_PATH", "/app/rocksdb-data")
    enable_advanced_probes = _coerce_bool(
        _safe_secret("ENABLE_ADVANCED_PROBES") or os.getenv("ENABLE_ADVANCED_PROBES"),
        default=False,
    )
    enable_ledger_management = _coerce_bool(
        _safe_secret("ENABLE_LEDGER_MANAGEMENT") or os.getenv("ENABLE_LEDGER_MANAGEMENT"),
        default=False,
    )
    return AppSettings(
        api_base=api_base.rstrip("/"),
        api_key=api_key or "demo-key",
        default_entity=default_entity,
        default_ledger_id=default_ledger_id,
        metric_floors=metric_floors,
        genai_api_key=genai_api_key,
        openai_api_key=openai_api_key,
        rocksdb_data_path=rocksdb_data_path,
        enable_advanced_probes=enable_advanced_probes,
        enable_ledger_management=enable_ledger_management,
    )


__all__ = ["AppSettings", "DEFAULT_METRIC_FLOORS", "load_settings"]
