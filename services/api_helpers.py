"""Helper functions for Streamlit tabs to interact with API services."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import requests

from services.api import ApiService
from services.memory_service import MemoryService
from services.prime_service import PrimeService

ADMIN_CONFIG_KEY = "__admin_config__"
_API_SERVICE_KEY = "__api_service__"
_API_SERVICE_BASE_KEY = "__api_service_base__"
_MEMORY_SERVICE_KEY = "__memory_service__"
_PRIME_SERVICE_KEY = "__prime_service__"
_PRIME_SERVICE_BASE_KEY = "__prime_service_base__"
_DEFAULT_TIMEOUT = 5


def _config(session_state: MutableMapping[str, Any]) -> Mapping[str, Any]:
    raw = session_state.get(ADMIN_CONFIG_KEY)
    if isinstance(raw, Mapping):
        return raw
    return {}


def get_ledger_id(session_state: MutableMapping[str, Any], default: str | None = None) -> str | None:
    return session_state.get("ledger_id") or default


def build_headers(
    session_state: MutableMapping[str, Any],
    *,
    include_ledger: bool = True,
) -> dict[str, str]:
    cfg = _config(session_state)
    api_key = cfg.get("api_key")
    headers = {"x-api-key": api_key} if api_key else {}
    if include_ledger:
        ledger_id = get_ledger_id(session_state)
        if ledger_id:
            headers["X-Ledger-ID"] = ledger_id
    return headers


def _prime_fallback(session_state: MutableMapping[str, Any]) -> int:
    cfg = _config(session_state)
    fallback = cfg.get("fallback_prime")
    if isinstance(fallback, int):
        return fallback
    return 2


def get_api_service(session_state: MutableMapping[str, Any]) -> ApiService:
    cfg = _config(session_state)
    base_url = cfg.get("api_url")
    api_key = cfg.get("api_key")
    if not base_url:
        raise ValueError("API base URL is not configured")
    cached_base = session_state.get(_API_SERVICE_BASE_KEY)
    service = session_state.get(_API_SERVICE_KEY)
    if not isinstance(service, ApiService) or cached_base != base_url:
        service = ApiService(base_url, api_key)
        session_state[_API_SERVICE_KEY] = service
        session_state[_API_SERVICE_BASE_KEY] = base_url
    return service


def get_memory_service(session_state: MutableMapping[str, Any]) -> MemoryService:
    cfg = _config(session_state)
    prime_weights = cfg.get("prime_weights") or {}
    service = session_state.get(_MEMORY_SERVICE_KEY)
    if not isinstance(service, MemoryService):
        service = MemoryService(get_api_service(session_state), prime_weights)
        session_state[_MEMORY_SERVICE_KEY] = service
    return service


def get_prime_service(session_state: MutableMapping[str, Any]) -> PrimeService:
    fallback_prime = _prime_fallback(session_state)
    cached_fallback = session_state.get(_PRIME_SERVICE_BASE_KEY)
    service = session_state.get(_PRIME_SERVICE_KEY)
    if not isinstance(service, PrimeService) or cached_fallback != fallback_prime:
        service = PrimeService(get_api_service(session_state), fallback_prime)
        session_state[_PRIME_SERVICE_KEY] = service
        session_state[_PRIME_SERVICE_BASE_KEY] = fallback_prime
    return service


def ingest_text(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    text: str,
    schema: Mapping[int, Mapping[str, object]],
    ledger_id: str | None,
    metadata: Mapping[str, object] | None = None,
    factors_override: Sequence[Mapping[str, int]] | None = None,
):
    prime_service = get_prime_service(session_state)
    return prime_service.ingest(
        entity,
        text,
        schema,
        ledger_id=ledger_id,
        factors_override=factors_override,
        metadata=metadata,
    )


def search_probe(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    question: str,
    ledger_id: str | None,
    mode: str = "s1",
    limit: int = 5,
) -> Mapping[str, Any]:
    api_service = get_api_service(session_state)
    return api_service.search(
        entity,
        question,
        ledger_id=ledger_id,
        mode=mode,
        limit=limit,
    )


def fetch_traversal_paths(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    ledger_id: str | None,
    limit: int = 10,
) -> Mapping[str, Any]:
    memory_service = get_memory_service(session_state)
    return memory_service.traversal_paths(
        entity,
        ledger_id=ledger_id,
        limit=limit,
    )


def fetch_inference_state(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    ledger_id: str | None,
    include_history: bool = True,
    limit: int = 10,
) -> Mapping[str, Any]:
    memory_service = get_memory_service(session_state)
    return memory_service.fetch_inference_state(
        entity,
        ledger_id=ledger_id,
        include_history=include_history,
        limit=limit,
    )


def load_ledger_snapshot(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    ledger_id: str | None,
) -> Mapping[str, Any]:
    api_service = get_api_service(session_state)
    payload = api_service.fetch_ledger(entity, ledger_id=ledger_id)
    return payload if isinstance(payload, Mapping) else {}


def coerce_ledger_records(payload: object) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    source = payload
    if isinstance(payload, Mapping):
        if isinstance(payload.get("ledgers"), list):
            source = payload.get("ledgers")  # type: ignore[assignment]
        else:
            source = [
                {"ledger_id": key, "path": value}
                for key, value in payload.items()
                if isinstance(key, str)
            ]
    if not isinstance(source, Sequence):
        return records
    for item in source:
        if isinstance(item, str):
            records.append({"ledger_id": item})
            continue
        if not isinstance(item, Mapping):
            continue
        ledger_id = item.get("ledger_id") or item.get("id") or item.get("name")
        if not ledger_id:
            continue
        records.append({"ledger_id": str(ledger_id), "path": item.get("path") or item.get("base_path")})
    return records


def refresh_ledgers(
    session_state: MutableMapping[str, Any],
    *,
    requester=requests,
) -> tuple[list[dict[str, str]], str | None]:
    cfg = _config(session_state)
    base_url = cfg.get("api_url")
    if not base_url:
        return [], "API base URL missing"
    try:
        response = requester.get(
            f"{base_url}/admin/ledgers",
            headers=build_headers(session_state, include_ledger=False),
            timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return session_state.get("ledgers", []), str(exc)
    except ValueError:
        payload = []
    records = coerce_ledger_records(payload) or session_state.get("ledgers", [])
    return records, None


def create_or_switch_ledger(
    session_state: MutableMapping[str, Any],
    ledger_id: str,
    *,
    requester=requests,
) -> tuple[bool, str | None]:
    cfg = _config(session_state)
    base_url = cfg.get("api_url")
    if not base_url:
        return False, "API base URL missing"
    try:
        response = requester.post(
            f"{base_url}/admin/ledgers",
            json={"ledger_id": ledger_id},
            headers=build_headers(session_state, include_ledger=False),
            timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, str(exc)
    memory_service = get_memory_service(session_state)
    memory_service.clear_entity_cache(ledger_id=ledger_id)
    session_state["ledger_id"] = ledger_id
    return True, None


def reset_entity_factors(
    session_state: MutableMapping[str, Any],
    *,
    entity: str,
    schema: Mapping[int, Mapping[str, object]],
    ledger_id: str | None,
) -> bool:
    api_service = get_api_service(session_state)
    try:
        payload = api_service.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException:
        return False
    if not isinstance(payload, Mapping):
        return False
    factors = payload.get("factors") if isinstance(payload.get("factors"), Sequence) else []
    prime_service = get_prime_service(session_state)
    for entry in factors:
        if not isinstance(entry, Mapping):
            continue
        prime = entry.get("prime")
        value = entry.get("value")
        if not isinstance(prime, int) or value is None:
            continue
        delta = -abs(int(value))
        try:
            prime_service.anchor(
                entity,
                f"[reset] prime {prime}",
                schema,
                ledger_id=ledger_id,
                factors_override=[{"prime": prime, "delta": delta}],
            )
        except requests.RequestException:
            return False
    return True
