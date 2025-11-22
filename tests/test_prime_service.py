import time
from typing import Mapping

from prime_schema import DEFAULT_PRIME_SCHEMA
from services.prime_service import PrimeService


class DummyApiService:
    def fetch_ledger(self, *_args, **_kwargs):
        return {}


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def write_ledger_entry(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
            "state": {
                "metadata": dict(kwargs.get("metadata") or {}),
                "coordinates": dict(kwargs.get("coordinates") or {}),
            },
            "timestamp": time.time(),
        }


def test_ingest_writes_single_ledger_entry_with_default_entity(monkeypatch):
    monkeypatch.setattr("services.prime_service.time.time", lambda: 1234.0)
    backend = DummyBackend()
    service = PrimeService(
        api_service=DummyApiService(),
        fallback_prime=2,
        backend_client=backend,
        default_entity="Demo_dev",
    )

    result = service.ingest(
        None,  # exercise default entity fallback
        "Team sync about roadmap",
        DEFAULT_PRIME_SCHEMA,
        ledger_id="alpha-ledger",
    )

    assert backend.calls and len(backend.calls) == 1
    call = backend.calls[0]
    assert call["key_namespace"] == "alpha-ledger"
    assert call["key_identifier"] == "demo_dev-structured"
    assert call["entity"] == "Demo_dev"
    assert call["phase"] == "ingest"
    assert call["metadata"]["entity"] == "Demo_dev"
    assert call["metadata"]["text"] == "Team sync about roadmap"
    assert call["metadata"]["structured"]["slots"]
    assert call["metadata"]["timestamp"] == 1234.0

    ledger_entry = result.get("ledger_entry")
    assert isinstance(ledger_entry, Mapping)
    assert ledger_entry.get("entry_id") == "alpha-ledger:demo_dev-structured"


def test_ingest_metadata_includes_structured_bundle(monkeypatch):
    backend = DummyBackend()
    monkeypatch.setattr("services.prime_service.time.time", lambda: 1234.0)

    service = PrimeService(
        api_service=DummyApiService(),
        fallback_prime=2,
        backend_client=backend,
    )

    result = service.ingest(
        "Demo_dev",
        " Ledger ingest payload with metadata ",
        DEFAULT_PRIME_SCHEMA,
    )

    assert backend.calls and len(backend.calls) == 1
    payload = backend.calls[0]
    metadata = payload.get("metadata", {})
    assert metadata.get("text") == "Ledger ingest payload with metadata"
    assert metadata.get("entity") == "Demo_dev"
    assert metadata.get("structured")
    assert metadata.get("factors")
    assert metadata.get("timestamp") == 1234.0

    ledger_entry = result.get("ledger_entry")
    assert isinstance(ledger_entry, Mapping)
    assert ledger_entry.get("entry_id") == f"{payload['key_namespace']}:{payload['key_identifier']}"
    state = ledger_entry.get("state") if hasattr(ledger_entry, "get") else None
    assert isinstance(state, Mapping)
    assert state.get("metadata", {}).get("structured")
