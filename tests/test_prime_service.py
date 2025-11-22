import time
from typing import Mapping

from prime_schema import DEFAULT_PRIME_SCHEMA
from services.prime_service import PrimeService


class DummyApiService:
    def fetch_ledger(self, *_args, **_kwargs):
        return {}


def test_ingest_writes_single_ledger_entry(monkeypatch):
    backend_calls: list[dict] = []

    class DummyBackend:
        def write_ledger_entry(self, **kwargs):
            backend_calls.append(kwargs)
            return {
                "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
                "state": {"metadata": dict(kwargs.get("metadata") or {})},
                "timestamp": time.time(),
            }

    service = PrimeService(
        api_service=DummyApiService(),
        fallback_prime=2,
        backend_client=DummyBackend(),
    )

    result = service.ingest(
        "Demo_dev",
        "Team sync about roadmap",
        DEFAULT_PRIME_SCHEMA,
        ledger_id="alpha-ledger",
    )

    assert result["text"] == "Team sync about roadmap"
    assert result["factors"]
    assert backend_calls and len(backend_calls) == 1

    call = backend_calls[0]
    assert call["key_namespace"] == "alpha-ledger"
    assert call["key_identifier"] == "demo_dev-structured"
    assert call["phase"] == "ingest"
    assert call["metadata"]["text"] == "Team sync about roadmap"
    assert call["metadata"]["entity"] == "Demo_dev"
    assert call["metadata"]["structured"]["slots"]
    assert result["ledger_entry"]["entry_id"] == "alpha-ledger:demo_dev-structured"


def test_ingest_metadata_includes_structured_bundle(monkeypatch):
    backend_calls: list[dict] = []

    class DummyBackend:
        def write_ledger_entry(self, **kwargs):
            backend_calls.append(kwargs)
            return {
                "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
                "state": {
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "coordinates": dict(kwargs.get("coordinates") or {}),
                },
            }

    monkeypatch.setattr("services.prime_service.time.time", lambda: 1234.0)

    service = PrimeService(
        api_service=DummyApiService(),
        fallback_prime=2,
        backend_client=DummyBackend(),
    )

    result = service.ingest(
        "Demo_dev",
        " Ledger ingest payload with metadata ",
        DEFAULT_PRIME_SCHEMA,
    )

    assert backend_calls and len(backend_calls) == 1
    payload = backend_calls[0]
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
