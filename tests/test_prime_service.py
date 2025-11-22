import time

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
