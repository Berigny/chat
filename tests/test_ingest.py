from typing import Any, Iterable, Mapping

import pytest

from services.prime_service import PrimeService


class RecordingApiService:
    def __init__(self) -> None:
        self.anchor_calls: list[tuple[str, list[dict[str, Any]], str | None, str]] = []

    def anchor(
        self,
        entity: str,
        factors: Iterable[Mapping[str, Any]],
        *,
        ledger_id: str | None = None,
        text: str | None = None,
        modifiers: Iterable[int] | None = None,
    ) -> Mapping[str, Any]:
        payload = [dict(item) for item in factors]
        self.anchor_calls.append((entity, payload, ledger_id, text or ""))
        return {"edges": [], "energy": 1.0, "text": text or ""}


class RecordingBackendClient:
    def __init__(self) -> None:
        self.write_calls: list[dict[str, Any]] = []

    def write_ledger_entry(self, **kwargs) -> Mapping[str, Any]:
        self.write_calls.append(kwargs)
        metadata = kwargs.get("metadata") or {}
        return {
            "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
            "state": {"metadata": metadata},
        }


SCHEMA = {
    2: {"tier": "S"},
    3: {"tier": "S"},
    5: {"tier": "S"},
    7: {"tier": "S"},
    11: {"tier": "A"},
    13: {"tier": "A"},
    17: {"tier": "A"},
    19: {"tier": "A"},
    37: {"tier": "C"},
}


@pytest.fixture()
def prime_service() -> tuple[PrimeService, RecordingApiService, RecordingBackendClient]:
    api = RecordingApiService()
    backend = RecordingBackendClient()
    service = PrimeService(api_service=api, fallback_prime=23, backend_client=backend)
    return service, api, backend


def _ingest(service: PrimeService, text: str) -> Mapping[str, Any]:
    return service.ingest(
        "demo",
        text,
        SCHEMA,
        factors_override=[
            {"prime": 2, "delta": 1},
            {"prime": 37, "delta": 1},
            {"prime": 11, "delta": 1},
        ],
    )


def test_ingest_writes_single_ledger_entry(
    prime_service: tuple[PrimeService, RecordingApiService, RecordingBackendClient]
) -> None:
    service, api, backend = prime_service

    result = _ingest(service, "Meeting recap with immutable storage")

    assert not api.anchor_calls, "Anchor should be skipped for ingest"  # /anchor is deprecated here
    assert len(backend.write_calls) == 1
    call = backend.write_calls[0]
    assert call["phase"] == "ingest"
    assert call["entity"] == "demo"
    assert isinstance(call.get("metadata", {}), Mapping)
    structured = call["metadata"].get("structured")
    assert structured and structured.get("s2")
    ledger_entry = result.get("ledger_entry")
    assert isinstance(ledger_entry, Mapping)
    assert ledger_entry.get("entry_id") == f"{call['key_namespace']}:{call['key_identifier']}"


def test_ingest_persists_factors_in_metadata(
    prime_service: tuple[PrimeService, RecordingApiService, RecordingBackendClient]
) -> None:
    service, api, backend = prime_service

    _ingest(service, "Second body copy")

    assert backend.write_calls, "Expected a ledger write"
    metadata = backend.write_calls[-1]["metadata"]
    factor_primes = {item.get("prime") for item in metadata.get("factors", []) if isinstance(item, Mapping)}
    assert {2, 11, 37}.issubset(factor_primes)


def test_ingest_blocks_flow_violations_without_writes(
    prime_service: tuple[PrimeService, RecordingApiService, RecordingBackendClient]
) -> None:
    service, api, backend = prime_service

    result = service.ingest(
        "demo",
        "hello",
        SCHEMA,
        factors_override=[{"prime": 2, "delta": 1}, {"prime": 11, "delta": 1}],
    )

    assert result["flow_errors"]
    assert backend.write_calls == []
    assert not api.anchor_calls
