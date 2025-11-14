from __future__ import annotations

from typing import Mapping

from services.api_service import EnrichmentHelper


class DummyApiService:
    def __init__(self, supported: bool = True) -> None:
        self.body_calls: list[dict[str, object]] = []
        self.enrich_calls: list[tuple[str, dict, str | None]] = []
        self._supported = supported

    def put_ledger_body(self, entity: str, prime: int, body_payload, *, ledger_id=None, metadata=None):
        if isinstance(body_payload, Mapping):
            payload = dict(body_payload)
        else:
            payload = {"body": body_payload}
        self.body_calls.append(
            {
                "entity": entity,
                "prime": prime,
                "payload": payload,
                "ledger_id": ledger_id,
                "metadata_arg": metadata,
            }
        )

    def enrich(self, entity: str, payload: dict, *, ledger_id=None):
        self.enrich_calls.append((entity, payload, ledger_id))
        return {"structured": {"echo": True}}

    def supports_enrich(self) -> bool:
        return self._supported


class DummyPrimeService:
    def __init__(self) -> None:
        self._next = 23
        self.fallback_prime = 23

    def next_body_prime(self, *, reserved=None, entity=None, ledger_id=None):
        reserved = set(reserved or [])
        candidate = self._next
        while candidate in reserved:
            candidate += 2
        self._next = candidate + 2
        return candidate


def test_enrichment_helper_mints_bodies_and_calls_enrich():
    api = DummyApiService()
    primes = DummyPrimeService()
    helper = EnrichmentHelper(api, primes)

    result = helper.submit(
        "demo",
        ref_prime=29,
        deltas=[{"prime": "2", "delta": "1"}, {"prime": "x"}],
        body_chunks=[" first body ", "second"],
        metadata={"workflow": "unit"},
        ledger_id="alpha",
    )

    assert len(api.body_calls) == 2
    seen_bodies = {call["payload"].get("body") for call in api.body_calls}
    assert seen_bodies == {"first body", "second"}
    for call in api.body_calls:
        assert call["ledger_id"] == "alpha"
        assert call["metadata_arg"] is None
        metadata = call["payload"].get("metadata")
        assert isinstance(metadata, dict)
        assert metadata["workflow"] == "unit"
        assert metadata["superseded_by"] == 29
        assert metadata["source"] == "enrichment"
        assert call["prime"] in {23, 25}

    assert api.enrich_calls
    enrich_entity, payload, ledger_id = api.enrich_calls[0]
    assert enrich_entity == "demo"
    assert ledger_id == "alpha"
    assert payload["ref_prime"] == 29
    assert payload["deltas"] == [{"prime": 2, "delta": 1}]
    assert payload["bodies"] == [
        {"prime": 23, "superseded_by": 29},
        {"prime": 25, "superseded_by": 29},
    ]

    assert result["deltas"] == [{"prime": 2, "delta": 1}]
    assert result["bodies"][0]["metadata"]["superseded_by"] == 29
    assert result["enrichment_supported"] is True
    assert result["request"]
    assert result["response"]


def test_enrichment_helper_blocks_flow_violation():
    api = DummyApiService()
    primes = DummyPrimeService()
    helper = EnrichmentHelper(api, primes)

    schema = {2: {"tier": "S"}, 11: {"tier": "A"}, 37: {"tier": "C"}}

    result = helper.submit(
        "demo",
        ref_prime=2,
        deltas=[{"prime": 11, "delta": 1}],
        body_chunks=["text"],
        metadata={},
        schema=schema,
    )

    assert result["flow_errors"]
    assert api.enrich_calls == []
    assert api.body_calls == []


def test_enrichment_helper_handles_missing_enrich_endpoint():
    api = DummyApiService(supported=False)
    primes = DummyPrimeService()
    helper = EnrichmentHelper(api, primes)

    result = helper.submit(
        "demo",
        ref_prime=29,
        deltas=[{"prime": 2, "delta": 1}],
        body_chunks=["body"],
    )

    assert not api.enrich_calls
    assert len(api.body_calls) == 1
    assert result["enrichment_supported"] is False
    assert result["request"] is None
    assert result["response"] == {}
    assert result["bodies"]
