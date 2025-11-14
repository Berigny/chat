from typing import Any, Mapping

import requests

import pytest

from services.prime_service import PrimeService


class RecordingApiService:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, Mapping[str, Any], str | None]] = []
        self.body_calls: list[dict[str, Any]] = []

    def ingest(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> Mapping[str, Any]:
        self.ingest_calls.append((entity, dict(payload), ledger_id))
        return {"ok": True}

    def put_ledger_body(
        self,
        entity: str,
        prime: int,
        body_payload,
        *,
        ledger_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
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

    def fetch_ledger(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
    ) -> Mapping[str, Any]:
        return {}


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
def prime_service() -> tuple[PrimeService, RecordingApiService]:
    api = RecordingApiService()
    service = PrimeService(api_service=api, fallback_prime=23)
    return service, api


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


def test_ingest_mints_body_primes_for_structured_payload(prime_service: tuple[PrimeService, RecordingApiService]) -> None:
    service, api = prime_service

    result = _ingest(service, "Meeting recap with immutable storage")

    assert api.body_calls, "Expected minted body primes"
    first_call = api.body_calls[0]
    minted_prime = first_call["prime"]
    assert minted_prime >= service.body_prime_floor
    assert first_call["metadata_arg"] is None
    payload = first_call["payload"]
    assert payload["body"].startswith("Meeting recap")
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("kind") == "memory"

    structured = result["structured"]
    bodies = structured["bodies"]
    assert bodies and bodies[0]["prime"] == minted_prime

    for slot in structured["s1"]:
        assert slot["body_prime"] == minted_prime
    for slot in structured["s2"]:
        assert slot["body_prime"] == minted_prime


def test_ingest_preserves_s1_s2_reference_integrity(prime_service: tuple[PrimeService, RecordingApiService]) -> None:
    service, api = prime_service

    _ingest(service, "First body copy")
    structured = _ingest(service, "Second body copy")["structured"]

    minted_prime = structured["bodies"][0]["prime"]
    assert minted_prime == api.body_calls[-1]["prime"]

    s1_body_primes = {slot["body_prime"] for slot in structured["s1"]}
    s2_body_primes = {slot["body_prime"] for slot in structured["s2"]}
    assert s1_body_primes == {minted_prime}
    assert s2_body_primes == {minted_prime}


def test_ingest_does_not_overwrite_existing_body_primes(prime_service: tuple[PrimeService, RecordingApiService]) -> None:
    service, api = prime_service

    _ingest(service, "Legacy transcript one")
    _ingest(service, "Legacy transcript two")
    _ingest(service, "Legacy transcript three")

    minted_primes = [call["prime"] for call in api.body_calls]
    assert len(minted_primes) == len(set(minted_primes)), "Body primes should not be reused"
    assert minted_primes == sorted(minted_primes), "Body primes should grow monotonically"


class SeededApiService(RecordingApiService):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_payload: dict[str, Any] = {
            "bodies": [{"prime": 23}, {"prime": 29}],
            "slots": [{"body_prime": 31}],
        }
        self.fetch_calls = 0

    def fetch_ledger(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
    ) -> Mapping[str, Any]:
        self.fetch_calls += 1
        return self.fetch_payload


def test_ingest_skips_primes_already_in_ledger() -> None:
    api = SeededApiService()
    service = PrimeService(api_service=api, fallback_prime=23)

    result = _ingest(service, "Prime collision guard")

    minted = [body["prime"] for body in result["structured"]["bodies"]]
    assert minted == [37]
    assert api.fetch_calls == 1
    assert api.body_calls[0]["prime"] == 37


class ConflictingApiService(SeededApiService):
    def __init__(self) -> None:
        super().__init__()
        self.conflict_triggered = False

    def put_ledger_body(
        self,
        entity: str,
        prime: int,
        body_payload,
        *,
        ledger_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
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
        if not self.conflict_triggered:
            self.conflict_triggered = True
            self.fetch_payload.setdefault("bodies", []).append({"prime": prime})
            response = type(
                "DummyResponse",
                (),
                {"status_code": 422, "text": "duplicate body prime"},
            )()
            raise requests.HTTPError("duplicate body prime", response=response)


def test_ingest_retries_when_body_prime_conflicts() -> None:
    api = ConflictingApiService()
    service = PrimeService(api_service=api, fallback_prime=23)

    result = _ingest(service, "Handle remote conflict")

    minted = [body["prime"] for body in result["structured"]["bodies"]]
    assert minted == [41]
    assert len(api.body_calls) == 2
    assert api.fetch_calls >= 2
