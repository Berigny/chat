from typing import Any, Mapping

from services.prime_service import PrimeService


class DummyApiService:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, Mapping[str, Any], str | None]] = []
        self.body_calls: list[tuple[str, int, str, str | None]] = []

    def ingest(self, entity: str, payload: Mapping[str, Any], *, ledger_id: str | None = None):
        self.ingest_calls.append((entity, dict(payload), ledger_id))
        return {"ok": True}

    def put_ledger_body(self, entity: str, prime: int, body: str, *, ledger_id=None, metadata=None):
        self.body_calls.append((entity, prime, body, ledger_id))

    def fetch_ledger(self, entity: str, *, ledger_id: str | None = None) -> Mapping[str, Any]:
        return {}


SCHEMA = {
    2: {"tier": "S"},
    3: {"tier": "S"},
    11: {"tier": "A"},
    37: {"tier": "C"},
}


def test_ingest_blocks_flow_violation() -> None:
    api = DummyApiService()
    service = PrimeService(api_service=api, fallback_prime=23)

    result = service.ingest(
        "demo",
        "hello",
        SCHEMA,
        factors_override=[{"prime": 2, "delta": 1}, {"prime": 11, "delta": 1}],
    )

    assert api.ingest_calls == []
    assert api.body_calls == []
    assert result["flow_errors"]


def test_ingest_allows_mediated_flow() -> None:
    api = DummyApiService()
    service = PrimeService(api_service=api, fallback_prime=23)

    result = service.ingest(
        "demo",
        "hello",
        SCHEMA,
        factors_override=[
            {"prime": 2, "delta": 1},
            {"prime": 37, "delta": 1},
            {"prime": 11, "delta": 1},
        ],
    )

    assert not result.get("flow_errors")
    assert api.ingest_calls
    assert api.body_calls
