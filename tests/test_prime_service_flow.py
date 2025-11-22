from typing import Any, Iterable, Mapping

from services.prime_service import PrimeService


class DummyApiService:
    def __init__(self) -> None:
        self.anchor_calls: list[tuple[str, list[dict[str, Any]], str | None, str]] = []
        self.body_calls: list[dict[str, Any]] = []

    def anchor(
        self,
        entity: str,
        factors: Iterable[Mapping[str, Any]],
        *,
        ledger_id: str | None = None,
        text: str | None = None,
        modifiers: Iterable[int] | None = None,
    ) -> Mapping[str, Any]:
        factors_list = [dict(item) for item in factors]
        self.anchor_calls.append((entity, factors_list, ledger_id, text or ""))
        return {"edges": [], "energy": 1.0, "text": text or ""}

    def write_body_entry(self, entity: str, prime: int, text: str, *, ledger_id=None, metadata=None):
        metadata_payload = {"text": text}
        if isinstance(metadata, Mapping):
            metadata_payload.update(metadata)
        self.body_calls.append(
            {
                "entity": entity,
                "prime": prime,
                "metadata": metadata_payload,
                "ledger_id": ledger_id,
                "metadata_arg": metadata,
            }
        )
        return {"entry_id": f"{prime}:1", "state": {"metadata": metadata_payload}}

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

    assert api.anchor_calls == []
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
    assert api.anchor_calls
    assert api.body_calls
    call = api.body_calls[0]
    assert isinstance(call["metadata_arg"], Mapping)
    metadata = call.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("kind") == "memory"
