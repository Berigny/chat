from typing import Any, Mapping

from services.memory_service import MemoryService


class DummyApiService:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def assemble_context(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        k: int | None = None,
        quote_safe: bool | None = None,
        since: int | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "entity": entity,
                "ledger_id": ledger_id,
                "k": k,
                "quote_safe": quote_safe,
                "since": since,
            }
        )
        return self.payload


def test_assemble_returns_bodies_in_recency_order() -> None:
    payload = {
        "s1": {
            "bodies": [
                {"prime": 101, "body": ["older"], "timestamp": 1_000},
                {"prime": 102, "body": ["newer"], "timestamp": 5_000},
            ]
        }
    }
    api = DummyApiService(payload)
    service = MemoryService(api_service=api, prime_weights={2: 1.0})

    result = service.assemble_context("demo")

    bodies = result["bodies"]
    assert [entry["prime"] for entry in bodies] == [102, 101]


def test_assemble_honors_since_parameter() -> None:
    api = DummyApiService({})
    service = MemoryService(api_service=api, prime_weights={2: 1.0})

    service.assemble_context("demo", since=123456)

    assert api.calls
    assert api.calls[0]["since"] == 123456


def test_assemble_filters_non_quote_safe_entries() -> None:
    payload = {
        "s1": {
            "bodies": [
                {
                    "prime": 201,
                    "body": ["allowed"],
                    "timestamp": 4_000,
                    "quote_safe": True,
                },
                {
                    "prime": 202,
                    "body": ["blocked"],
                    "timestamp": 5_000,
                    "quote_safe": False,
                },
            ]
        }
    }
    api = DummyApiService(payload)
    service = MemoryService(api_service=api, prime_weights={2: 1.0})

    result = service.assemble_context("demo", quote_safe=True)

    bodies = result["bodies"]
    assert len(bodies) == 1
    assert bodies[0]["prime"] == 201
    assert api.calls[0]["quote_safe"] is True
