from typing import Any, Mapping

from services.ledger_service import persist_structured_views
from services.prime_service import PrimeService


class RecordingApi:
    def __init__(self) -> None:
        self.body_calls: list[tuple[str, int, str, str | None, Mapping[str, Any] | None]] = []
        self.s1_calls: list[tuple[str, Mapping[str, Any], str | None]] = []

    def put_ledger_body(
        self,
        entity: str,
        prime: int,
        body_text: str,
        *,
        ledger_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.body_calls.append((entity, prime, body_text, ledger_id, dict(metadata or {})))

    def put_ledger_s1(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> Mapping[str, Any]:
        self.s1_calls.append((entity, dict(payload), ledger_id))
        return {}


def test_backfill_persists_body_primes_without_data_loss() -> None:
    api = RecordingApi()
    service = PrimeService(api_service=api, fallback_prime=23)

    legacy_slot = {
        "prime": 2,
        "title": "Sprint review",
        "tags": ["review", "sprint"],
        "value": 3,
        "body": ["Discussed progress and blockers."],
        "metadata": {"owner": "pm"},
        "score": 0.82,
        "timestamp": 1_700_000_000_000,
    }

    body_plan = [
        {
            "key": "2:0",
            "body": legacy_slot["body"][0],
            "metadata": {"source_prime": 2, "index": 0, "backfill": True},
        }
    ]

    minted = service.persist_bodies("demo", body_plan)

    assert len(minted) == 1
    minted_prime = minted[0]["prime"]
    assert api.body_calls[0][1] == minted_prime
    assert api.body_calls[0][2] == legacy_slot["body"][0]
    assert api.body_calls[0][-1]["source_prime"] == 2

    structured = {
        "s1": [
            {
                "prime": legacy_slot["prime"],
                "value": legacy_slot["value"],
                "title": legacy_slot["title"],
                "tags": legacy_slot["tags"],
                "body_prime": minted_prime,
                "metadata": {**legacy_slot["metadata"], "backfill": True, "source_prime": 2},
                "score": legacy_slot["score"],
                "timestamp": legacy_slot["timestamp"],
            }
        ],
        "s2": [],
        "bodies": minted,
    }

    result = persist_structured_views(api, "demo", structured)

    assert api.s1_calls, "Expected S1 persistence during backfill"
    persisted_slots = api.s1_calls[0][1]["slots"]
    assert persisted_slots[0]["body_prime"] == minted_prime
    assert persisted_slots[0]["metadata"]["backfill"] is True
    assert persisted_slots[0]["metadata"]["owner"] == "pm"
    assert persisted_slots[0]["value"] == 3
    assert persisted_slots[0]["title"] == "Sprint review"
    assert result["bodies"][0]["metadata"]["source_prime"] == 2
