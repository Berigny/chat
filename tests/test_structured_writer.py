from __future__ import annotations

from typing import Any, Mapping

import pytest

from services.structured_writer import write_s2_slots


class _RecordingApiService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def put_ledger_s2(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> Mapping[str, Any]:
        record = {
            "entity": entity,
            "payload": dict(payload),
            "ledger_id": ledger_id,
        }
        self.calls.append(record)
        return payload


def test_write_s2_slots_includes_structured_views_payload() -> None:
    api = _RecordingApiService()
    slots = [
        {
            "prime": 17,
            "body_prime": 101,
            "summary": "Launch recap",
            "metadata": {"tier": "S2"},
            "score": 0.85,
            "timestamp": 1700000000,
            "tags": ["roadmap", "launch"],
        }
    ]

    sanitized = write_s2_slots(api, "Demo_dev", slots, ledger_id="alpha")

    assert sanitized == [
        {
            "prime": 17,
            "body_prime": 101,
            "summary": "Launch recap",
            "metadata": {"tier": "S2"},
            "score": pytest.approx(0.85),
            "timestamp": 1700000000,
            "tags": ["roadmap", "launch"],
        }
    ]
    assert api.calls and api.calls[0]["entity"] == "Demo_dev"
    payload = api.calls[0]["payload"]
    assert payload["slots"] == sanitized
    assert "views" in payload
    view_entry = payload["views"][0]
    assert view_entry["entity_prime"] == 17
    assert view_entry["body_prime"] == 101
    assert view_entry["summary"] == "Launch recap"
    assert view_entry["metadata"] == {"tier": "S2"}
    assert view_entry["tags"] == ["roadmap", "launch"]
    assert view_entry["view_id"] == "17"

