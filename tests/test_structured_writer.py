from __future__ import annotations

import json
from pathlib import Path
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


CONTRACT_PATH = Path(__file__).parent / "contracts" / "s2_payload.json"


def test_write_s2_slots_collapses_to_prime_map() -> None:
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

    expected = json.loads(CONTRACT_PATH.read_text())
    assert sanitized == expected
    assert api.calls and api.calls[0]["entity"] == "Demo_dev"
    payload = api.calls[0]["payload"]
    assert payload == expected
    assert set(payload.keys()).issubset({"11", "13", "17", "19"})
    assert payload["17"]["score"] == pytest.approx(0.85)

