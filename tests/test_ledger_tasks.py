"""Tests for ledger task helpers that do not touch external services."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest
import requests

from services.ledger_tasks import fetch_metrics_snapshot


class DummyApiService:
    """Minimal API service stub for exercising ``fetch_metrics_snapshot``."""

    def __init__(
        self,
        metrics_payload: Any,
        *,
        memories: Optional[List[Dict[str, Any]]] = None,
        inference_errors: Optional[str] = None,
        inference_missing: bool = False,
    ) -> None:
        self._metrics_payload = metrics_payload
        self._memories = memories if memories is not None else []
        self._inference_errors = inference_errors
        self._inference_missing = inference_missing

    def fetch_metrics(self, *, ledger_id: Optional[str] = None):
        return self._metrics_payload

    def fetch_memories(self, entity: str, *, ledger_id: Optional[str] = None, limit: int | None = None):
        return list(self._memories)

    def _maybe_raise(self):
        if self._inference_missing:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError(response=response)
        if self._inference_errors:
            raise requests.RequestException(self._inference_errors)

    def fetch_inference_state(self, entity: str, *, ledger_id: Optional[str] = None):
        self._maybe_raise()
        return {"status": "idle", "ts": int(time.time())}

    def fetch_inference_traverse(self, entity: str, *, ledger_id: Optional[str] = None):
        self._maybe_raise()
        return [{"node": "prime", "weight": 0.7}]

    def fetch_inference_memories(self, entity: str, *, ledger_id: Optional[str] = None):
        self._maybe_raise()
        return [{"text": "hello", "score": 0.92}]

    def fetch_inference_retrieve(self, entity: str, *, ledger_id: Optional[str] = None):
        self._maybe_raise()
        return {"hits": 3}


def test_fetch_metrics_snapshot_parses_prometheus_payload():
    metrics_text = """
    # HELP dualsubstrate_tokens_deduped_total Tokens saved across anchors
    dualsubstrate_tokens_deduped_total 2048
    dualsubstrate_ledger_integrity_ratio 0.98
    dualsubstrate_durability_hours 6.5
    """.strip()

    api = DummyApiService(metrics_text)
    floors = {"tokens_deduped": 1.0, "ledger_integrity": 0.75, "durability_h": 0.0}

    snapshot = fetch_metrics_snapshot(api, "demo", ledger_id="alpha", metric_floors=floors)

    assert snapshot["tokens_saved"] == 2048
    assert snapshot["ledger_integrity"] == pytest.approx(0.98)
    assert snapshot["durability_hours"] == pytest.approx(6.5)
    assert snapshot["metrics_source"] == "prometheus"
    assert snapshot["inference_state"]["status"] == "idle"
    assert snapshot["inference_supported"] is True
    assert snapshot["inference_errors"] == []


def test_fetch_metrics_snapshot_handles_missing_inference_endpoints():
    api = DummyApiService({}, inference_missing=True)
    floors = {"tokens_deduped": 8.0, "ledger_integrity": 0.5, "durability_h": 3.0}

    snapshot = fetch_metrics_snapshot(api, "demo", ledger_id="alpha", metric_floors=floors)

    assert snapshot["tokens_saved"] == floors["tokens_deduped"]
    assert snapshot["ledger_integrity"] == floors["ledger_integrity"]
    assert snapshot["durability_hours"] == floors["durability_h"]
    assert snapshot["inference_state"] is None
    assert snapshot["inference_supported"] is False
    assert snapshot["inference_errors"] == []


def test_fetch_metrics_snapshot_records_inference_errors():
    api = DummyApiService({}, inference_errors="boom")
    floors = {"tokens_deduped": 5.0, "ledger_integrity": 0.9, "durability_h": 2.0}

    snapshot = fetch_metrics_snapshot(api, "demo", ledger_id="alpha", metric_floors=floors)

    assert snapshot["inference_supported"] is True
    assert any("boom" in err for err in snapshot["inference_errors"])
