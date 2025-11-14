from __future__ import annotations

from typing import Any

import pytest

from api_client import DualSubstrateClient


class DummyResponse:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {"status": "ok"}

    def raise_for_status(self) -> None:  # pragma: no cover - no-op
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_put_ledger_s1_forwards_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_put(url: str, *, json=None, headers=None, timeout=None, params=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout, params=params)
        return DummyResponse({"ok": True})

    monkeypatch.setattr("requests.put", fake_put)

    client = DualSubstrateClient("https://example.test", "api-key")
    payload = {"entity": "demo", "slots": [{"prime": 2}]}

    response = client.put_ledger_s1(payload, ledger_id="ledger-1")

    assert captured["url"].endswith("/ledger/s1")
    assert captured["json"] is payload
    assert captured["headers"]["X-Ledger-ID"] == "ledger-1"
    assert response == {"ok": True}


def test_put_ledger_body_allows_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_put(url: str, *, json=None, headers=None, timeout=None, params=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout, params=params)
        return DummyResponse({"prime": 23})

    monkeypatch.setattr("requests.put", fake_put)

    client = DualSubstrateClient("https://example.test", "api-key")
    payload = {"entity": "demo", "body": "text", "prime": 23}
    params = {"prime": 23, "dry_run": True}

    response = client.put_ledger_body(payload, ledger_id="ledger-2", params=params)

    assert captured["url"].endswith("/ledger/body")
    assert captured["json"] is payload
    assert captured["params"] is params
    assert response["prime"] == 23


def test_enrich_passes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return DummyResponse({"result": "ok"})

    monkeypatch.setattr("requests.post", fake_post)

    client = DualSubstrateClient("https://example.test", "api-key")
    payload = {"ref_prime": 11, "deltas": [{"prime": 2, "delta": 1}]}

    response = client.enrich(payload, ledger_id="ledger-3")

    assert captured["url"].endswith("/enrich")
    assert captured["json"] is payload
    assert captured["headers"]["X-Ledger-ID"] == "ledger-3"
    assert response == {"result": "ok"}


def test_put_ledger_s2_does_not_mutate_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_put(url: str, *, json=None, headers=None, timeout=None, params=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout, params=params)
        return DummyResponse({"ok": True})

    monkeypatch.setattr("requests.put", fake_put)

    client = DualSubstrateClient("https://example.test", "api-key")
    payload = {"entity": "demo", "slots": [{"prime": 11}]}

    response = client.put_ledger_s2(payload, ledger_id="ledger-4")

    assert captured["url"].endswith("/ledger/s2")
    assert captured["json"] is payload
    assert captured["params"] is None
    assert response == {"ok": True}
