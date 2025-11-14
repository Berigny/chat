import requests

from api_client import DualSubstrateClient


def test_search_forwards_query_and_headers(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"slots": []}

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured.update({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    client = DualSubstrateClient("https://api.example", "secret", timeout=7)
    client.search("demo", "meeting recap", ledger_id="alpha", mode="slots", limit=4)

    assert captured["url"].endswith("/search")
    assert captured["params"] == {
        "entity": "demo",
        "q": "meeting recap",
        "mode": "slots",
        "limit": 4,
    }
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["timeout"] == 7


def test_fetch_inference_state_includes_flags(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "idle"}

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured.update({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    client = DualSubstrateClient("https://api.example", "secret", timeout=9)
    client.fetch_inference_state(
        "demo",
        ledger_id="alpha",
        include_history=True,
        limit=5,
    )

    assert captured["url"].endswith("/inference/state")
    assert captured["params"] == {
        "entity": "demo",
        "include_history": "true",
        "limit": 5,
    }
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["timeout"] == 9


def test_traverse_forwards_optional_parameters(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"paths": []}

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured.update({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    client = DualSubstrateClient("https://api.example", "secret")
    client.traverse(
        "demo",
        ledger_id="alpha",
        origin=23,
        limit=4,
        depth=2,
        direction="forward",
        include_metadata=True,
    )

    assert captured["url"].endswith("/traverse")
    assert captured["params"] == {
        "entity": "demo",
        "origin": 23,
        "limit": 4,
        "depth": 2,
        "direction": "forward",
        "include_metadata": "true",
    }
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
