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

    def fake_post(url, *, params=None, headers=None, timeout=None, json=None):
        captured.update({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
            "json": json,
        })
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

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
    assert captured["json"] is None


def test_put_ledger_body_includes_entity_prime_and_body(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "ok"}

    def fake_put(url, *, json=None, params=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "json": json,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "put", fake_put)

    client = DualSubstrateClient("https://api.example", "secret")
    client.put_ledger_body(
        "demo",
        41,
        "Anchored body text",
        ledger_id="alpha",
        metadata={"kind": "memory"},
    )

    assert captured["url"].endswith("/ledger/body")
    assert captured["params"] == {"entity": "demo", "prime": 41}
    payload = captured["json"]
    assert payload["entity"] == "demo"
    assert payload["prime"] == 41
    assert payload["body"] == "Anchored body text"
    assert payload["metadata"] == {"kind": "memory"}
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
