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
