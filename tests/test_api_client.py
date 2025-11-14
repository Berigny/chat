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


def test_traverse_uses_query_parameters_only(monkeypatch):
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

    client = DualSubstrateClient("https://api.example", "secret", timeout=11)
    client.traverse(start=17, depth=5, ignored="value")

    assert captured["url"].endswith("/traverse")
    assert captured["params"] == {"start": 17, "depth": 5}
    assert captured["headers"] == {"x-api-key": "secret"}
    assert captured["timeout"] == 11
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


def test_anchor_posts_json_payload(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, object]:  # pragma: no cover - trivial
            return {"status": "ok"}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = DualSubstrateClient("https://api.example", "secret", timeout=6)
    client.anchor(
        "demo",
        [
            {"prime": 2, "weight": 0.7},
            {"prime": 3, "weight": 0.2},
        ],
        ledger_id="alpha",
        text="Hello",
        modifiers=[11],
    )

    assert captured["url"].endswith("/anchor")
    assert captured["json"] == {
        "entity": "demo",
        "factors": [
            {"prime": 2, "weight": 0.7},
            {"prime": 3, "weight": 0.2},
        ],
        "text": "Hello",
        "modifiers": [11],
    }
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["timeout"] == 5


def test_enrich_posts_payload(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, object]:  # pragma: no cover - trivial
            return {"status": "ok"}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    payload = {"entity": "demo", "plan": {"prime": 2}}
    client = DualSubstrateClient("https://api.example", "secret", timeout=12)
    client.enrich(payload, ledger_id="alpha")

    assert captured["url"].endswith("/enrich")
    assert captured["json"] == payload
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["timeout"] == 12


def test_retrieve_gets_with_entity_query(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, object]:  # pragma: no cover - trivial
            return {"entries": []}

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    client = DualSubstrateClient("https://api.example", "secret", timeout=9)
    client.retrieve("demo", ledger_id="alpha")

    assert captured["url"].endswith("/retrieve")
    assert captured["params"] == {"entity": "demo"}
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["timeout"] == 9


def test_rotate_posts_axis_and_angle(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, object]:  # pragma: no cover - trivial
            return {"status": "ok"}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = DualSubstrateClient("https://api.example", "secret", timeout=13)
    client.rotate("demo", ledger_id="alpha", axis=(1.0, 0.0, 0.0), angle=0.5)

    assert captured["url"].endswith("/rotate")
    assert captured["json"] == {
        "entity": "demo",
        "axis": [1.0, 0.0, 0.0],
        "angle": 0.5,
    }
    assert captured["headers"]["X-Ledger-ID"] == "alpha"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["timeout"] == 10

