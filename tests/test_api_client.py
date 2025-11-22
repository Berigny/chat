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
    client.search(
        "demo",
        "meeting recap",
        ledger_id="alpha",
        mode="slots",
        limit=4,
        semantic_weight=0.45,
        delta=2,
    )

    assert captured["url"].endswith("/search")
    assert captured["params"] == {
        "entity": "demo",
        "q": "meeting recap",
        "mode": "slots",
        "limit": 4,
        "fuzzy": "true",
        "semantic_weight": 0.45,
        "delta": 2,
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


def test_write_and_read_ledger_entry(monkeypatch):
    captured_write: dict[str, object] = {}
    captured_read: dict[str, object] = {}

    class DummyResponse:
        def __init__(self, payload: dict[str, object]):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url, *, json=None, params=None, headers=None, timeout=None):
        captured_write.update(
            {
                "url": url,
                "json": json,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse({"entry_id": "abc123"})

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured_read.update(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse({"state": {"metadata": {"text": "Anchored body text"}}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    client = DualSubstrateClient("https://api.example", "secret")
    write_response = client.write_ledger_entry(
        key_namespace="demo",
        key_identifier="demo-body",
        text="Anchored body text",
        phase="body",
        ledger_id="alpha",
        metadata={"kind": "memory"},
        coordinates={"prime_41": 1.0},
        created_at=1234,
        notes="important",
    )
    read_response = client.read_ledger_entry("abc123", ledger_id="alpha")

    assert write_response == {"entry_id": "abc123"}
    assert read_response["state"]["metadata"]["text"] == "Anchored body text"
    assert captured_write["url"].endswith("/ledger/write")
    payload = captured_write["json"]
    assert payload["key"] == {"namespace": "demo", "identifier": "demo-body"}
    assert payload["created_at"] == 1234
    assert payload["state"]["phase"] == "body"
    assert payload["state"]["coordinates"] == {"prime_41": 1.0}
    assert payload["state"]["metadata"] == {"text": "Anchored body text", "kind": "memory"}
    assert payload["state"]["notes"] == ["important"]
    assert captured_write["headers"]["X-Ledger-ID"] == "alpha"
    assert captured_read["url"].endswith("/ledger/read/abc123")
    assert captured_read["headers"]["X-Ledger-ID"] == "alpha"


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


def test_update_metrics_uses_query_entity(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, object]:  # pragma: no cover - trivial
            return {"status": "ok"}

    def fake_patch(url, *, json=None, headers=None, timeout=None, params=None):
        captured.update(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
                "params": params,
            }
        )
        return DummyResponse()

    monkeypatch.setattr(requests, "patch", fake_patch)

    client = DualSubstrateClient("https://api.example", "secret", timeout=5)
    client.update_metrics("Demo_dev", {"dE": -1.5, "K": 0.0}, ledger_id="alpha")

    assert captured["url"].endswith("/ledger/metrics")
    assert captured["params"] == {"entity": "Demo_dev"}
    assert captured["json"] == {"dE": -1.5, "K": 0.0}
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

