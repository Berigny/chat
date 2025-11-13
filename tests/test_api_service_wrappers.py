from __future__ import annotations

from typing import Any, Mapping

from services.api import ApiService
from services.api_service import EnrichmentHelper


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []

    def put_ledger_s1(self, payload: Mapping[str, Any], *, ledger_id=None):
        self.calls.append(("s1", payload, {"ledger_id": ledger_id}))
        return {"status": "ok"}

    def put_ledger_body(self, payload: Mapping[str, Any], *, ledger_id=None, params=None):
        self.calls.append(("body", payload, {"ledger_id": ledger_id, "params": params}))
        return {"status": "ok"}

    def enrich(self, payload: Mapping[str, Any], *, ledger_id=None):
        self.calls.append(("enrich", payload, {"ledger_id": ledger_id}))
        return {"result": True}

    def put_ledger_s2(self, payload: Mapping[str, Any], *, ledger_id=None):
        self.calls.append(("s2", payload, {"ledger_id": ledger_id}))
        return {"status": "ok"}


def test_api_service_passthrough_methods(monkeypatch):
    client = RecordingClient()
    service = ApiService.__new__(ApiService)
    service._client = client  # type: ignore[attr-defined]

    payload = {"example": True}
    assert service.put_ledger_s1(payload, ledger_id="one") == {"status": "ok"}
    assert service.put_ledger_body(payload, ledger_id="two", params={"prime": 23}) == {"status": "ok"}
    assert service.put_ledger_s2(payload, ledger_id="three") == {"status": "ok"}
    assert service.enrich(payload, ledger_id="four") == {"result": True}

    call_names = [name for name, *_ in client.calls]
    assert call_names == ["s1", "body", "s2", "enrich"]
    assert client.calls[0][1] is payload
    assert client.calls[1][2]["params"] == {"prime": 23}


def test_enrichment_helper_returns_request() -> None:
    class DummyApi(ApiService):
        def __init__(self) -> None:
            self._calls: list[tuple[str, Mapping[str, Any], str | None]] = []

        def enrich(self, payload: Mapping[str, Any], *, ledger_id: str | None = None):
            self._calls.append(("enrich", payload, ledger_id))
            return {"echo": payload}

    api = DummyApi()
    helper = EnrichmentHelper(api)
    payload = {"ref_prime": 29, "body": "memo"}

    result = helper.submit("demo", payload, ledger_id="alpha")

    assert result["entity"] == "demo"
    assert result["request"] == payload
    assert result["response"] == {"echo": payload}
    assert api._calls == [("enrich", payload, "alpha")]
