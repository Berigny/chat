"""Unit tests covering compatibility helpers in :mod:`services.api`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import requests

from services.api import ApiService


class _BaseClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []


class _KeywordOnlyClient(_BaseClient):
    def fetch_inference_state(
        self,
        *,
        ledger_id: Optional[str] = None,
        include_history: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "ledger_id": ledger_id,
                "include_history": include_history,
                "limit": limit,
            }
        )
        return {"ledger_id": ledger_id, "include_history": include_history, "limit": limit}


class _LegacyEntityClient(_BaseClient):
    def fetch_inference_state(self, entity: str) -> Dict[str, Any]:
        self.calls.append({"entity": entity})
        return {"entity": entity}


class _HybridClient(_BaseClient):
    def fetch_inference_state(self, entity: str, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        self.calls.append({"entity": entity, "ledger_id": ledger_id})
        return {"entity": entity, "ledger_id": ledger_id}


def _service_with_client(client) -> ApiService:
    service = object.__new__(ApiService)
    service._client = client  # type: ignore[attr-defined]
    service._enrich_supported = None  # type: ignore[attr-defined]
    service._traverse_supported = None  # type: ignore[attr-defined]
    return service


def test_fetch_inference_state_supports_keyword_only_arguments():
    client = _KeywordOnlyClient()
    service = _service_with_client(client)

    payload = service.fetch_inference_state(
        "demo",
        ledger_id="alpha",
        include_history=True,
        limit=10,
    )

    assert payload == {"ledger_id": "alpha", "include_history": True, "limit": 10}
    assert client.calls == [
        {"ledger_id": "alpha", "include_history": True, "limit": 10}
    ]


def test_fetch_inference_state_falls_back_to_positional_entity():
    client = _LegacyEntityClient()
    service = _service_with_client(client)

    payload = service.fetch_inference_state("demo", ledger_id="unused")

    assert payload == {"entity": "demo"}
    assert client.calls == [{"entity": "demo"}]


def test_fetch_inference_state_retains_ledger_id_when_supported():
    client = _HybridClient()
    service = _service_with_client(client)

    payload = service.fetch_inference_state("demo", ledger_id="beta")

    assert payload == {"entity": "demo", "ledger_id": "beta"}
    assert client.calls == [{"entity": "demo", "ledger_id": "beta"}]


def test_fetch_inference_state_raises_original_typeerror_when_unhandled():
    class _ErrorClient(_BaseClient):
        def fetch_inference_state(self, *, ledger_id: Optional[str] = None):
            raise TypeError("unexpected")

    client = _ErrorClient()
    service = _service_with_client(client)

    with pytest.raises(TypeError, match="unexpected"):
        service.fetch_inference_state("demo", ledger_id="alpha")


class _TraverseClient(_BaseClient):
    def traverse(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        origin: Optional[int] = None,
        limit: Optional[int] = None,
        depth: Optional[int] = None,
        direction: Optional[str] = None,
        include_metadata: Optional[bool] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "entity": entity,
                "ledger_id": ledger_id,
                "origin": origin,
                "limit": limit,
                "depth": depth,
                "direction": direction,
                "include_metadata": include_metadata,
                "payload": payload,
            }
        )
        return {"paths": [{"nodes": [{"prime": 2}]}]}


def test_traverse_forwards_parameters_and_marks_capability():
    client = _TraverseClient()
    service = _service_with_client(client)

    payload = service.traverse(
        "demo",
        ledger_id="alpha",
        origin=7,
        limit=5,
        depth=3,
        direction="forward",
        include_metadata=True,
    )

    assert payload == {"paths": [{"nodes": [{"prime": 2}]}]}
    assert client.calls == [
        {
            "entity": "demo",
            "ledger_id": "alpha",
            "origin": 7,
            "limit": 5,
            "depth": 3,
            "direction": "forward",
            "include_metadata": True,
            "payload": None,
        }
    ]
    assert service._traverse_supported is True


class _ListTraverseClient(_BaseClient):
    def traverse(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
        return [{"nodes": [{"prime": 13}]}]


def test_traverse_converts_list_payloads_to_mapping():
    client = _ListTraverseClient()
    service = _service_with_client(client)

    payload = service.traverse("demo")

    assert payload == {"paths": [{"nodes": [{"prime": 13}]}]}
    assert service._traverse_supported is True


class _MissingTraverseClient(_BaseClient):
    def traverse(self, *_: Any, **__: Any) -> Dict[str, Any]:
        error = requests.HTTPError("missing")
        error.response = type("_Resp", (), {"status_code": 404})()
        raise error


def test_traverse_marks_capability_false_on_404():
    client = _MissingTraverseClient()
    service = _service_with_client(client)

    with pytest.raises(requests.HTTPError):
        service.traverse("demo")

    assert service._traverse_supported is False

