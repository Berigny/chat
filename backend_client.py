"""Thin client for the backend/main FastAPI surface."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import requests

BASE_URL = "https://dualsubstrate-commercial.fly.dev"


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def build_action_request(
    *,
    actor: str,
    action: str,
    key_namespace: str | None,
    key_identifier: str | None,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a payload compatible with ActionRequestSchema."""

    action_request: dict[str, Any] = {
        "actor": actor,
        "action": action,
        "key": (
            {"namespace": key_namespace, "identifier": key_identifier}
            if key_namespace and key_identifier
            else None
        ),
        "parameters": {},
    }
    if parameters:
        sanitized: dict[str, float] = {}
        for key, value in parameters.items():
            try:
                sanitized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        action_request["parameters"] = sanitized
    return action_request


@dataclass
class BackendAPIClient:
    """REST client that targets the backend/main FastAPI deployment."""

    base_url: str = BASE_URL
    api_key: str | None = None
    timeout: int = 10

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    # Internal helpers -----------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_payload: Any | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json_payload,
            headers=_build_headers(self.api_key),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_payload: Any | None = None,
    ) -> Mapping[str, Any]:
        response = self._request(method, path, params=params, json_payload=json_payload)
        body = response.json()
        return body if isinstance(body, Mapping) else {}

    # Public API -----------------------------------------------------------
    def write_ledger_entry(
        self,
        *,
        key_namespace: str,
        key_identifier: str,
        text: str,
        phase: str,
        entity: str,
        metadata: Mapping[str, Any] | None = None,
        coordinates: Mapping[str, float] | None = None,
    ) -> Mapping[str, Any]:
        now = time.time()
        payload = {
            "key": {
                "namespace": key_namespace,
                "identifier": key_identifier,
            },
            "state": {
                "coordinates": dict(coordinates or {"prime_2": 1.0}),
                "phase": phase,
                "metadata": {
                    "source": "chat-demo",
                    "entity": entity,
                    "text": text,
                    "timestamp": now,
                    **(dict(metadata or {})),
                },
            },
        }
        return self._json("post", "/ledger/write", json_payload=payload)

    def read_ledger_entry(self, entry_id: str) -> Mapping[str, Any]:
        return self._json("get", f"/ledger/read/{entry_id}")

    def search_memories(
        self,
        *,
        entity: str | None,
        query: str,
        limit: int = 10,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "limit": max(1, limit),
            "fuzzy": "true" if fuzzy else "false",
            "semantic_weight": semantic_weight,
            "delta": delta,
        }
        if entity:
            params["entity"] = entity
        return self._json("get", "/search", params=params)

    def evaluate_coherence(self, action_request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json("post", "/coherence/evaluate", json_payload=dict(action_request))

    def evaluate_ethics(self, action_request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json("post", "/ethics/evaluate", json_payload=dict(action_request))

    def reindex_ledger(self, entity: str | None = None) -> Mapping[str, Any]:
        params = {"entity": entity} if entity else None
        return self._json("get", "/admin/reindex", params=params)


__all__ = ["BackendAPIClient", "BASE_URL", "build_action_request"]
