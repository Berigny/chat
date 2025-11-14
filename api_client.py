"""HTTP client helpers for the dual-substrate API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import requests


@dataclass
class DualSubstrateClient:
    """Lightweight helper for calling the commercial demo API."""

    base_url: str
    api_key: str | None = None
    timeout: int = 10

    def _headers(self, *, ledger_id: str | None = None, include_ledger: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if include_ledger and ledger_id:
            headers["X-Ledger-ID"] = ledger_id
        return headers

    def list_ledgers(self) -> list[dict[str, str]]:
        """Return ledger metadata records from the admin endpoint."""

        resp = requests.get(
            f"{self.base_url}/admin/ledgers",
            headers=self._headers(include_ledger=False),
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
        return _coerce_ledger_records(payload)

    def create_ledger(self, ledger_id: str) -> None:
        """Create or switch to a ledger via the admin surface."""

        resp = requests.post(
            f"{self.base_url}/admin/ledgers",
            json={"ledger_id": ledger_id},
            headers=self._headers(include_ledger=False),
            timeout=5,
        )
        resp.raise_for_status()

    def fetch_prime_schema(self, entity: str, *, ledger_id: str | None = None) -> dict[int, dict[str, Any]]:
        """Fetch the remote prime schema for an entity."""

        resp = requests.get(
            f"{self.base_url}/schema",
            params={"entity": entity},
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        raw = resp.json()
        schema: dict[int, dict[str, Any]] = {}
        for entry in raw.get("primes", []):
            prime = entry.get("prime")
            if isinstance(prime, int):
                schema[prime] = {
                    "name": entry.get("name") or entry.get("symbol") or f"Prime {prime}",
                    "tier": entry.get("tier") or "",
                    "mnemonic": entry.get("mnemonic") or "",
                    "description": entry.get("description") or "",
                }
        return schema

    def fetch_memories(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        limit: int = 3,
        since: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories for an entity."""

        params: dict[str, Any] = {"entity": entity, "limit": limit}
        if since is not None:
            params["since"] = since
        resp = requests.get(
            f"{self.base_url}/memories",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def assemble_context(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        k: int | None = None,
        quote_safe: bool | None = None,
        since: int | None = None,
    ) -> dict[str, Any]:
        """Call the `/assemble` endpoint for pre-formatted prompt material."""

        params: dict[str, Any] = {"entity": entity}
        if k is not None:
            params["k"] = int(k)
        if quote_safe is not None:
            params["quote_safe"] = "true" if quote_safe else "false"
        if since is not None:
            params["since"] = since

        resp = requests.get(
            f"{self.base_url}/assemble",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def search(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: str | None = None,
        mode: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Call the `/search` endpoint for recall and slot lookups."""

        params: dict[str, Any] = {"entity": entity, "q": query}
        if mode:
            params["mode"] = mode
        if limit is not None:
            params["limit"] = int(limit)

        resp = requests.get(
            f"{self.base_url}/search",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def traverse(
        self,
        *,
        start: int,
        depth: int = 3,
        **_: Any,
    ) -> dict[str, Any] | list[Any]:
        """Call the `/traverse` endpoint using query parameters only."""

        params: dict[str, Any] = {"start": start, "depth": depth}

        resp = requests.post(
            f"{self.base_url}/traverse",
            params=params,
            headers=self._headers(include_ledger=False),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            return {}
        if isinstance(data, Mapping):
            return dict(data)
        if isinstance(data, list):
            return list(data)
        return {}

    def fetch_inference_state(
        self,
        entity: str | None = None,
        *,
        ledger_id: str | None = None,
        include_history: bool | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the active inference state for the entity or ledger."""

        params: dict[str, Any] = {}
        if entity is not None:
            params["entity"] = entity
        if include_history is not None:
            params["include_history"] = "true" if include_history else "false"
        if limit is not None:
            params["limit"] = int(limit)

        resp = requests.get(
            f"{self.base_url}/inference/state",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def latest_memory_text(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        since: int | None = None,
    ) -> str | None:
        """Return the most recent memory text if available."""

        entries = self.fetch_memories(
            entity,
            ledger_id=ledger_id,
            limit=1,
            since=since,
        )
        if not entries:
            return None
        latest = entries[0]
        text = latest.get("text") if isinstance(latest, dict) else None
        return text if isinstance(text, str) else None

    def fetch_ledger(self, entity: str, *, ledger_id: str | None = None) -> dict[str, Any]:
        """Load the ledger factors for an entity."""

        resp = requests.get(
            f"{self.base_url}/ledger",
            params={"entity": entity},
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def query_ledger(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: str | None = None,
        limit: int | None = None,
        topic: str | None = None,
        required: Iterable[int] | None = None,
        preferred: Iterable[int] | None = None,
        modifiers: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Query the remote ledger for structured recall."""

        params: dict[str, Any] = {"entity": entity, "query": query}
        if limit is not None:
            params["limit"] = limit
        if topic:
            params["topic"] = topic
        if required:
            params["required"] = list(required)
        if preferred:
            params["preferred"] = list(preferred)
        if modifiers:
            params["modifiers"] = list(modifiers)
        resp = requests.get(
            f"{self.base_url}/query",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def anchor(
        self,
        entity: str,
        factors: Iterable[dict[str, Any]],
        *,
        ledger_id: str | None = None,
        text: str | None = None,
        modifiers: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Anchor factors (and optional text) into the ledger."""

        payload: dict[str, Any] = {"entity": entity, "factors": list(factors)}
        if text:
            payload["text"] = text
        if modifiers:
            payload["modifiers"] = list(modifiers)
        resp = requests.post(
            f"{self.base_url}/anchor",
            json=payload,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def put_ledger_s1(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Write S1 facets to the ledger without mutating the payload."""

        body: dict[str, Any] = {"entity": entity}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if key == "entity":
                    continue
                body[key] = value

        resp = requests.put(
            f"{self.base_url}/ledger/s1",
            json=body,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def put_ledger_body(
        self,
        entity: str,
        prime: int,
        body_text: str | Mapping[str, Any],
        *,
        ledger_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store long-form body text using the refreshed `/ledger/body` contract."""

        payload: dict[str, Any] = {"entity": entity, "prime": int(prime)}

        if isinstance(body_text, Mapping):
            for key, value in body_text.items():
                if key in {"entity", "prime"}:
                    continue
                payload[key] = value
        else:
            payload["body"] = body_text

        if metadata:
            existing = payload.get("metadata")
            if isinstance(existing, Mapping):
                merged = {key: value for key, value in existing.items()}
                merged.update(metadata)
                payload["metadata"] = merged
            else:
                payload["metadata"] = dict(metadata)

        resp = requests.put(
            f"{self.base_url}/ledger/body",
            json=payload,
            params={"entity": entity, "prime": prime},
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def enrich(
        self,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Call the enrichment endpoint while forwarding the supplied payload."""

        resp = requests.post(
            f"{self.base_url}/enrich",
            json=payload,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def put_ledger_s2(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Store enrichment output while leaving the payload untouched."""

        body: dict[str, Any] = {"entity": entity}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if key == "entity":
                    continue
                body[key] = value

        resp = requests.put(
            f"{self.base_url}/ledger/s2",
            json=body,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def update_lawfulness(
        self,
        entity: str,
        payload: dict[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Adjust lawfulness metrics via the engine guardrail endpoint."""

        body = {"entity": entity, **(payload or {})}
        resp = requests.put(
            f"{self.base_url}/ledger/lawfulness",
            json=body,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def update_metrics(
        self,
        entity: str,
        payload: dict[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Update ℝ metrics; server enforces bounds."""

        body = {"entity": entity, **(payload or {})}
        resp = requests.patch(
            f"{self.base_url}/ledger/metrics",
            json=body,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def retrieve(self, entity: str, *, ledger_id: str | None = None) -> dict[str, Any]:
        """Retrieve ledger memories for the entity."""

        resp = requests.get(
            f"{self.base_url}/retrieve",
            params={"entity": entity},
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def fetch_metrics(self, *, ledger_id: str | None = None) -> dict[str, Any] | str:
        """Fetch system-wide metrics for the current ledger."""

        resp = requests.get(
            f"{self.base_url}/metrics",
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            data = resp.json()
            return data if isinstance(data, dict) else {}
        return resp.text

    def fetch_inference_traverse(self, *, ledger_id: str | None = None) -> list[dict[str, Any]]:
        """Return recent traversal operations captured during inference."""

        resp = requests.get(
            f"{self.base_url}/inference/traverse",
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def fetch_inference_memories(self, *, ledger_id: str | None = None) -> list[dict[str, Any]]:
        """Return the inference-layer memory transcript if exposed."""

        resp = requests.get(
            f"{self.base_url}/inference/memories",
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def fetch_inference_retrieve(self, *, ledger_id: str | None = None) -> dict[str, Any]:
        """Return the most recent inference-layer retrieval payload."""

        resp = requests.get(
            f"{self.base_url}/inference/retrieve",
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def rotate(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        axis: tuple[float, float, float] | None = None,
        angle: float | None = None,
    ) -> dict[str, Any]:
        """Invoke the rotation endpoint used by the Möbius transform button."""

        payload: dict[str, Any] = {"entity": entity}
        if axis is not None:
            payload["axis"] = list(axis)
        if angle is not None:
            payload["angle"] = angle
        resp = requests.post(
            f"{self.base_url}/rotate",
            json=payload,
            headers=self._headers(ledger_id=ledger_id),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}


def _coerce_ledger_records(payload: Any) -> list[dict[str, str]]:
    """Normalise ledger payloads to a list of dictionaries."""

    if isinstance(payload, dict):
        ledgers = payload.get("ledgers")
        if isinstance(ledgers, list):
            payload = ledgers
        else:
            payload = [
                {"ledger_id": str(key), "path": str(value)}
                for key, value in payload.items()
                if isinstance(key, str)
            ]
    records: list[dict[str, str]] = []
    if not isinstance(payload, list):
        return records
    for item in payload:
        if isinstance(item, str):
            records.append({"ledger_id": item})
            continue
        if not isinstance(item, dict):
            continue
        ledger_id = item.get("ledger_id") or item.get("id") or item.get("name")
        if not ledger_id:
            continue
        records.append(
            {
                "ledger_id": str(ledger_id),
                "path": item.get("path") or item.get("base_path"),
            }
        )
    return records


__all__ = ["DualSubstrateClient"]
