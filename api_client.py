"""HTTP client helpers for the dual-substrate API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import requests

from models import CoherenceResponse, LedgerEntry, PolicyDecisionResponse


logger = logging.getLogger(__name__)


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
            f"{self.base_url}/search",
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

    def _request_search(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: str | None = None,
        mode: str | None = None,
        limit: int | None = None,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> requests.Response:
        params: dict[str, Any] = {
            "entity": entity,
            "q": query,
            "limit": int(limit) if limit is not None else 5,
            "fuzzy": str(fuzzy).lower(),
            "semantic_weight": semantic_weight,
            "delta": delta,
        }
        if mode:
            params["mode"] = mode

        return requests.get(
            f"{self.base_url}/search",
            params=params,
            headers=self._headers(ledger_id=ledger_id),
            timeout=self.timeout,
        )

    def search(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: str | None = None,
        mode: str | None = None,
        limit: int | None = None,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> dict[str, Any]:
        """Call the `/search` endpoint for recall and slot lookups."""

        resp = self._request_search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
            fuzzy=fuzzy,
            semantic_weight=semantic_weight,
            delta=delta,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def search_with_response(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: str | None = None,
        mode: str | None = None,
        limit: int | None = None,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> tuple[dict[str, Any], requests.Response]:
        """Return the parsed search payload along with the raw HTTP response."""

        resp = self._request_search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
            fuzzy=fuzzy,
            semantic_weight=semantic_weight,
            delta=delta,
        )
        resp.raise_for_status()
        payload = resp.json()
        parsed = payload if isinstance(payload, dict) else {}
        return parsed, resp

    def traverse(
        self,
        *,
        start: int,
        depth: int = 3,
        **_: Any,
    ) -> dict[str, Any] | list[Any]:
        """Call the `/traverse` endpoint using query parameters only."""

        params: dict[str, Any] = {"start": start, "depth": depth}

        body = {"params": params}
        logger.error("TRAVERSE_REQUEST payload=%s", json.dumps(body, indent=2))
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

    def write_ledger_entry(
        self,
        entity: str,
        prime: int,
        *,
        text: str,
        ledger_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write body text into the ledger and return the write response."""

        combined_metadata: dict[str, Any] = {"text": text}
        if metadata:
            combined_metadata.update(metadata)
            combined_metadata.setdefault("text", text)

        payload: dict[str, Any] = {
            "entity": entity,
            "prime": int(prime),
            "metadata": combined_metadata,
        }

        resp = requests.post(
            f"{self.base_url}/ledger/write",
            json=payload,
            headers=self._headers(ledger_id=ledger_id),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def read_ledger_entry(
        self,
        entry_id: str | int,
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Read a ledger entry using its ``entry_id``."""

        resp = requests.get(
            f"{self.base_url}/ledger/read/{entry_id}",
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

        body: dict[str, Any] = {}
        if isinstance(payload, Mapping):
            payload = {k: v for k, v in payload.items() if k in {"11", "13", "17", "19"}}
            for key, value in payload.items():
                body[key] = value

        resp = requests.put(
            f"{self.base_url}/ledger/s2",
            json=body,
            params={"entity": entity},
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

        params = {"entity": entity}
        body = dict(payload or {})
        resp = requests.patch(
            f"{self.base_url}/ledger/metrics",
            params=params,
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


@dataclass
class DualSubstrateV2Client:
    """HTTP client for the evaluation endpoints."""

    base_url: str
    api_key: str | None = None
    timeout: int = 10

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _full_url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_payload: Any | None = None,
    ) -> requests.Response:
        url = self._full_url(path)
        try:
            response = requests.request(
                method,
                url,
                params=dict(params or {}),
                json=json_payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException:
            logger.exception("Request to %s failed", url)
            raise

        if not response.ok:
            message = _extract_error_message(response)
            logger.error("API error (%s): %s", response.status_code, message)
            response.raise_for_status()
        return response

    def write_ledger(
        self,
        *,
        entity: str,
        text: str | None = None,
        factors: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        identifier: str | None = None,
        phase: str | None = None,
        coordinates: Mapping[str, Any] | None = None,
        notes: Sequence[str] | str | None = None,
    ) -> LedgerEntry:
        metadata = metadata or {}

        key_payload = metadata.get("key") if isinstance(metadata, Mapping) else {}
        namespace_value = namespace or (key_payload.get("namespace") if isinstance(key_payload, Mapping) else None)
        identifier_value = identifier or (key_payload.get("identifier") if isinstance(key_payload, Mapping) else None)

        state_phase = phase or metadata.get("phase") or "structured-ledger"
        state_coordinates = coordinates
        if state_coordinates is None:
            coords_field = metadata.get("coordinates") if isinstance(metadata, Mapping) else None
            state_coordinates = coords_field if isinstance(coords_field, Mapping) else None
        if state_coordinates is None and factors:
            state_coordinates = {"factors": list(factors)}

        extra_notes: list[str] = []
        if isinstance(notes, str):
            extra_notes = [notes]
        elif isinstance(notes, Sequence):
            extra_notes = [str(item) for item in notes]
        elif isinstance(metadata.get("notes"), Sequence) and not isinstance(metadata.get("notes"), str):
            extra_notes = [str(item) for item in metadata.get("notes")]
        elif isinstance(metadata.get("notes"), str):
            extra_notes = [str(metadata.get("notes"))]

        metadata_payload: dict[str, Any] = {}
        if isinstance(metadata, Mapping):
            metadata_payload = {
                key: value
                for key, value in metadata.items()
                if key not in {"key", "phase", "coordinates", "notes"}
            }
        metadata_payload.setdefault("entity", entity)
        if text is not None:
            metadata_payload.setdefault("text", text)
        metadata_payload.setdefault("timestamp", int(time.time()))

        payload: dict[str, Any] = {
            "key": {
                "namespace": namespace_value or "default",
                "identifier": identifier_value or entity,
            },
            "state": {
                "phase": state_phase,
                "coordinates": state_coordinates or {},
                "metadata": metadata_payload,
            },
        }
        if extra_notes:
            payload["state"]["notes"] = extra_notes

        response = self._request("post", "/ledger/write", json_payload=payload)
        body = _safe_json(response)
        return LedgerEntry.from_dict(body)

    def read_ledger(self, entry_id: str) -> LedgerEntry:
        response = self._request("get", f"/ledger/read/{entry_id}")
        body = _safe_json(response)
        return LedgerEntry.from_dict(body)

    def evaluate_coherence(self, payload: Mapping[str, Any]) -> CoherenceResponse:
        response = self._request("post", "/coherence/evaluate", json_payload=dict(payload))
        body = _safe_json(response)
        return CoherenceResponse.from_dict(body)

    def evaluate_ethics(self, payload: Mapping[str, Any]) -> PolicyDecisionResponse:
        response = self._request("post", "/ethics/evaluate", json_payload=dict(payload))
        body = _safe_json(response)
        return PolicyDecisionResponse.from_dict(body)


def _safe_json(response: requests.Response) -> Mapping[str, Any]:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, Mapping):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        return str(detail) if detail is not None else json.dumps(payload)
    return str(payload)


__all__ = ["DualSubstrateClient", "DualSubstrateV2Client"]
