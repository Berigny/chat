"""API facade ensuring consistent DualSubstrate interactions."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional


# Derived from the DualSubstrate Swagger documentation – demo entities map to
# fixed traversal node IDs. Extend this map as new workspaces are provisioned.
_TRAVERSAL_NODE_OVERRIDES: dict[str, int] = {
    "demo": 1,
    "demo_dev": 2,
    "demo_new": 3,
    "Demo_dev": 2,
    "Demo_new": 3,
    "Demo": 1,
}
_DEFAULT_TRAVERSAL_NODE = 0

# Export a stable, prompt-friendly list of supported ledger entities.
TRAVERSAL_ENTITY_SLUGS: tuple[str, ...] = tuple(sorted(_TRAVERSAL_NODE_OVERRIDES))

_ALLOWED_S2_PRIME_KEYS = {"11", "13", "17", "19"}

import requests

from api_client import DualSubstrateClient


logger = logging.getLogger(__name__)


class ApiService:
    """Lightweight facade wrapping :class:`DualSubstrateClient`.

    The Streamlit layer should rely on this service so that retries, error
    logging, and surface specific defaults stay consistent. Only thin wrappers
    around the REST endpoints we consume today are provided; higher level
    orchestration lives in dedicated service modules.
    """

    def __init__(self, base_url: str, api_key: Optional[str]) -> None:
        self._client = DualSubstrateClient(base_url, api_key)
        self._enrich_supported: bool | None = None
        self._traverse_supported: bool | None = None
        self._inference_supported: bool | None = None

    @property
    def client(self) -> DualSubstrateClient:
        return self._client

    # Capability ----------------------------------------------------------
    def supports_enrich(self, *, refresh: bool = False) -> bool:
        """Return ``True`` when the remote ``/enrich`` endpoint is available."""

        if refresh:
            self._enrich_supported = None
        if self._enrich_supported is None:
            self._enrich_supported = self._probe_endpoint("enrich")
        return bool(self._enrich_supported)

    def supports_traverse(self, *, refresh: bool = False) -> bool:
        """Return ``True`` if the ``/traverse`` endpoint is reachable."""

        if refresh:
            self._traverse_supported = None
        if self._traverse_supported is None:
            self._traverse_supported = self._probe_endpoint("traverse")
        return bool(self._traverse_supported)

    def supports_inference_state(self, *, refresh: bool = False) -> bool:
        """Return ``True`` if ``/inference/state`` is available."""

        if refresh:
            self._inference_supported = None
        if self._inference_supported is None:
            self._inference_supported = self._probe_endpoint("inference/state")
        return bool(self._inference_supported)

    def _probe_endpoint(self, path: str) -> bool:
        """Probe a single endpoint via ``OPTIONS`` to detect support."""

        try:
            response = requests.options(
                f"{self._client.base_url}/{path}",
                headers=self._client._headers(include_ledger=False),
                timeout=3,
            )
        except requests.RequestException:
            return False
        if response.status_code == 404:
            return False
        return True

    # Ledger management -------------------------------------------------
    def list_ledgers(self) -> List[Dict[str, Any]]:
        return self._client.list_ledgers()

    def create_ledger(self, ledger_id: str) -> Dict[str, Any]:
        return self._client.create_ledger(ledger_id)

    # Schema ------------------------------------------------------------
    def fetch_prime_schema(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
        return self._client.fetch_prime_schema(entity, ledger_id=ledger_id)

    # Anchoring ---------------------------------------------------------
    def anchor(
        self,
        entity: str,
        factors: Iterable[Dict[str, Any]],
        *,
        ledger_id: Optional[str] = None,
        text: Optional[str] = None,
        modifiers: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        return self._client.anchor(
            entity,
            factors,
            ledger_id=ledger_id,
            text=text,
            modifiers=modifiers,
        )

    def rotate(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        axis: tuple[float, float, float] | None = None,
        angle: float | None = None,
    ) -> Dict[str, Any]:
        return self._client.rotate(entity, ledger_id=ledger_id, axis=axis, angle=angle)

    # Ledger and memory -------------------------------------------------
    def fetch_memories(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        limit: int | None = None,
        since: int | None = None,
    ) -> List[Dict[str, Any]]:
        return self._client.fetch_memories(entity, ledger_id=ledger_id, limit=limit, since=since)

    def assemble_context(
        self,
        entity: str,
        k: int = 3,
        quote_safe: bool | None = True,
        since: int | None = None,
        ledger_id: str | None = None,
    ) -> Dict[str, Any]:
        """Call the ``/assemble`` endpoint using the shared client defaults."""

        params: Dict[str, Any] = {"entity": entity}
        if k is not None:
            params["k"] = int(k)
        if quote_safe is not None:
            params["quote_safe"] = "true" if quote_safe else "false"
        if since is not None:
            params["since"] = since

        response = requests.get(
            f"{self._client.base_url}/assemble",
            params=params,
            headers=self._client._headers(ledger_id=ledger_id),
            timeout=self._client.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}

    def fetch_assembly(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        k: int | None = None,
        quote_safe: bool | None = None,
        since: int | None = None,
    ) -> Dict[str, Any]:
        return self.assemble_context(
            entity,
            ledger_id=ledger_id,
            k=k if k is not None else 3,
            quote_safe=quote_safe,
            since=since,
        )

    def _resolve_traversal_start(self, entity: Any) -> int:
        """Resolve the traversal start node from a UI-facing entity label."""

        if isinstance(entity, int):
            return entity
        if isinstance(entity, str):
            candidate = entity.strip()
            if candidate.isdigit():
                return int(candidate)
            override = _TRAVERSAL_NODE_OVERRIDES.get(candidate)
            if override is not None:
                return override
            override = _TRAVERSAL_NODE_OVERRIDES.get(candidate.lower())
            if override is not None:
                return override
        return _DEFAULT_TRAVERSAL_NODE

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
        payload: Mapping[str, Any] | list[Any] | None = None,
    ) -> Dict[str, Any]:
        """Return traversal paths while tracking capability support."""

        raw_start = entity
        resolved_start = self._resolve_traversal_start(entity)
        traverse_kwargs: Dict[str, Any] = {"start": resolved_start}

        depth_value: Any | None = depth if depth is not None else limit
        if depth_value is not None:
            traverse_kwargs["depth"] = depth_value

        traverse_kwargs.update(
            {
                "ledger_id": ledger_id,
                "origin": origin,
                "limit": limit,
                "direction": direction,
                "include_metadata": include_metadata,
                "payload": payload,
            }
        )

        logger.error("API_SVC_TRAVERSE start=%r resolved=%r", raw_start, resolved_start)

        try:
            response = self._client.traverse(**traverse_kwargs)
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None:
                status = getattr(response, "status_code", None)
                if status == 404:
                    self._traverse_supported = False
                elif status is not None:
                    self._traverse_supported = True

                message = self._summarize_traverse_error(response)
                if message and message != str(exc):
                    raise requests.HTTPError(
                        message,
                        response=response,
                        request=getattr(exc, "request", None),
                    ) from exc
            raise
        except requests.RequestException:
            raise
        else:
            self._traverse_supported = True

        if isinstance(response, Mapping):
            return dict(response)
        if isinstance(response, list):
            return {"paths": list(response)}
        return {}

    @staticmethod
    def _summarize_traverse_error(response: Any) -> str | None:
        """Extract a useful error message from a traverse failure response."""

        def _stringify(value: Any) -> str | None:
            if isinstance(value, str):
                text = value.strip()
                return text or None
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, Mapping):
                for item in value.values():
                    text = _stringify(item)
                    if text:
                        return text
                return None
            if isinstance(value, (list, tuple, set)):
                parts = [text for item in value if (text := _stringify(item))]
                if parts:
                    return "; ".join(parts)
            return None

        for extractor in (getattr(response, "json", None),):
            if callable(extractor):
                try:
                    data = extractor()
                except ValueError:
                    continue
                if isinstance(data, Mapping):
                    for key in ("detail", "message", "error", "errors"):
                        if key in data:
                            text = _stringify(data[key])
                            if text:
                                return text
        text_content = getattr(response, "text", None)
        if isinstance(text_content, str) and text_content.strip():
            return text_content.strip()
        return None

    def latest_memory_text(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        since: int | None = None,
    ) -> Optional[str]:
        return self._client.latest_memory_text(entity, ledger_id=ledger_id, since=since)

    def fetch_ledger(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.fetch_ledger(entity, ledger_id=ledger_id)

    def fetch_inference_state(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        include_history: bool | None = None,
        limit: int | None = None,
    ) -> Dict[str, Any]:
        try:
            payload = self._client.fetch_inference_state(
                entity,
                ledger_id=ledger_id,
                include_history=include_history,
                limit=limit,
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self._inference_supported = False
                return {}
            raise
        else:
            self._inference_supported = True
            return payload

    def fetch_inference_traverse(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        _ = entity
        return self._client.fetch_inference_traverse(ledger_id=ledger_id)

    def fetch_inference_memories(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        _ = entity
        return self._client.fetch_inference_memories(ledger_id=ledger_id)

    def fetch_inference_retrieve(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = entity
        return self._client.fetch_inference_retrieve(ledger_id=ledger_id)

    def query_ledger(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: Optional[str] = None,
        limit: int | None = None,
        topic: str | None = None,
        required: Iterable[int] | None = None,
        preferred: Iterable[int] | None = None,
        modifiers: Iterable[int] | None = None,
    ) -> Dict[str, Any]:
        return self._client.query_ledger(
            entity,
            query,
            ledger_id=ledger_id,
            limit=limit,
            topic=topic,
            required=required,
            preferred=preferred,
            modifiers=modifiers,
        )

    def search(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> Dict[str, Any]:
        return self._client.search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
            fuzzy=fuzzy,
            semantic_weight=semantic_weight,
            delta=delta,
        )

    def search_with_response(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        fuzzy: bool = True,
        semantic_weight: float = 0.45,
        delta: int = 2,
    ) -> tuple[Dict[str, Any], "requests.Response"]:
        """Call ``/memories`` and return both parsed JSON and raw HTTP response."""

        return self._client.search_with_response(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
            fuzzy=fuzzy,
            semantic_weight=semantic_weight,
            delta=delta,
        )

    def search_slots(
        self,
        entity: str,
        query: str,
        *,
        ledger_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload = self.search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
        )
        slots = payload.get("slots") if isinstance(payload, dict) else None
        return [slot for slot in slots if isinstance(slot, dict)] if isinstance(slots, list) else []

    def retrieve(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.retrieve(entity, ledger_id=ledger_id)

    def fetch_metrics(self, *, ledger_id: Optional[str] = None) -> Dict[str, Any] | str:
        return self._client.fetch_metrics(ledger_id=ledger_id)

    def _call_inference_endpoint(
        self,
        method_name: str,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
        **extra_kwargs: Any,
    ):
        """Invoke an inference telemetry helper while tolerating older clients."""

        method = getattr(self._client, method_name)
        signature = inspect.signature(method)
        accepts_variadic_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        allowed_kwargs: Dict[str, Any] = {}
        for key, value in extra_kwargs.items():
            if value is None:
                continue
            if accepts_variadic_kwargs or key in signature.parameters:
                allowed_kwargs[key] = value

        if ledger_id is not None and (
            accepts_variadic_kwargs or "ledger_id" in signature.parameters
        ):
            allowed_kwargs["ledger_id"] = ledger_id

        entity_supported = accepts_variadic_kwargs or "entity" in signature.parameters

        def _invoke(*args, **kwargs):
            return method(*args, **kwargs)

        last_error: TypeError | None = None
        try:
            return _invoke(**allowed_kwargs)
        except TypeError as exc:
            last_error = exc

        if entity is not None and entity_supported:
            try:
                return _invoke(entity, **allowed_kwargs)
            except TypeError as exc:
                last_error = exc
                if "ledger_id" in allowed_kwargs:
                    trimmed_kwargs = {k: v for k, v in allowed_kwargs.items() if k != "ledger_id"}
                    try:
                        return _invoke(entity, **trimmed_kwargs)
                    except TypeError as fallback_exc:
                        last_error = fallback_exc

        if last_error is None:
            raise TypeError(f"Failed to call inference endpoint '{method_name}'")
        raise last_error

    def fetch_inference_state(
        self,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
        include_history: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._call_inference_endpoint(
            "fetch_inference_state",
            entity,
            ledger_id=ledger_id,
            include_history=include_history,
            limit=limit,
        )

    def fetch_inference_traverse(
        self,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._call_inference_endpoint(
            "fetch_inference_traverse",
            entity,
            ledger_id=ledger_id,
        )

    def fetch_inference_memories(
        self,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._call_inference_endpoint(
            "fetch_inference_memories",
            entity,
            ledger_id=ledger_id,
        )

    def fetch_inference_retrieve(
        self,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._call_inference_endpoint(
            "fetch_inference_retrieve",
            entity,
            ledger_id=ledger_id,
        )

    # Structured ledger writes -----------------------------------------
    def put_ledger_s1(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client.put_ledger_s1(entity, payload, ledger_id=ledger_id)

    def put_ledger_body(
        self,
        entity: str,
        prime: int,
        body_text: str | Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._client.put_ledger_body(
            entity,
            prime,
            body_text,
            ledger_id=ledger_id,
            metadata=metadata,
        )

    def enrich(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = entity  # Included for interface parity with helper usage.
        try:
            response = self._client.enrich(payload, ledger_id=ledger_id)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self._enrich_supported = False
            raise
        else:
            self._enrich_supported = True
            return response

    def put_ledger_s2(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        if isinstance(payload, Mapping):
            sanitized = {k: v for k, v in payload.items() if k in _ALLOWED_S2_PRIME_KEYS}

        response = self._client.put_ledger_s2(entity, sanitized, ledger_id=ledger_id)
        if sanitized:
            return sanitized
        if isinstance(response, Mapping):
            return dict(response)
        return {}

    def update_lawfulness(
        self,
        entity: str,
        payload: Dict[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client.update_lawfulness(entity, payload, ledger_id=ledger_id)

    def patch_lawfulness(
        self,
        entity: str,
        tier_or_payload: Mapping[str, Any] | int | float,
        *,
        ledger_id: Optional[str] = None,
        **extra_fields: Any,
    ) -> Dict[str, Any]:
        """Normalize promotion payloads before calling the guardrail endpoint."""

        payload: Dict[str, Any] = {}
        if isinstance(tier_or_payload, Mapping):
            payload.update(dict(tier_or_payload))
        else:
            payload["tier"] = tier_or_payload

        for key, value in extra_fields.items():
            if value is not None:
                payload[key] = value

        return self._client.update_lawfulness(entity, payload, ledger_id=ledger_id)

    def update_metrics(self, entity: str, payload: Dict[str, Any], *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.update_metrics(entity, payload, ledger_id=ledger_id)

    def patch_metrics(
        self,
        entity: str,
        metrics: Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = dict(metrics) if isinstance(metrics, Mapping) else dict(metrics or {})
        return self._client.update_metrics(entity, payload, ledger_id=ledger_id)


__all__ = ["ApiService", "TRAVERSAL_ENTITY_SLUGS", "requests"]
