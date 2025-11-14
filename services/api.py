"""API facade ensuring consistent DualSubstrate interactions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

from api_client import DualSubstrateClient


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

    @property
    def client(self) -> DualSubstrateClient:
        return self._client

    # Capability ----------------------------------------------------------
    def supports_enrich(self, *, refresh: bool = False) -> bool:
        """Return ``True`` when the remote ``/enrich`` endpoint is available."""

        if refresh:
            self._enrich_supported = None
        if self._enrich_supported is None:
            self._enrich_supported = self._probe_enrich_support()
        return bool(self._enrich_supported)

    def _probe_enrich_support(self) -> bool:
        """Probe the API once to determine whether ``/enrich`` exists."""

        try:
            response = requests.options(
                f"{self._client.base_url}/enrich",
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

    def fetch_assembly(
        self,
        entity: str,
        *,
        ledger_id: Optional[str] = None,
        k: int | None = None,
        quote_safe: bool | None = None,
        since: int | None = None,
    ) -> Dict[str, Any]:
        return self._client.assemble_context(
            entity,
            ledger_id=ledger_id,
            k=k,
            quote_safe=quote_safe,
            since=since,
        )

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
    ) -> Dict[str, Any]:
        return self._client.search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=mode,
            limit=limit,
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
    ):
        """Invoke an inference telemetry helper while tolerating older clients."""

        method = getattr(self._client, method_name)
        kwargs: Dict[str, Any] = {}
        if ledger_id is not None:
            kwargs["ledger_id"] = ledger_id

        try:
            # Newer clients accept keyword-only ``ledger_id``.
            return method(**kwargs)
        except TypeError as exc:
            if entity is None:
                raise
            try:
                # Fallback for deployments still expecting a positional entity argument.
                return method(entity, **kwargs)
            except TypeError:
                raise exc

    def fetch_inference_state(
        self,
        entity: Optional[str] = None,
        *,
        ledger_id: Optional[str] = None,
        include_history: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        # ``include_history`` and ``limit`` are currently unused but preserved for
        # backwards compatibility with callers that may provide them.
        _ = include_history, limit
        return self._call_inference_endpoint(
            "fetch_inference_state",
            entity,
            ledger_id=ledger_id,
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
        return self._client.put_ledger_s2(entity, payload, ledger_id=ledger_id)

    def update_lawfulness(self, entity: str, payload: Dict[str, Any], *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.update_lawfulness(entity, payload, ledger_id=ledger_id)

    def update_metrics(self, entity: str, payload: Dict[str, Any], *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.update_metrics(entity, payload, ledger_id=ledger_id)


__all__ = ["ApiService", "requests"]
