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

    @property
    def client(self) -> DualSubstrateClient:
        return self._client

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

    def fetch_metrics(self, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.fetch_metrics(ledger_id=ledger_id)

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
        payload: Mapping[str, Any],
        *,
        ledger_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client.enrich(payload, ledger_id=ledger_id)

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
