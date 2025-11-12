"""API facade ensuring consistent DualSubstrate interactions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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
    def anchor(self, entity: str, factors: Iterable[Dict[str, Any]], *, ledger_id: Optional[str] = None, text: Optional[str] = None) -> Dict[str, Any]:
        return self._client.anchor(entity, factors, ledger_id=ledger_id, text=text)

    def rotate(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.rotate(entity, ledger_id=ledger_id)

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

    def fetch_ledger(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.fetch_ledger(entity, ledger_id=ledger_id)

    def retrieve(self, entity: str, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.retrieve(entity, ledger_id=ledger_id)

    def fetch_metrics(self, *, ledger_id: Optional[str] = None) -> Dict[str, Any]:
        return self._client.fetch_metrics(ledger_id=ledger_id)


__all__ = ["ApiService", "requests"]
