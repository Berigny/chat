from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from services.api import ApiService
from services.memory_service import MemoryService
from services.prompt_service import LEDGER_SNIPPET_LIMIT


logger = logging.getLogger(__name__)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class LedgerTraversalService:
    """Execute traversal intents against ledger retrieval endpoints."""

    api_service: ApiService
    memory_service: MemoryService

    def _assemble(
        self,
        entity: str,
        *,
        ledger_id: str | None,
        k: int | None,
        quote_safe: bool,
        since: int | None,
    ) -> dict[str, list[dict]]:
        return self.memory_service.assemble_context(
            entity,
            ledger_id=ledger_id,
            k=k,
            quote_safe=quote_safe,
            since=since,
        )

    def _body_view(
        self,
        entity: str,
        *,
        ledger_id: str | None,
        k: int | None,
        quote_safe: bool,
        since: int | None,
    ) -> dict[str, list[dict]]:
        assembly = self._assemble(
            entity,
            ledger_id=ledger_id,
            k=k,
            quote_safe=True if quote_safe is None else quote_safe,
            since=since,
        )
        bodies = [
            entry
            for entry in assembly.get("bodies", [])
            if isinstance(entry, Mapping)
        ]
        if bodies:
            return {"summaries": [], "claims": [], "bodies": bodies}
        return assembly

    def execute_intent(
        self,
        intent: Mapping[str, Any] | None,
        *,
        default_entity: str | None,
        ledger_id: str | None = None,
        quote_safe_default: bool = False,
        since: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(intent, Mapping):
            return {"entity": None, "path": None, "assembly": None, "k": None}

        entity = intent.get("entity") or intent.get("target") or default_entity
        if isinstance(entity, str):
            entity = entity.strip()
        if not entity:
            return {"entity": None, "path": None, "assembly": None, "k": None}

        raw_path = intent.get("path") or intent.get("endpoint") or intent.get("intent")
        normalized_path = str(raw_path or "/assemble").strip().lower()
        if normalized_path not in {"/assemble", "assemble", "/body", "body"}:
            normalized_path = "/assemble"
        if normalized_path == "body":
            normalized_path = "/body"
        if normalized_path == "assemble":
            normalized_path = "/assemble"

        k_value = _coerce_int(intent.get("k") or intent.get("limit"))
        if k_value is None:
            k_value = LEDGER_SNIPPET_LIMIT

        quote_safe = intent.get("quote_safe")
        if quote_safe is None:
            quote_safe = quote_safe_default
        since_value = _coerce_int(intent.get("since")) or since

        try:
            if normalized_path == "/body":
                assembly = self._body_view(
                    entity,
                    ledger_id=ledger_id,
                    k=k_value,
                    quote_safe=bool(quote_safe),
                    since=since_value,
                )
            else:
                assembly = self._assemble(
                    entity,
                    ledger_id=ledger_id,
                    k=k_value,
                    quote_safe=bool(quote_safe),
                    since=since_value,
                )
        except Exception:
            logger.exception("Traversal intent failed for entity %s", entity)
            assembly = None

        return {
            "entity": entity,
            "path": normalized_path,
            "assembly": assembly,
            "k": k_value,
            "quote_safe": bool(quote_safe),
        }

