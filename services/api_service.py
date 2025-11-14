from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from services.api import ApiService


@dataclass
class EnrichmentHelper:
    """Thin helper that proxies enrichment payloads to the engine."""

    api_service: "ApiService"

    def submit(
        self,
        entity: str,
        payload: Mapping[str, Any],
        *,
        ledger_id: str | None = None,
    ) -> dict[str, Any]:
        """Forward the caller-provided payload to ``/enrich``."""

        response = self.api_service.enrich(payload, ledger_id=ledger_id)
        return {
            "entity": entity,
            "request": dict(payload),
            "response": response if isinstance(response, dict) else {},
        }


__all__ = ["EnrichmentHelper"]
