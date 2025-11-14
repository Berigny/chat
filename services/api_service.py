"""High-level enrichment helpers built on top of :mod:`services.api`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import requests


if TYPE_CHECKING:
    from services.api import ApiService
    from services.prime_service import PrimeService


from flow_rules import FlowAssessment, assess_enrichment_path

def _normalize_deltas(entries: Sequence[Mapping[str, Any]] | None) -> list[dict[str, int]]:
    """Return a sanitized list of deltas ready for the enrichment endpoint."""

    normalized: list[dict[str, int]] = []
    if not entries:
        return normalized

    for item in entries:
        if not isinstance(item, Mapping):
            continue
        prime = item.get("prime")
        delta = item.get("delta", item.get("value", 0))
        try:
            prime_int = int(prime)
            delta_int = int(delta)
        except (TypeError, ValueError):
            continue
        normalized.append({"prime": prime_int, "delta": delta_int})
    return normalized


@dataclass
class EnrichmentHelper:
    """Facilitates enrichment cycles with consistent metadata management."""

    api_service: "ApiService"
    prime_service: "PrimeService"

    def _mint_bodies(
        self,
        entity: str,
        *,
        ref_prime: int,
        body_chunks: Iterable[str] | None,
        ledger_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        minted: list[dict[str, Any]] = []
        if not body_chunks:
            return minted

        supplemental_meta = dict(metadata or {})
        supplemental_meta.setdefault("source", "enrichment")
        supplemental_meta["superseded_by"] = ref_prime

        reserved: set[int] = set()
        for chunk in body_chunks:
            if not isinstance(chunk, str):
                continue
            text = chunk.strip()
            if not text:
                continue
            prime = self.prime_service.next_body_prime(
                reserved=reserved,
                entity=entity,
                ledger_id=ledger_id,
            )
            reserved.add(prime)
            body_payload: dict[str, Any] = {"body": text}
            if supplemental_meta:
                body_payload["metadata"] = dict(supplemental_meta)
            self.api_service.put_ledger_body(
                entity,
                prime,
                body_payload,
                ledger_id=ledger_id,
            )
            minted.append(
                {
                    "prime": prime,
                    "body": text,
                    "metadata": dict(supplemental_meta),
                }
            )
        return minted

    def submit(
        self,
        entity: str,
        *,
        ref_prime: int,
        deltas: Sequence[Mapping[str, Any]] | None = None,
        body_chunks: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        ledger_id: str | None = None,
        schema: Mapping[int, Mapping[str, object]] | None = None,
    ) -> dict[str, Any]:
        """Call ``/enrich`` and return the request/response envelope."""

        normalized_deltas = _normalize_deltas(deltas)
        flow_assessment: FlowAssessment | None = None
        if schema:
            flow_assessment = assess_enrichment_path(ref_prime, normalized_deltas, schema)
        else:
            flow_assessment = assess_enrichment_path(ref_prime, normalized_deltas, {})
        if flow_assessment and not flow_assessment.ok:
            return {
                "ref_prime": int(ref_prime),
                "deltas": normalized_deltas,
                "bodies": [],
                "request": None,
                "response": {},
                "enrichment_supported": True,
                "flow_errors": flow_assessment.messages(),
                "flow_violations": [
                    violation.asdict() for violation in flow_assessment.violations
                ],
                "flow_assessment": flow_assessment.asdict(),
            }
        minted_bodies = self._mint_bodies(
            entity,
            ref_prime=ref_prime,
            body_chunks=body_chunks,
            ledger_id=ledger_id,
            metadata=metadata,
        )

        payload: dict[str, Any] = {
            "ref_prime": int(ref_prime),
            "deltas": normalized_deltas,
        }
        if minted_bodies:
            payload["bodies"] = [
                {"prime": entry["prime"], "superseded_by": ref_prime}
                for entry in minted_bodies
            ]
        if metadata:
            merged_meta = dict(metadata)
            merged_meta["superseded_by"] = ref_prime
            payload["metadata"] = merged_meta

        enrichment_supported = True
        request_payload: dict[str, Any] | None = None
        response: dict[str, Any] = {}

        try:
            enrichment_supported = self.api_service.supports_enrich()
        except AttributeError:
            enrichment_supported = True
        except Exception:
            enrichment_supported = True

        if enrichment_supported:
            request_payload = payload
            try:
                response = self.api_service.enrich(
                    entity,
                    payload,
                    ledger_id=ledger_id,
                )
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404:
                    enrichment_supported = False
                    request_payload = None
                    response = {}
                else:
                    raise
        else:
            response = {}

        return {
            "ref_prime": int(ref_prime),
            "deltas": normalized_deltas,
            "bodies": minted_bodies,
            "request": request_payload,
            "response": response or {},
            "enrichment_supported": enrichment_supported,
        }


__all__ = ["EnrichmentHelper"]
