"""High-level helpers for anchoring text via the DualSubstrate engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import requests

from prime_pipeline import (
    S1_PRIMES,
    S2_PRIMES,
    assess_factor_flow,
    build_anchor_factors,
    normalize_override_factors,
)

from services.body_prime_allocator import (
    BodyPrimeAllocator,
    BODY_PRIME_FLOOR as DEFAULT_BODY_PRIME_FLOOR,
    sanitize_metadata,
)


Payload = Mapping[str, object]
BODY_PRIME_FLOOR = DEFAULT_BODY_PRIME_FLOOR


def _merge_metadata(base: Mapping[str, Any] | None, extra: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, Mapping):
        for key, value in base.items():
            if isinstance(key, str):
                merged[key] = value
    for key, value in extra.items():
        if isinstance(key, str):
            merged.setdefault(key, value)
    return merged


def _derive_title(text: str, *, max_length: int = 96) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    first_line = cleaned.splitlines()[0]
    if len(first_line) <= max_length:
        return first_line
    trunc = first_line[:max_length].rstrip()
    return trunc


def _derive_summary(text: str, *, max_length: int = 160) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    summary = cleaned.replace("\n", " ")
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 1].rstrip() + "…"


def _prepare_ingest_plan(
    text: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = (text or "").strip()
    base_meta = _merge_metadata(metadata, {"length": len(cleaned), "kind": "memory"})
    body_key = "body-0"
    bodies: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    s1_slots: list[dict[str, Any]] = []
    s2_slots: list[dict[str, Any]] = []

    if cleaned:
        bodies.append(
            {
                "key": body_key,
                "body": cleaned,
                "metadata": _merge_metadata(
                    base_meta,
                    {
                        "source_primes": list(S1_PRIMES + S2_PRIMES),
                        "superseded_primes": list(S1_PRIMES + S2_PRIMES),
                    },
                ),
            }
        )
        title = _derive_title(cleaned)
        summary = _derive_summary(cleaned)

        for prime in S1_PRIMES:
            slot = {
                "prime": prime,
                "value": 1,
                "title": title,
                "tags": [],
                "body": [cleaned],
                "body_key": body_key,
                "metadata": _merge_metadata(base_meta, {"tier": "S1"}),
            }
            slots.append(slot)
            s1_slots.append(slot)
        for prime in S2_PRIMES:
            slot = {
                "prime": prime,
                "summary": summary,
                "body": [cleaned],
                "body_key": body_key,
                "metadata": _merge_metadata(base_meta, {"tier": "S2"}),
            }
            slots.append(slot)
            s2_slots.append(slot)

    return {
        "slots": slots,
        "s1": s1_slots,
        "s2": s2_slots,
        "bodies": bodies,
    }


@dataclass
class PrimeService:
    """Centralise prime tagging, anchoring, and ingest orchestration."""

    api_service: "ApiService"
    fallback_prime: int
    body_prime_floor: int = BODY_PRIME_FLOOR
    body_allocator: BodyPrimeAllocator | None = None

    def __post_init__(self) -> None:
        if self.body_allocator is None:
            self.body_allocator = BodyPrimeAllocator(
                api_service=self.api_service,
                floor=self.body_prime_floor,
            )
        else:
            self.body_allocator.api_service = self.api_service
            self.body_allocator.floor = self.body_prime_floor

    def build_factors(
        self,
        text: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        factors_override: Sequence[Mapping[str, int]] | None = None,
        llm_extractor=None,
    ) -> list[dict[str, int]]:
        """Return normalized factors ready for anchoring."""

        valid_primes = tuple(schema.keys()) or (self.fallback_prime,)
        override = (
            normalize_override_factors(factors_override, valid_primes)
            if factors_override
            else None
        )
        return build_anchor_factors(
            text,
            schema,
            fallback_prime=self.fallback_prime,
            factors_override=override,
            llm_extractor=llm_extractor,
        )

    # The ApiService dependency is imported lazily to avoid circular imports.
    def anchor(
        self,
        entity: str,
        text: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        factors_override: Sequence[Mapping[str, int]] | None = None,
        llm_extractor=None,
        modifiers: Sequence[int] | None = None,
    ) -> Payload:
        """Send the text and factors to the engine and return the payload."""

        factors = self.build_factors(
            text,
            schema,
            factors_override=factors_override,
            llm_extractor=llm_extractor,
        )
        response = self.api_service.anchor(
            entity,
            factors,
            ledger_id=ledger_id,
            text=text,
            modifiers=modifiers,
        )
        return {"text": text, "factors": factors, "response": response}

    def next_body_prime(
        self,
        *,
        reserved: Iterable[int] | None = None,
        entity: str | None = None,
        ledger_id: str | None = None,
    ) -> int:
        """Return the next unused prime for immutable body storage."""

        if not self.body_allocator:
            raise RuntimeError("Body prime allocator is not configured")
        return self.body_allocator.next_prime(
            reserved=reserved,
            entity=entity,
            ledger_id=ledger_id,
        )

    def ingest(
        self,
        entity: str,
        text: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        factors_override: Sequence[Mapping[str, int]] | None = None,
        llm_extractor=None,
        metadata: Mapping[str, object] | None = None,
    ) -> Payload:
        """Persist text via /ingest with structured metadata and bodies."""

        normalized_text = (text or "").strip()
        factors = self.build_factors(
            normalized_text,
            schema,
            factors_override=factors_override,
            llm_extractor=llm_extractor,
        )
        flow_assessment = assess_factor_flow(factors, schema)
        if not flow_assessment.ok:
            return {
                "text": normalized_text,
                "factors": factors,
                "structured": {},
                "flow_errors": flow_assessment.messages(),
                "flow_violations": [
                    violation.asdict() for violation in flow_assessment.violations
                ],
                "flow_assessment": flow_assessment.asdict(),
            }
        plan = _prepare_ingest_plan(normalized_text, metadata=metadata)
        anchor_payload = {
            "entity": entity,
            "factors": factors,
            "text": normalized_text,
        }
        anchor_response = self.api_service.anchor(
            entity,
            factors,
            ledger_id=ledger_id,
            text=normalized_text,
        )

        if not self.body_allocator:
            raise RuntimeError("Body prime allocator is not configured")
        minted_bodies = self.body_allocator.mint_bodies(
            entity,
            plan.get("bodies", []),
            ledger_id=ledger_id,
        )
        body_map = {
            item.get("key"): item.get("prime")
            for item in minted_bodies
            if isinstance(item.get("key"), str) and isinstance(item.get("prime"), int)
        }

        s1_slots: list[dict[str, Any]] = []
        for slot in plan.get("s1", []):
            if not isinstance(slot, Mapping):
                continue
            prime = slot.get("prime")
            if not isinstance(prime, int):
                continue
            body_key = slot.get("body_key")
            body_prime = body_map.get(body_key)
            if not body_prime:
                continue
            payload: dict[str, Any] = {
                "prime": prime,
                "value": int(slot.get("value", 1)),
                "body_prime": body_prime,
            }
            title = slot.get("title")
            if isinstance(title, str) and title.strip():
                payload["title"] = title.strip()
            tags = slot.get("tags")
            if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
                cleaned_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
                if cleaned_tags:
                    payload["tags"] = cleaned_tags
            slot_meta = sanitize_metadata(slot.get("metadata"))
            if slot_meta:
                payload["metadata"] = slot_meta
            s1_slots.append(payload)

        s2_slots: list[dict[str, Any]] = []
        for slot in plan.get("s2", []):
            if not isinstance(slot, Mapping):
                continue
            prime = slot.get("prime")
            if not isinstance(prime, int):
                continue
            body_key = slot.get("body_key")
            body_prime = body_map.get(body_key)
            if not body_prime:
                continue
            payload: dict[str, Any] = {
                "prime": prime,
                "body_prime": body_prime,
            }
            summary = slot.get("summary")
            if isinstance(summary, str) and summary.strip():
                payload["summary"] = summary.strip()
            slot_meta = sanitize_metadata(slot.get("metadata"))
            if slot_meta:
                payload["metadata"] = slot_meta
            s2_slots.append(payload)

        slots = []
        for slot in plan.get("slots", []):
            if not isinstance(slot, Mapping):
                continue
            body_key = slot.get("body_key")
            body_prime = body_map.get(body_key)
            enriched = dict(slot)
            if body_prime:
                enriched["body_prime"] = body_prime
            enriched.pop("body_key", None)
            slots.append(enriched)

        structured = {
            "slots": slots,
            "s1": s1_slots,
            "s2": s2_slots,
            "bodies": minted_bodies,
        }

        return {
            "text": normalized_text,
            "factors": factors,
            "structured": structured,
            "anchor": anchor_response,
            "anchor_request": anchor_payload,
        }


def create_prime_service(api_service: "ApiService", fallback_prime: int) -> PrimeService:
    """Factory to avoid import cycles when wiring from Streamlit."""

    return PrimeService(api_service=api_service, fallback_prime=fallback_prime)


__all__ = ["PrimeService", "create_prime_service"]
