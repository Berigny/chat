"""High-level helpers for anchoring text via the DualSubstrate engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from prime_pipeline import build_anchor_payload, normalize_override_factors


Payload = Mapping[str, object]


@dataclass
class PrimeService:
    """Centralise prime tagging and anchoring orchestration."""

    api_service: "ApiService"
    fallback_prime: int

    def build_payload(
        self,
        text: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        factors_override: Sequence[Mapping[str, int]] | None = None,
        llm_extractor=None,
    ) -> Payload:
        """Return the payload ready for the `/anchor` endpoint."""

        valid_primes = tuple(schema.keys()) or (self.fallback_prime,)
        override = (
            normalize_override_factors(factors_override, valid_primes)
            if factors_override
            else None
        )
        return build_anchor_payload(
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
    ) -> Payload:
        """Send the text and factors to the engine and return the payload."""

        payload = self.build_payload(
            text,
            schema,
            factors_override=factors_override,
            llm_extractor=llm_extractor,
        )
        self.api_service.anchor(
            entity,
            payload.get("factors", []),
            ledger_id=ledger_id,
            text=payload.get("text") or text,
        )
        return payload


def create_prime_service(api_service: "ApiService", fallback_prime: int) -> PrimeService:
    """Factory to avoid import cycles when wiring from Streamlit."""

    return PrimeService(api_service=api_service, fallback_prime=fallback_prime)


__all__ = ["PrimeService", "create_prime_service"]
