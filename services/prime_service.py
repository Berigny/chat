"""High-level helpers for anchoring text via the DualSubstrate engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from flow_safe import sequence as flow_safe_sequence
from prime_pipeline import build_anchor_batches, normalize_override_factors


Payload = Mapping[str, object]


@dataclass
class PrimeService:
    """Centralise prime tagging and anchoring orchestration."""

    api_service: "ApiService"
    fallback_prime: int

    def build_batches(
        self,
        text: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        factors_override: Sequence[Mapping[str, int]] | None = None,
        llm_extractor=None,
    ) -> list[list[dict[str, int]]]:
        """Return flow-safe factor batches ready for anchoring."""

        valid_primes = tuple(schema.keys()) or (self.fallback_prime,)
        override = (
            normalize_override_factors(factors_override, valid_primes)
            if factors_override
            else None
        )
        return build_anchor_batches(
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

        batches = self.build_batches(
            text,
            schema,
            factors_override=factors_override,
            llm_extractor=llm_extractor,
        )
        if not batches:
            batches = [flow_safe_sequence([self.fallback_prime])]
        for index, factors in enumerate(batches):
            self.api_service.anchor(
                entity,
                factors,
                ledger_id=ledger_id,
                text=text if index == 0 else None,
            )
        return {"text": text, "batches": batches}


def create_prime_service(api_service: "ApiService", fallback_prime: int) -> PrimeService:
    """Factory to avoid import cycles when wiring from Streamlit."""

    return PrimeService(api_service=api_service, fallback_prime=fallback_prime)


__all__ = ["PrimeService", "create_prime_service"]
