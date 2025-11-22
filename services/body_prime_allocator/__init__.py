"""Allocate ledger body primes while respecting existing ledger state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING


if TYPE_CHECKING:
    from services.api import ApiService


BODY_PRIME_FLOOR = 23


def _is_prime(candidate: int) -> bool:
    if candidate <= 1:
        return False
    if candidate == 2:
        return True
    if candidate % 2 == 0:
        return False
    limit = int(candidate**0.5) + 1
    for factor in range(3, limit, 2):
        if candidate % factor == 0:
            return False
    return True


def _collect_body_primes(node: Any, *, floor: int) -> set[int]:
    primes: set[int] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            candidate = value.get("body_prime")
            if isinstance(candidate, int) and candidate >= floor:
                primes.add(int(candidate))
            candidate = value.get("prime")
            if isinstance(candidate, int) and candidate >= floor:
                primes.add(int(candidate))
            for child in value.values():
                if isinstance(child, (Mapping, Sequence)) and not isinstance(
                    child, (str, bytes, bytearray)
                ):
                    _walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                _walk(item)

    _walk(node)
    return primes


def sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = []
            for element in value:
                if isinstance(element, (str, int, float, bool)) or element is None:
                    items.append(element)
            if items:
                sanitized[key] = items
        elif isinstance(value, Mapping):
            nested = sanitize_metadata(value)
            if nested:
                sanitized[key] = nested
    return sanitized


@dataclass
class BodyPrimeAllocator:
    """Manage body prime assignment and persistence for ledger bodies."""

    api_service: "ApiService" | None
    floor: int = BODY_PRIME_FLOOR
    max_attempts: int = 5
    _issued: set[int] = field(default_factory=set)
    _seeded_scopes: set[tuple[str, str | None]] = field(default_factory=set)

    def next_prime(
        self,
        *,
        reserved: Iterable[int] | None = None,
        entity: str | None = None,
        ledger_id: str | None = None,
    ) -> int:
        reserved_set = {
            int(prime)
            for prime in (reserved or [])
            if isinstance(prime, int)
        }
        reserved_set |= self._issued
        baseline = max(
            (prime for prime in reserved_set if prime >= self.floor),
            default=self.floor - 2,
        )
        candidate = max(self.floor, baseline + 2)
        if candidate % 2 == 0:
            candidate += 1
        while candidate in reserved_set or not _is_prime(candidate):
            candidate += 2
        self._issued.add(candidate)
        return candidate

    def mint_bodies(
        self,
        entity: str,
        body_plan: Sequence[Mapping[str, Any]],
        *,
        ledger_id: str | None = None,
    ) -> list[dict[str, Any]]:
        # Legacy body minting is deprecated; bodies are now carried in ledger
        # metadata. This method is retained for compatibility but intentionally
        # returns an empty collection to avoid network calls or side effects.
        return []


__all__ = ["BodyPrimeAllocator", "BODY_PRIME_FLOOR", "sanitize_metadata"]
 
