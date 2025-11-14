"""Flow-control heuristics for ledger operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

TierSchema = Mapping[int, Mapping[str, object]]

SOURCE_TIERS = frozenset({"S"})
TARGET_TIERS = frozenset({"A", "B"})
MEDIATOR_TIERS = frozenset({"C"})


def _tier_for(prime: int, schema: TierSchema | None) -> str:
    if not schema:
        return ""
    meta = schema.get(prime)
    if not isinstance(meta, Mapping):
        return ""
    tier = meta.get("tier")
    if isinstance(tier, str):
        return tier.strip().upper()
    return ""


def _prime_label(prime: int, schema: TierSchema | None) -> str:
    if schema:
        meta = schema.get(prime)
        if isinstance(meta, Mapping):
            name = meta.get("name")
            tier = _tier_for(prime, schema)
            if isinstance(name, str) and name.strip():
                if tier:
                    return f"{prime} ({name.strip()} / {tier})"
                return f"{prime} ({name.strip()})"
            if tier:
                return f"{prime} ({tier})"
    return str(prime)


def _normalize_primes(entries: Sequence[int | Mapping[str, object]] | None) -> list[int]:
    normalized: list[int] = []
    if not entries:
        return normalized
    for entry in entries:
        prime: object
        if isinstance(entry, Mapping):
            prime = entry.get("prime")
        else:
            prime = entry
        try:
            normalized.append(int(prime))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return normalized


@dataclass(frozen=True)
class FlowViolation:
    """Represents a discrete flow constraint failure."""

    code: str
    message: str
    primes: tuple[int, ...] = field(default_factory=tuple)

    def asdict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.primes:
            payload["primes"] = list(self.primes)
        return payload


@dataclass
class FlowAssessment:
    """Container describing whether a flow sequence is permissible."""

    ok: bool
    violations: list[FlowViolation]

    @classmethod
    def success(cls) -> "FlowAssessment":
        return cls(ok=True, violations=[])

    @classmethod
    def failure(cls, violations: Iterable[FlowViolation]) -> "FlowAssessment":
        return cls(ok=False, violations=list(violations))

    def messages(self) -> list[str]:
        return [violation.message for violation in self.violations]

    def asdict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "violations": [violation.asdict() for violation in self.violations],
        }


def _evaluate_sequence(
    primes: Sequence[int],
    schema: TierSchema | None,
    *,
    rule_code: str,
) -> FlowAssessment:
    if not primes:
        return FlowAssessment.success()

    tiers = [_tier_for(prime, schema) for prime in primes]
    violations: list[FlowViolation] = []

    for idx, tier in enumerate(tiers):
        if tier not in SOURCE_TIERS:
            continue
        mediator_present = False
        for cursor in range(idx + 1, len(primes)):
            next_tier = tiers[cursor]
            if next_tier in SOURCE_TIERS:
                break
            if next_tier in MEDIATOR_TIERS:
                mediator_present = True
                continue
            if next_tier in TARGET_TIERS:
                if not mediator_present:
                    source_prime = primes[idx]
                    target_prime = primes[cursor]
                    message = (
                        f"{_prime_label(source_prime, schema)} requires a "
                        f"C-tier mediator before {_prime_label(target_prime, schema)}."
                    )
                    violations.append(
                        FlowViolation(
                            code=rule_code,
                            message=message,
                            primes=(source_prime, target_prime),
                        )
                    )
                break
    if violations:
        return FlowAssessment.failure(violations)
    return FlowAssessment.success()


def assess_write_path(
    factors: Sequence[Mapping[str, object] | int] | None,
    schema: TierSchema | None,
) -> FlowAssessment:
    """Evaluate ingest factors for mediator compliance."""

    normalized = _normalize_primes(factors or ())
    return _evaluate_sequence(normalized, schema, rule_code="flow.write.mediator")


def assess_enrichment_path(
    ref_prime: int | Mapping[str, object],
    deltas: Sequence[Mapping[str, object] | int] | None,
    schema: TierSchema | None,
) -> FlowAssessment:
    """Verify enrichment payloads respect mediator requirements."""

    ref_value = ref_prime.get("prime") if isinstance(ref_prime, Mapping) else ref_prime
    try:
        ref_int = int(ref_value)
    except (TypeError, ValueError):
        ref_int = None

    normalized = _normalize_primes(deltas or ())
    if ref_int is not None:
        normalized.insert(0, ref_int)
    return _evaluate_sequence(normalized, schema, rule_code="flow.enrich.mediator")


__all__ = [
    "FlowAssessment",
    "FlowViolation",
    "assess_enrichment_path",
    "assess_write_path",
]
