"""Flow validation utilities for prime write paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TierSchema = Mapping[int, Mapping[str, object]]

SHELL_ORDER: tuple[str, ...] = ("S", "A", "B", "C", "D", "E", "F")
SHELL_INDEX = {label: idx for idx, label in enumerate(SHELL_ORDER)}


def _base_tier_letter(tier: object) -> str | None:
    if not isinstance(tier, str):
        return None
    for char in tier.upper():
        if char.isalpha():
            return char
    return None


def _shell_index(prime: int, schema: TierSchema) -> tuple[int | None, bool]:
    meta = schema.get(prime, {}) if isinstance(schema, Mapping) else {}
    base = _base_tier_letter(meta.get("tier"))
    if not base:
        return None, False
    index = SHELL_INDEX.get(base)
    if index is None:
        return None, base == "C"
    return index, base == "C"


@dataclass(frozen=True)
class FlowViolation:
    """Represents a sequencing violation detected in a write path."""

    index: int
    prime: int | None
    previous_prime: int | None
    reason: str

    def asdict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "prime": self.prime,
            "previous_prime": self.previous_prime,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FlowAssessment:
    """Aggregates flow evaluation results for a given write path."""

    sequence: tuple[int, ...]
    violations: tuple[FlowViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def messages(self) -> list[str]:
        return [violation.reason for violation in self.violations]

    def asdict(self) -> dict[str, object]:
        return {
            "sequence": list(self.sequence),
            "violations": [violation.asdict() for violation in self.violations],
        }


def _evaluate_even_c_odd(sequence: Sequence[int], schema: TierSchema) -> list[FlowViolation]:
    if not sequence or not isinstance(schema, Mapping):
        return []

    violations: list[FlowViolation] = []
    last_non_conductor_index = -1
    last_conductor_index = -1
    previous_prime: int | None = None
    previous_parity: int | None = None

    for idx, prime in enumerate(sequence):
        shell_info = _shell_index(prime, schema)
        if not shell_info:
            previous_prime = prime
            previous_parity = None
            last_non_conductor_index = idx
            continue

        shell_index, is_conductor = shell_info
        if is_conductor:
            last_conductor_index = idx
            continue

        if shell_index is None:
            previous_prime = prime
            previous_parity = None
            last_non_conductor_index = idx
            continue

        parity = shell_index % 2
        if previous_parity is None:
            previous_parity = parity
            previous_prime = prime
            last_non_conductor_index = idx
            continue

        if parity != previous_parity and last_conductor_index <= last_non_conductor_index:
            from_label = "even-shell" if previous_parity == 0 else "odd-shell"
            to_label = "even-shell" if parity == 0 else "odd-shell"
            violations.append(
                FlowViolation(
                    index=idx,
                    prime=prime,
                    previous_prime=previous_prime,
                    reason=(
                        f"Transition from {from_label} prime {previous_prime} "
                        f"to {to_label} prime {prime} lacks C-tier mediation."
                    ),
                )
            )

        previous_parity = parity
        previous_prime = prime
        last_non_conductor_index = idx

    return violations


def _normalized_sequence(entries: Iterable[object]) -> tuple[int, ...]:
    sequence: list[int] = []
    for item in entries:
        if isinstance(item, Mapping):
            prime = item.get("prime")
        else:
            prime = item
        if isinstance(prime, int):
            sequence.append(prime)
    return tuple(sequence)


def assess_write_path(entries: Iterable[object], schema: TierSchema) -> FlowAssessment:
    sequence = _normalized_sequence(entries)
    violations = tuple(_evaluate_even_c_odd(sequence, schema))
    return FlowAssessment(sequence=sequence, violations=violations)


def assess_enrichment_path(
    ref_prime: int,
    deltas: Iterable[Mapping[str, object]] | None,
    schema: TierSchema,
) -> FlowAssessment:
    seed: list[int] = []
    if isinstance(ref_prime, int):
        seed.append(int(ref_prime))
    if deltas:
        seed.extend(_normalized_sequence(deltas))
    return assess_write_path(seed, schema)


__all__ = [
    "FlowAssessment",
    "FlowViolation",
    "assess_enrichment_path",
    "assess_write_path",
]

