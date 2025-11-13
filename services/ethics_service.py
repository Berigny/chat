"""Evaluate enrichment ethics metrics derived from ledger snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from validators import validate_prime_sequence


_DEFAULT_HARMFUL_TERMS = {
    "attack",
    "abuse",
    "exploit",
    "harm",
    "kill",
    "self-harm",
}


@dataclass(frozen=True)
class EthicsAssessment:
    """Container for ethics scores and accompanying reviewer notes."""

    lawfulness: float
    evidence: float
    non_harm: float
    coherence: float
    notes: tuple[str, ...]

    def asdict(self) -> dict[str, Any]:
        return {
            "lawfulness": self.lawfulness,
            "evidence": self.evidence,
            "non_harm": self.non_harm,
            "coherence": self.coherence,
            "notes": list(self.notes),
        }


class EthicsService:
    """Computes heuristic ethics scores for enrichment responses."""

    def __init__(
        self,
        *,
        schema: Mapping[int, Mapping[str, Any]] | None = None,
        harmful_terms: Iterable[str] | None = None,
    ) -> None:
        self._schema = schema or {}
        terms = {term.lower() for term in (harmful_terms or _DEFAULT_HARMFUL_TERMS)}
        self._harmful_terms = {term for term in terms if term}

    def evaluate(
        self,
        ledger_snapshot: Mapping[str, Any] | None,
        *,
        deltas: Sequence[Mapping[str, Any]] | None = None,
        minted_bodies: Sequence[Mapping[str, Any]] | None = None,
    ) -> EthicsAssessment:
        """Return an ethics assessment for the supplied enrichment output."""

        notes: list[str] = []
        lawfulness = self._lawfulness_score(ledger_snapshot, deltas, notes)
        evidence = self._evidence_score(minted_bodies, notes)
        non_harm = self._non_harm_score(minted_bodies, notes)
        coherence = self._coherence_score(deltas, minted_bodies, notes)
        return EthicsAssessment(
            lawfulness=round(lawfulness, 3),
            evidence=round(evidence, 3),
            non_harm=round(non_harm, 3),
            coherence=round(coherence, 3),
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------
    def _lawfulness_score(
        self,
        ledger_snapshot: Mapping[str, Any] | None,
        deltas: Sequence[Mapping[str, Any]] | None,
        notes: list[str],
    ) -> float:
        sequence: list[Mapping[str, Any]] = []
        if ledger_snapshot:
            factors = ledger_snapshot.get("factors")
            if isinstance(factors, Sequence):
                for entry in factors:
                    if isinstance(entry, Mapping):
                        sequence.append(entry)
        if deltas:
            for entry in deltas:
                if isinstance(entry, Mapping):
                    sequence.append(entry)

        if not sequence:
            notes.append("No ledger factors available; defaulting to cautious lawfulness score.")
            return 0.6

        if not validate_prime_sequence(sequence, self._schema):
            notes.append("Prime flow check flagged a tier escalation; review required.")
            return 0.35

        notes.append("Prime flow checks passed.")
        return 0.9

    def _evidence_score(
        self,
        minted_bodies: Sequence[Mapping[str, Any]] | None,
        notes: list[str],
    ) -> float:
        if not minted_bodies:
            notes.append("No new body primes minted; limited evidence captured.")
            return 0.4

        clean_chunks = [
            (entry.get("body") or "").strip()
            for entry in minted_bodies
            if isinstance(entry, Mapping) and isinstance(entry.get("body"), str)
        ]
        clean_chunks = [chunk for chunk in clean_chunks if chunk]
        if not clean_chunks:
            notes.append("Minted bodies lacked usable text; evidence score reduced.")
            return 0.5

        notes.append(f"Minted {len(clean_chunks)} body prime(s) as evidence.")
        return min(1.0, 0.5 + 0.2 * len(clean_chunks))

    def _non_harm_score(
        self,
        minted_bodies: Sequence[Mapping[str, Any]] | None,
        notes: list[str],
    ) -> float:
        if not minted_bodies:
            return 0.9

        harmful_hits = 0
        for entry in minted_bodies:
            if not isinstance(entry, Mapping):
                continue
            body = entry.get("body")
            if not isinstance(body, str):
                continue
            text = body.lower()
            if any(term in text for term in self._harmful_terms):
                harmful_hits += 1

        if harmful_hits:
            notes.append(f"Detected {harmful_hits} potential harmful snippet(s) during enrichment.")
            return max(0.1, 0.9 - 0.25 * harmful_hits)

        notes.append("No harmful language detected in minted bodies.")
        return 0.95

    def _coherence_score(
        self,
        deltas: Sequence[Mapping[str, Any]] | None,
        minted_bodies: Sequence[Mapping[str, Any]] | None,
        notes: list[str],
    ) -> float:
        primes = {
            int(entry.get("prime"))
            for entry in (deltas or [])
            if isinstance(entry, Mapping) and isinstance(entry.get("prime"), int)
        }
        body_count = len(minted_bodies or [])
        if not primes:
            notes.append("No prime deltas supplied; coherence derived from body coverage only.")
            return 0.55 if body_count else 0.4

        if not body_count:
            notes.append("Prime deltas supplied without corresponding body content.")
            return 0.45

        notes.append(
            "Prime deltas aligned with body updates; enrichment remains coherent."
        )
        return min(1.0, 0.6 + 0.1 * min(len(primes), body_count))


__all__ = ["EthicsService", "EthicsAssessment"]

