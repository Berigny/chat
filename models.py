"""Shared dataclasses for Streamlit dashboards and API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LedgerEntry:
    """Represents a stored ledger entry returned by the v2 API."""

    entry_id: str
    entity: str
    text: str | None = None
    factors: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LedgerEntry":
        entry_id = str(payload.get("id") or payload.get("entry_id") or "")
        entity = str(payload.get("entity") or "")
        text = payload.get("text") if isinstance(payload.get("text"), str) else None
        factors_field = payload.get("factors") or payload.get("entries") or []
        factors: list[Mapping[str, Any]] = []
        if isinstance(factors_field, Sequence):
            factors = [entry for entry in factors_field if isinstance(entry, Mapping)]
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        return cls(
            entry_id=entry_id,
            entity=entity,
            text=text,
            factors=tuple(factors),
            metadata=metadata,
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entity": self.entity,
            "text": self.text,
            "factors": [dict(entry) for entry in self.factors],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Strain:
    """Represents an individual coherence strain and its score."""

    name: str
    score: float
    weight: float | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Strain":
        name = str(payload.get("name") or payload.get("label") or "strain")
        score_raw = payload.get("score") if payload.get("score") is not None else payload.get("value")
        score = float(score_raw) if isinstance(score_raw, (int, float, str)) else 0.0
        weight_raw = payload.get("weight")
        weight = float(weight_raw) if isinstance(weight_raw, (int, float, str)) else None
        notes_field = payload.get("notes") or payload.get("reason")
        if isinstance(notes_field, Sequence) and not isinstance(notes_field, str):
            notes = "; ".join(str(item) for item in notes_field)
        else:
            notes = str(notes_field) if notes_field is not None else None
        return cls(name=name, score=score, weight=weight, notes=notes)


@dataclass(frozen=True)
class Score:
    """Generic score container for evaluation endpoints."""

    name: str
    value: float
    rationale: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, default_name: str = "score") -> "Score":
        name = str(payload.get("name") or payload.get("label") or default_name)
        raw_value = payload.get("value") if payload.get("value") is not None else payload.get("score")
        value = float(raw_value) if isinstance(raw_value, (int, float, str)) else 0.0
        rationale_field = payload.get("rationale") or payload.get("notes")
        if isinstance(rationale_field, Sequence) and not isinstance(rationale_field, str):
            rationale = "; ".join(str(item) for item in rationale_field)
        else:
            rationale = str(rationale_field) if rationale_field is not None else None
        return cls(name=name, value=value, rationale=rationale)


@dataclass(frozen=True)
class CoherenceResponse:
    """Structured response for ``/coherence/evaluate``."""

    score: Score
    strains: tuple[Strain, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoherenceResponse":
        score_payload: Mapping[str, Any] = {}
        if isinstance(payload.get("score"), Mapping):
            score_payload = payload.get("score")  # type: ignore[assignment]
        else:
            score_payload = {"value": payload.get("coherence") or payload.get("score")}
        score = Score.from_dict(score_payload, default_name="coherence")

        strains_payload = payload.get("strains") or payload.get("components") or []
        strains: list[Strain] = []
        if isinstance(strains_payload, Sequence):
            strains = [Strain.from_dict(item) for item in strains_payload if isinstance(item, Mapping)]

        notes_field = payload.get("notes") or payload.get("messages") or []
        if isinstance(notes_field, Sequence) and not isinstance(notes_field, str):
            notes = tuple(str(item) for item in notes_field)
        elif notes_field is None:
            notes = tuple()
        else:
            notes = (str(notes_field),)

        return cls(score=score, strains=tuple(strains), notes=notes, raw=payload)


@dataclass(frozen=True)
class PolicyDecisionResponse:
    """Structured policy decision returned by ``/ethics/evaluate``."""

    decision: str
    scores: tuple[Score, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyDecisionResponse":
        decision = str(payload.get("decision") or payload.get("policy") or "review")
        scores_payload = payload.get("scores") or payload.get("assessments") or []
        scores: list[Score] = []
        if isinstance(scores_payload, Sequence):
            scores = [Score.from_dict(item) for item in scores_payload if isinstance(item, Mapping)]

        notes_field = payload.get("notes") or payload.get("messages") or payload.get("rationale") or []
        if isinstance(notes_field, Sequence) and not isinstance(notes_field, str):
            notes = tuple(str(item) for item in notes_field)
        elif notes_field is None:
            notes = tuple()
        else:
            notes = (str(notes_field),)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        return cls(
            decision=decision,
            scores=tuple(scores),
            notes=notes,
            metadata=metadata,
            raw=payload,
        )


__all__ = [
    "LedgerEntry",
    "Strain",
    "CoherenceResponse",
    "Score",
    "PolicyDecisionResponse",
]
