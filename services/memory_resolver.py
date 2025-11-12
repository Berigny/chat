"""Utilities for contextual ledger memory retrieval.

This module centralises the heavier reasoning required to transform raw
ledger entries into helpful recall summaries. Streamlit surfaces can import
the :class:`MemoryResolver` and hand off query text, recent conversation
history, and the raw ledger payload; the resolver handles keyword expansion,
scoring, and summarisation so UI layers stay thin.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Sequence


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class MemorySentence:
    """Represents a ranked sentence extracted from a ledger entry."""

    text: str
    timestamp: int | None
    source: str | None
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class MemoryResolution:
    """Structured result returned by :class:`MemoryResolver`."""

    summary: str | None
    matched_terms: tuple[str, ...]
    focus_terms: tuple[str, ...]
    used_entries: int


class MemoryResolver:
    """Ranks ledger entries against a conversational query.

    Parameters
    ----------
    keyword_extractor:
        Callable that yields candidate keywords for a piece of text.
    text_normalizer:
        Callable responsible for stripping ledger boilerplate from entry text.
    filler_terms:
        Optional iterable of low-information keywords that should not appear
        in feedback messages (for example ``{"did", "talk"}``).
    """

    DEFAULT_FILLER_TERMS = {
        "about",
        "conversation",
        "day",
        "days",
        "did",
        "discuss",
        "discussed",
        "discussion",
        "have",
        "last",
        "ledger",
        "talk",
        "talked",
        "talking",
        "topic",
        "topics",
        "we",
        "what",
    }

    def __init__(
        self,
        keyword_extractor: Callable[[str], Iterable[str]],
        text_normalizer: Callable[[str], str],
        *,
        filler_terms: Iterable[str] | None = None,
    ) -> None:
        self._keyword_extractor = keyword_extractor
        self._text_normalizer = text_normalizer
        provided_fillers = {term.lower() for term in (filler_terms or [])}
        self._filler_terms = self.DEFAULT_FILLER_TERMS | provided_fillers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(
        self,
        query: str,
        history: Sequence[tuple[str, str]] | None,
        entries: Sequence[dict],
        *,
        keywords: Iterable[str] | None = None,
        limit: int = 5,
    ) -> MemoryResolution:
        """Return a ranked ledger summary for the provided query."""

        prepared_keywords = self._prepare_keywords(query, keywords, history)
        ranked_sentences = self._rank_sentences(entries, prepared_keywords, query)
        focus_terms = tuple(self._focus_terms(prepared_keywords))

        if not ranked_sentences:
            return MemoryResolution(
                summary=None,
                matched_terms=(),
                focus_terms=focus_terms,
                used_entries=0,
            )

        summary_lines: list[str] = []
        seen_sentences: set[str] = set()
        matched: set[str] = set()

        for sentence in ranked_sentences:
            normalized = self._normalize_sentence(sentence.text)
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            summary_lines.append(self._format_sentence(sentence))
            matched.update(sentence.matched_terms)
            if len(summary_lines) >= limit:
                break

        summary = "\n".join(summary_lines)
        if summary:
            summary = f"Here’s what the ledger currently recalls:\n{summary}"

        return MemoryResolution(
            summary=summary or None,
            matched_terms=tuple(sorted(matched)),
            focus_terms=focus_terms,
            used_entries=len(summary_lines),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_keywords(
        self,
        query: str,
        explicit_keywords: Iterable[str] | None,
        history: Sequence[tuple[str, str]] | None,
        *,
        max_terms: int = 16,
    ) -> list[str]:
        """Combine query keywords with relevant conversational context."""

        prepared: list[str] = []
        seen: set[str] = set()

        def _add_terms(terms: Iterable[str], *, cap: int | None = None) -> None:
            count = 0
            for term in terms:
                lowered = term.lower()
                if not lowered or lowered in seen:
                    continue
                seen.add(lowered)
                prepared.append(lowered)
                count += 1
                if cap is not None and count >= cap:
                    break

        if explicit_keywords:
            _add_terms(explicit_keywords)
        else:
            _add_terms(self._keyword_extractor(query))

        if history:
            for role, content in reversed(history):
                if len(prepared) >= max_terms:
                    break
                if role not in {"You", "Bot"}:
                    continue
                if not content:
                    continue
                cap = 6 if role == "You" else 3
                _add_terms(self._keyword_extractor(content), cap=cap)

        return prepared[:max_terms]

    def _rank_sentences(
        self,
        entries: Sequence[dict],
        keywords: Sequence[str],
        query: str,
    ) -> list[MemorySentence]:
        sentences: list[MemorySentence] = []
        normalized_keywords = [kw.lower() for kw in keywords if kw]
        keyword_set = set(normalized_keywords)
        now_ms = int(time.time() * 1000)

        for entry in entries:
            text = self._text_normalizer((entry.get("text") or "").strip())
            if not text:
                continue

            timestamp = self._coerce_timestamp(entry.get("timestamp"))
            source = self._extract_source(entry)

            lowered = text.lower()
            keyword_score = sum(lowered.count(term) for term in normalized_keywords)
            recency_bonus = 0.0
            if timestamp is not None:
                age_hours = max(1.0, (now_ms - timestamp) / 3600000)
                recency_bonus = 1.0 / age_hours

            base_score = keyword_score + recency_bonus
            if self._is_definition_query(query) and self._looks_like_definition(text):
                base_score += 1.5

            segments = self._split_sentences(text)
            if not segments:
                segments = [text]

            for segment in segments:
                sentence = segment.strip()
                if not sentence:
                    continue

                lowered_sentence = sentence.lower()
                matched_terms = tuple(sorted({term for term in keyword_set if term in lowered_sentence}))
                match_bonus = sum(lowered_sentence.count(term) for term in matched_terms)
                score = base_score + match_bonus
                if self._looks_like_definition(sentence):
                    score += 0.75
                if not matched_terms and not normalized_keywords:
                    score += 0.5

                sentences.append(
                    MemorySentence(
                        text=sentence,
                        timestamp=timestamp,
                        source=source,
                        score=score,
                        matched_terms=matched_terms,
                    )
                )

        sentences.sort(key=lambda item: (item.score, item.timestamp or 0), reverse=True)
        return sentences

    def _focus_terms(self, keywords: Sequence[str]) -> list[str]:
        focus = [term for term in keywords if term not in self._filler_terms]
        return focus[:5]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return _SENTENCE_SPLIT_PATTERN.split(text)

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        return re.sub(r"\s+", " ", sentence.strip().lower())

    @staticmethod
    def _coerce_timestamp(candidate) -> int | None:
        if isinstance(candidate, (int, float)):
            return int(candidate)
        if isinstance(candidate, str):
            try:
                return int(candidate)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_source(entry: dict) -> str | None:
        meta = entry.get("meta") if isinstance(entry, dict) else None
        if isinstance(meta, dict):
            source = meta.get("source") or meta.get("name")
            if isinstance(source, str) and source.strip():
                return source.strip()
        for key in ("name", "attachment"):
            candidate = entry.get(key) if isinstance(entry, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @staticmethod
    def _format_sentence(sentence: MemorySentence) -> str:
        prefix_parts: list[str] = []
        if sentence.timestamp:
            dt = datetime.fromtimestamp(sentence.timestamp / 1000)
            prefix_parts.append(dt.strftime("%Y-%m-%d %H:%M"))
        if sentence.source:
            prefix_parts.append(sentence.source)
        prefix = f"[{' • '.join(prefix_parts)}] " if prefix_parts else ""
        return f"- {prefix}{sentence.text.strip()}"

    @staticmethod
    def _is_definition_query(query: str) -> bool:
        lowered = (query or "").lower()
        definition_hints = (
            "define",
            "definition",
            "meaning",
            "what is",
            "who is",
            "describe",
        )
        return any(hint in lowered for hint in definition_hints)

    @staticmethod
    def _looks_like_definition(text: str) -> bool:
        lowered = text.lower()
        return (
            " is " in lowered
            or " refers to" in lowered
            or " means " in lowered
            or " defined" in lowered
            or ":" in text
        )


__all__ = ["MemoryResolver", "MemoryResolution", "MemorySentence"]

